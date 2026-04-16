#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gpytorch
except ImportError as e:
    raise ImportError(
        "gpytorch is required for FullOutputDKLRegressor. "
        "Please install gpytorch before using this model."
    ) from e

from ofml_alfn.models.dkl_feature_extractors import make_mlp_feature_extractor
from ofml_alfn.training.train_protocol_predictor import (
    ProtocolEvaluationResult,
    ProtocolTrainingConfig,
    ProtocolTrainingResult,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


def _as_float_tensor(
    x: Any,
    *,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(x):
        out = x.detach().to(dtype=torch.float32, device=device)
    else:
        out = torch.as_tensor(x, dtype=torch.float32, device=device)
    return out


def _stack_rows(
    xs: Sequence[Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for x in xs:
        t = _as_float_tensor(x, device=device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        elif t.ndim > 1:
            t = t.reshape(-1)
        rows.append(t)
    return torch.stack(rows, dim=0)


def _extract_condition_x(
    sample: BenchmarkSample,
    *,
    protocol: ProtocolSpec,
    device: torch.device,
) -> torch.Tensor:
    if "x" in sample.metadata:
        return _as_float_tensor(sample.metadata["x"], device=device).reshape(-1)

    return torch.tensor(
        [float(sample.condition.values[k]) for k in protocol.condition_keys],
        dtype=torch.float32,
        device=device,
    )


def _extract_target_y(
    sample: BenchmarkSample,
    *,
    device: torch.device,
) -> torch.Tensor:
    y = _as_float_tensor(sample.target_value, device=device)
    if y.ndim == 0:
        y = y.unsqueeze(0)
    elif y.ndim > 1:
        y = y.reshape(-1)
    return y


class _ExactDKLGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        feature_extractor: nn.Module,
        feature_dim: int,
        kernel_type: str = "rbf",
    ):
        super().__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor
        self.feature_dim = int(feature_dim)
        self.kernel_type = str(kernel_type).lower()

        self.mean_module = gpytorch.means.ConstantMean()
        if self.kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=self.feature_dim
            )
        elif self.kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=self.feature_dim,
            )
        else:
            raise ValueError(
                f"Unsupported kernel_type: {kernel_type}. Use 'rbf' or 'matern'."
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        phi = self.feature_extractor(x)
        mean_x = self.mean_module(phi)
        covar_x = self.covar_module(phi)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _ScalarOutputDKL(nn.Module):
    """
    One scalar-output DKL regressor.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden: int = 256,
        depth: int = 2,
        feature_dim: int = 32,
        kernel_type: str = "rbf",
        noise_constraint: Optional[gpytorch.constraints.Interval] = None,
    ) -> None:
        super().__init__()

        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)

        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.feature_dim = int(feature_dim)
        self.kernel_type = str(kernel_type).lower()

        self.feature_extractor = make_mlp_feature_extractor(
            in_dim=self.input_dim,
            hidden=self.hidden,
            depth=self.depth,
            out_dim=self.feature_dim,
            activation="relu",
            output_activation="identity",
            dropout=0.0,
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        )

        dummy_x = torch.zeros(1, self.input_dim, dtype=torch.get_default_dtype())
        dummy_y = torch.zeros(1, dtype=torch.get_default_dtype())
        self.gp = _ExactDKLGP(
            train_x=dummy_x,
            train_y=dummy_y,
            likelihood=self.likelihood,
            feature_extractor=self.feature_extractor,
            feature_dim=self.feature_dim,
            kernel_type=self.kernel_type,
        )

        self._has_real_train_data = False

    @property
    def has_real_train_data(self) -> bool:
        return bool(self._has_real_train_data)

    def set_train_data(self, x: torch.Tensor, y: torch.Tensor, strict: bool = False) -> None:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.input_dim}], got {tuple(x.shape)}"
            )

        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(f"Expected y of shape [N, 1] or [N], got {tuple(y.shape)}")
            y = y[:, 0]
        elif y.ndim != 1:
            raise ValueError(f"Expected y of shape [N, 1] or [N], got {tuple(y.shape)}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have same batch size, got {x.shape[0]} and {y.shape[0]}"
            )

        self.gp.set_train_data(inputs=x, targets=y, strict=strict)
        self._has_real_train_data = True

    def fit_exact_gp(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        n_steps: int,
        lr: float,
        weight_decay: float,
        grad_clip_norm: Optional[float] = None,
        verbose: bool = False,
        prefix: str = "",
    ) -> Dict[str, float]:
        self.set_train_data(x=x, y=y, strict=False)

        self.gp.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            list(self.gp.parameters()) + list(self.likelihood.parameters()),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

        last_loss = float("nan")
        for step in range(1, int(n_steps) + 1):
            optimizer.zero_grad()
            output = self.gp(self.gp.train_inputs[0])
            loss = -mll(output, self.gp.train_targets)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.gp.parameters()) + list(self.likelihood.parameters()),
                    max_norm=float(grad_clip_norm),
                )

            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

            if verbose and (step == 1 or step == int(n_steps)):
                tag = f"[{prefix}] " if prefix else ""
                print(f"{tag}step={step} exact_mll_loss={last_loss:.6f}")

        return {"final_exact_mll_loss": float(last_loss)}

    @torch.no_grad()
    def predict_mean_var(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._has_real_train_data:
            raise RuntimeError("No real training data has been set.")

        self.gp.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            posterior = self.gp(x)
            mean = posterior.mean.unsqueeze(-1)
            var = posterior.variance.unsqueeze(-1)
        return mean, var

    @torch.no_grad()
    def sample_observation(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if not self._has_real_train_data:
            raise RuntimeError("No real training data has been set.")

        self.gp.eval()
        self.likelihood.eval()
        posterior_y = self.likelihood(self.gp(x))
        samples = posterior_y.rsample(torch.Size([n_samples]))  # [S, N]
        return samples.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_mean_var(x)
        return mean


class FullOutputDKLRegressor(nn.Module):
    """
    Exact DKL regressor for full outputs.

    This is a vector-output wrapper built from independent scalar DKL heads.
    It is useful when the target is a full output vector rather than a node wise
    function network.

    Hooks are provided so train_protocol_predictor.py can delegate to this class.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden: int = 256,
        depth: int = 2,
        feature_dim: int = 32,
        kernel_type: str = "rbf",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.feature_dim = int(feature_dim)
        self.kernel_type = str(kernel_type).lower()
        self.predictor_type = "dkl"

        self.output_models = nn.ModuleList(
            [
                _ScalarOutputDKL(
                    input_dim=self.input_dim,
                    hidden=self.hidden,
                    depth=self.depth,
                    feature_dim=self.feature_dim,
                    kernel_type=self.kernel_type,
                )
                for _ in range(self.output_dim)
            ]
        )

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward_target(
        self,
        *,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
    ) -> torch.Tensor:
        del protocol
        means = []
        for model in self.output_models:
            means.append(model(condition_x))
        return torch.cat(means, dim=-1)

    def forward_protocol(
        self,
        *,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_target(protocol=protocol, condition_x=condition_x)

    @torch.no_grad()
    def sample_protocol_fantasy_targets(
        self,
        *,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
        n_fantasies: int,
    ) -> torch.Tensor:
        del protocol
        if n_fantasies <= 0:
            raise ValueError(f"n_fantasies must be positive, got {n_fantasies}")

        samples_per_output = [
            model.sample_observation(condition_x, n_samples=n_fantasies)
            for model in self.output_models
        ]
        return torch.cat(samples_per_output, dim=-1)

    def fit_protocol_dataset(
        self,
        *,
        protocols: Mapping[str, ProtocolSpec],
        train_samples: Sequence[BenchmarkSample],
        val_samples: Optional[Sequence[BenchmarkSample]],
        config: ProtocolTrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> ProtocolTrainingResult:
        del optimizer

        if len(train_samples) == 0:
            raise ValueError("train_samples must be non-empty")

        device = self._device()

        grouped: Dict[str, List[BenchmarkSample]] = {}
        for sample in train_samples:
            grouped.setdefault(sample.protocol_id, []).append(sample)

        train_x_rows: List[torch.Tensor] = []
        train_y_rows: List[torch.Tensor] = []

        for protocol_id, samples_this_protocol in grouped.items():
            protocol = protocols[protocol_id]
            for sample in samples_this_protocol:
                train_x_rows.append(
                    _extract_condition_x(sample, protocol=protocol, device=device)
                )
                train_y_rows.append(
                    _extract_target_y(sample, device=device)
                )

        train_x = torch.stack(train_x_rows, dim=0)
        train_y = torch.stack(train_y_rows, dim=0)

        if train_y.shape[1] != self.output_dim:
            raise ValueError(
                f"Target dim {train_y.shape[1]} does not match output_dim={self.output_dim}"
            )

        history: List[Dict[str, float]] = []
        for j, model in enumerate(self.output_models):
            metrics = model.fit_exact_gp(
                x=train_x,
                y=train_y[:, j:j+1],
                n_steps=int(config.n_steps),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                grad_clip_norm=config.grad_clip_norm,
                verbose=bool(config.verbose),
                prefix=f"full_output_{j}",
            )
            history.append({"stage": f"output_{j}", **metrics})

        final_train_eval = self.evaluate_protocol_dataset(
            protocols=protocols,
            samples=train_samples,
            config=config,
        )
        final_train_loss = float(final_train_eval.loss)

        final_val_loss: Optional[float] = None
        best_val_loss: Optional[float] = None
        if val_samples is not None and len(val_samples) > 0:
            val_eval = self.evaluate_protocol_dataset(
                protocols=protocols,
                samples=val_samples,
                config=config,
            )
            final_val_loss = float(val_eval.loss)
            best_val_loss = final_val_loss

        return ProtocolTrainingResult(
            optimizer=None,
            history=history,
            best_step=int(config.n_steps),
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_state_dict=None,
        )

    @torch.no_grad()
    def evaluate_protocol_dataset(
        self,
        *,
        protocols: Mapping[str, ProtocolSpec],
        samples: Sequence[BenchmarkSample],
        config: ProtocolTrainingConfig,
    ) -> ProtocolEvaluationResult:
        if len(samples) == 0:
            return ProtocolEvaluationResult(
                loss=float("nan"),
                loss_by_protocol={},
                n_by_protocol={},
                n_total=0,
            )

        device = self._device()
        grouped: Dict[str, List[BenchmarkSample]] = {}
        for sample in samples:
            grouped.setdefault(sample.protocol_id, []).append(sample)

        loss_by_protocol: Dict[str, float] = {}
        n_by_protocol: Dict[str, int] = {}
        total_weighted_loss = 0.0
        total_count = 0

        for protocol_id, samples_this_protocol in grouped.items():
            protocol = protocols[protocol_id]

            x = _stack_rows(
                [
                    _extract_condition_x(s, protocol=protocol, device=device)
                    for s in samples_this_protocol
                ],
                device=device,
            )
            y = _stack_rows(
                [_extract_target_y(s, device=device) for s in samples_this_protocol],
                device=device,
            )

            pred = self.forward_target(protocol=protocol, condition_x=x)

            if pred.shape != y.shape:
                if pred.numel() == y.numel():
                    pred = pred.view_as(y)
                else:
                    raise ValueError(
                        f"Prediction shape {tuple(pred.shape)} does not match target shape "
                        f"{tuple(y.shape)} for protocol {protocol_id!r}"
                    )

            if config.loss_name == "mse":
                loss_this = F.mse_loss(pred, y)
            elif config.loss_name == "l1":
                loss_this = F.l1_loss(pred, y)
            elif config.loss_name == "smooth_l1":
                loss_this = F.smooth_l1_loss(pred, y)
            else:
                raise ValueError(f"Unsupported loss_name: {config.loss_name!r}")

            n_this = len(samples_this_protocol)
            loss_val = float(loss_this.detach().cpu().item())

            loss_by_protocol[protocol_id] = loss_val
            n_by_protocol[protocol_id] = n_this
            total_weighted_loss += n_this * loss_val
            total_count += n_this

        total_loss = total_weighted_loss / max(total_count, 1)
        return ProtocolEvaluationResult(
            loss=float(total_loss),
            loss_by_protocol=loss_by_protocol,
            n_by_protocol=n_by_protocol,
            n_total=total_count,
        )


__all__ = [
    "FullOutputDKLRegressor",
]