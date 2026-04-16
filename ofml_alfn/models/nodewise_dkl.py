#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gpytorch
except ImportError as e:
    raise ImportError(
        "gpytorch is required for MultiHeadNodewiseDKL. "
        "Please install gpytorch before using this model."
    ) from e

from ofml_alfn.models.full_output_dkl import FullOutputDKLRegressor
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


def _extract_sample_x(
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


def _extract_sample_z(
    sample: BenchmarkSample,
    *,
    device: torch.device,
) -> torch.Tensor:
    if "z" not in sample.metadata:
        raise KeyError(
            "Problem 1A DKL fitting expects sample.metadata['z'] to exist."
        )
    return _as_float_tensor(sample.metadata["z"], device=device).reshape(-1)


def _extract_sample_y(
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


class _NodeFeatureExtractor(nn.Module):
    """
    Per-node feature extractor: x_node -> phi_node
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        feature_dim: int = 32,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.feature_dim = int(feature_dim)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.in_dim}], got {tuple(x.shape)}"
            )
        return self.net(x)


class _NodeExactDKLGP(gpytorch.models.ExactGP):
    """
    Exact GP on top of a learned feature extractor.
    """

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
        self.kernel_type = kernel_type.lower()

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
                f"Unsupported kernel_type: {kernel_type}. "
                "Use 'rbf' or 'matern'."
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        phi = self.feature_extractor(x)
        mean_x = self.mean_module(phi)
        covar_x = self.covar_module(phi)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NodewiseDKLRegressor(nn.Module):
    """
    One DKL regressor for one scalar output.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        feature_dim: int = 32,
        kernel_type: str = "rbf",
        noise_constraint: Optional[gpytorch.constraints.Interval] = None,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.feature_dim = int(feature_dim)
        self.kernel_type = str(kernel_type).lower()

        self.feature_extractor = _NodeFeatureExtractor(
            in_dim=self.in_dim,
            hidden=self.hidden,
            feature_dim=self.feature_dim,
        )

        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        )

        dummy_x = torch.zeros(1, self.in_dim, dtype=torch.get_default_dtype())
        dummy_y = torch.zeros(1, dtype=torch.get_default_dtype())
        self.gp = _NodeExactDKLGP(
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

    def set_train_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        strict: bool = False,
    ) -> None:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.in_dim}], got {tuple(x.shape)}"
            )

        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(
                    f"Expected y of shape [N, 1] or [N], got {tuple(y.shape)}"
                )
            y_flat = y[:, 0]
        elif y.ndim == 1:
            y_flat = y
        else:
            raise ValueError(
                f"Expected y of shape [N, 1] or [N], got {tuple(y.shape)}"
            )

        if x.shape[0] != y_flat.shape[0]:
            raise ValueError(
                f"x and y must have the same batch size, got "
                f"{x.shape[0]} and {y_flat.shape[0]}"
            )

        self.gp.set_train_data(inputs=x, targets=y_flat, strict=strict)
        self._has_real_train_data = True

    def marginal_log_likelihood(self) -> gpytorch.mlls.ExactMarginalLogLikelihood:
        return gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self.gp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_mean_var(x)
        return mean

    @torch.no_grad()
    def predict_mean_var(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._has_real_train_data:
            raise RuntimeError(
                "This node DKL regressor has no real training data yet. "
                "Call set_train_data(...) before prediction."
            )

        self.gp.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            posterior = self.gp(x)
            mean = posterior.mean.unsqueeze(-1)
            var = posterior.variance.unsqueeze(-1)
        return mean, var

    @torch.no_grad()
    def sample_latent(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if not self._has_real_train_data:
            raise RuntimeError(
                "This node DKL regressor has no real training data yet. "
                "Call set_train_data(...) before sampling."
            )

        self.gp.eval()
        self.likelihood.eval()
        posterior = self.gp(x)
        samples = posterior.rsample(torch.Size([n_samples]))  # [S, N]
        return samples.unsqueeze(-1)

    @torch.no_grad()
    def sample_observation(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if not self._has_real_train_data:
            raise RuntimeError(
                "This node DKL regressor has no real training data yet. "
                "Call set_train_data(...) before sampling."
            )

        self.gp.eval()
        self.likelihood.eval()
        posterior_y = self.likelihood(self.gp(x))
        samples = posterior_y.rsample(torch.Size([n_samples]))  # [S, N]
        return samples.unsqueeze(-1)

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
        mll = self.marginal_log_likelihood()

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

            if verbose and (step == 1 or step == n_steps):
                tag = f"[{prefix}] " if prefix else ""
                print(f"{tag}step={step} exact_mll_loss={last_loss:.6f}")

        return {"final_exact_mll_loss": float(last_loss)}


class MultiHeadNodewiseDKL(nn.Module):
    """
    Problem 1A oriented DKL predictor.

    Structure
    ---------
    - shared upstream: x -> z  (vector output)
    - protocol-specific sink: z -> y  (scalar output)

    Notes
    -----
    This class is intentionally specialized for the current Problem 1A path.
    """

    def __init__(
        self,
        *,
        external_input_dim: int,
        node_input_dims: Sequence[int],
        parent_nodes: Optional[Sequence[Sequence[int]]] = None,
        active_input_indices: Optional[Sequence[Sequence[int]]] = None,
        hidden: int = 256,
        depth: int = 2,
        feature_dim: int = 32,
        kernel_type: str = "rbf",
        sink_idx: Optional[int] = None,
    ):
        super().__init__()

        if external_input_dim <= 0:
            raise ValueError(
                f"external_input_dim must be positive, got {external_input_dim}"
            )

        self.external_input_dim = int(external_input_dim)
        self.node_input_dims = [int(d) for d in node_input_dims]
        self.n_nodes = len(self.node_input_dims)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.feature_dim = int(feature_dim)
        self.kernel_type = str(kernel_type).lower()
        self.predictor_type = "dkl"

        if self.n_nodes <= 0:
            raise ValueError("node_input_dims must contain at least one node.")

        self.sink_idx = (self.n_nodes - 1) if sink_idx is None else int(sink_idx)
        if not (0 <= self.sink_idx < self.n_nodes):
            raise ValueError(
                f"sink_idx must be in [0, {self.n_nodes - 1}], got {self.sink_idx}"
            )

        # Kept only for compatibility; Problem 1A hooks below do not rely on them.
        self.parent_nodes = (
            [list(p) for p in parent_nodes] if parent_nodes is not None else None
        )
        self.active_input_indices = (
            [list(a) for a in active_input_indices]
            if active_input_indices is not None
            else None
        )

        # Shared upstream x -> z (vector output).
        # node_input_dims[sink_idx] is the latent z dimension.
        self.shared_upstream_output_dim = int(self.node_input_dims[self.sink_idx])
        self.shared_upstream_model = FullOutputDKLRegressor(
            input_dim=self.external_input_dim,
            output_dim=self.shared_upstream_output_dim,
            hidden=self.hidden,
            depth=self.depth,
            feature_dim=self.feature_dim,
            kernel_type=self.kernel_type,
        )

        # Protocol-specific observer heads z -> y.
        self.protocol_sink_models = nn.ModuleDict()

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _ensure_protocol_sink_model(
        self,
        protocol_id: str,
    ) -> NodewiseDKLRegressor:
        if protocol_id not in self.protocol_sink_models:
            self.protocol_sink_models[protocol_id] = NodewiseDKLRegressor(
                in_dim=self.shared_upstream_output_dim,
                hidden=self.hidden,
                feature_dim=self.feature_dim,
                kernel_type=self.kernel_type,
            )
        return self.protocol_sink_models[protocol_id]

    def _problem1a_fit_datasets(
        self,
        *,
        protocols: Mapping[str, ProtocolSpec],
        train_samples: Sequence[BenchmarkSample],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        if len(train_samples) == 0:
            raise ValueError("train_samples must be non-empty")

        upstream_x_rows: List[torch.Tensor] = []
        upstream_z_rows: List[torch.Tensor] = []
        sink_data: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

        for sample in train_samples:
            if sample.protocol_id not in protocols:
                raise KeyError(
                    f"Unknown protocol_id in train_samples: {sample.protocol_id!r}"
                )

            protocol = protocols[sample.protocol_id]
            x = _extract_sample_x(sample, protocol=protocol, device=device)
            z = _extract_sample_z(sample, device=device)
            y = _extract_sample_y(sample, device=device)

            upstream_x_rows.append(x)
            upstream_z_rows.append(z)

            if sample.protocol_id not in sink_data:
                sink_data[sample.protocol_id] = ([], [])
            sink_data[sample.protocol_id][0].append(z)
            sink_data[sample.protocol_id][1].append(y)

        upstream_x = torch.stack(upstream_x_rows, dim=0)
        upstream_z = torch.stack(upstream_z_rows, dim=0)

        sink_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for protocol_id, (zs, ys) in sink_data.items():
            sink_tensors[protocol_id] = (
                torch.stack(zs, dim=0),
                torch.stack(ys, dim=0),
            )

        return upstream_x, upstream_z, sink_tensors

    def forward_target(
        self,
        *,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
    ) -> torch.Tensor:
        if condition_x.ndim != 2 or condition_x.shape[1] != self.external_input_dim:
            raise ValueError(
                f"Expected condition_x of shape [N, {self.external_input_dim}], "
                f"got {tuple(condition_x.shape)}"
            )

        if protocol.protocol_id not in self.protocol_sink_models:
            raise RuntimeError(
                f"No protocol-specific sink model is available for protocol "
                f"{protocol.protocol_id!r}. Fit the predictor on protocol "
                f"samples first."
            )

        z_mean = self.shared_upstream_model.forward_target(
            protocol=protocol,
            condition_x=condition_x,
        )
        sink_model = self.protocol_sink_models[protocol.protocol_id]
        return sink_model(z_mean)

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
        if n_fantasies <= 0:
            raise ValueError(f"n_fantasies must be positive, got {n_fantasies}")
        if protocol.protocol_id not in self.protocol_sink_models:
            raise RuntimeError(
                f"No protocol-specific sink model is available for protocol "
                f"{protocol.protocol_id!r}. Fit the predictor on protocol "
                f"samples first."
            )

        z_samples = self.shared_upstream_model.sample_protocol_fantasy_targets(
            protocol=protocol,
            condition_x=condition_x,
            n_fantasies=n_fantasies,
        )  # [S, N, Dz]

        sink_model = self.protocol_sink_models[protocol.protocol_id]
        y_samples: List[torch.Tensor] = []
        for s in range(n_fantasies):
            z_s = z_samples[s]              # [N, Dz]
            y_s = sink_model.sample_observation(z_s, n_samples=1)[0]  # [N, 1]
            y_samples.append(y_s)

        return torch.stack(y_samples, dim=0)  # [S, N, 1]

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
        self.eval()

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
                    _extract_sample_x(s, protocol=protocol, device=device)
                    for s in samples_this_protocol
                ],
                device=device,
            )
            y = _stack_rows(
                [_extract_sample_y(s, device=device) for s in samples_this_protocol],
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
        protocol_map = dict(protocols)

        upstream_x, upstream_z, sink_tensors = self._problem1a_fit_datasets(
            protocols=protocol_map,
            train_samples=train_samples,
            device=device,
        )

        history: List[Dict[str, float]] = []

        # shared upstream: x -> z (vector output)
        for d, model in enumerate(self.shared_upstream_model.output_models):
            metrics = model.fit_exact_gp(
                x=upstream_x,
                y=upstream_z[:, d : d + 1],
                n_steps=int(config.n_steps),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                grad_clip_norm=config.grad_clip_norm,
                verbose=bool(config.verbose),
                prefix=f"shared_upstream_dim_{d}",
            )
            history.append({"stage": f"shared_upstream_dim_{d}", **metrics})

        # protocol-specific observer: z -> y
        for protocol_id, (z_train, y_train) in sink_tensors.items():
            sink_model = self._ensure_protocol_sink_model(protocol_id)
            sink_metrics = sink_model.fit_exact_gp(
                x=z_train,
                y=y_train,
                n_steps=int(config.n_steps),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                grad_clip_norm=config.grad_clip_norm,
                verbose=bool(config.verbose),
                prefix=f"{protocol_id}_observer",
            )
            history.append({"stage": f"{protocol_id}_observer", **sink_metrics})

        final_train_eval = self.evaluate_protocol_dataset(
            protocols=protocol_map,
            samples=train_samples,
            config=config,
        )
        final_train_loss = float(final_train_eval.loss)

        final_val_loss: Optional[float] = None
        best_val_loss: Optional[float] = None
        if val_samples is not None and len(val_samples) > 0:
            val_eval = self.evaluate_protocol_dataset(
                protocols=protocol_map,
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