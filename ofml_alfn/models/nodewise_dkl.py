#!/usr/bin/env python3
from __future__ import annotations

import copy
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
            "Problem 1A latent supervision expects sample.metadata['z'] to exist."
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


class _SharedEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(max(self.depth - 1, 0)):
            layers.append(nn.Linear(in_dim, self.hidden))
            layers.append(nn.ReLU())
            in_dim = self.hidden
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        return self.net(x)


class MultiHeadNodewiseDKL(nn.Module):
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
        use_true_latent_supervision: bool = False,
        latent_supervision_weight: float = 1.0,
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

        self.parent_nodes = (
            [list(p) for p in parent_nodes] if parent_nodes is not None else None
        )
        self.active_input_indices = (
            [list(a) for a in active_input_indices]
            if active_input_indices is not None
            else None
        )

        self.shared_upstream_output_dim = int(self.node_input_dims[self.sink_idx])

        self.use_true_latent_supervision = bool(use_true_latent_supervision)
        self.latent_supervision_weight = float(latent_supervision_weight)

        self.shared_encoder = _SharedEncoder(
            input_dim=self.external_input_dim,
            output_dim=self.shared_upstream_output_dim,
            hidden=self.hidden,
            depth=self.depth,
        )

        self.protocol_sink_models = nn.ModuleDict()

        self._init_kwargs = {
            "external_input_dim": int(external_input_dim),
            "node_input_dims": [int(d) for d in node_input_dims],
            "parent_nodes": None if parent_nodes is None else [list(p) for p in parent_nodes],
            "active_input_indices": None if active_input_indices is None else [list(a) for a in active_input_indices],
            "hidden": int(hidden),
            "depth": int(depth),
            "feature_dim": int(feature_dim),
            "kernel_type": str(kernel_type),
            "sink_idx": None if sink_idx is None else int(sink_idx),
            "use_true_latent_supervision": bool(use_true_latent_supervision),
            "latent_supervision_weight": float(latent_supervision_weight),
        }

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _unique_trainable_parameters(self) -> List[nn.Parameter]:
        unique: List[nn.Parameter] = []
        seen = set()
        for p in self.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append(p)
        return unique

    def __deepcopy__(self, memo):
        copied = self.__class__(**copy.deepcopy(self._init_kwargs, memo))
        copied.to(self._device())

        for protocol_id, heads in self.protocol_sink_models.items():
            copied._ensure_protocol_sink_models(protocol_id, len(heads))

        copied.load_state_dict(copy.deepcopy(self.state_dict(), memo), strict=False)
        memo[id(self)] = copied
        return copied

    def _ensure_protocol_sink_models(
        self,
        protocol_id: str,
        output_dim: int,
    ) -> nn.ModuleList:
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        if protocol_id not in self.protocol_sink_models:
            self.protocol_sink_models[protocol_id] = nn.ModuleList(
                [
                    NodewiseDKLRegressor(
                        in_dim=self.shared_upstream_output_dim,
                        hidden=self.hidden,
                        feature_dim=self.feature_dim,
                        kernel_type=self.kernel_type,
                    )
                    for _ in range(int(output_dim))
                ]
            )

        heads = self.protocol_sink_models[protocol_id]
        if len(heads) != int(output_dim):
            raise ValueError(
                f"Protocol {protocol_id!r} already has {len(heads)} sink heads, "
                f"but output_dim={output_dim} was requested."
            )

        return heads

    def _group_problem1a_train_data(
        self,
        *,
        protocols: Mapping[str, ProtocolSpec],
        train_samples: Sequence[BenchmarkSample],
        device: torch.device,
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if len(train_samples) == 0:
            raise ValueError("train_samples must be non-empty")

        grouped_xy: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        all_x_rows: List[torch.Tensor] = []
        all_z_rows: List[torch.Tensor] = []

        for sample in train_samples:
            if sample.protocol_id not in protocols:
                raise KeyError(
                    f"Unknown protocol_id in train_samples: {sample.protocol_id!r}"
                )

            protocol = protocols[sample.protocol_id]
            x = _extract_sample_x(sample, protocol=protocol, device=device)
            y = _extract_sample_y(sample, device=device)

            if sample.protocol_id not in grouped_xy:
                grouped_xy[sample.protocol_id] = ([], [])
            grouped_xy[sample.protocol_id][0].append(x)
            grouped_xy[sample.protocol_id][1].append(y)

            if self.use_true_latent_supervision:
                all_x_rows.append(x)
                all_z_rows.append(_extract_sample_z(sample, device=device))

        grouped_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for protocol_id, (xs, ys) in grouped_xy.items():
            grouped_tensors[protocol_id] = (
                torch.stack(xs, dim=0),
                torch.stack(ys, dim=0),
            )

        all_x = None
        all_true_z = None
        if self.use_true_latent_supervision:
            all_x = torch.stack(all_x_rows, dim=0)
            all_true_z = torch.stack(all_z_rows, dim=0)

        return grouped_tensors, all_x, all_true_z

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

        z_mean = self.shared_encoder(condition_x)
        heads = self.protocol_sink_models[protocol.protocol_id]

        preds = [head(z_mean) for head in heads]
        return torch.cat(preds, dim=-1)

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

        z_mean = self.shared_encoder(condition_x)
        heads = self.protocol_sink_models[protocol.protocol_id]

        samples_per_output = [
            head.sample_observation(z_mean, n_samples=n_fantasies)
            for head in heads
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

        grouped_xy, all_x, all_true_z = self._group_problem1a_train_data(
            protocols=protocols,
            train_samples=train_samples,
            device=device,
        )

        protocol_output_dims: Dict[str, int] = {}
        for protocol_id, (_, y) in grouped_xy.items():
            if y.ndim != 2:
                raise ValueError(
                    f"Expected y tensor to be 2D for protocol {protocol_id!r}, "
                    f"got {tuple(y.shape)}"
                )
            protocol_output_dims[protocol_id] = int(y.shape[1])
            self._ensure_protocol_sink_models(protocol_id, int(y.shape[1]))

        opt = torch.optim.Adam(
            self._unique_trainable_parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

        history: List[Dict[str, float]] = []
        last_joint_loss = float("nan")
        last_latent_loss = float("nan")

        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        best_val_loss: Optional[float] = None
        best_step = int(config.n_steps)

        for step in range(1, int(config.n_steps) + 1):
            opt.zero_grad()

            joint_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            per_protocol_losses: Dict[str, float] = {}

            for protocol_id, (x_this, y_this) in grouped_xy.items():
                z_hat = self.shared_encoder(x_this)
                heads = self.protocol_sink_models[protocol_id]

                if y_this.shape[1] != len(heads):
                    raise ValueError(
                        f"Output dimension mismatch for protocol {protocol_id!r}: "
                        f"y dim={y_this.shape[1]}, n_heads={len(heads)}"
                    )

                protocol_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                for j, head in enumerate(heads):
                    y_j = y_this[:, j:j + 1]
                    head.set_train_data(x=z_hat, y=y_j, strict=False)

                    head.gp.train()
                    head.likelihood.train()

                    output = head.gp(z_hat)
                    mll = head.marginal_log_likelihood()
                    loss_j = -mll(output, head.gp.train_targets)

                    protocol_loss = protocol_loss + loss_j

                joint_loss = joint_loss + protocol_loss
                per_protocol_losses[protocol_id] = float(protocol_loss.detach().cpu().item())

            latent_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if self.use_true_latent_supervision:
                if all_x is None or all_true_z is None:
                    raise RuntimeError(
                        "use_true_latent_supervision=True but latent targets are missing."
                    )
                z_pred_all = self.shared_encoder(all_x)
                latent_loss = F.mse_loss(z_pred_all, all_true_z)
                joint_loss = joint_loss + float(self.latent_supervision_weight) * latent_loss

            joint_loss.backward()

            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._unique_trainable_parameters(),
                    max_norm=float(config.grad_clip_norm),
                )

            opt.step()

            last_joint_loss = float(joint_loss.detach().cpu().item())
            last_latent_loss = float(latent_loss.detach().cpu().item())

            row: Dict[str, float] = {
                "step": float(step),
                "joint_loss": float(last_joint_loss),
            }
            if self.use_true_latent_supervision:
                row["latent_supervision_loss"] = float(last_latent_loss)
            for protocol_id, loss_val in per_protocol_losses.items():
                row[f"{protocol_id}_loss"] = float(loss_val)
            history.append(row)

            if bool(config.verbose) and (step == 1 or step == int(config.n_steps)):
                print(
                    f"[multihead_dkl] step={step} "
                    f"joint_loss={last_joint_loss:.6f}"
                    + (
                        f" latent_supervision_loss={last_latent_loss:.6f}"
                        if self.use_true_latent_supervision
                        else ""
                    )
                )

            should_eval_val = (
                val_samples is not None
                and len(val_samples) > 0
                and int(config.val_every) > 0
                and (step % int(config.val_every) == 0 or step == int(config.n_steps))
            )

            if should_eval_val:
                val_eval = self.evaluate_protocol_dataset(
                    protocols=protocols,
                    samples=val_samples,
                    config=config,
                )
                val_loss = float(val_eval.loss)

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = int(step)
                    best_state_dict = copy.deepcopy(self.state_dict())

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        final_train_eval = self.evaluate_protocol_dataset(
            protocols=protocols,
            samples=train_samples,
            config=config,
        )
        final_train_loss = float(final_train_eval.loss)

        final_val_loss: Optional[float] = None
        if val_samples is not None and len(val_samples) > 0:
            final_val_eval = self.evaluate_protocol_dataset(
                protocols=protocols,
                samples=val_samples,
                config=config,
            )
            final_val_loss = float(final_val_eval.loss)
            if best_val_loss is None:
                best_val_loss = final_val_loss

        return ProtocolTrainingResult(
            optimizer=None,
            history=history,
            best_step=int(best_step),
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_state_dict=best_state_dict,
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
            if protocol_id not in protocols:
                raise KeyError(f"Unknown protocol_id {protocol_id!r} in evaluation.")

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


__all__ = [
    "NodewiseDKLRegressor",
    "MultiHeadNodewiseDKL",
]