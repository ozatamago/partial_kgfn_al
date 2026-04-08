#!/usr/bin/env python3

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import gpytorch
except ImportError as e:
    raise ImportError(
        "gpytorch is required for MultiHeadNodewiseDKL. "
        "Please install gpytorch before using this model."
    ) from e


class _NodeFeatureExtractor(nn.Module):
    """
    Per-node feature extractor:
        x_node -> phi_node
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
    One DKL regressor for one node:
        x_node -> z_node

    This wraps:
        - feature extractor
        - Gaussian likelihood
        - Exact GP head

    Notes
    -----
    - Training is handled outside this class.
    - This class exposes helper methods for:
        * setting train data
        * obtaining posterior mean / variance
        * drawing latent posterior samples
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
        """
        Backward-compatible default:
        return posterior mean with shape [N, 1]
        """
        mean, _ = self.predict_mean_var(x)
        return mean

    @torch.no_grad()
    def predict_mean_var(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns latent posterior mean / variance with shape [N, 1].
        """
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
        """
        Draw samples from the latent posterior f(x).

        Returns
        -------
        samples: torch.Tensor
            Shape [S, N, 1]
        """
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
        """
        Draw samples from p(y | x), i.e. with likelihood noise included.

        Returns
        -------
        samples: torch.Tensor
            Shape [S, N, 1]
        """
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


class MultiHeadNodewiseDKL(nn.Module):
    """
    Node-wise DKL predictor for a function network.

    Important:
    - one conditional DKL surrogate per node
    - each node has its own input dimension in node-input space
    - API intentionally mirrors MultiHeadMCDropoutMLP

    Example for a 2-node chain:
        node 0: x_ext (dim 3) -> z0
        node 1: z0 (dim 1)    -> z1
    """

    def __init__(
        self,
        *,
        external_input_dim: int,
        node_input_dims: Sequence[int],
        parent_nodes: Optional[Sequence[Sequence[int]]] = None,
        active_input_indices: Optional[Sequence[Sequence[int]]] = None,
        hidden: int = 256,
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

        if self.parent_nodes is not None and len(self.parent_nodes) != self.n_nodes:
            raise ValueError(
                "parent_nodes must have length n_nodes, "
                f"got {len(self.parent_nodes)} and {self.n_nodes}"
            )

        if (
            self.active_input_indices is not None
            and len(self.active_input_indices) != self.n_nodes
        ):
            raise ValueError(
                "active_input_indices must have length n_nodes, "
                f"got {len(self.active_input_indices)} and {self.n_nodes}"
            )

        self.node_models = nn.ModuleList(
            [
                NodewiseDKLRegressor(
                    in_dim=self.node_input_dims[j],
                    hidden=self.hidden,
                    feature_dim=self.feature_dim,
                    kernel_type=self.kernel_type,
                )
                for j in range(self.n_nodes)
            ]
        )

        if self.parent_nodes is not None and self.active_input_indices is not None:
            for j in range(self.n_nodes):
                implied_dim = len(self.parent_nodes[j]) + len(self.active_input_indices[j])
                if implied_dim != self.node_input_dims[j]:
                    raise ValueError(
                        f"Node {j}: node_input_dims[{j}]={self.node_input_dims[j]} "
                        f"but implied dim from graph is {implied_dim}"
                    )

    def set_node_train_data(
        self,
        node_idx: int,
        x: torch.Tensor,
        y: torch.Tensor,
        strict: bool = False,
    ) -> None:
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"node_idx must be in [0, {self.n_nodes - 1}], got {node_idx}"
            )
        self.node_models[node_idx].set_train_data(x=x, y=y, strict=strict)

    def node_mll(
        self,
        node_idx: int,
    ) -> gpytorch.mlls.ExactMarginalLogLikelihood:
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"node_idx must be in [0, {self.n_nodes - 1}], got {node_idx}"
            )
        return self.node_models[node_idx].marginal_log_likelihood()

    def forward_node(
        self,
        x_node: torch.Tensor,
        node_idx: int,
    ) -> torch.Tensor:
        """
        Direct conditional prediction in node-input space:
            x_node -> z_node mean

        Returns shape [N, 1].
        """
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"node_idx must be in [0, {self.n_nodes - 1}], got {node_idx}"
            )
        return self.node_models[node_idx](x_node)

    @torch.no_grad()
    def predict_node_mean_var(
        self,
        x_node: torch.Tensor,
        node_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns latent posterior mean / variance of node_idx from node-input space.
        Shapes: [N, 1], [N, 1]
        """
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"node_idx must be in [0, {self.n_nodes - 1}], got {node_idx}"
            )
        return self.node_models[node_idx].predict_mean_var(x_node)

    def make_node_input_from_base(
        self,
        *,
        base_x: torch.Tensor,
        node_idx: int,
        parent_outputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Construct the input of node_idx from:
        - parent outputs
        - directly active external inputs from base_x

        parent_outputs[p] is expected to be shape [N, 1] for parent node p.
        """
        if self.parent_nodes is None or self.active_input_indices is None:
            raise ValueError(
                "Graph metadata is required. "
                "Please provide parent_nodes and active_input_indices "
                "when constructing the model."
            )

        if base_x.ndim != 2 or base_x.shape[1] != self.external_input_dim:
            raise ValueError(
                f"Expected base_x of shape [N, {self.external_input_dim}], "
                f"got {tuple(base_x.shape)}"
            )

        parts: List[torch.Tensor] = []

        parents = self.parent_nodes[node_idx]
        for p in parents:
            yp = parent_outputs[p]
            if yp is None:
                raise ValueError(
                    f"Parent output for node {p} is missing while constructing "
                    f"the input for node {node_idx}."
                )
            if yp.ndim != 2 or yp.shape[1] != 1:
                raise ValueError(
                    f"Parent output for node {p} must be [N,1], got {tuple(yp.shape)}"
                )
            parts.append(yp)

        active_idx = self.active_input_indices[node_idx]
        if len(active_idx) > 0:
            parts.append(base_x[:, active_idx])

        if len(parts) == 0:
            raise ValueError(
                f"Node {node_idx} has neither parents nor active external inputs."
            )

        x_node = torch.cat(parts, dim=-1)
        expected_dim = self.node_input_dims[node_idx]
        if x_node.shape[1] != expected_dim:
            raise ValueError(
                f"Constructed input for node {node_idx} has dim {x_node.shape[1]}, "
                f"expected {expected_dim}"
            )

        return x_node

    def rollout_means_from_base(
        self,
        base_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic mean rollout through the function network using
        node-wise DKL posterior means in topological order.

        Returns
        -------
        y_all: torch.Tensor
            Shape [N, n_nodes]
        """
        if self.parent_nodes is None or self.active_input_indices is None:
            raise ValueError(
                "rollout_means_from_base requires parent_nodes and active_input_indices."
            )

        node_outputs: List[Optional[torch.Tensor]] = [None] * self.n_nodes
        collected = []

        for j in range(self.n_nodes):
            xj = self.make_node_input_from_base(
                base_x=base_x,
                node_idx=j,
                parent_outputs=node_outputs,
            )
            mean_j = self.forward_node(xj, j)
            node_outputs[j] = mean_j
            collected.append(mean_j)

        return torch.cat(collected, dim=-1)

    @torch.no_grad()
    def rollout_samples_from_base(
        self,
        base_x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Ancestral latent posterior sampling through the DAG.

        For each Monte Carlo path:
            node 0 sample -> node 1 sample -> ...

        Returns
        -------
        samples: torch.Tensor
            Shape [S, N, n_nodes]
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if self.parent_nodes is None or self.active_input_indices is None:
            raise ValueError(
                "rollout_samples_from_base requires parent_nodes and active_input_indices."
            )

        all_samples: List[torch.Tensor] = []

        for _ in range(n_samples):
            node_outputs: List[Optional[torch.Tensor]] = [None] * self.n_nodes
            collected = []

            for j in range(self.n_nodes):
                xj = self.make_node_input_from_base(
                    base_x=base_x,
                    node_idx=j,
                    parent_outputs=node_outputs,
                )
                sample_j = self.node_models[j].sample_latent(xj, n_samples=1)[0]
                node_outputs[j] = sample_j
                collected.append(sample_j)

            all_samples.append(torch.cat(collected, dim=-1))

        return torch.stack(all_samples, dim=0)

    def forward_all(
        self,
        base_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward-compatible convenience:
        returns all node mean predictions from external input base_x.
        """
        return self.rollout_means_from_base(base_x)

    def forward_sink_from_base(
        self,
        base_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns sink prediction from external input base_x.
        """
        return self.rollout_means_from_base(base_x)[:, [self.sink_idx]]

    def forward_sink(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience wrapper.

        If x is in external input space:
            compute sink prediction from base_x.

        If x is already in sink-input space:
            compute sink conditional model directly.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {tuple(x.shape)}")

        if x.shape[1] == self.external_input_dim:
            return self.forward_sink_from_base(x)

        if x.shape[1] == self.node_input_dims[self.sink_idx]:
            return self.forward_node(x, self.sink_idx)

        raise ValueError(
            f"Input dim {x.shape[1]} matches neither external_input_dim="
            f"{self.external_input_dim} nor sink input dim="
            f"{self.node_input_dims[self.sink_idx]}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default behavior: return sink prediction.
        """
        return self.forward_sink(x)