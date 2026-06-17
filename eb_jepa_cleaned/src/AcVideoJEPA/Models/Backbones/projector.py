"""
MLP projector.

Re-implemented from EB-JEPA (Decision 30). An expander head built from a spec
string like ``"512-1024-1024"`` (Linear -> BatchNorm -> ReLU per hidden layer,
final Linear without bias). Used by the anti-collapse regularizer metrics to
project embeddings before computing variance/covariance.

This is a plain building block, not a registered model: the regularizer metrics
construct it via their own field resolvers (it depends on the encoder feature
dimension, resolved at module-build time).
"""

import torch
from torch import nn


class Projector(nn.Module):
    """Expander MLP built from a ``"d0-d1-...-dk"`` spec string."""

    def __init__(self, mlp_spec: str) -> None:
        super().__init__()
        dims = [int(d) for d in mlp_spec.split("-")]
        if len(dims) < 2:
            raise ValueError(
                f"Projector mlp_spec must list at least two dims, got '{mlp_spec}'."
            )

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=False))

        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
