"""Project-compatible, non-recurrent GPPO-style actor-critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.model import ActorHead, CriticHead, RHGraphEncoder


class GPPOActorCritic(nn.Module):
    """Graph PPO baseline over the current project's five-split action space.

    This is a project-compatible GPPO-style ablation, not an exact
    reproduction of Ngo's GPPO architecture or split semantics.
    """

    def __init__(
        self,
        rh_feature_dim: int,
        hidden_dim: int,
        num_actions_per_rh: int,
    ) -> None:
        super().__init__()
        self.rh_feature_dim = rh_feature_dim
        self.graph_encoder = RHGraphEncoder(rh_feature_dim, hidden_dim)
        self.actor = ActorHead(hidden_dim, num_actions_per_rh)
        self.critic = CriticHead(hidden_dim)

    def forward(
        self,
        rh_feature_sequence: Tensor,
        handover_edge_index_sequence: list[Tensor],
        handover_edge_attr_sequence: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Return last-snapshot per-RH logits and a global value estimate."""

        if rh_feature_sequence.ndim != 4:
            raise ValueError("rh_feature_sequence must have shape [batch, time, rh, features]")
        batch, time, _, _ = rh_feature_sequence.shape
        if batch != 1:
            raise ValueError("GPPOActorCritic currently expects batch size 1 for PyG edge inputs")
        if time <= 0:
            raise ValueError("rh_feature_sequence must contain at least one timestep")
        if len(handover_edge_index_sequence) != time or len(handover_edge_attr_sequence) != time:
            raise ValueError("handover edge sequences must match the temporal length")

        context = self.graph_encoder(
            rh_feature_sequence[:, -1].squeeze(0),
            handover_edge_index_sequence[-1],
            handover_edge_attr_sequence[-1],
        ).unsqueeze(0)
        return self.actor(context), self.critic(context)
