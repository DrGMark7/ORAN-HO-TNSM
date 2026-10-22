"""Modular ST-GNN actor-critic model for handover-aware orchestration."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv


STGNN_ARCHITECTURE_VERSION = "stgnn_full_episode_recurrent_v2"
STGNN_TEMPORAL_FUSION_MODE = "projected_sum_layernorm"
STGNN_HIDDEN_RESET_MODE = "episode_boundary_only"
STGNN_PPO_REPLAY_MODE = "full_episode"


class RHGraphEncoder(nn.Module):
    """Encode RH features with handover-edge message passing."""

    def __init__(self, rh_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.rh_feature_dim = rh_feature_dim
        self.input_proj = nn.Sequential(
            nn.Linear(rh_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.handover_conv = GATv2Conv(
            hidden_dim,
            hidden_dim,
            edge_dim=2,
            add_self_loops=False,
        )

    def forward(
        self,
        rh_x: Tensor,
        handover_edge_index: Tensor,
        handover_edge_attr: Tensor,
    ) -> Tensor:
        encoded = self.input_proj(rh_x)
        return torch.relu(self.handover_conv(encoded, handover_edge_index, handover_edge_attr))


class TemporalGRUEncoder(nn.Module):
    """GRU over per-RH graph embeddings across time.

    Hidden state shape is ``[num_layers, batch * num_rh, hidden_dim]``.
    This is per-RH temporal memory after graph encoding, not one global
    graph-level hidden state.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        rh_sequence: Tensor,
        hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, time, num_rh, hidden = rh_sequence.shape
        flattened = rh_sequence.permute(0, 2, 1, 3).reshape(batch * num_rh, time, hidden)
        encoded, next_hidden_state = self.gru(flattened, hidden_state)
        last = encoded[:, -1, :]
        return last.reshape(batch, num_rh, hidden), next_hidden_state


class ActorHead(nn.Module):
    """Per-RH categorical logits over flattened split/ES/RC tuples."""

    def __init__(self, hidden_dim: int, num_actions_per_rh: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions_per_rh),
        )

    def forward(self, rh_context: Tensor) -> Tensor:
        return self.net(rh_context)


class CriticHead(nn.Module):
    """Global value estimate from pooled RH context."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rh_context: Tensor) -> Tensor:
        pooled = rh_context.mean(dim=1)
        return self.net(pooled).squeeze(-1)


class STGNNActorCritic(nn.Module):
    """Proposal-consistent graph encoder + GRU actor-critic."""

    def __init__(
        self,
        rh_feature_dim: int,
        hidden_dim: int,
        num_actions_per_rh: int,
    ) -> None:
        super().__init__()
        self.rh_feature_dim = rh_feature_dim
        self.graph_encoder = RHGraphEncoder(rh_feature_dim, hidden_dim)
        self.temporal_encoder = TemporalGRUEncoder(hidden_dim)
        self.spatial_fusion_projection = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_fusion_projection = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_fusion_norm = nn.LayerNorm(hidden_dim)
        self.actor = ActorHead(hidden_dim, num_actions_per_rh)
        self.critic = CriticHead(hidden_dim)
        self.architecture_version = STGNN_ARCHITECTURE_VERSION
        self.temporal_fusion_mode = STGNN_TEMPORAL_FUSION_MODE
        self.hidden_reset_mode = STGNN_HIDDEN_RESET_MODE
        self.ppo_replay_mode = STGNN_PPO_REPLAY_MODE
        self.hidden_dim = hidden_dim
        self.num_gru_layers = self.temporal_encoder.gru.num_layers

    def forward(
        self,
        rh_feature_sequence: Tensor,
        handover_edge_index_sequence: list[Tensor],
        handover_edge_attr_sequence: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Return per-RH action logits and a state-value estimate.

        ``rh_feature_sequence`` has shape ``[batch, time, rh, features]``.
        Handover edge sequences are applied for batch-size one graph snapshots,
        which is the phase-1 mock-up training path.
        """

        logits, value, _ = self.forward_recurrent(
            rh_feature_sequence,
            handover_edge_index_sequence,
            handover_edge_attr_sequence,
            hidden_state=None,
        )
        return logits, value

    def forward_recurrent(
        self,
        rh_feature_sequence: Tensor,
        handover_edge_index_sequence: list[Tensor],
        handover_edge_attr_sequence: list[Tensor],
        hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, value, and the next per-RH recurrent hidden state.

        ``rh_feature_sequence`` has shape ``[batch, time, rh, features]``.
        ``hidden_state`` has shape ``[num_layers, batch * num_rh, hidden_dim]``.
        In the current phase this hidden state is per-RH temporal memory after
        graph encoding, not one global graph-level hidden state.

        Hidden-state detaching is intentionally not done inside the model. A
        recurrent trainer or rollout buffer should decide when to detach.
        """

        batch, time, _, _ = rh_feature_sequence.shape
        if batch != 1:
            raise ValueError("STGNNActorCritic currently expects batch size 1 for PyG edge inputs")
        if len(handover_edge_index_sequence) != time or len(handover_edge_attr_sequence) != time:
            raise ValueError("handover edge sequences must match the temporal length")

        encoded_steps: list[Tensor] = []
        for step in range(time):
            encoded = self.graph_encoder(
                rh_feature_sequence[:, step].squeeze(0),
                handover_edge_index_sequence[step],
                handover_edge_attr_sequence[step],
            ).unsqueeze(0)
            encoded_steps.append(encoded)
        temporal_input = torch.stack(encoded_steps, dim=1)
        temporal_context, next_hidden_state = self.temporal_encoder(temporal_input, hidden_state)
        current_spatial = temporal_input[:, -1]
        fused_context = self.temporal_fusion_norm(
            self.spatial_fusion_projection(current_spatial)
            + self.hidden_fusion_projection(temporal_context)
        )
        return self.actor(fused_context), self.critic(fused_context), next_hidden_state
