from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class FactorState:
    anatomy: torch.Tensor
    lesion: torch.Tensor
    uncertainty: torch.Tensor
    state: torch.Tensor
    lesion_presence_logits: torch.Tensor
    lesion_presence: torch.Tensor
    uncertainty_score: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "anatomy": self.anatomy,
            "lesion": self.lesion,
            "uncertainty": self.uncertainty,
            "state": self.state,
            "lesion_presence_logits": self.lesion_presence_logits,
            "lesion_presence": self.lesion_presence,
            "uncertainty_score": self.uncertainty_score,
        }


class FactorHeads(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_anat: int = 256,
        d_lesion: int = 192,
        d_unc: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_anat = d_anat
        self.d_lesion = d_lesion
        self.d_unc = d_unc
        self.f_anat = mlp(d_model, d_model, d_anat, dropout)
        self.f_les = mlp(d_model, d_model, d_lesion, dropout)
        self.f_unc = mlp(d_model, d_model, d_unc, dropout)
        self.f_occ = nn.Linear(d_lesion, 1)
        self.g_unc = mlp(d_unc, max(32, d_unc), 1, dropout)

    @property
    def state_dim(self) -> int:
        return self.d_anat + self.d_lesion + self.d_unc

    def forward(self, hidden_states: torch.Tensor) -> FactorState:
        anatomy = self.f_anat(hidden_states)
        lesion = self.f_les(hidden_states)
        uncertainty = self.f_unc(hidden_states)
        state = torch.cat([anatomy, lesion, uncertainty], dim=-1)
        lesion_logits = self.f_occ(lesion).squeeze(-1)
        return FactorState(
            anatomy=anatomy,
            lesion=lesion,
            uncertainty=uncertainty,
            state=state,
            lesion_presence_logits=lesion_logits,
            lesion_presence=torch.sigmoid(lesion_logits),
            uncertainty_score=self.g_unc(uncertainty).squeeze(-1),
        )

    def lesion_zero_state(self, factors: FactorState) -> torch.Tensor:
        return torch.cat(
            [factors.anatomy, torch.zeros_like(factors.lesion), factors.uncertainty],
            dim=-1,
        )


class WorldTokenProjector(nn.Module):
    def __init__(self, state_dim: int = 512, world_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = mlp(state_dim, world_dim, world_dim, dropout)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class LLMProjector(nn.Module):
    def __init__(self, world_dim: int = 512, lm_hidden_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = mlp(world_dim, max(world_dim, lm_hidden_dim), lm_hidden_dim, dropout)

    def forward(self, world_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(world_tokens)


class FutureSlicePredictionHeads(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        world_dim: int = 512,
        visual_dim: int = 1024,
        future_horizon: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.future_horizon = future_horizon
        self.f_pred_h = mlp(d_model, d_model, visual_dim * future_horizon, dropout)
        self.f_pred_w = mlp(world_dim, world_dim, visual_dim * future_horizon, dropout)

    def from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        prediction = self.f_pred_h(hidden_states)
        return prediction.view(*prediction.shape[:-1], self.future_horizon, self.visual_dim)

    def from_world(self, world_tokens: torch.Tensor) -> torch.Tensor:
        prediction = self.f_pred_w(world_tokens)
        return prediction.view(*prediction.shape[:-1], self.future_horizon, self.visual_dim)
