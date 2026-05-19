import math
from typing import Optional

import torch
from torch import nn


def sequence_mask(lengths: Optional[torch.Tensor], max_length: int, device: torch.device) -> torch.Tensor:
    if lengths is None:
        return torch.ones((1, max_length), dtype=torch.bool, device=device)
    steps = torch.arange(max_length, device=device).unsqueeze(0)
    return steps < lengths.to(device).unsqueeze(1)


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    enc = torch.zeros((length, dim), dtype=torch.float32, device=device)
    enc[:, 0::2] = torch.sin(positions * div)
    enc[:, 1::2] = torch.cos(positions * div[: enc[:, 1::2].shape[1]])
    return enc


class DinoV2SliceEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        hidden_dim: int = 1024,
        freeze: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze = freeze
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.pooler = CrossAttentionSlicePooler(hidden_dim)

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("DINOv2 encoding requires transformers.") from exc
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if getattr(self.model.config, "hidden_size", self.hidden_dim) != self.hidden_dim:
            raise ValueError(f"{self.model_name} hidden size is not {self.hidden_dim}.")
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, slices = pixel_values.shape[:2]
        flat = pixel_values.reshape(batch * slices, *pixel_values.shape[2:])
        self.load()
        self.model.to(flat.device)
        if self.freeze:
            self.model.eval()
        with torch.set_grad_enabled(not self.freeze):
            output = self.model(pixel_values=flat)
            tokens = output.last_hidden_state
        if self.freeze:
            tokens = tokens.detach()
        features = self.pooler(tokens[:, 1:] if tokens.shape[1] > 1 else tokens)
        return features.reshape(batch, slices, -1)


class CrossAttentionSlicePooler(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(patch_tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query, patch_tokens, patch_tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class IdentitySliceEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        if slice_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected precomputed features with dim {self.hidden_dim}.")
        return slice_features


class TinySliceEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, slices = pixel_values.shape[:2]
        flat = pixel_values.reshape(batch * slices, *pixel_values.shape[2:])
        features = self.net(flat)
        return features.reshape(batch, slices, self.hidden_dim)


class SmokeMambaLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        return x + self.mix(y)


class MambaSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 512,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        backend: str = "mamba_ssm",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.backend = backend
        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        if backend == "mamba_ssm":
            try:
                from mamba_ssm.modules.mamba_simple import Mamba
            except ImportError as exc:
                raise ImportError("Install mamba-ssm or set backend='smoke' for dry runs.") from exc
            for _ in range(n_layers):
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(d_model),
                            "mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                        }
                    )
                )
        elif backend == "smoke":
            for _ in range(n_layers):
                self.layers.append(SmokeMambaLayer(d_model, dropout))
        else:
            raise ValueError(f"Unknown Mamba backend: {backend}")

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.input_projection(features)
        hidden = hidden + sinusoidal_position_encoding(hidden.shape[1], hidden.shape[2], hidden.device).unsqueeze(0)
        hidden = self.dropout(hidden)
        if lengths is not None:
            mask = sequence_mask(lengths, hidden.shape[1], hidden.device)
            hidden = hidden * mask.unsqueeze(-1)

        for layer in self.layers:
            if self.backend == "mamba_ssm":
                hidden = hidden + layer["mamba"](layer["norm"](hidden))
            else:
                hidden = layer(hidden)

        hidden = self.output_norm(hidden)
        if lengths is not None:
            hidden = hidden * sequence_mask(lengths, hidden.shape[1], hidden.device).unsqueeze(-1)
        return hidden


def build_slice_encoder(config: dict) -> nn.Module:
    encoder_type = config.get("type", "dinov2")
    hidden_dim = int(config.get("hidden_dim", 1024))
    if encoder_type == "dinov2":
        return DinoV2SliceEncoder(
            model_name=config.get("model_name", "facebook/dinov2-large"),
            hidden_dim=hidden_dim,
            freeze=bool(config.get("freeze", True)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    if encoder_type == "identity":
        return IdentitySliceEncoder(hidden_dim=hidden_dim)
    if encoder_type == "tiny":
        return TinySliceEncoder(hidden_dim=hidden_dim)
    raise ValueError(f"Unknown slice encoder type: {encoder_type}")
