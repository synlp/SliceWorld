from typing import Dict, Optional

import torch
from torch import nn

from sliceworld.models.encoders import MambaSequenceEncoder, build_slice_encoder
from sliceworld.models.heads import FactorHeads, FutureSlicePredictionHeads, LLMProjector, WorldTokenProjector


class SliceWorldCore(nn.Module):
    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        config = config or {}
        slice_cfg = config.get("slice_encoder", {})
        seq_cfg = config.get("sequence_encoder", {})
        state_cfg = config.get("state", {})
        dropout = float(config.get("dropout", 0.1))

        visual_dim = int(slice_cfg.get("hidden_dim", 1024))
        d_model = int(seq_cfg.get("d_model", 512))
        d_world = int(state_cfg.get("world_dim", 512))
        self.future_horizon = int(config.get("future_horizon", 5))

        self.slice_encoder = build_slice_encoder(slice_cfg)
        self.sequence_encoder = MambaSequenceEncoder(
            input_dim=visual_dim,
            d_model=d_model,
            n_layers=int(seq_cfg.get("layers", 12)),
            d_state=int(seq_cfg.get("d_state", 16)),
            d_conv=int(seq_cfg.get("d_conv", 4)),
            expand=int(seq_cfg.get("expand", 2)),
            dropout=dropout,
            backend=seq_cfg.get("backend", "mamba_ssm"),
        )
        self.factor_heads = FactorHeads(
            d_model=d_model,
            d_anat=int(state_cfg.get("d_anat", 256)),
            d_lesion=int(state_cfg.get("d_lesion", 192)),
            d_unc=int(state_cfg.get("d_unc", 64)),
            dropout=dropout,
        )
        self.world_projector = WorldTokenProjector(
            state_dim=self.factor_heads.state_dim,
            world_dim=d_world,
            dropout=dropout,
        )
        self.prediction_heads = FutureSlicePredictionHeads(
            d_model=d_model,
            world_dim=d_world,
            visual_dim=visual_dim,
            future_horizon=self.future_horizon,
            dropout=dropout,
        )

    def encode_slices(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        slice_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if slice_features is not None:
            return self.slice_encoder(slice_features) if self.slice_encoder.__class__.__name__ == "IdentitySliceEncoder" else slice_features
        if pixel_values is None:
            raise ValueError("Provide pixel_values or precomputed slice_features.")
        return self.slice_encoder(pixel_values)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        slice_features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.encode_slices(pixel_values=pixel_values, slice_features=slice_features)
        hidden = self.sequence_encoder(features, lengths=lengths)
        factors = self.factor_heads(hidden)
        world_tokens = self.world_projector(factors.state)
        lesion_zero_state = self.factor_heads.lesion_zero_state(factors)
        lesion_zero_world_tokens = self.world_projector(lesion_zero_state)

        output = {
            "slice_features": features,
            "hidden_states": hidden,
            "world_tokens": world_tokens,
            "lesion_zero_world_tokens": lesion_zero_world_tokens,
            "pred_future_from_hidden": self.prediction_heads.from_hidden(hidden),
            "pred_future_from_world": self.prediction_heads.from_world(world_tokens),
            "pred_future_from_lesion_zero": self.prediction_heads.from_world(lesion_zero_world_tokens),
        }
        output.update(factors.as_dict())
        return output

    def freeze_world_model(self) -> None:
        for module in [self.slice_encoder, self.sequence_encoder, self.factor_heads, self.world_projector, self.prediction_heads]:
            module.requires_grad_(False)


class SliceWorldForCTRG(nn.Module):
    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        config = config or {}
        self.core = SliceWorldCore(config.get("world_model", config))
        lm_cfg = config.get("llm", {})
        self.llm_name = lm_cfg.get("model_name", "Qwen/Qwen3-1.7B")
        self.lm_hidden_dim = int(lm_cfg.get("hidden_dim", 2048))
        self.load_llm = bool(lm_cfg.get("load", True))
        self.tokenizer = None
        self.llm = None
        self.lm_projector = LLMProjector(
            world_dim=int(config.get("world_model", config).get("state", {}).get("world_dim", 512)),
            lm_hidden_dim=self.lm_hidden_dim,
            dropout=float(config.get("world_model", config).get("dropout", 0.1)),
        )
        if bool(config.get("freeze_world_model", False)):
            self.core.freeze_world_model()
        if self.load_llm:
            self._load_llm(lm_cfg)

    def _load_llm(self, lm_cfg: Dict) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Report generation requires transformers.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=bool(lm_cfg.get("trust_remote_code", True)))
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            trust_remote_code=bool(lm_cfg.get("trust_remote_code", True)),
            torch_dtype=getattr(torch, lm_cfg.get("torch_dtype", "bfloat16")),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if lm_cfg.get("use_lora", True):
            self._attach_lora(lm_cfg.get("lora", {}))

    def _attach_lora(self, lora_cfg: Dict) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA fine-tuning requires peft.") from exc
        config = LoraConfig(
            r=int(lora_cfg.get("rank", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        )
        self.llm = get_peft_model(self.llm, config)

    def world_embeddings(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        slice_features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        lesion_zero: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.core(pixel_values=pixel_values, slice_features=slice_features, lengths=lengths)
        tokens = outputs["lesion_zero_world_tokens"] if lesion_zero else outputs["world_tokens"]
        outputs["lm_prefix"] = self.lm_projector(tokens)
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        slice_features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded.")
        outputs = self.world_embeddings(pixel_values=pixel_values, slice_features=slice_features, lengths=lengths)
        prefix = outputs["lm_prefix"]
        token_embeddings = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
        prefix_attention = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype if attention_mask is not None else torch.long, device=prefix.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        full_attention = torch.cat([prefix_attention, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            ignore_prefix = torch.full(prefix.shape[:2], -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([ignore_prefix, labels], dim=1)
        lm_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attention, labels=full_labels)
        outputs["ctrg_loss"] = lm_output.loss
        outputs["logits"] = lm_output.logits
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        slice_features: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        lesion_zero: bool = False,
        **generate_kwargs,
    ) -> torch.Tensor:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded.")
        outputs = self.world_embeddings(
            pixel_values=pixel_values,
            slice_features=slice_features,
            lengths=lengths,
            lesion_zero=lesion_zero,
        )
        prefix = outputs["lm_prefix"]
        token_embeddings = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_attention = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=prefix.device)
        full_attention = torch.cat([prefix_attention, attention_mask], dim=1)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=full_attention, **generate_kwargs)
