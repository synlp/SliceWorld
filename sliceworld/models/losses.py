from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sliceworld.models.encoders import sequence_mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def _transition_mask(lengths: Optional[torch.Tensor], max_length: int, device: torch.device) -> torch.Tensor:
    if lengths is None:
        return torch.ones((1, max_length - 1), dtype=torch.bool, device=device)
    mask = sequence_mask(lengths, max_length, device)
    return mask[:, :-1] & mask[:, 1:]


def _future_source_mask(
    lengths: Optional[torch.Tensor],
    batch_size: int,
    max_length: int,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    source_length = max(max_length - horizon, 0)
    if lengths is None:
        return torch.ones((batch_size, source_length), dtype=torch.bool, device=device)
    valid_lengths = (lengths.to(device) - horizon).clamp_min(0)
    steps = torch.arange(source_length, device=device).unsqueeze(0)
    return steps < valid_lengths.unsqueeze(1)


def future_targets(features: torch.Tensor, horizon: int) -> torch.Tensor:
    source_length = max(features.shape[1] - horizon, 0)
    if source_length == 0:
        return features.new_empty((features.shape[0], 0, horizon, features.shape[-1]))
    return torch.stack([features[:, k : k + source_length] for k in range(1, horizon + 1)], dim=2)


def squared_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).sum(dim=-1)


def normalized_prediction_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).mean(dim=-1)


@dataclass
class ObjectiveSwitches:
    alpha_nsp: float = 1.0
    alpha_fas: float = 1.0
    alpha_cf: float = 1.0
    alpha_ctrg: float = 0.0

    @classmethod
    def world_pretraining(cls) -> "ObjectiveSwitches":
        return cls(alpha_nsp=1.0, alpha_fas=1.0, alpha_cf=1.0, alpha_ctrg=0.0)

    @classmethod
    def report_finetuning(cls) -> "ObjectiveSwitches":
        return cls(alpha_nsp=0.0, alpha_fas=0.0, alpha_cf=0.0, alpha_ctrg=1.0)


@dataclass
class ComponentWeights:
    lambda_h: float = 1.0
    lambda_w: float = 1.0
    lambda_smooth: float = 1.0
    lambda_sparse: float = 1.0
    lambda_unc: float = 1.0
    lambda_occ: float = 1.0
    lambda_inv: float = 1.0
    lambda_eff: float = 1.0
    margin_delta: float = 0.1


@dataclass
class LossConfig:
    switches: ObjectiveSwitches = field(default_factory=ObjectiveSwitches.world_pretraining)
    weights: ComponentWeights = field(default_factory=ComponentWeights)


class SliceWorldObjective(nn.Module):
    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def nsp_loss(self, outputs: Dict[str, torch.Tensor], lengths: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = outputs["slice_features"]
        horizon = outputs["pred_future_from_hidden"].shape[2]
        target = future_targets(features, horizon)
        source_length = target.shape[1]
        source_valid = _future_source_mask(lengths, features.shape[0], features.shape[1], horizon, features.device)
        valid = source_valid.unsqueeze(-1).expand(-1, -1, horizon)
        pred_h = outputs["pred_future_from_hidden"][:, :source_length]
        pred_w = outputs["pred_future_from_world"][:, :source_length]
        loss_h = _masked_mean(squared_l2(pred_h, target), valid)
        loss_w = _masked_mean(squared_l2(pred_w, target), valid)
        total = self.config.weights.lambda_h * loss_h + self.config.weights.lambda_w * loss_w
        return {"loss_nsp_h": loss_h, "loss_nsp_w": loss_w, "loss_nsp": total}

    def fas_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        lesion_labels: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        anatomy = outputs["anatomy"]
        lesion = outputs["lesion"]
        uncertainty_score = outputs["uncertainty_score"]
        logits = outputs["lesion_presence_logits"]
        features = outputs["slice_features"]
        max_length = features.shape[1]
        lesion_labels = lesion_labels.to(device=features.device, dtype=logits.dtype)
        slice_valid = sequence_mask(lengths, max_length, features.device) if lengths is not None else torch.ones_like(lesion_labels, dtype=torch.bool)
        trans_valid = _transition_mask(lengths, max_length, features.device)
        horizon = outputs["pred_future_from_hidden"].shape[2]
        target = future_targets(features, horizon)
        source_length = target.shape[1]
        source_valid = _future_source_mask(lengths, features.shape[0], features.shape[1], horizon, features.device)

        smooth = _masked_mean(squared_l2(anatomy[:, 1:], anatomy[:, :-1]), trans_valid)
        sparse = _masked_mean(lesion.abs().sum(dim=-1), slice_valid)
        pred_h = outputs["pred_future_from_hidden"][:, :source_length]
        target_error = normalized_prediction_error(pred_h, target).mean(dim=-1).detach()
        uncertainty = _masked_mean((uncertainty_score[:, :source_length] - target_error).pow(2), source_valid)
        occ = F.binary_cross_entropy_with_logits(logits, lesion_labels.to(logits.dtype), reduction="none")
        occ = _masked_mean(occ, slice_valid)

        total = (
            self.config.weights.lambda_smooth * smooth
            + self.config.weights.lambda_sparse * sparse
            + self.config.weights.lambda_unc * uncertainty
            + self.config.weights.lambda_occ * occ
        )
        return {
            "loss_smooth": smooth,
            "loss_sparse": sparse,
            "loss_unc": uncertainty,
            "loss_occ": occ,
            "loss_fas": total,
        }

    def cf_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        lesion_labels: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        features = outputs["slice_features"]
        horizon = outputs["pred_future_from_world"].shape[2]
        target = future_targets(features, horizon)
        source_length = target.shape[1]
        source_valid = _future_source_mask(lengths, features.shape[0], features.shape[1], horizon, features.device)
        labels = lesion_labels.to(device=features.device, dtype=features.dtype)[:, :source_length]
        factual = outputs["pred_future_from_world"][:, :source_length]
        counterfactual = outputs["pred_future_from_lesion_zero"][:, :source_length]

        inv_mask = (source_valid & (labels <= 0.5)).unsqueeze(-1).expand(-1, -1, horizon)
        eff_mask = (source_valid & (labels > 0.5)).unsqueeze(-1).expand(-1, -1, horizon)
        inv = _masked_mean(squared_l2(factual, counterfactual), inv_mask)
        fact_err = squared_l2(factual, target)
        cf_err = squared_l2(counterfactual, target)
        eff = _masked_mean(F.relu(self.config.weights.margin_delta + fact_err - cf_err), eff_mask)
        total = self.config.weights.lambda_inv * inv + self.config.weights.lambda_eff * eff
        return {"loss_cf_inv": inv, "loss_cf_eff": eff, "loss_cf": total}

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        lesion_labels: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        ctrg_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        switches = self.config.switches
        device = outputs["slice_features"].device
        total = torch.zeros((), dtype=outputs["slice_features"].dtype, device=device)
        losses: Dict[str, torch.Tensor] = {}

        if switches.alpha_nsp:
            part = self.nsp_loss(outputs, lengths)
            losses.update(part)
            total = total + switches.alpha_nsp * part["loss_nsp"]

        if switches.alpha_fas:
            if lesion_labels is None:
                raise ValueError("FAS loss requires lesion_labels.")
            part = self.fas_loss(outputs, lesion_labels, lengths)
            losses.update(part)
            total = total + switches.alpha_fas * part["loss_fas"]

        if switches.alpha_cf:
            if lesion_labels is None:
                raise ValueError("CF loss requires lesion_labels.")
            part = self.cf_loss(outputs, lesion_labels, lengths)
            losses.update(part)
            total = total + switches.alpha_cf * part["loss_cf"]

        if switches.alpha_ctrg:
            if ctrg_loss is None:
                raise ValueError("CTRG loss requires ctrg_loss.")
            losses["loss_ctrg"] = ctrg_loss
            total = total + switches.alpha_ctrg * ctrg_loss

        losses["loss"] = total
        return losses


def loss_config_from_dict(config: Dict) -> LossConfig:
    switches = ObjectiveSwitches(
        alpha_nsp=float(config.get("alpha_nsp", 1.0)),
        alpha_fas=float(config.get("alpha_fas", 1.0)),
        alpha_cf=float(config.get("alpha_cf", 1.0)),
        alpha_ctrg=float(config.get("alpha_ctrg", 0.0)),
    )
    weights = ComponentWeights(
        lambda_h=float(config.get("lambda_h", 1.0)),
        lambda_w=float(config.get("lambda_w", 1.0)),
        lambda_smooth=float(config.get("lambda_smooth", 1.0)),
        lambda_sparse=float(config.get("lambda_sparse", 1.0)),
        lambda_unc=float(config.get("lambda_unc", 1.0)),
        lambda_occ=float(config.get("lambda_occ", 1.0)),
        lambda_inv=float(config.get("lambda_inv", 1.0)),
        lambda_eff=float(config.get("lambda_eff", 1.0)),
        margin_delta=float(config.get("margin_delta", 0.1)),
    )
    return LossConfig(switches=switches, weights=weights)
