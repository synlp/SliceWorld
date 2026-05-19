import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sliceworld.data.deeplesion import DeepLesionSequenceDataset, collate_deeplesion
from sliceworld.models.sliceworld import SliceWorldCore
from sliceworld.utils.config import get_device, load_config
from sliceworld.utils.io import load_model_weights, write_json


def _summary(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


@torch.no_grad()
def prefix_only_nsp(model: SliceWorldCore, features: torch.Tensor, length: int, head: str = "world") -> List[Dict]:
    rows = []
    for t in range(1, length):
        prefix = features[:1, :t]
        outputs = model(slice_features=prefix, lengths=torch.tensor([t], device=features.device))
        key = "pred_future_from_world" if head == "world" else "pred_future_from_hidden"
        max_horizon = min(outputs[key].shape[2], length - t)
        for k in range(1, max_horizon + 1):
            prediction = outputs[key][0, t - 1, k - 1]
            target = features[0, t + k - 1]
            persistence = features[0, t - 1]
            if t >= 2:
                linear = features[0, t - 1] + k * (features[0, t - 1] - features[0, t - 2])
            else:
                linear = persistence
            rows.append(
                {
                    "t": t,
                    "k": k,
                    "model_mse": F.mse_loss(prediction, target).item(),
                    "model_cosine": F.cosine_similarity(prediction, target, dim=0).item(),
                    "persistence_mse": F.mse_loss(persistence, target).item(),
                    "linear_mse": F.mse_loss(linear, target).item(),
                }
            )
    return rows


@torch.no_grad()
def evaluate(config: dict) -> None:
    device = get_device(config.get("device", "auto"))
    model = SliceWorldCore(config["model"]).to(device).eval()
    load_model_weights(model, config["checkpoint"], strict=False)
    dataset = DeepLesionSequenceDataset(
        manifest_path=config["data"]["manifest_path"],
        split=config["data"].get("split", "val"),
        max_slices=int(config["data"].get("max_sampled_slices", 196)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_deeplesion)
    all_rows = []
    head = config.get("prediction_head", "world")
    for batch in tqdm(loader, desc="prefix nsp"):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        features = model.encode_slices(pixel_values=batch["pixel_values"])
        rows = prefix_only_nsp(model, features, int(batch["lengths"][0].item()), head=head)
        for row in rows:
            row["sample_id"] = batch["sample_ids"][0]
        all_rows.extend(rows)
    summary = {
        "model_mse": _summary([row["model_mse"] for row in all_rows]),
        "model_cosine": _summary([row["model_cosine"] for row in all_rows]),
        "persistence_mse": _summary([row["persistence_mse"] for row in all_rows]),
        "linear_mse": _summary([row["linear_mse"] for row in all_rows]),
    }
    summary["relative_mse_vs_persistence"] = summary["model_mse"] / max(summary["persistence_mse"], 1e-8)
    summary["win_rate_vs_persistence"] = _summary([float(row["model_mse"] < row["persistence_mse"]) for row in all_rows])
    summary["win_rate_vs_linear"] = _summary([float(row["model_mse"] < row["linear_mse"]) for row in all_rows])
    horizons = sorted({row["k"] for row in all_rows})
    summary["per_horizon"] = {
        str(k): {
            "model_mse": _summary([row["model_mse"] for row in all_rows if row["k"] == k]),
            "persistence_mse": _summary([row["persistence_mse"] for row in all_rows if row["k"] == k]),
            "linear_mse": _summary([row["linear_mse"] for row in all_rows if row["k"] == k]),
        }
        for k in horizons
    }
    write_json(config.get("output_path", "outputs/eval_nsp.json"), {"summary": summary, "rows": all_rows})
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
