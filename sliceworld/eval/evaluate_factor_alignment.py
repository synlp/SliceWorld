import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sliceworld.data.deeplesion import DeepLesionSequenceDataset, collate_deeplesion
from sliceworld.models.losses import future_targets
from sliceworld.models.sliceworld import SliceWorldCore
from sliceworld.utils.config import get_device, load_config
from sliceworld.utils.io import load_model_weights, write_json


@torch.no_grad()
def export_factors(config: dict):
    device = get_device(config.get("device", "auto"))
    model = SliceWorldCore(config["model"]).to(device).eval()
    load_model_weights(model, config["checkpoint"], strict=False)
    dataset = DeepLesionSequenceDataset(
        manifest_path=config["data"]["manifest_path"],
        split=config["data"].get("split", "val"),
        max_slices=int(config["data"].get("max_sampled_slices", 196)),
    )
    loader = DataLoader(dataset, batch_size=int(config["data"].get("batch_size", 1)), shuffle=False, collate_fn=collate_deeplesion)
    anatomy = []
    lesion = []
    uncertainty = []
    lesion_labels = []
    z_bins = []
    future_error = []
    for batch in tqdm(loader, desc="export factors"):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(pixel_values=batch["pixel_values"], lengths=batch["lengths"])
        for row in range(batch["lengths"].shape[0]):
            length = int(batch["lengths"][row].item())
            horizon = int(outputs["pred_future_from_world"].shape[2])
            source_length = max(length - horizon, 0)
            if source_length == 0:
                continue
            anatomy.append(outputs["anatomy"][row, :source_length].cpu().numpy())
            lesion.append(outputs["lesion"][row, :source_length].cpu().numpy())
            uncertainty.append(outputs["uncertainty"][row, :source_length].cpu().numpy())
            lesion_labels.append(batch["lesion_labels"][row, :source_length].cpu().numpy())
            z_bins.append(np.floor(np.linspace(0, 1, length)[:source_length] * int(config.get("z_bins", 10))).clip(0, int(config.get("z_bins", 10)) - 1))
            targets = future_targets(outputs["slice_features"][row : row + 1, :length], horizon)
            pred = outputs["pred_future_from_world"][row : row + 1, :source_length]
            err = ((pred - targets) ** 2).mean(dim=-1).mean(dim=-1).squeeze(0)
            future_error.append(err.cpu().numpy())
    payload = {
        "anatomy": np.concatenate(anatomy, axis=0),
        "lesion": np.concatenate(lesion, axis=0),
        "uncertainty": np.concatenate(uncertainty, axis=0),
        "lesion_labels": np.concatenate(lesion_labels, axis=0),
        "z_bins": np.concatenate(z_bins, axis=0),
        "future_error": np.concatenate(future_error, axis=0),
    }
    output_path = Path(config.get("factor_output", "outputs/factors.npz"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path


def fit_probe_summary(factor_path: Path, output_path: str) -> None:
    try:
        from scipy.stats import spearmanr
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, average_precision_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        write_json(output_path, {"factor_file": str(factor_path), "probe_status": "skipped_missing_sklearn_or_scipy"})
        return

    data = np.load(factor_path)
    summary = {"factor_file": str(factor_path)}
    for factor_name, target_name, metric_name in [
        ("lesion", "lesion_labels", "lesion_auprc"),
        ("anatomy", "z_bins", "z_bin_accuracy"),
    ]:
        x_train, x_test, y_train, y_test = train_test_split(data[factor_name], data[target_name], test_size=0.2, random_state=7)
        probe = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        if metric_name == "lesion_auprc":
            score = average_precision_score(y_test, probe.predict_proba(x_test)[:, 1])
        else:
            score = accuracy_score(y_test, probe.predict(x_test))
        summary[metric_name] = float(score)
    x_train, x_test, y_train, y_test = train_test_split(data["uncertainty"], data["future_error"], test_size=0.2, random_state=7)
    probe = Ridge().fit(x_train, y_train)
    summary["error_spearman"] = float(spearmanr(y_test, probe.predict(x_test)).correlation)
    write_json(output_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    factor_path = export_factors(config)
    fit_probe_summary(factor_path, config.get("probe_output", "outputs/factor_probe_summary.json"))


if __name__ == "__main__":
    main()
