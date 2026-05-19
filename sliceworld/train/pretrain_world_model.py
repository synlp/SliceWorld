import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sliceworld.data.deeplesion import DeepLesionSequenceBuilder, DeepLesionSequenceDataset, collate_deeplesion
from sliceworld.models.losses import SliceWorldObjective, loss_config_from_dict
from sliceworld.models.sliceworld import SliceWorldCore
from sliceworld.utils.config import get_device, load_config, set_seed
from sliceworld.utils.io import save_checkpoint


def make_synthetic_batch(config: dict, device: torch.device) -> dict:
    batch_size = int(config.get("data", {}).get("batch_size", 2))
    length = min(8, int(config.get("data", {}).get("max_sampled_slices", 8)))
    visual_dim = int(config.get("model", {}).get("slice_encoder", {}).get("hidden_dim", 1024))
    return {
        "slice_features": torch.randn(batch_size, length, visual_dim, device=device),
        "lesion_labels": torch.randint(0, 2, (batch_size, length), device=device).float(),
        "lengths": torch.full((batch_size,), length, dtype=torch.long, device=device),
    }


def build_dataset(config: dict):
    data = config["data"]
    manifest_path = data.get("manifest_path")
    if not manifest_path and data.get("images_root") and data.get("dl_info_csv"):
        manifest_path = data.get("generated_manifest_path", "manifests/deeplesion_train.jsonl")
        DeepLesionSequenceBuilder(data["images_root"], data["dl_info_csv"]).write_manifest(manifest_path, split=data.get("split", "train"))
    return DeepLesionSequenceDataset(
        manifest_path=manifest_path,
        split=data.get("split", "train"),
        max_slices=int(data.get("max_sampled_slices", 196)),
        image_size=int(data.get("image_size", 224)),
    )


def train(config: dict, dry_run: bool = False) -> None:
    set_seed(int(config.get("seed", 7)))
    device = get_device(config.get("device", "auto"))
    model = SliceWorldCore(config["model"]).to(device)
    objective = SliceWorldObjective(loss_config_from_dict(config["loss"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    if dry_run:
        batch = make_synthetic_batch(config, device)
        outputs = model(slice_features=batch["slice_features"], lengths=batch["lengths"])
        losses = objective(outputs, lesion_labels=batch["lesion_labels"], lengths=batch["lengths"])
        losses["loss"].backward()
        optimizer.step()
        print({key: round(value.detach().float().item(), 4) for key, value in losses.items()})
        return

    dataset = build_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_deeplesion,
    )
    model.train()
    step = 0
    for epoch in range(int(config["optimizer"]["epochs"])):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            lesion_labels = batch["lesion_labels"].to(device)
            lengths = batch["lengths"].to(device)
            outputs = model(pixel_values=pixel_values, lengths=lengths)
            losses = objective(outputs, lesion_labels=lesion_labels, lengths=lengths)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimizer"].get("max_grad_norm", 1.0)))
            optimizer.step()
            step += 1
            progress.set_postfix(loss=f"{losses['loss'].detach().float().item():.4f}")
        checkpoint_dir = Path(config.get("output_dir", "outputs/world_pretraining"))
        save_checkpoint(str(checkpoint_dir / f"epoch_{epoch + 1}.pt"), model, optimizer, step, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
