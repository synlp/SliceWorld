import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sliceworld.data.ctrate import CTRATEDataset, collate_ctrg
from sliceworld.data.m3dcap import M3DCapDataset, collate_m3dcap
from sliceworld.models.generation import DEFAULT_PROMPT, tokenize_prompts, tokenize_reports
from sliceworld.models.losses import SliceWorldObjective, loss_config_from_dict
from sliceworld.models.sliceworld import SliceWorldForCTRG
from sliceworld.utils.config import get_device, load_config, set_seed
from sliceworld.utils.io import load_model_weights, save_checkpoint


def build_dataset(config: dict):
    data = config["data"]
    if data["dataset"] == "ctrate":
        return CTRATEDataset(
            preprocessed_root=data["preprocessed_root"],
            manifest_path=data["manifest_path"],
            split=data.get("split", "train"),
            max_slices=int(data.get("max_sampled_slices", 480)),
            report_field=data.get("report_field", "findings_impressions"),
        ), collate_ctrg
    if data["dataset"] == "m3dcap":
        return M3DCapDataset(
            data_root=data["data_root"],
            annotation_path=data["annotation_path"],
            split=data.get("split", "train"),
            max_slices=int(data.get("max_sampled_slices", 196)),
            slice_order_path=data.get("slice_order_path"),
        ), collate_m3dcap
    raise ValueError(f"Unknown dataset: {data['dataset']}")


def train(config: dict) -> None:
    set_seed(int(config.get("seed", 7)))
    device = get_device(config.get("device", "auto"))
    model = SliceWorldForCTRG(config["model"]).to(device)
    checkpoint = config["model"].get("world_checkpoint")
    if checkpoint:
        load_model_weights(model.core, checkpoint, strict=False)
    objective = SliceWorldObjective(loss_config_from_dict(config["loss"]))
    dataset, collate_fn = build_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    grad_accum = int(config["optimizer"].get("gradient_accumulation", 1))
    max_text_length = int(config["data"]["max_text_length"])
    prompt = config.get("generation", {}).get("prompt", DEFAULT_PROMPT)
    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(config["optimizer"]["epochs"])):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}")
        for micro_step, batch in enumerate(progress, start=1):
            prompts = [prompt] * len(batch["texts"])
            prompt_inputs = tokenize_prompts(model.tokenizer, prompts, device)
            report_inputs, labels = tokenize_reports(model.tokenizer, batch["texts"], max_text_length, device)
            input_ids = torch.cat([prompt_inputs["input_ids"], report_inputs["input_ids"]], dim=1)
            attention_mask = torch.cat([prompt_inputs["attention_mask"], report_inputs["attention_mask"]], dim=1)
            prompt_ignore = torch.full_like(prompt_inputs["input_ids"], -100)
            labels = torch.cat([prompt_ignore, labels], dim=1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=batch["pixel_values"].to(device),
                lengths=batch["lengths"].to(device),
            )
            losses = objective(outputs, lengths=batch["lengths"].to(device), ctrg_loss=outputs["ctrg_loss"])
            (losses["loss"] / grad_accum).backward()
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimizer"].get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            progress.set_postfix(loss=f"{losses['loss'].detach().float().item():.4f}")
        checkpoint_dir = Path(config.get("output_dir", "outputs/ctrg_finetuning"))
        save_checkpoint(str(checkpoint_dir / f"epoch_{epoch + 1}.pt"), model, optimizer, step, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
