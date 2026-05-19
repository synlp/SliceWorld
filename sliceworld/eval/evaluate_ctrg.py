import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sliceworld.data.ctrate import CTRATEDataset, collate_ctrg
from sliceworld.data.m3dcap import M3DCapDataset, collate_m3dcap
from sliceworld.models.generation import DEFAULT_PROMPT, generate_reports
from sliceworld.models.sliceworld import SliceWorldForCTRG
from sliceworld.utils.config import get_device, load_config
from sliceworld.utils.io import load_model_weights, write_json
from sliceworld.utils.metrics import text_metrics


def build_dataset(config: dict):
    data = config["data"]
    if data["dataset"] == "ctrate":
        return CTRATEDataset(
            preprocessed_root=data["preprocessed_root"],
            manifest_path=data["manifest_path"],
            split=data.get("split", "valid"),
            max_slices=int(data.get("max_sampled_slices", 480)),
            report_field=data.get("report_field", "findings_impressions"),
        ), collate_ctrg
    if data["dataset"] == "m3dcap":
        return M3DCapDataset(
            data_root=data["data_root"],
            annotation_path=data["annotation_path"],
            split=data.get("split", "test"),
            max_slices=int(data.get("max_sampled_slices", 196)),
            slice_order_path=data.get("slice_order_path"),
        ), collate_m3dcap
    raise ValueError(f"Unknown dataset: {data['dataset']}")


@torch.no_grad()
def evaluate(config: dict) -> None:
    device = get_device(config.get("device", "auto"))
    model = SliceWorldForCTRG(config["model"]).to(device).eval()
    load_model_weights(model, config["checkpoint"], strict=False)
    dataset, collate_fn = build_dataset(config)
    loader = DataLoader(dataset, batch_size=int(config["data"].get("batch_size", 1)), shuffle=False, collate_fn=collate_fn)
    predictions = []
    references = []
    rows = []
    prompt = config.get("generation", {}).get("prompt", DEFAULT_PROMPT)
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 256))
    for batch in tqdm(loader, desc="evaluate ctrg"):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        generated = generate_reports(model, batch, prompt=prompt, max_new_tokens=max_new_tokens)
        predictions.extend(generated)
        references.extend(batch["texts"])
        for sample_id, prediction, reference in zip(batch["sample_ids"], generated, batch["texts"]):
            rows.append({"sample_id": sample_id, "prediction": prediction, "reference": reference})
    metrics = text_metrics(predictions, references)
    output = {"metrics": metrics, "rows": rows}
    write_json(config.get("output_path", "outputs/eval_ctrg.json"), output)
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
