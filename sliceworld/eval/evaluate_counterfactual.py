import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sliceworld.data.ctrate import CTRATEDataset, collate_ctrg
from sliceworld.models.generation import DEFAULT_PROMPT, generate_reports
from sliceworld.models.sliceworld import SliceWorldForCTRG
from sliceworld.utils.config import get_device, load_config
from sliceworld.utils.io import load_model_weights, write_json
from sliceworld.utils.metrics import counterfactual_lesion_summary


@torch.no_grad()
def evaluate(config: dict) -> None:
    device = get_device(config.get("device", "auto"))
    model = SliceWorldForCTRG(config["model"]).to(device).eval()
    load_model_weights(model, config["checkpoint"], strict=False)
    dataset = CTRATEDataset(
        preprocessed_root=config["data"]["preprocessed_root"],
        manifest_path=config["data"]["manifest_path"],
        split=config["data"].get("split", "valid"),
        max_slices=int(config["data"].get("max_sampled_slices", 480)),
        report_field=config["data"].get("report_field", "findings_impressions"),
    )
    loader = DataLoader(dataset, batch_size=int(config["data"].get("batch_size", 1)), shuffle=False, collate_fn=collate_ctrg)
    factual = []
    lesion_zero = []
    rows = []
    prompt = config.get("generation", {}).get("prompt", DEFAULT_PROMPT)
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 256))
    for batch in tqdm(loader, desc="counterfactual"):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        factual_reports = generate_reports(model, batch, prompt=prompt, max_new_tokens=max_new_tokens, lesion_zero=False)
        zero_reports = generate_reports(model, batch, prompt=prompt, max_new_tokens=max_new_tokens, lesion_zero=True)
        factual.extend(factual_reports)
        lesion_zero.extend(zero_reports)
        for sample_id, fact, zero in zip(batch["sample_ids"], factual_reports, zero_reports):
            rows.append({"sample_id": sample_id, "factual": fact, "lesion_zero": zero})
    metrics = counterfactual_lesion_summary(factual, lesion_zero)
    write_json(config.get("output_path", "outputs/eval_counterfactual.json"), {"metrics": metrics, "rows": rows})
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
