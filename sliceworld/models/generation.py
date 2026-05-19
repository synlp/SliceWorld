from typing import Iterable, List, Optional

import torch


DEFAULT_PROMPT = "Generate a concise CT radiology report from the SliceWorld tokens."


def tokenize_prompts(tokenizer, prompts: Iterable[str], device: torch.device):
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def tokenize_reports(tokenizer, reports: Iterable[str], max_length: int, device: torch.device):
    encoded = tokenizer(
        list(reports),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return {key: value.to(device) for key, value in encoded.items()}, labels.to(device)


@torch.no_grad()
def generate_reports(
    model,
    batch: dict,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 256,
    lesion_zero: bool = False,
) -> List[str]:
    device = next(model.parameters()).device
    prompts = [prompt] * int(batch["lengths"].shape[0])
    prompt_inputs = tokenize_prompts(model.tokenizer, prompts, device=device)
    generated = model.generate(
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs["attention_mask"],
        pixel_values=batch.get("pixel_values", None).to(device) if batch.get("pixel_values", None) is not None else None,
        slice_features=batch.get("slice_features", None).to(device) if batch.get("slice_features", None) is not None else None,
        lengths=batch["lengths"].to(device),
        lesion_zero=lesion_zero,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return model.tokenizer.batch_decode(generated, skip_special_tokens=True)
