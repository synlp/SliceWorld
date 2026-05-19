import json
from pathlib import Path
from typing import Iterable

import torch


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str, data) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_checkpoint(path: str, model, optimizer=None, step: int = 0, config=None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "step": step, "config": config}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, output)


def load_model_weights(model, checkpoint_path: str, strict: bool = True):
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model", payload)
    return model.load_state_dict(state, strict=strict)
