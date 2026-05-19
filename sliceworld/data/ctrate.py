import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sliceworld.data.deeplesion import IMAGE_MEAN, IMAGE_STD, _uniform_indices


def _read_records(path: str) -> List[Dict]:
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open("r") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open("r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    records = []
    for split, items in data.items():
        for item in items:
            item = dict(item)
            item.setdefault("split", split)
            records.append(item)
    return records


def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    values = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (values - IMAGE_MEAN) / IMAGE_STD


def report_from_entry(entry: Dict, field: str = "findings_impressions") -> str:
    reports = entry.get("reports", entry)
    findings = (reports.get("Findings_EN") or reports.get("findings") or "").strip()
    impressions = (reports.get("Impressions_EN") or reports.get("impressions") or "").strip()
    if field == "findings":
        return findings
    if field == "impressions":
        return impressions
    if field == "findings_impressions":
        return "\n\n".join(part for part in [findings, impressions] if part)
    return (reports.get(field) or "").strip()


class CTRATEDataset(Dataset):
    def __init__(
        self,
        preprocessed_root: str,
        manifest_path: str,
        split: str = "train",
        max_slices: int = 480,
        report_field: str = "findings_impressions",
        min_report_chars: int = 5,
    ) -> None:
        self.preprocessed_root = Path(preprocessed_root)
        self.max_slices = max_slices
        self.report_field = report_field
        self.samples = []
        for entry in _read_records(manifest_path):
            if entry.get("split", split) != split:
                continue
            report = report_from_entry(entry, report_field)
            if len(report) < min_report_chars:
                continue
            npy_path = Path(entry.get("npy", entry.get("slice_array", "")))
            if not npy_path.is_absolute():
                npy_path = self.preprocessed_root / npy_path
            if npy_path.exists():
                item = dict(entry)
                item["npy_path"] = str(npy_path)
                item["report"] = report
                self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        array = np.load(item["npy_path"], mmap_mode="r")
        keep = _uniform_indices(int(array.shape[0]), self.max_slices)
        pixel_values = torch.stack([array_to_tensor(np.asarray(array[i])) for i in keep])
        return {
            "sample_id": item.get("volume", item.get("sample_id", Path(item["npy_path"]).stem)),
            "pixel_values": pixel_values,
            "text": item["report"],
            "positions": torch.linspace(0.0, 1.0, steps=len(keep), dtype=torch.float32),
            "length": len(keep),
            "labels": item.get("labels", {}),
        }


def collate_ctrg(batch: Iterable[Dict]) -> Dict:
    items = list(batch)
    max_length = max(item["length"] for item in items)
    channels, height, width = items[0]["pixel_values"].shape[1:]
    pixel_values = torch.zeros((len(items), max_length, channels, height, width), dtype=torch.float32)
    positions = torch.zeros((len(items), max_length), dtype=torch.float32)
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.long)
    for row, item in enumerate(items):
        length = item["length"]
        pixel_values[row, :length] = item["pixel_values"]
        positions[row, :length] = item["positions"]
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "pixel_values": pixel_values,
        "positions": positions,
        "lengths": lengths,
        "texts": [item["text"] for item in items],
        "labels": [item.get("labels", {}) for item in items],
    }
