import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SPLIT_MAP = {"1": "train", "2": "val", "3": "test", 1: "train", 2: "val", 3: "test"}
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _parse_int(value: str) -> int:
    return int(float(str(value).strip()))


def _parse_slice_range(value: str) -> Tuple[int, int]:
    parts = [int(float(x.strip())) for x in str(value).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid DeepLesion Slice_range: {value}")
    return min(parts), max(parts)


def _slice_index_from_name(path: Path) -> Optional[int]:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return None
    return int(numbers[-1])


def _series_dir_name(patient_id: int, study_id: int, series_id: int) -> str:
    return f"{patient_id:06d}_{study_id:02d}_{series_id:02d}"


def _uniform_indices(length: int, max_items: int) -> List[int]:
    if length <= max_items:
        return list(range(length))
    values = np.linspace(0, length - 1, max_items)
    return [int(round(x)) for x in values]


def image_to_tensor(path: Path, image_size: int = 224) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


class DeepLesionSequenceBuilder:
    def __init__(self, images_root: str, dl_info_csv: str) -> None:
        self.images_root = Path(images_root)
        self.dl_info_csv = Path(dl_info_csv)

    def read_rows(self) -> List[Dict]:
        rows = []
        with self.dl_info_csv.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                patient_id = _parse_int(row["Patient_index"])
                study_id = _parse_int(row["Study_index"])
                series_id = _parse_int(row["Series_ID"])
                split = SPLIT_MAP[row["Train_Val_Test"]]
                slice_start, slice_end = _parse_slice_range(row["Slice_range"])
                rows.append(
                    {
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "series_id": series_id,
                        "split": split,
                        "series_dir": _series_dir_name(patient_id, study_id, series_id),
                        "key_slice_index": _parse_int(row["Key_slice_index"]),
                        "slice_range": [slice_start, slice_end],
                    }
                )
        return rows

    def build_entries(self, split: Optional[str] = None) -> List[Dict]:
        grouped: Dict[Tuple[int, int, int], Dict] = defaultdict(lambda: {"annotations": []})
        for row in self.read_rows():
            if split is not None and row["split"] != split:
                continue
            key = (row["patient_id"], row["study_id"], row["series_id"])
            grouped[key].update(
                {
                    "sample_id": row["series_dir"],
                    "patient_id": row["patient_id"],
                    "study_id": row["study_id"],
                    "series_id": row["series_id"],
                    "split": row["split"],
                    "series_dir": row["series_dir"],
                }
            )
            grouped[key]["annotations"].append(row)

        entries = []
        for item in grouped.values():
            series_dir = self.images_root / item["series_dir"]
            if not series_dir.is_dir():
                continue
            slices = []
            for path in sorted(series_dir.glob("*.png")):
                index = _slice_index_from_name(path)
                if index is not None:
                    slices.append((index, path))
            slices.sort(key=lambda pair: pair[0])
            if len(slices) < 2:
                continue
            lesion_ranges = [annotation["slice_range"] for annotation in item["annotations"]]
            labels = []
            for slice_index, _ in slices:
                present = any(start <= slice_index <= end for start, end in lesion_ranges)
                labels.append(1 if present else 0)
            entries.append(
                {
                    "sample_id": item["sample_id"],
                    "patient_id": item["patient_id"],
                    "study_id": item["study_id"],
                    "series_id": item["series_id"],
                    "split": item["split"],
                    "slice_paths": [str(path) for _, path in slices],
                    "slice_indices": [index for index, _ in slices],
                    "lesion_presence": labels,
                }
            )
        return entries

    def write_manifest(self, output_path: str, split: Optional[str] = None) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            for entry in self.build_entries(split=split):
                handle.write(json.dumps(entry) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    with Path(path).open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DeepLesionSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        max_slices: int = 196,
        image_size: int = 224,
    ) -> None:
        self.samples = [item for item in read_jsonl(manifest_path) if item.get("split", split) == split]
        self.max_slices = max_slices
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        keep = _uniform_indices(len(item["slice_paths"]), self.max_slices)
        paths = [Path(item["slice_paths"][i]) for i in keep]
        images = torch.stack([image_to_tensor(path, self.image_size) for path in paths])
        labels = torch.tensor([item["lesion_presence"][i] for i in keep], dtype=torch.float32)
        positions = torch.linspace(0.0, 1.0, steps=len(keep), dtype=torch.float32)
        return {
            "sample_id": item["sample_id"],
            "pixel_values": images,
            "lesion_labels": labels,
            "positions": positions,
            "length": len(keep),
        }


def collate_deeplesion(batch: Iterable[Dict]) -> Dict:
    items = list(batch)
    max_length = max(item["length"] for item in items)
    channels, height, width = items[0]["pixel_values"].shape[1:]
    pixel_values = torch.zeros((len(items), max_length, channels, height, width), dtype=torch.float32)
    lesion_labels = torch.zeros((len(items), max_length), dtype=torch.float32)
    positions = torch.zeros((len(items), max_length), dtype=torch.float32)
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.long)
    sample_ids = []
    for row, item in enumerate(items):
        length = item["length"]
        pixel_values[row, :length] = item["pixel_values"]
        lesion_labels[row, :length] = item["lesion_labels"]
        positions[row, :length] = item["positions"]
        sample_ids.append(item["sample_id"])
    return {
        "sample_ids": sample_ids,
        "pixel_values": pixel_values,
        "lesion_labels": lesion_labels,
        "positions": positions,
        "lengths": lengths,
    }
