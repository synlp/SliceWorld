import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sliceworld.data.ctrate import collate_ctrg
from sliceworld.data.deeplesion import IMAGE_MEAN, IMAGE_STD, _uniform_indices


def _read_json_or_jsonl(path: str):
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open("r") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open("r") as handle:
        return json.load(handle)


def _image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def _text_from_path(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    lowered = text.lower()
    for marker in ["study_findings:", "findings:", "discussion:"]:
        index = lowered.find(marker)
        if index >= 0:
            text = text[index + len(marker) :]
            break
    return " ".join(text.split())


class M3DCapDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        split: str = "train",
        max_slices: int = 196,
        image_size: int = 224,
        slice_order_path: Optional[str] = None,
        min_report_chars: int = 5,
    ) -> None:
        self.data_root = Path(data_root)
        self.max_slices = max_slices
        self.image_size = image_size
        self.slice_order = _read_json_or_jsonl(slice_order_path) if slice_order_path else {}
        raw = _read_json_or_jsonl(annotation_path)
        if isinstance(raw, dict):
            records = raw.get(split, raw.get("validation" if split == "val" else split, []))
        else:
            records = [entry for entry in raw if entry.get("split", split) == split]
        self.samples = []
        for entry in records:
            item = self._normalize_entry(entry)
            if item is None:
                continue
            if len(item["text"]) >= min_report_chars:
                self.samples.append(item)

    def _normalize_entry(self, entry: Dict) -> Optional[Dict]:
        image_dir = entry.get("image_dir")
        npy_path = entry.get("npy")
        if image_dir is None and entry.get("image"):
            image_dir = entry["image"].replace("M3D_Cap_npy/", "M3D_Cap/")
            if image_dir.endswith(".npy"):
                image_dir = image_dir[:-4]
        text = entry.get("report") or entry.get("text_value")
        if text is None:
            text_path = entry.get("report_path") or entry.get("text")
            text = _text_from_path(self.data_root / text_path) if text_path else ""
        if image_dir is None and npy_path is None:
            return None
        if image_dir is not None and not (self.data_root / image_dir).is_dir():
            image_dir = None
        if npy_path is not None and not (self.data_root / npy_path).exists() and not Path(npy_path).exists():
            npy_path = None
        if image_dir is None and npy_path is None:
            return None
        return {
            "sample_id": entry.get("sample_id", entry.get("id", image_dir or npy_path or "")),
            "image_dir": image_dir,
            "npy": npy_path,
            "text": " ".join(str(text).split()),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _load_from_dir(self, image_dir: str) -> torch.Tensor:
        directory = self.data_root / image_dir
        files = sorted([path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        key = str(Path(image_dir))
        if key in self.slice_order:
            order = [directory / name for name in self.slice_order[key]]
            files = [path for path in order if path.exists()]
        keep = _uniform_indices(len(files), self.max_slices)
        return torch.stack([_image_tensor(files[i], self.image_size) for i in keep])

    def _load_from_npy(self, npy_path: str) -> torch.Tensor:
        path = Path(npy_path)
        if not path.is_absolute():
            path = self.data_root / path
        array = np.load(path, mmap_mode="r")
        keep = _uniform_indices(int(array.shape[0]), self.max_slices)
        tensors = []
        for i in keep:
            value = np.asarray(array[i])
            if value.ndim == 2:
                value = np.stack([value, value, value], axis=-1)
            image = Image.fromarray(value.astype(np.uint8), mode="RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
            tensors.append((tensor - IMAGE_MEAN) / IMAGE_STD)
        return torch.stack(tensors)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        if item.get("npy"):
            pixel_values = self._load_from_npy(item["npy"])
        else:
            pixel_values = self._load_from_dir(item["image_dir"])
        return {
            "sample_id": item["sample_id"],
            "pixel_values": pixel_values,
            "text": item["text"],
            "positions": torch.linspace(0.0, 1.0, steps=pixel_values.shape[0], dtype=torch.float32),
            "length": int(pixel_values.shape[0]),
        }


collate_m3dcap = collate_ctrg
