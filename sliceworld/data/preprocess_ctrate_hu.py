import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np
from PIL import Image


WINDOWS = {
    "lung": (-600.0, 1500.0),
    "soft_tissue": (40.0, 400.0),
    "bone": (400.0, 1500.0),
}


def window_hu(slice_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    scaled = (np.clip(slice_hu, low, high) - low) / (high - low)
    return np.round(scaled * 255.0).astype(np.uint8)


def hu_slice_to_rgb(slice_hu: np.ndarray, size: int = 224) -> np.ndarray:
    channels = [window_hu(slice_hu, *WINDOWS[name]) for name in ["lung", "soft_tissue", "bone"]]
    rgb = np.stack(channels, axis=-1)
    image = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.BOX)
    return np.asarray(image, dtype=np.uint8)


def choose_axial_axis(array: np.ndarray, requested: str, affine: Optional[np.ndarray] = None) -> int:
    if requested != "auto":
        return int(requested)
    if affine is not None:
        direction = np.abs(affine[:3, :3][2, :])
        if np.any(direction > 0):
            return int(np.argmax(direction))
    return int(np.argmin(array.shape))


def convert_nifti_to_array(nifti_path: str, size: int = 224, axial_axis: str = "auto") -> np.ndarray:
    image = nib.load(nifti_path)
    hu = np.asanyarray(image.dataobj).astype(np.float32)
    axis = choose_axial_axis(hu, axial_axis, image.affine)
    slices = np.moveaxis(hu, axis, 0)
    if image.affine[2, axis] < 0:
        slices = slices[::-1]
    return np.stack([hu_slice_to_rgb(slice_hu, size=size) for slice_hu in slices], axis=0)


def iter_nifti_files(input_root: str) -> Iterable[Path]:
    root = Path(input_root)
    yield from sorted(root.rglob("*.nii"))
    yield from sorted(root.rglob("*.nii.gz"))


def preprocess_tree(input_root: str, output_root: str, size: int, axial_axis: str) -> List[Dict]:
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)
    records = []
    for nifti_path in iter_nifti_files(input_root):
        relative = nifti_path.relative_to(input_root_path)
        stem = relative.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = output_root_path / relative.parent / f"{stem}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        array = convert_nifti_to_array(str(nifti_path), size=size, axial_axis=axial_axis)
        np.save(output_path, array)
        records.append(
            {
                "volume": str(relative),
                "npy": str(output_path.relative_to(output_root_path)),
                "n_slices": int(array.shape[0]),
                "shape": list(array.shape),
                "windows": WINDOWS,
                "target_size": size,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--axial-axis", default="auto")
    args = parser.parse_args()
    records = preprocess_tree(args.input_root, args.output_root, args.size, args.axial_axis)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
