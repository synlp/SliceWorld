# SliceWorld Release Code

This directory is a clean implementation of the SliceWorld method in the accompanying manuscript.

## Install

```bash
cd release_code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Not included assets: DINOv2 weights, the Mamba runtime dependency, Qwen or Ministral weights, LoRA or full checkpoints, official DeepLesion files, official M3D-Cap files, and official CT-RATE files.

## Data Preparation

DeepLesion world-model pretraining:

```bash
python - <<'PY'
from sliceworld.data.deeplesion import DeepLesionSequenceBuilder
DeepLesionSequenceBuilder(
    images_root="data/deeplesion/Images_png",
    dl_info_csv="data/deeplesion/DL_info.csv",
).write_manifest("manifests/deeplesion_train.jsonl", split="train")
PY
```

CT-RATE HU preprocessing:

```bash
python -m sliceworld.data.preprocess_ctrate_hu \
  --input-root data/ctrate_nifti \
  --output-root data/ctrate_preprocessed \
  --manifest data/ctrate_preprocessed/manifest_train.jsonl
```

For CT-RATE training, join the generated slice-array paths with the official report metadata so each manifest row also contains `reports` and `split`.

M3D-Cap training reads paired CT-report entries from `data/m3d_cap/M3D_Cap.json`.

Expected manifests:

```json
{"sample_id":"case_001","split":"train","slice_paths":["/path/001.png"],"slice_indices":[1],"lesion_presence":[0]}
{"volume":"valid_001.nii.gz","split":"valid","npy":"valid/valid_001.npy","reports":{"Findings_EN":"...","Impressions_EN":"..."},"labels":{}}
{"sample_id":"case_001","split":"train","image_dir":"M3D_Cap/ct_case/case_001/Axial","text_value":"..."}
```

The CT-RATE preprocessor reads HU-valued NIfTI arrays directly. With `--axial-axis auto`, the slice axis is selected from the affine direction closest to superior-inferior z and reversed when needed so the saved array follows ascending physical z.

## Train

World-model pretraining on DeepLesion:

```bash
bash scripts/run_pretrain_deeplesion.sh
```

CT-RATE report fine-tuning with Qwen3-1.7B:

```bash
bash scripts/run_finetune_ctrate.sh
```

M3D-Cap fine-tuning uses `configs/finetune_m3dcap_qwen3_1p7b.yaml`.

## Evaluate

```bash
bash scripts/run_eval_ctrate.sh
python -m sliceworld.eval.evaluate_nsp --config configs/eval_deeplesion_world.yaml
python -m sliceworld.eval.evaluate_factor_alignment --config configs/eval_deeplesion_world.yaml
python -m sliceworld.eval.evaluate_counterfactual --config configs/eval_ctrate.yaml
```

CTRG evaluation exports generated reports and lexical metrics. NSP evaluation is prefix-only, predicts the next `K=5` slice features from each prefix state, and compares SliceWorld with persistence and linear extrapolation baselines. Factor evaluation exports frozen anatomy, lesion, and uncertainty factors. Counterfactual evaluation generates factual and lesion-zero reports using the same decoder prompt and decoding strategy.

## Smoke Test

```bash
python -m pytest tests/test_smoke_forward.py
```

The smoke test uses synthetic DINOv2-shaped features and a small local Mamba-compatible dry-run backend to verify factor construction, lesion-zero projection, multi-step NSP/FAS/CF losses, prefix causality, and LLM-prefix projection.
