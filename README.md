# SegMam — Camera + Pseudo-LiDAR BEV Segmentation with a Lightweight Mamba Lifting Block

SegMam performs joint **BEV map and object segmentation** on the nuScenes dataset by fusing surround-view camera features with **pseudo-LiDAR** (radar-format) point features through a transformer-based lifter whose latent BEV grid is refined by a **lightweight Mamba state-space block**. The lightweight Mamba block is obtained by **structural pruning of a trained heavy variant** (`d_state` 8→4, `expand` 2→1) and is used **eval-time only** — no additional training is required to reproduce the headline result.

## Result (nuScenes trainval, DRN val split, single RTX PRO 6000)

| Metric                                  | Value          |
| --------------------------------------- | -------------- |
| **Drivable-area IoU — ALL**             | **81.6**       |
| Drivable-area IoU — DAY / RAIN / NIGHT  | 83.5 / 77.6 / 70.3 |
| Mean map IoU — ALL                      | 51.7           |
| Object IoU — ALL                        | 50.0           |
| Object IoU — 0–20 m / 20–35 m / 35–50 m | 70.2 / 48.1 / 29.0 |
| **Forward latency (1 frame)**           | **~335 ms**    |

All numbers come from the released checkpoint `segmam_pseudolidar_mambalight_sliced/model-000075000.pth` evaluated with `SEGMAM_MAMBA_LIGHT=1`. The drivable-area IoU is reported at the class-specific best threshold (`drivable_th=40`), matching the protocol of the baseline paper.

## Method (brief)

1. **Image encoder.** Frozen DINOv2 ViT (with adapter), producing multi-scale image features at the patch-aligned input resolution `448 × 896`.
2. **Pseudo-LiDAR encoder.** Sparse voxel encoder (VoxelNet-style SVFE) that turns the 5-sweep accumulated pseudo-LiDAR point cloud into a BEV feature map. The input modality is structurally the same 18-channel sweep stack used by radar-based BEV fusion methods; in this release we treat it as pseudo-LiDAR (depth-derived 3D points lifted into the radar feature layout).
3. **Transformer lift + fuse.** Deformable-attention lifting from image features into a 200 × 200 BEV grid, fused with the pseudo-LiDAR feature map. The BEV queries are initialized from image features and combined with a learned query bank.
4. **Lightweight self-Mamba refinement.** A 6-layer Mamba state-space block refines the BEV latents. The released model uses the **light** variant (`d_state=4`, `expand=1`, `d_inner=128`) obtained by slicing the trained heavy block's weights — no additional training. Selective-scan cost scales with `d_inner × d_state`, so the light variant is ≈ 4 × cheaper at inference than the heavy one.
5. **BEV decoders.** Two heads share the BEV trunk: a binary BCE drivable/object head (pos-weight ≈ 2.13) and a multi-label sigmoid focal loss (`alpha=0.25, gamma=3`) map head.

## Repository layout

```
SegMam/
├── train.py                 # single-GPU training (with fine-tune knobs)
├── eval.py                  # nuScenes val mIoU + DRN split
├── vis_eval.py              # qualitative BEV renderings + per-frame latency
├── nuscenes_data.py         # NuscData dataset + compile_data
├── saverloader.py           # checkpoint save / load
├── custom_nuscenes_splits.py
├── nuscenes_image_converter.py  # optional: pre-scale nuScenes images
├── configs/
│   ├── train/train_segmam_mambalight.yaml
│   ├── eval/eval_segmam_mambalight.yaml
│   └── vis/vis_segmam_lightfinal.yaml
├── nets/
│   ├── segnet_transformer_lift_fuse_new_decoders.py   # SegnetTransformerLiftFuse
│   ├── voxelnet.py                                    # pseudo-LiDAR encoder
│   ├── dino_v2_with_adapter/                          # frozen image encoder
│   └── ops/                                           # Multi-Scale Deformable Attention CUDA op
├── utils/                   # geom / vox / improc helpers
├── CrossMamba/example.py    # self_mamba block (Mamba SSM wrapper)
├── mamba/                   # vendored mamba_ssm source (used by CrossMamba)
├── checkpoints/             # released checkpoint goes here
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

## Installation

### 1. Conda env

```bash
conda env create -f environment.yml
conda activate segmam   # the env name in environment.yml is 'bevcar' historically; rename if you like
```

Pip-only fallback: `pip install -r requirements.txt`.

System CUDA toolkit `12.6` is required to build the deformable-attention op; PyTorch CUDA `11.8` wheels are compatible with this toolkit.

### 2. Build the Multi-Scale Deformable Attention CUDA op

This is required for the transformer lifter (the only model variant in this release).

```bash
cd nets/ops
bash make.sh        # runs `python setup.py build install`
cd ../..
```

### 3. Install `mamba_ssm`

The lightweight Mamba block depends on `mamba_ssm`. The repo ships a vendored copy under `mamba/`; install it from source:

```bash
pip install -e mamba/
```

(Alternatively `pip install mamba-ssm==1.x` from PyPI — the vendored copy is provided to pin a tested version.)

### 4. PYTHONPATH

The training/eval scripts expect `nets/ops` to be importable:

```bash
export PYTHONPATH=$(pwd)/nets/ops:$PYTHONPATH
```

## Dataset preparation

### 1. Download nuScenes trainval

Get the full nuScenes trainval (≈ 350 GB) from <https://www.nuscenes.org/nuscenes#download>. After extraction the layout should be:

```
/path/to/nuscenes/trainval/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/        # metadata JSONs
└── ...
```

Point every `data_dir:` field in `configs/*/*.yaml` to this path.

### 2. (Optional) Pre-scale the camera images

To avoid resizing on every dataloader call, pre-scale the camera images once to the model input resolution (`448 × 896`):

```bash
python nuscenes_image_converter.py \
    --dataroot /path/to/nuscenes/trainval \
    --out_dataroot datasets/scaled_images \
    --image_size 448 896
```

Then set `use_pre_scaled_imgs: true` and `custom_dataroot: datasets/scaled_images` in your config. Training/eval reads the pre-scaled JPGs at full disk-load speed.

### 3. (Optional) DRN val split

`do_drn_val_split: true` (default in the eval config) groups the val set into **Day / Rain / Night / All** subsets defined in `custom_nuscenes_splits.py` and reports per-condition mIoU.

## Pre-trained checkpoint

The released checkpoint (≈ 549 MB) is shipped with this repository via **Git LFS**:

```
./checkpoints/segmam_pseudolidar_mambalight_sliced/model-000075000.pth   # drivable IoU 81.6 / latency ~335 ms
```

Make sure Git LFS is installed before cloning so the `.pth` file resolves to the real binary:

```bash
# (one-time, system-wide)
git lfs install

# clone with LFS objects fetched
git clone https://github.com/KimJunHan/SegMam.git
cd SegMam

# if you already cloned without LFS installed:
git lfs pull
```

The configs in this repo already reference this checkpoint via the `init_dir` + `load_step: 75000` fields, so no path edits are needed.

## Evaluation

```bash
SEGMAM_MAMBA_LIGHT=1 python eval.py --config configs/eval/eval_segmam_mambalight.yaml
```

Notes:

* The `SEGMAM_MAMBA_LIGHT=1` environment variable is **required** — it builds the model with the lightweight self-Mamba block (`d_state=4`, `expand=1`). Without it the model is constructed with the heavy shapes, and the released checkpoint will fail to load with a shape-mismatch error.
* Expected output: a table of per-class IoU for **ALL / DAY / RAIN / NIGHT**, drivable-area IoU at threshold `40`, plus the per-range object IoU (0–20 m / 20–35 m / 35–50 m).
* Eval runs on a single GPU (`device_ids: [0]` in the config). One full DRN-val pass takes about 55 min on a single RTX PRO 6000.

## Training (optional — only needed to fine-tune)

The released checkpoint reproduces the paper result without retraining. The training entrypoint is provided in case you want to fine-tune the lightweight variant further:

```bash
SEGMAM_MAMBA_LIGHT=1 python train.py --config configs/train/train_segmam_mambalight.yaml
```

* Single-GPU training, `batch_size=1` with `grad_acc=5` (effective batch 5).
* Warm-starts from the released slice-initialized checkpoint.
* Checkpoints are written to `./checkpoints/<exp_name-...>/model-<step:09d>.pth`; `keep_latest=20` prunes older ones.
* TensorBoard logs go to `logs_nuscenes/<run-name>/{t,v,ev}`.

To train the heavy variant from scratch, **drop the env var** and re-point `init_dir: ''` (no warm-start). The default model in code is the heavy one; the env var is the only thing that swaps in the light shapes.

## Qualitative visualization

```bash
SEGMAM_MAMBA_LIGHT=1 python vis_eval.py --config configs/vis/vis_segmam_lightfinal.yaml
```

Writes per-frame BEV map+object renderings into `inference_img/<step>_<exp_name>_scene_<NNN>/` and prints per-stage forward latency (image encoder / lifting / fuser / pseudo-LiDAR / total).

## Reproducing the headline numbers, end to end

```bash
# (1) Build the CUDA op once
cd nets/ops && bash make.sh && cd ../..
pip install -e mamba/

# (2) Get nuScenes trainval, set data_dir in configs/eval/eval_segmam_mambalight.yaml

# (3) Drop the released checkpoint at:
#     ./checkpoints/segmam_pseudolidar_mambalight_sliced/model-000075000.pth

# (4) Evaluate
SEGMAM_MAMBA_LIGHT=1 python eval.py --config configs/eval/eval_segmam_mambalight.yaml

# (5) Check forward latency
SEGMAM_MAMBA_LIGHT=1 python vis_eval.py --config configs/vis/vis_segmam_lightfinal.yaml
```

You should see drivable IoU **ALL ≈ 81.6 / DAY ≈ 83.5 / RAIN ≈ 77.6 / NIGHT ≈ 70.3** at threshold `40`, and a forward time of **~335 ms** per frame on an RTX PRO 6000.

## Hardware

The released numbers are measured on:

* GPU: RTX PRO 6000 (Blackwell)
* CUDA toolkit 12.6, PyTorch 2.x with CUDA 11.8 wheels
* OS: Ubuntu (Linux 6.14)

Other modern GPUs (≥ 24 GB VRAM) should also work; the forward-latency numbers will differ.

## Voxel grid (hard-coded in `train.py` / `eval.py` / `vis_eval.py`)

```
XMIN, XMAX = -50, 50      # ego-centric meters
ZMIN, ZMAX = -50, 50
YMIN, YMAX =  -5, 5
Z, Y, X    = 200, 8, 200
scene_centroid = (0, 1, 0) # 1 m below camera ego
```

These are not exposed via YAML — edit the scripts if you need a different range/resolution.

## Acknowledgements

This repository builds on the BEVCar codebase (Schramm et al., 2024, *BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation*, arXiv:2403.11761) and on the Mamba state-space block (Gu & Dao, 2023). The deformable-attention CUDA op is from Deformable DETR. nuScenes is © Motional.

## License

See [`LICENSE`](LICENSE).
