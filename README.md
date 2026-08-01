# UIE3_VN

UIE3_VN is a new, independent PyTorch project for versioned underwater image
enhancement experiments. Version 1 is the previously verified UIE3
NAFNet-small baseline migrated without the old ORDER_STUDY phase runtime
governance.

NAFNet-small is this project's research configuration of the ordinary NAFNet
class. It is not an official upstream fixed model and no official pretrained
weights are loaded.

## v1 semantics

The fixed model configuration is:

```yaml
img_channel: 3
width: 32
enc_blk_nums: [2, 2, 2]
middle_blk_num: 4
dec_blk_nums: [2, 2, 2]
```

Inputs and targets are Pillow-decoded RGB float32 tensors in `[0,1]`. The
model accepts and returns `[B,3,H,W]`. Its three encoder scales require an
internal multiple of 8, so ordinary NAFNet pads only the right and bottom with
zeros, applies exactly one global residual inside `NAFNet.forward`, and crops
back to the original height and width. No wrapper residual, sigmoid, or model
clamp is added. Predictions are clamped to `[0,1]` only for metrics and saved
enhanced images.

## Layout

```text
configs/configV1.yaml       the default v1 configuration
splits/lsui19/              migrated train/validation/test TSV manifests
src/common/                 data, metrics, run, checkpoint, and grid utilities
src/v1/                     v1 model, loss, trainer, train/test entry points
third_party/nafnet/         vendored NAFNet, license, and provenance
tests/                      static-flow and optional PyTorch runtime tests
experiments/                ignored run output root
```

## Configure and train

Before each new seed run, edit these two matching values in
`configs/configV1.yaml`:

```yaml
experiment:
  name: NAFNet_small_seed1234
  seed: 1234
```

The `seed<number>` token in the name must equal `experiment.seed`; mismatches
are rejected. The dataset path and the single v1 protocol are also stored in
that YAML.

Start training from the repository root:

```bash
sh src/v1/run.sh
```

The Python entry point uses `configs/configV1.yaml` by default. A v1 YAML at
any other repository or filesystem location can be selected explicitly; its
`version`, `variant`, `model_version`, or other version-like fields must all
identify v1:

```bash
python -m src.v1.train_v1 --config /tmp/UIE3_VN_v1_smoke/configV1_smoke.yaml
```

A tmux session is optional and remains under user control:

```bash
tmux new -s v1_seed1234
sh src/v1/run.sh
```

Every epoch automatically runs the complete validation manifest without
random crop or augmentation. Validation PSNR, SSIM, and validation loss update
three independent best checkpoints. The formal primary checkpoint is always
`best/best_psnr.pt`.

Only after all 200 epochs complete successfully, training automatically runs
the complete test manifest exactly once with `best/best_psnr.pt`. Test results
cannot update checkpoints, change training length, or select hyperparameters.

Each invocation creates a collision-safe directory such as:

```text
experiments/v1_NAFNet_small_seed1234_20260730_225800/
```

Its immutable `config.json` snapshot makes historical runs independent of
later YAML edits. Metrics, logs, statuses, errors, checkpoints, and results
are kept under that run. `experiments/*` is ignored by Git except for
`experiments/.gitkeep`.

## Test visualization

After full test inference, the code sorts candidates by `sample_id` and then
uses `random.Random(3407)` to draw ten distinct samples without looking at
quality or metrics. It saves:

```text
result/test_visualization_samples.json
result/test_samples/{sample_id}_enhanced.png
result/test_grid_10x3.png
```

The grid is ten rows by three columns in `Input / Enhanced / GT` order. Each
cell is 512×512, aspect ratio is preserved with uniform padding, and the
default Pillow font labels the sample and column. If fewer than ten test
samples exist, all are used once and unused grid rows remain blank. Resizing
is visualization-only and never feeds PSNR or SSIM.

## Standalone test

To retest an existing run, set its path in the current YAML:

```yaml
test:
  run_dir: experiments/v1_NAFNet_small_seed1234_20260730_225800
  checkpoint: best_psnr
```

Then run:

```bash
sh src/v1/test.sh
```

As with training, a different v1 YAML can be selected with
`python -m src.v1.test_v1 --config /path/to/config.yaml`.

The standalone entry reads the run's own `config.json` for model, data,
metric, and visualization semantics. It only accepts `best_psnr` and writes
into that run's `result/`. Existing test outputs are rejected while
`allow_overwrite: false`; set it to `true` only when replacement is intended.

## Resume

Set `training.resume` to an existing v1 checkpoint, normally:

```yaml
training:
  resume: experiments/<run>/checkpoint/last.pt
```

Then run `sh src/v1/run.sh`. Resume strictly restores model, optimizer,
GradScaler, global step, all tracked best values, and Python/NumPy/PyTorch/CUDA
RNG states. It creates a new unique run and carries forward available metric
history and independent best files from the source run. Apart from the resume
path and standalone-test controls, checkpoint and current configuration
semantics must match.

## Checks

Install a compatible PyTorch separately for the machine; do not let the
project choose or upgrade a CUDA wheel. Then run:

```bash
python -m compileall -q src tests third_party
bash -n src/v1/run.sh
bash -n src/v1/test.sh
python -m pytest -q
git diff --check
```

The test suite covers config validation, unique run creation, independent
best selection, fixed metrics, model shapes/padding/residual behavior,
checkpoint save/load/resume, reproducible visualization, and automatic test
lifecycle behavior.

Test-set results are final reporting only. Do not use them for checkpoint
selection, early stopping, model choice, or hyperparameter tuning.

See `MIGRATION_V1_REPORT.md` for provenance and validation status and
`GITHUB_PUSH_GUIDE.md` for local review and optional GitHub publication.
