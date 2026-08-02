# UIE3_VN

UIE3_VN is an independent PyTorch project for versioned underwater image
enhancement experiments. Version 1 preserves the completed three-seed
NAFNet-small baseline. Version 2 adds auditable RGB-input color/scattering
order experiments without modifying NAFNet internals or V1 behavior.

NAFNet-small is this project's research configuration of the ordinary NAFNet
class. It is not an official upstream fixed model and no official pretrained
weights are loaded.

## v1 semantics

The default NAFNet-small configuration is:

```yaml
img_channel: 3
width: 32
enc_blk_nums: [2, 2, 2]
middle_blk_num: 4
dec_blk_nums: [2, 2, 2]
```

Inputs and targets are Pillow-decoded RGB float32 tensors in `[0,1]`. Training
uses paired random `patch_size` crops (padding smaller images first when
enabled); it does not resize every source image. Validation and test use the
complete original image. The model accepts and returns `[B,3,H,W]`. Its three
encoder scales require an
internal multiple of 8, so ordinary NAFNet pads only the right and bottom with
zeros, applies exactly one global residual inside `NAFNet.forward`, and crops
back to the original height and width. No wrapper residual, sigmoid, or model
clamp is added. Predictions are clamped to `[0,1]` only for metrics and saved
enhanced images.

## v2 order study

V2 keeps the formal V1 data split, patch size, batch size, backbone, loss,
optimizer, 200-epoch schedule, metrics, and best-validation-PSNR test policy.
Its operators act only in the RGB input domain before NAFNet-small `B`:

```text
baseline:                B(x)
color_only:              B(C(x))
scatter_only:            B(S(x))
color_then_scatter:      B(S(C(x)))   # C first, then S
scatter_then_color:      B(C(S(x)))   # S first, then C
shared_order_diagnostic: Y_cs=B(S(C(x))), Y_sc=B(C(S(x)))
```

`C` predicts a bounded per-image 3×3 color matrix and RGB bias and is exactly
the identity at initialization. `S` predicts a transmission map and global
atmospheric light, then applies a bounded inverse-scattering residual. Neither
operator contains BatchNorm, Dropout, random behavior, an output sigmoid, or a
hard clamp. The shared diagnostic owns only one C, one S, and one NAFNet-small;
the two orders do not duplicate weights.

The formal seeds are `1234`, `3407`, and `3520`. V2 `baseline` exists to prove
state-dict and numerical compatibility with V1; the completed V1 baseline does
not need to be trained again under the V2 entry point.

## Layout

```text
configs/configV1.yaml       the default v1 configuration
configs/configV2.yaml       the default v2 order-study configuration
splits/lsui19/              migrated train/validation/test TSV manifests
src/common/                 data, metrics, run, checkpoint, and grid utilities
src/v1/                     v1 model, loss, trainer, train/test entry points
src/v2/                     v2 operators, shared trainer, train/test entry points
third_party/nafnet/         vendored NAFNet, license, and provenance
tests/                      static-flow and optional PyTorch runtime tests
experiments/                ignored run output root
```

## Configure and train

Before each new seed run, edit these values in
`configs/configV1.yaml`:

```yaml
experiment:
  name: NAFNet_small_seed1234
  seed: 1234
```

The experiment name is used in the run-directory name, while the numeric seed
is read independently from `experiment.seed`. Dataset paths, model dimensions,
training hyperparameters, checkpoint switches, metrics, and visualization
settings are all read from YAML.

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

The repository also includes a configurable shorter-run file. Its epoch and
batch values are ordinary YAML parameters and can be edited without changing
Python code:

```bash
python -m src.v1.train_v1 --config configs/configV1_smoke.yaml
```

On a 24 GB GPU, if `batch_size: 8` peaks near 6 GB, try 16 next and then 24 if
peak memory still leaves several GB of headroom. Batch 32 is too close to the
limit under a simple linear estimate and is not the recommended next jump.
The code only requires a positive batch size and does not impose an artificial
upper bound, but a larger batch changes optimization behavior and is not
automatically better.

With `logging.console: true`, train and validation progress is printed to the
terminal. Every epoch writes one training summary, and every validation run
writes one validation summary. Set `logging.log_every_steps: 0` to disable
per-batch messages (the checked-in v1 configs use this setting). A positive
value enables batch progress at that interval, including the first and final
batch. With `training.validate_every: 1`, both `train.log` and `val.log` receive
one metric summary per epoch.

A tmux session is optional and remains under user control:

```bash
tmux new -s v1_seed1234
sh src/v1/run.sh
```

Validation runs according to `training.validate_every` and always runs on the
final epoch, without random crop or augmentation. Validation PSNR, SSIM, and
validation loss update three independent best checkpoints when their save
switches are enabled. The default primary checkpoint is
`best/best_psnr.pt`.

After all configured epochs complete successfully, training optionally runs
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
result/test_all_enhanced/{sample_id}_enhanced.png
result/test_grid_10x3.png
```

By default, saved test enhancements are resized to 256×256 through
`test.output_size`, and all test enhancements are saved because
`test.save_all_enhanced_images` is enabled. The grid is ten rows by three
columns in `Input / Enhanced / GT` order, with every cell occupying exactly
256×256. Test inference and PSNR/SSIM still use the original full resolution;
resizing happens only while saving output images. The output name follows
`test_grid_{rows}x3.png`.

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

## Run one v2 experiment

Run exactly one variant and one seed per command:

```bash
bash src/v2/run.sh color_only 1234
bash src/v2/run.sh scatter_only 1234
bash src/v2/run.sh color_then_scatter 1234
bash src/v2/run.sh scatter_then_color 1234
bash src/v2/run.sh shared_order_diagnostic 1234
```

The equivalent explicit command is:

```bash
python -m src.v2.train_v2 \
  --config configs/configV2.yaml \
  --variant color_only \
  --seed 1234
```

`--variant` and `--seed` are optional overrides. Before creating the run, the
resolved configuration is strictly validated and its name is set to
`{variant}_seed{seed}`. The complete resolved configuration is saved as the
run's immutable `config.json`. The shell script never loops over variants or
seeds.

For every variant, validation—not test—selects checkpoints. Formal training
automatically evaluates `best/best_psnr.pt` on the complete test manifest only
after every epoch succeeds. Test output is final reporting and must not be used
to revise the order, architecture, or hyperparameters.

The five single-output variants retain the V1 training and checkpoint
semantics. For `shared_order_diagnostic`, each batch runs two fresh graphs and
accumulates:

```text
joint_loss = 0.5 * loss_color_then_scatter
           + 0.5 * loss_scatter_then_color
```

There is one `zero_grad`, one optimizer step, one scaler update, and at most
one global-step increment per batch. Its `best_psnr.pt` is selected only by the
validation mean of the two path PSNR values, never by either individual path.

Shared validation writes:

```text
result/validation_metrics_color_then_scatter.csv
result/validation_metrics_scatter_then_color.csv
result/validation_order_comparison.csv
result/validation_summary.json
```

The `validation_*` files always describe the latest validation. Whenever
validation PSNR selects a new `best_psnr.pt`, V2 also atomically updates a
matching `best_psnr_validation_*` snapshot. Single-output runs save
`best_psnr_validation_metrics.csv` and
`best_psnr_validation_summary.json`; shared runs save two path metric files,
the paired order-comparison CSV, and the matching best summary. Resume copies
these files when an older run already has them and remains compatible with
older checkpoints that predate the snapshots.

Shared final test writes:

```text
result/test_metrics_color_then_scatter.csv
result/test_metrics_scatter_then_color.csv
result/test_order_comparison.csv
result/test_summary.json
result/test_all_enhanced/color_then_scatter/
result/test_all_enhanced/scatter_then_color/
result/test_grid_10x4.png
```

The comparison CSV reports per-image C→S minus S→C PSNR/SSIM deltas and winner
labels. Both paths use the same checkpoint and the same randomly selected
visualization samples. Inference and metrics remain full-resolution; only
saved images and grid cells use `test.output_size`.

Before test inference, V2 requires `best/best_psnr.json` and verifies that the
run config, checkpoint config, seed, sidecar fields, selection metric, and
shared mean-path/joint-loss provenance all agree. A structurally compatible
checkpoint copied from another variant is therefore rejected before the test
DataLoader is constructed.

Retest an existing V2 run without editing source code:

```bash
bash src/v2/test.sh experiments/<v2-run-directory>
```

or:

```bash
python -m src.v2.test_v2 \
  --config configs/configV2.yaml \
  --run-dir experiments/<v2-run-directory>
```

V2 resume uses `training.resume` and strictly restores model, optimizer,
GradScaler, scheduler, global step, independent best validation values, metric
history, and Python/NumPy/PyTorch/CUDA RNG state. Only `training.resume`,
`test.run_dir`, and `test.allow_overwrite` may differ from the checkpoint
configuration.

## Checks

Install a compatible PyTorch separately for the machine; do not let the
project choose or upgrade a CUDA wheel. Then run:

```bash
python -m compileall -q src tests third_party
bash -n src/v1/run.sh
bash -n src/v1/test.sh
bash -n src/v2/run.sh
bash -n src/v2/test.sh
python -m pytest -q
git diff --check
```

The test suite covers both schemas, V1/V2 rejection, C/S bounds and gradients,
all six variant structures, shared optimizer-step accounting, independent best
selection, checkpoint/resume, full-resolution metrics, paired order reports,
and three/four-column visualization.

Test-set results are final reporting only. Do not use them for checkpoint
selection, early stopping, model choice, or hyperparameter tuning.

See `MIGRATION_V1_REPORT.md` for provenance and validation status and
`GITHUB_PUSH_GUIDE.md` for local review and optional GitHub publication.
