# UIE3 v1 Migration Report

## Result and scope

This report records the read-only audit and migration of the verified
NAFNet-small baseline into a new independent project. No formal training,
full validation, test inference, dataset transfer, checkpoint transfer from
the old project, dependency installation, remote creation, commit, or push
was performed.

- Source project:
  `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace/UIE3`
- Read-only upstream:
  `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace/NAFNet`
- Target project:
  `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace/UIE3_VN`
- Source UIE3 commit:
  `dbe7d43ab3e2a5baf10818767a271feb897bdd54`
- Source UIE3 initial status: clean, `main...origin/main`
- Upstream NAFNet commit:
  `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Upstream NAFNet initial status: clean, `main...origin/main`

The final source-repository status audit is recorded under “Checks”.

## Source-to-target mapping

| Read-only source | Target | Migration treatment |
|---|---|---|
| `UIE3/configs/nafnet_small_lsui_formal_seed1234.yaml` | `configs/configV1.yaml` | Preserved baseline values; added independent run, test, visualization, status, and logging fields. |
| `UIE3/src/models/backbones/nafnet_small.py` | `src/v1/model.py` | Preserved the verified small defaults and inherited ordinary NAFNet forward; architecture dimensions are YAML-configurable. |
| `UIE3/third_party/nafnet/nafnet_arch.py` | `third_party/nafnet/nafnet_arch.py` | Migrated implementation with source header retained; normalized one trailing-whitespace line and one extra EOF blank line so Git whitespace checks pass. No token or computation changed. |
| `UIE3/third_party/nafnet/LICENSE` | `third_party/nafnet/LICENSE` | Exact composite MIT/Apache-2.0 license copy. |
| `UIE3/third_party/nafnet/UPSTREAM.md` | `third_party/nafnet/UPSTREAM.md` | Preserved provenance and appended the independent-project migration note. |
| `UIE3/src/data/paired_image_dataset.py` | `src/common/data/paired_dataset.py` | Preserved strict three-column manifest, sample ID, RGB float32, range, pairing, and full-image evaluation semantics. |
| `UIE3/src/data/paired_image_dataset.py` | `src/common/data/transforms.py` | Extracted the same paired reflect-pad, crop, flip, and rot90 operations. |
| `UIE3/tools/train_baseline.py` DataLoader construction | `src/common/data/dataloader.py` | Preserved seeded workers, shuffled training, and ordered batch-size-one validation; added ordered test. |
| `UIE3/src/metrics/image_metrics.py` | `src/common/metrics/image_metrics.py` | Preserved joint-RGB per-image PSNR and valid-window RGB SSIM definitions. |
| `UIE3/src/utils/seed.py` | `src/common/experiment/seed.py` | Preserved Python, NumPy, torch, CUDA, cuDNN, and worker seeding; added deterministic-algorithm enforcement when requested. |
| `UIE3/src/losses/charbonnier.py` | `src/v1/loss.py` | Preserved `sqrt(error² + epsilon²)` with epsilon `0.001`. |
| `UIE3/src/engine/trainer.py` | `src/v1/trainer.py` | Preserved finite checks and corrected AMP-overflow step accounting; added epoch aggregation and validation loss. |
| `UIE3/src/engine/checkpoint.py` | `src/common/experiment/checkpoint.py` | Preserved atomic torch persistence and strict restore; expanded to all required metrics, three best values, and RNG states. |
| `UIE3/tools/train_baseline.py` | `src/v1/train_v1.py` | Replaced phase CLI governance with the single v1 YAML, unique runs, per-epoch validation, three best files, statuses, errors, and automatic test. |
| `UIE3/tools/evaluate_baseline.py` | `src/v1/test_v1.py` | Preserved ordered per-image CSV evaluation; authorized only full test with formal best-validation-PSNR selection and added visualization. |
| `UIE3/splits/lsui19/train.tsv` | `splits/lsui19/train.tsv` | Exact copy; 3,466 rows; SHA-256 `5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab`. |
| `UIE3/splits/lsui19/validation.tsv` | `splits/lsui19/validation.tsv` | Exact copy; 385 rows; SHA-256 `e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18`. |
| `UIE3/splits/lsui19/test.tsv` | `splits/lsui19/test.tsv` | Exact copy; 428 rows; SHA-256 `cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c`. |
| Existing UIE3 unit-test intent | `tests/test_*.py` | Reframed for the independent v1 config, run layout, three best rules, auto-test flow, and 10×3 output. |

New project-only modules implement simple config/JSON handling, unique
experiment creation, logging, status/error recording, and visualization. They
do not migrate the old phase runtime state machine.

## Preserved model semantics

The v1 model is configured as `img_channel=3`, `width=32`,
`enc_blk_nums=[2,2,2]`, `middle_blk_num=4`, and
`dec_blk_nums=[2,2,2]`. Three encoder levels make the internal padder multiple
8. `NAFNet.check_image_size` pads zeros only on the right and bottom. Ordinary
`NAFNet.forward` performs one and only one global residual against the padded
input and crops the result back to the original `H,W`.

The v1 wrapper does not add another residual. It adds no sigmoid and no hard
clamp. NAFNetLocal, Local_Base, BasicSR registry/discovery, BasicSR
train/test, and pretrained weights are absent. “NAFNet-small” denotes the
project's research configuration, not an official upstream fixed model.

## License and attribution

Vendored NAFNet material retains the upstream 2022 megvii-model header and MIT
License. The consolidated LayerNorm implementation retains BasicSR
2018–2020 attribution and the composite license includes Apache License 2.0.
The pinned upstream repository and source hashes remain documented in
`third_party/nafnet/UPSTREAM.md`.

## Preserved data and metric semantics

Each TSV row is exactly:

```text
sample_id<TAB>input_relative_path<TAB>gt_relative_path
```

Paths resolve under the configured dataset root and may not escape it. Pillow
decodes both files, converts them to RGB, creates NumPy float32 arrays, and
divides by `255.0`. Paired random crop, reflect padding, horizontal flip,
vertical flip, and rot90 are training-only. Validation and test always use
the complete original pair.

Before metrics only, prediction is clamped to `[0,1]`; target must already be
in range. PSNR is calculated independently per image from joint RGB MSE with
`data_range=1.0`, then averaged over images. SSIM uses an 11×11 Gaussian
window, sigma 1.5, valid convolution, standard constants 0.01² and 0.03²,
then averages spatial locations and RGB channels per image. Crop border is
zero.

## Training, checkpoint, and automatic test flow

`configs/configV1.yaml` is the default v1 configuration; the train and test
entry points also accept an explicitly selected v1 YAML from any filesystem
location. The name's `seed<number>` token must match the numeric seed. Each
launch immediately creates a unique,
second-precision run directory and atomically snapshots the complete config.
Python, NumPy, torch, CUDA, workers, runtime metadata, logs, status, and errors
are recorded.

Every epoch trains with Charbonnier loss and AdamW, then runs complete
validation. AMP overflow is allowed to make GradScaler skip an optimizer
update; such a skip does not increase `global_step`, and later finite updates
continue. Metrics history is written by temporary file plus atomic replace.

Three validation-only best checkpoints are independent:

- `best_psnr.pt`: maximum validation PSNR and the formal primary checkpoint;
- `best_ssim.pt`: maximum validation SSIM;
- `best_loss.pt`: minimum validation loss.

Last and periodic checkpoints are also saved. Payloads contain the full
training/selection values, model/optimizer/scaler state, config, seed, and all
required RNG states. Test metrics cannot call the validation best tracker.

Only successful completion of all epochs triggers full test. Automatic test
loads only `best/best_psnr.pt`; `checkpoint_selection_source` is
`formal_validation_psnr`. It creates the per-image CSV and summary without an
optimizer, scheduler step, backward pass, checkpoint update, or config update.
A test failure leaves training complete and marks the run
`PARTIAL_FAILURE`.

## 10×3 visualization flow

Candidates are sorted by sample ID and sampled only with
`random.Random(3407)`. No quality metric enters selection. Ten distinct IDs
are used, or all IDs once when fewer than ten exist. Selected enhanced RGB
PNGs, selection provenance JSON, and a default ten-row by three-column grid are
saved under `result/`. Columns are Input, Enhanced, and GT. Each 512×512 cell
uses aspect-preserving resize, uniform padding, and Pillow's default font.
Only the visualization copies are resized.

## New experiment layout

```text
run_dir/
├── config.json
├── run_info.json
├── status.json
├── error.json                 # failure only
├── log/{train.log,val.log,test.log,metrics_history.json}
├── best/{best_psnr,best_ssim,best_loss}.{pt,json}
├── checkpoint/{last.pt,epoch_NNNN.pt}
└── result/
    ├── validation_summary.json
    ├── validation_metrics.csv
    ├── test_summary.json
    ├── test_metrics.csv
    ├── test_visualization_samples.json
    ├── test_grid_10x3.png
    └── test_samples/{sample_id}_enhanced.png
```

## Deliberately not migrated

- `UIE3/.git`, old experiments, checkpoints, GitHub results, runtime reports,
  caches, and temporary files: historical/runtime artifacts are out of scope.
- `ORDER_STUDY_PROTOCOL.md`, `ORDER_STUDY_DECISIONS.md`, and
  `ORDER_STUDY_STATE.yaml`: the new project uses simple status JSON, not the
  old phase governance.
- Phase ID guards and validation-only test prohibition: v1 has direct
  train/validation/test behavior authorized by this migration.
- NAFNetLocal, Local_Base, BasicSR framework code, official task YAMLs, and
  pretrained weights: not part of the ordinary NAFNet-small baseline.
- Color operators, scattering operators, routers, v2+, and unified registries
  or `tools/run.py`: explicitly outside v1 scope.
- Dataset files and old checkpoints: external data remains referenced by the
  configured dataset root and no weights were copied.

## Added files

The target inventory below lists every project file (`.git/` internals are
omitted):

```text
.gitignore
GITHUB_PUSH_GUIDE.md
MIGRATION_V1_REPORT.md
README.md
configs/configV1.yaml
experiments/.gitkeep
requirements/README.md
requirements/dev.txt
requirements/runtime.txt
splits/lsui19/test.tsv
splits/lsui19/train.tsv
splits/lsui19/validation.tsv
src/__init__.py
src/common/__init__.py
src/common/data/__init__.py
src/common/data/dataloader.py
src/common/data/paired_dataset.py
src/common/data/transforms.py
src/common/experiment/__init__.py
src/common/experiment/checkpoint.py
src/common/experiment/config.py
src/common/experiment/experiment.py
src/common/experiment/logging_utils.py
src/common/experiment/seed.py
src/common/experiment/visualization.py
src/common/metrics/__init__.py
src/common/metrics/image_metrics.py
src/v1/__init__.py
src/v1/loss.py
src/v1/model.py
src/v1/run.sh
src/v1/test.sh
src/v1/test_v1.py
src/v1/train_v1.py
src/v1/trainer.py
tests/test_auto_test_flow.py
tests/test_checkpoint_resume.py
tests/test_checkpoint_selection.py
tests/test_config_v1.py
tests/test_experiment_creation.py
tests/test_metrics.py
tests/test_v1_model.py
tests/test_visualization_grid.py
third_party/nafnet/LICENSE
third_party/nafnet/UPSTREAM.md
third_party/nafnet/__init__.py
third_party/nafnet/nafnet_arch.py
```

No source-project or upstream file was modified.

## Checks

Executed in the target project:

- YAML load and full v1 validation: passed.
- `python -m compileall -q src tests third_party`: passed.
- `bash -n src/v1/run.sh`: passed.
- `bash -n src/v1/test.sh`: passed.
- `python -m pytest -q`: 20 passed, 3 skipped.
- `git diff --check`: passed for the current unstaged index/worktree state.
- Full target-file whitespace audit equivalent to checking newly added files:
  passed after two source-copy-only whitespace normalizations documented above.
- Manifest structural audit: 3,466/385/428 rows, unique IDs within each
  split, and no sample-ID overlap between train, validation, and test; passed.
- License and all three manifest hashes match the source copies; passed.
- Vendored NAFNet source matches the source copy under whitespace-insensitive
  comparison; passed.
- Both shell scripts have executable mode `-rwxr-xr-x`.
- Local Git repository initialized on unborn branch `main`; no commit and no
  remote.
- Target Git status: all 47 intended project files remain untracked for user
  review; ignored cache/checkpoint/experiment patterns behave as specified.
- Ending source UIE3 status: commit
  `dbe7d43ab3e2a5baf10818767a271feb897bdd54`, clean
  `main...origin/main`.
- Ending upstream NAFNet status: commit
  `2b4af71ebe098a92a75910c233a3965a3e93ede4`, clean
  `main...origin/main`.

The three skipped modules require PyTorch: model runtime, metrics runtime, and
checkpoint save/load/resume. The creation environment reported
`ModuleNotFoundError: No module named 'torch'`; PyTorch was not installed or
upgraded. The configured cloud dataset root was not available locally.
Therefore model forward/backward, optimizer/GradScaler execution,
checkpoint runtime I/O, full dataset loading, formal training, full
validation, test, and grid generation from formal outputs were not run.
Current implementation status is `IMPLEMENTED_NOT_RUNTIME_VALIDATED`.

## Current runtime blockers

The implementation itself is complete for static handoff. Runtime validation
is blocked locally by the absent PyTorch package and absent configured cloud
dataset root. No dependency was installed and no dataset was downloaded or
copied. Final task classification is `PASS_STATIC_ONLY_TORCH_UNAVAILABLE`.

## Cloud preflight

Run before any formal training:

```bash
cd /Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace/UIE3_VN
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
test -d /root/autodl-tmp/pro/publicdata/LSUI19_dup_train
python -m pytest -q
python -m compileall -q src tests third_party
bash -n src/v1/run.sh
bash -n src/v1/test.sh
git diff --check
```

Only after those checks pass and the YAML name/seed pair is reviewed should
the user intentionally start `sh src/v1/run.sh`.

GitHub instructions are in `GITHUB_PUSH_GUIDE.md`. They require user review,
an explicit local commit, creation of an empty GitHub repository, and manual
remote/push commands with user-supplied names.
