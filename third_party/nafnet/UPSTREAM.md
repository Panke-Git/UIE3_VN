# NAFNet Upstream Record

## Upstream

- Repository: `https://github.com/megvii-research/NAFNet`
- Commit: `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Acquisition date: 2026-07-18
- Local reference checkout: `UIE3_workspace/NAFNet` (read-only)

## Source Files and Imported Symbols

| Upstream source | Imported symbols |
|---|---|
| `basicsr/models/archs/arch_util.py` | `LayerNormFunction`, `LayerNorm2d` |
| `basicsr/models/archs/NAFNet_arch.py` | `SimpleGate`, `NAFBlock`, `NAFNet` |
| `LICENSE` | complete composite license text |

Source SHA-256 values at the pinned commit:

- `basicsr/models/archs/arch_util.py`: `5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005`
- `basicsr/models/archs/NAFNet_arch.py`: `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`
- `LICENSE`: `a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad`

## Destination Files

- `third_party/nafnet/nafnet_arch.py`
- `third_party/nafnet/LICENSE`
- this record, `third_party/nafnet/UPSTREAM.md`
- project-specific configuration wrapper: `src/models/backbones/nafnet_small.py`
- runtime validation plan: `tests/test_nafnet_import.py`

## Modifications

The two LayerNorm symbols were consolidated into the same vendored file as the three ordinary NAFNet symbols. Imports were reduced to PyTorch. The following upstream dependencies and code paths were deliberately removed:

- `from basicsr.models.archs.arch_util import LayerNorm2d` (the exact audited LayerNorm implementation is now co-located);
- `from basicsr.models.archs.local_arch import Local_Base`;
- `NAFNetLocal`, `Local_Base`, `AvgPool2d`, and Local conversion code;
- BasicSR dynamic architecture discovery/registration;
- BasicSR training, data, loss, metric, logger, and checkpoint frameworks;
- official task YAML files and pretrained weights.

No mathematical refactor, optimization, operator substitution, module rename, parameter rename, clamp, normalization, or additional residual connection was intentionally introduced. The ordinary NAFNet computation semantics are intended to be unchanged. This statement is based on source extraction and static inspection only; numerical equivalence has **not** been established locally because Phase B1a has no PyTorch runtime.

## License and Attribution

- NAFNet material retains `Copyright (c) 2022 megvii-model` and is distributed under the upstream MIT License.
- The extracted LayerNorm source comes from an upstream file marked as modified from BasicSR and retains `Copyright 2018-2020 BasicSR Authors`; the BasicSR portion is distributed under Apache License 2.0.
- `third_party/nafnet/LICENSE` is an exact copy of the pinned upstream composite `LICENSE` and contains both license texts.
- The vendored source header records extraction, consolidation, import decoupling, the pinned commit, and the fact that runtime equivalence remains pending.

## Phase B1b Runtime Validation

Run from the UIE3 repository root in a cloud environment that has a compatible CPU PyTorch installation and pytest:

```bash
NAFNET_UPSTREAM_ROOT=../NAFNet PYTHONPATH=. python -m pytest -q -s tests/test_nafnet_import.py
```

The command must use the upstream checkout at the pinned commit. It is expected to execute official-versus-vendored output/state-dict comparisons, NAFNet-small shape and non-multiple-size behavior, finite forward/backward gradients, and the duplicate-global-residual guard. Phase B1a did not execute this command and records no runtime result.

## UIE3_VN Migration Note

This provenance record was migrated from UIE3 commit
`dbe7d43ab3e2a5baf10818767a271feb897bdd54`. In the independent project, the
configuration wrapper is `src/v1/model.py`, and the relevant architecture
tests are in `tests/test_v1_model.py`. The old UIE3 repository later archived
runtime validation of the vendored implementation; this migration itself did
not repeat PyTorch runtime validation because PyTorch was unavailable in the
local creation environment. See `MIGRATION_V1_REPORT.md` for the exact
validation status. The source header in `nafnet_arch.py` is intentionally
preserved unchanged.
