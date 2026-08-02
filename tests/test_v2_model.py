from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.v1.model import NAFNetSmall, build_nafnet_small  # noqa: E402
from src.v2.config import load_v2_config  # noqa: E402
from src.v2.model import OrderStudyModel, build_v2_model  # noqa: E402


class Spy(torch.nn.Module):
    def __init__(self, name: str, calls: list[str]):
        super().__init__()
        self.name = name
        self.calls = calls
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, value):
        self.calls.append(self.name)
        return value * self.weight


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("color_only", ["C", "B"]),
        ("scatter_only", ["S", "B"]),
        ("color_then_scatter", ["C", "S", "B"]),
        ("scatter_then_color", ["S", "C", "B"]),
    ],
)
def test_fixed_variant_calls_explicit_order(variant: str, expected: list[str]) -> None:
    calls: list[str] = []
    model = OrderStudyModel(
        variant=variant,
        backbone=Spy("B", calls),
        color_operator=Spy("C", calls) if "color" in variant else None,
        scattering_operator=Spy("S", calls) if "scatter" in variant else None,
    )
    output = model(torch.ones(1, 3, 5, 7))
    assert output.shape == (1, 3, 5, 7)
    assert calls == expected


def _small_model_config():
    config = load_v2_config()["model"]
    config["backbone"].update(
        width=4, enc_blk_nums=[0], middle_blk_num=0, dec_blk_nums=[0]
    )
    config["color_operator"]["hidden_channels"] = 4
    config["scattering_operator"]["hidden_channels"] = 4
    return config


def test_v2_baseline_is_raw_v1_model_and_state_compatible() -> None:
    model_config = _small_model_config()
    v1 = build_nafnet_small(**model_config["backbone"]).eval()
    v2 = build_v2_model(variant="baseline", model_config=model_config).eval()
    assert isinstance(v2, NAFNetSmall)
    assert type(v1) is type(v2)
    assert list(v1.state_dict()) == list(v2.state_dict())
    assert sum(p.numel() for p in v1.parameters()) == sum(
        p.numel() for p in v2.parameters()
    )
    v2.load_state_dict(v1.state_dict(), strict=True)
    value = torch.rand(1, 3, 9, 11)
    with torch.no_grad():
        torch.testing.assert_close(v1(value), v2(value), rtol=0.0, atol=0.0)


def test_shared_has_exactly_one_of_each_module_and_two_explicit_paths() -> None:
    model = build_v2_model(
        variant="shared_order_diagnostic", model_config=_small_model_config()
    )
    color_id = id(model.color_operator)
    scatter_id = id(model.scattering_operator)
    backbone_id = id(model.backbone)
    value = torch.rand(1, 3, 9, 11)
    assert model.forward_color_then_scatter(value).shape == value.shape
    assert model.forward_scatter_then_color(value).shape == value.shape
    assert id(model.color_operator) == color_id
    assert id(model.scattering_operator) == scatter_id
    assert id(model.backbone) == backbone_id
    with pytest.raises(RuntimeError, match="no single forward path"):
        model(value)


def test_orders_have_equal_parameter_count_and_unused_modules_are_absent() -> None:
    config = _small_model_config()
    cs = build_v2_model(variant="color_then_scatter", model_config=config)
    sc = build_v2_model(variant="scatter_then_color", model_config=config)
    assert sum(p.numel() for p in cs.parameters()) == sum(
        p.numel() for p in sc.parameters()
    )
    color_only = build_v2_model(variant="color_only", model_config=config)
    scatter_only = build_v2_model(variant="scatter_only", model_config=config)
    assert color_only.scattering_operator is None
    assert scatter_only.color_operator is None
    baseline = build_v2_model(variant="baseline", model_config=config)
    assert not hasattr(baseline, "color_operator")
    assert not hasattr(baseline, "scattering_operator")
