"""Unified v2 model construction for fixed and shared RGB-domain orders."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from torch import Tensor, nn

from src.v1.model import build_nafnet_small
from src.v2.config import VALID_VARIANTS
from src.v2.operators import (
    ColorCorrectionOperator,
    ScatteringRemovalOperator,
)


class OrderStudyModel(nn.Module):
    """Compose exactly one requested set of C, S, and backbone modules."""

    def __init__(
        self,
        *,
        variant: str,
        backbone: nn.Module,
        color_operator: Optional[nn.Module] = None,
        scattering_operator: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if variant not in VALID_VARIANTS or variant == "baseline":
            raise ValueError(
                "OrderStudyModel requires a non-baseline valid v2 variant."
            )
        needs_color = variant in {
            "color_only",
            "color_then_scatter",
            "scatter_then_color",
            "shared_order_diagnostic",
        }
        needs_scatter = variant in {
            "scatter_only",
            "color_then_scatter",
            "scatter_then_color",
            "shared_order_diagnostic",
        }
        if needs_color != (color_operator is not None):
            raise ValueError(
                f"variant={variant} has an invalid color operator allocation."
            )
        if needs_scatter != (scattering_operator is not None):
            raise ValueError(
                f"variant={variant} has an invalid scattering operator allocation."
            )
        self.variant = variant
        self.backbone = backbone
        self.color_operator = color_operator
        self.scattering_operator = scattering_operator

    def _color(self, value: Tensor) -> Tensor:
        if self.color_operator is None:
            raise RuntimeError(f"variant={self.variant} has no color operator.")
        return self.color_operator(value)

    def _scatter(self, value: Tensor) -> Tensor:
        if self.scattering_operator is None:
            raise RuntimeError(f"variant={self.variant} has no scattering operator.")
        return self.scattering_operator(value)

    def forward_color_only(self, value: Tensor) -> Tensor:
        return self.backbone(self._color(value))

    def forward_scatter_only(self, value: Tensor) -> Tensor:
        return self.backbone(self._scatter(value))

    def forward_color_then_scatter(self, value: Tensor) -> Tensor:
        """Compute B(S(C(x))): color first, then scattering removal."""

        return self.backbone(self._scatter(self._color(value)))

    def forward_scatter_then_color(self, value: Tensor) -> Tensor:
        """Compute B(C(S(x))): scattering removal first, then color."""

        return self.backbone(self._color(self._scatter(value)))

    def forward(self, value: Tensor) -> Tensor:
        if self.variant == "color_only":
            return self.forward_color_only(value)
        if self.variant == "scatter_only":
            return self.forward_scatter_only(value)
        if self.variant == "color_then_scatter":
            return self.forward_color_then_scatter(value)
        if self.variant == "scatter_then_color":
            return self.forward_scatter_then_color(value)
        raise RuntimeError(
            "shared_order_diagnostic has no single forward path; call "
            "forward_color_then_scatter() and forward_scatter_then_color() "
            "explicitly."
        )


def build_v2_model(
    *, variant: str, model_config: Mapping[str, Any]
) -> nn.Module:
    """Instantiate only the modules used by the selected v2 variant."""

    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"variant must be one of {sorted(VALID_VARIANTS)}, got {variant!r}."
        )
    backbone = build_nafnet_small(**dict(model_config["backbone"]))
    if variant == "baseline":
        return backbone
    needs_color = variant in {
        "color_only",
        "color_then_scatter",
        "scatter_then_color",
        "shared_order_diagnostic",
    }
    needs_scatter = variant in {
        "scatter_only",
        "color_then_scatter",
        "scatter_then_color",
        "shared_order_diagnostic",
    }
    color = (
        ColorCorrectionOperator(**dict(model_config["color_operator"]))
        if needs_color
        else None
    )
    scatter = (
        ScatteringRemovalOperator(**dict(model_config["scattering_operator"]))
        if needs_scatter
        else None
    )
    return OrderStudyModel(
        variant=variant,
        backbone=backbone,
        color_operator=color,
        scattering_operator=scatter,
    )
