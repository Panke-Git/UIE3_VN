from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.v1.model import build_nafnet_small  # noqa: E402


@pytest.mark.parametrize(
    "shape",
    [(1, 3, 256, 256), (2, 3, 128, 192), (1, 3, 257, 259)],
)
def test_fixed_model_preserves_shape_and_is_finite(shape) -> None:
    model = build_nafnet_small().cpu().eval()
    with torch.no_grad():
        output = model(torch.randn(shape))
    assert output.shape == shape
    assert torch.isfinite(output).all()


def test_non_multiple_padding_is_right_bottom_zero_and_cropped() -> None:
    model = build_nafnet_small().cpu().eval()
    value = torch.randn(1, 3, 257, 259)
    padded = model.check_image_size(value)
    assert padded.shape == (1, 3, 264, 264)
    torch.testing.assert_close(padded[:, :, :257, :259], value)
    assert torch.count_nonzero(padded[:, :, 257:, :]) == 0
    assert torch.count_nonzero(padded[:, :, :, 259:]) == 0
    with torch.no_grad():
        assert model(value).shape == value.shape


def test_forward_backward_are_finite() -> None:
    model = build_nafnet_small().cpu().train()
    value = torch.randn(1, 3, 16, 24, requires_grad=True)
    loss = model(value).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_wrapper_does_not_add_a_second_global_residual() -> None:
    model = build_nafnet_small().cpu().eval()
    with torch.no_grad():
        model.ending.weight.zero_()
        model.ending.bias.zero_()
    value = torch.full((1, 3, 17, 19), 0.25)
    with torch.no_grad():
        output = model(value)
    torch.testing.assert_close(output, value, rtol=0.0, atol=0.0)


def test_yaml_model_dimensions_are_configurable() -> None:
    model = build_nafnet_small(
        width=8,
        enc_blk_nums=[1, 1],
        middle_blk_num=1,
        dec_blk_nums=[1, 1],
    ).cpu().eval()
    value = torch.randn(1, 3, 17, 19)
    with torch.no_grad():
        output = model(value)
    assert model.padder_size == 4
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
