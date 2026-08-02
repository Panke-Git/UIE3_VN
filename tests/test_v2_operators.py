from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.v2.operators import ColorCorrectionOperator, ScatteringRemovalOperator  # noqa: E402


def _forbidden_modules(module):
    return [
        child
        for child in module.modules()
        if isinstance(child, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
    ]


@pytest.mark.parametrize("shape", [(1, 3, 7, 9), (2, 3, 11, 13)])
def test_color_operator_is_initially_identity_and_preserves_shape(shape) -> None:
    operator = ColorCorrectionOperator(hidden_channels=4)
    value = torch.rand(shape, requires_grad=True)
    original = value.detach().clone()
    output = operator(value)
    assert output.shape == value.shape
    torch.testing.assert_close(output, value, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(value.detach(), original)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in operator.parameters()
    )


def test_color_matrix_and_bias_are_bounded_and_no_stochastic_layers() -> None:
    operator = ColorCorrectionOperator(
        hidden_channels=4, matrix_scale=0.1, bias_scale=0.05
    )
    with torch.no_grad():
        operator.parameter_head.bias.fill_(100.0)
    value = torch.rand(2, 3, 5, 7)
    matrix, bias = operator.predict_parameters(value)
    identity = torch.eye(3).unsqueeze(0)
    assert torch.max(torch.abs(matrix - identity)) <= 0.1 + 1.0e-7
    assert torch.max(torch.abs(bias)) <= 0.05 + 1.0e-7
    assert not _forbidden_modules(operator)


@pytest.mark.parametrize("shape", [(1, 3, 9, 11), (2, 3, 13, 15)])
def test_scattering_operator_ranges_residual_and_backward(shape) -> None:
    operator = ScatteringRemovalOperator(hidden_channels=4)
    value = torch.rand(shape, requires_grad=True)
    original = value.detach().clone()
    output, diagnostics = operator.forward_with_diagnostics(value)
    transmission = diagnostics["transmission"]
    atmospheric = diagnostics["atmospheric_light"]
    assert output.shape == value.shape
    assert transmission.shape == (shape[0], 1, shape[2], shape[3])
    assert atmospheric.shape == (shape[0], 3, 1, 1)
    assert torch.all(transmission >= operator.t_min)
    assert torch.all(transmission < 1.0)
    assert torch.all(atmospheric > 0.0) and torch.all(atmospheric < 1.0)
    torch.testing.assert_close(
        transmission, torch.full_like(transmission, 0.95), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        atmospheric, torch.full_like(atmospheric, 0.5), atol=1.0e-6, rtol=0.0
    )
    assert torch.max(torch.abs(output - value)) <= operator.residual_max + 1.0e-6
    assert torch.isfinite(output).all()
    torch.testing.assert_close(value.detach(), original)
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in operator.parameters()
    )
    assert not _forbidden_modules(operator)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_min": 0.0},
        {"t_min": 0.95, "initial_t": 0.95},
        {"initial_t": 1.0},
        {"initial_A": 0.0},
        {"eps": 0.0},
        {"residual_max": -1.0},
    ],
)
def test_scattering_invalid_parameters_fail(kwargs) -> None:
    with pytest.raises(ValueError):
        ScatteringRemovalOperator(**kwargs)


@pytest.mark.parametrize(
    "value",
    [torch.zeros(3, 8, 8), torch.zeros(1, 1, 8, 8), torch.zeros(1, 3, 8, 8, dtype=torch.int64)],
)
def test_operators_reject_non_rgb_or_non_float_inputs(value) -> None:
    for operator in (ColorCorrectionOperator(), ScatteringRemovalOperator()):
        with pytest.raises((TypeError, ValueError)):
            operator(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
def test_scattering_operator_cuda_amp_output_is_finite() -> None:
    operator = ScatteringRemovalOperator(hidden_channels=4).cuda()
    value = torch.rand(1, 3, 17, 19, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = operator(value)
    assert torch.isfinite(output).all()
