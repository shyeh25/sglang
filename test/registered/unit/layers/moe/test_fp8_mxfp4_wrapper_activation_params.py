"""Regression tests for Fp8MoEMethod.process_weights_after_loading()."""

import sys

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-c-test-cpu")

_TRTLLM_BACKENDS = [
    MoeRunnerBackend.FLASHINFER_TRTLLM,
    MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
]


@pytest.fixture
def _runner_backend_flag():
    moe = get_flags().moe
    saved = moe.runner_backend
    yield moe
    moe.runner_backend = saved


class _FakeDispatcher:
    def __init__(self):
        self.set_quant_config_calls = []

    def set_quant_config(self, config):
        self.set_quant_config_calls.append(config)


def _make_layer(with_dispatcher: bool = False):
    layer = torch.nn.Module()
    layer.num_local_experts = 2
    layer.w13_weight = torch.zeros(2, 4, 4)
    if with_dispatcher:
        layer.dispatcher = _FakeDispatcher()
    return layer


def _make_fp8_method() -> Fp8MoEMethod:
    fp8_method = object.__new__(Fp8MoEMethod)
    fp8_method._owned_moe_runner = None
    fp8_method.block_quant = True
    fp8_method.process_weights_after_loading_block_quant = lambda layer: None
    return fp8_method


@pytest.mark.parametrize("backend", _TRTLLM_BACKENDS)
def test_borrowed_fp8_method_skips_activation_params_under_trtllm_backend(
    _runner_backend_flag, backend
):
    _runner_backend_flag.runner_backend = backend

    fp8_method = _make_fp8_method()
    assert fp8_method._owned_moe_runner is None

    layer = _make_layer()
    fp8_method.process_weights_after_loading(layer)

    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_alpha")
    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_beta")
    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_clamp_limit")


@pytest.mark.parametrize("backend", _TRTLLM_BACKENDS)
def test_owned_trtllm_runner_materializes_activation_params(
    _runner_backend_flag, backend
):
    _runner_backend_flag.runner_backend = backend

    config = MoeRunnerConfig(gemm1_alpha=1.5, gemm1_beta=2.5, gemm1_clamp_limit=3.5)
    fp8_method = _make_fp8_method()
    layer = _make_layer()
    fp8_method.create_moe_runner(layer, config)

    assert fp8_method._owned_moe_runner is fp8_method.runner

    fp8_method.process_weights_after_loading(layer)

    assert layer._flashinfer_trtllm_gemm1_alpha.tolist() == [1.5, 1.5]
    assert layer._flashinfer_trtllm_gemm1_beta.tolist() == [2.5, 2.5]
    assert layer._flashinfer_trtllm_gemm1_clamp_limit.tolist() == [3.5, 3.5]


def test_owned_trtllm_runner_materializes_none_valued_params(_runner_backend_flag):
    _runner_backend_flag.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM

    config = MoeRunnerConfig()  # gemm1_alpha/beta/clamp_limit all default to None
    fp8_method = _make_fp8_method()
    layer = _make_layer()
    fp8_method.create_moe_runner(layer, config)

    fp8_method.process_weights_after_loading(layer)

    assert layer._flashinfer_trtllm_gemm1_alpha is None
    assert layer._flashinfer_trtllm_gemm1_beta is None
    assert layer._flashinfer_trtllm_gemm1_clamp_limit is None


def test_owned_triton_runner_does_not_materialize_activation_params(
    _runner_backend_flag,
):
    _runner_backend_flag.runner_backend = MoeRunnerBackend.TRITON

    config = MoeRunnerConfig(gemm1_alpha=1.5, gemm1_beta=2.5, gemm1_clamp_limit=3.5)
    fp8_method = _make_fp8_method()
    layer = _make_layer()
    fp8_method.create_moe_runner(layer, config)

    assert fp8_method._owned_moe_runner is not None

    fp8_method.process_weights_after_loading(layer)

    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_alpha")
    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_beta")
    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_clamp_limit")


@pytest.mark.parametrize(
    "backend",
    [
        MoeRunnerBackend.CUTLASS,
        MoeRunnerBackend.MARLIN,
        MoeRunnerBackend.FLASHINFER_CUTEDSL,
    ],
)
def test_direct_kernel_backend_never_owns_a_runner(_runner_backend_flag, backend):
    _runner_backend_flag.runner_backend = backend

    fp8_method = _make_fp8_method()
    layer = _make_layer()
    fp8_method.create_moe_runner(layer, MoeRunnerConfig())

    assert fp8_method._owned_moe_runner is None
    assert not hasattr(fp8_method, "runner")


def test_wrapper_call_chain_does_not_crash_on_borrowed_fp8_method(
    _runner_backend_flag,
):
    """Reproduces the reported stack trace: wrapper -> borrowed _fp8."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )

    _runner_backend_flag.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM

    wrapper = object.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    wrapper._fp8 = _make_fp8_method()
    assert wrapper._fp8._owned_moe_runner is None

    layer = _make_layer()
    # Skip the real FP4 reorder path; unrelated to this fix.
    layer._mega_moe_weights_built = True

    wrapper.process_weights_after_loading(layer)

    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_alpha")


def test_dispatcher_set_quant_config_still_runs(_runner_backend_flag):
    _runner_backend_flag.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM

    fp8_method = _make_fp8_method()
    layer = _make_layer(with_dispatcher=True)

    fp8_method.process_weights_after_loading(layer)

    assert layer.dispatcher.set_quant_config_calls == [
        {"weight_dtype": layer.w13_weight.dtype}
    ]


def test_hpc_ops_preparation_unaffected_by_runner_ownership(_runner_backend_flag):
    _runner_backend_flag.runner_backend = MoeRunnerBackend.HPC_OPS

    fp8_method = _make_fp8_method()
    assert fp8_method._owned_moe_runner is None

    calls = []
    fp8_method._prepare_hpc_ops_weights = lambda layer: calls.append(layer)

    layer = _make_layer()
    fp8_method.process_weights_after_loading(layer)

    assert calls == [layer]
    assert not hasattr(layer, "_flashinfer_trtllm_gemm1_alpha")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
