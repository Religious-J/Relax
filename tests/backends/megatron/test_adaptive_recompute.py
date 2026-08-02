# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CI guard and runtime checks for memory-adaptive recompute elision."""

from pathlib import Path

import pytest


PATCH = Path(__file__).resolve().parents[3] / "docker" / "patch" / "latest" / "megatron.patch"
MODEL = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "model.py"
HINT = f"the adaptive recompute hunk is missing from {PATCH}"

try:
    from megatron.core.transformer.transformer_block import (
        _evenly_spaced_layer_indices,
        _memory_budgeted_elision_count,
    )
except ImportError:  # CI installs no training dependencies
    _evenly_spaced_layer_indices = None
    _memory_budgeted_elision_count = None

needs_megatron = pytest.mark.skipif(
    _memory_budgeted_elision_count is None,
    reason="requires patched Megatron",
)


def test_megatron_patch_carries_adaptive_recompute():
    assert PATCH.is_file(), HINT
    patch = PATCH.read_text()
    assert "megatron/core/transformer/transformer_block.py" in patch, HINT
    assert "_memory_budgeted_elision_count" in patch, HINT
    assert "reset_adaptive_recompute" in patch, HINT
    assert "_ADAPTIVE_RECOMPUTE_MIN_RESERVE_BYTES" in patch, HINT
    assert "baseline_transient_bytes" in patch, HINT
    assert "get_pipeline_model_parallel_world_size() != 1" in patch, HINT
    assert "get_data_parallel_world_size() != 1" in patch, HINT
    assert "current_input_numel" in patch, HINT
    assert "state.planned_input_numel" in patch, HINT
    assert "return set()" in patch, HINT


def test_relax_opts_in_at_each_optimizer_step():
    source = MODEL.read_text()

    assert "def _reset_adaptive_recompute" in source
    train_start = source.index("def train(")
    loop_start = source.index("for step_id in range(num_steps_per_rollout):", train_start)
    reset = source.index("_reset_adaptive_recompute(model)", loop_start)
    train_step = source.index("train_one_step(", loop_start)
    assert loop_start < reset < train_step
    assert "if len(resetters) != 1:" in source


@needs_megatron
def test_memory_budget_reserves_hbm_before_eliding_recompute():
    gib = 1024**3
    count = _memory_budgeted_elision_count(
        num_layers=48,
        effective_free_bytes=30 * gib,
        baseline_transient_bytes=5 * gib,
        layer_activation_bytes=1 * gib,
        total_bytes=96 * gib,
    )

    # 30 GiB free - 5 GiB transient - 9.6 GiB reserve = 15.4 layers.
    assert count == 15


@needs_megatron
def test_memory_budget_caps_retained_layers_at_three_quarters():
    gib = 1024**3
    count = _memory_budgeted_elision_count(
        num_layers=48,
        effective_free_bytes=96 * gib,
        baseline_transient_bytes=0,
        layer_activation_bytes=1 * gib,
        total_bytes=96 * gib,
    )

    assert count == 36


@needs_megatron
def test_even_layer_selection_is_deterministic_and_unbiased():
    selected = _evenly_spaced_layer_indices(num_layers=48, count=35)

    assert len(selected) == 35
    assert min(selected) == 0
    assert max(selected) == 47
    assert selected == _evenly_spaced_layer_indices(num_layers=48, count=35)
