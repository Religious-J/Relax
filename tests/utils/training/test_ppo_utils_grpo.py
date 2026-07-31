# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.utils.training.ppo_utils import (
    INLINE_OLD_LOG_PROBS_KEY,
    can_inline_first_step_old_log_probs,
    get_grpo_returns,
    requires_reference_log_probs,
    use_inline_old_log_probs,
)
from relax.utils.training.ppo_utils import compute_approx_kl as _compiled_compute_approx_kl


compute_approx_kl = torch.compiler.disable(_compiled_compute_approx_kl)


@pytest.mark.parametrize(
    ("kl_coef", "use_kl_loss", "kl_loss_coef", "expected"),
    [
        (0.0, False, 0.0, False),
        (0.0, True, 0.0, False),
        (0.1, False, 0.0, True),
        (0.0, True, 0.1, True),
    ],
)
def test_requires_reference_log_probs_only_for_effective_kl(kl_coef, use_kl_loss, kl_loss_coef, expected):
    args = Namespace(
        kl_coef=kl_coef,
        use_kl_loss=use_kl_loss,
        kl_loss_coef=kl_loss_coef,
    )

    assert requires_reference_log_probs(args) is expected


def _inline_args(**overrides):
    values = {
        "colocate": True,
        "fully_async": False,
        "hybrid": False,
        "loss_type": "policy_loss",
        "compute_advantages_and_returns": True,
        "true_on_policy_mode": False,
        "kl_coef": 0.0,
        "use_rollout_logprobs": False,
        "keep_old_actor": False,
        "use_opd": False,
        "use_routing_replay": False,
        "use_rollout_routing_replay": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "num_experts": None,
        "fp8": None,
        "multimodal_keys": None,
        "is_vl_model": False,
        "custom_megatron_before_log_prob_hook_path": None,
        "custom_megatron_before_train_step_hook_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_inline_first_step_old_log_probs_accepts_safe_multi_step_text_policy_loss():
    assert can_inline_first_step_old_log_probs(_inline_args(), 2)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("colocate", False),
        ("fully_async", True),
        ("hybrid", True),
        ("loss_type", "value_loss"),
        ("compute_advantages_and_returns", False),
        ("true_on_policy_mode", True),
        ("kl_coef", 0.1),
        ("use_rollout_logprobs", True),
        ("keep_old_actor", True),
        ("use_opd", True),
        ("use_routing_replay", True),
        ("use_rollout_routing_replay", True),
        ("hidden_dropout", 0.1),
        ("attention_dropout", 0.1),
        ("num_experts", 8),
        ("fp8", "hybrid"),
        ("multimodal_keys", ["pixel_values"]),
        ("is_vl_model", True),
        ("custom_megatron_before_log_prob_hook_path", "hooks.before_log_prob"),
        ("custom_megatron_before_train_step_hook_path", "hooks.before_train"),
        ("rollout_data_postprocess_path", "hooks.postprocess"),
    ],
)
def test_inline_first_step_old_log_probs_rejects_non_equivalent_paths(override, value):
    assert not can_inline_first_step_old_log_probs(_inline_args(**{override: value}), 2)


def test_inline_first_step_old_log_probs_requires_more_than_one_train_step():
    assert not can_inline_first_step_old_log_probs(_inline_args(), 1)


def test_inline_old_log_probs_marker_is_uniform_per_microbatch():
    assert not use_inline_old_log_probs({})
    assert use_inline_old_log_probs({INLINE_OLD_LOG_PROBS_KEY: [True, True]})
    assert not use_inline_old_log_probs({INLINE_OLD_LOG_PROBS_KEY: [False, False]})

    with pytest.raises(RuntimeError, match="cannot mix inline and cached"):
        use_inline_old_log_probs({INLINE_OLD_LOG_PROBS_KEY: [True, False]})
    with pytest.raises(RuntimeError, match="non-empty sequence"):
        use_inline_old_log_probs({INLINE_OLD_LOG_PROBS_KEY: []})


def test_get_grpo_returns_broadcasts_rewards_to_token_shapes():
    torch.manual_seed(0)
    rewards = torch.tensor([1.5, -0.25])
    kl = [torch.randn(3), torch.randn(2, 2, dtype=torch.float64)]

    returns = get_grpo_returns(rewards, kl)

    expected = [torch.full_like(kl[0], 1.5), torch.full_like(kl[1], -0.25)]
    assert len(returns) == len(expected)
    for actual, expected_return in zip(returns, expected):
        assert torch.allclose(actual, expected_return, atol=1e-6)


def test_compute_approx_kl_k1_returns_signed_log_ratio():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.2, -0.4, -1.0])
    log_probs_base = torch.tensor([-0.3, -0.1, -1.0])

    actual = compute_approx_kl(log_probs, log_probs_base, "k1")
    expected = log_probs.float() - log_probs_base.float()

    assert torch.allclose(actual, expected, atol=1e-6)


def test_compute_approx_kl_k2_matches_formula_and_is_non_negative():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.7, -0.2, -1.4])
    log_probs_base = torch.tensor([-0.1, -0.8, -1.0])
    log_ratio = log_probs.float() - log_probs_base.float()

    actual = compute_approx_kl(log_probs, log_probs_base, "k2")
    expected = log_ratio.square() / 2.0

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.all(actual >= 0)


def test_compute_approx_kl_k3_matches_formula_and_is_non_negative():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.8, -0.2, -1.5])
    log_probs_base = torch.tensor([-0.3, -0.7, -1.0])
    log_ratio = log_probs.float() - log_probs_base.float()

    actual = compute_approx_kl(log_probs, log_probs_base, "k3")
    expected = torch.exp(-log_ratio) - 1.0 + log_ratio

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.all(actual >= 0)


def test_compute_approx_kl_low_var_matches_formula_and_clamps_large_values():
    torch.manual_seed(0)
    log_probs = torch.tensor([-20.0, -0.5, 0.5])
    log_probs_base = torch.zeros_like(log_probs)
    log_ratio = log_probs.float() - log_probs_base.float()

    actual = compute_approx_kl(log_probs, log_probs_base, "low_var_kl")
    expected = torch.clamp(torch.exp(-log_ratio) - 1.0 + log_ratio, min=-10, max=10)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.all(actual >= 0)
    assert torch.all(actual <= 10)


def test_compute_approx_kl_rejects_unknown_estimator():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.2, -0.4])
    log_probs_base = torch.tensor([-0.3, -0.1])

    with pytest.raises(ValueError, match="Unknown kl_loss_type: unsupported"):
        compute_approx_kl(log_probs, log_probs_base, "unsupported")
