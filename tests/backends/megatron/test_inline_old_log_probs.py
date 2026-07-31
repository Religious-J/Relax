# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.backends.megatron import loss as loss_module
from relax.utils.training.ppo_utils import INLINE_OLD_LOG_PROBS_KEY


def _policy_args(*, use_tis: bool) -> Namespace:
    return Namespace(
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        get_mismatch_metrics=False,
        use_tis=use_tis,
        tis_clip_low=0.0,
        tis_clip=2.0,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=True,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=True,
        use_unbiased_kl=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.0,
        use_opd=False,
        opd_loss_coef=0.0,
        opd_token_selection="all",
    )


def _run_policy_loss(monkeypatch, *, inline: bool, use_tis: bool):
    def _fake_get_log_probs_and_entropy(logits, **kwargs):
        return logits.new_empty((0,)), {
            "log_probs": [logits],
            "entropy": [torch.zeros_like(logits)],
        }

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", _fake_get_log_probs_and_entropy)
    monkeypatch.setattr(
        loss_module,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: lambda tensor: tensor.mean(),
    )

    current_log_probs = torch.tensor([-0.4, -0.7, -1.1], requires_grad=True)
    rollout_log_probs = [torch.tensor([-0.45, -0.65, -1.0])]
    batch = {
        "advantages": [torch.tensor([1.2, -0.3, 0.7])],
        "log_probs": [current_log_probs.detach().clone()],
        "rollout_log_probs": rollout_log_probs,
        "unconcat_tokens": [torch.tensor([1, 2, 3, 4])],
        "response_lengths": [3],
        "total_lengths": [4],
        "loss_masks": [torch.ones(3)],
        "ref_log_probs": [torch.tensor([-0.5, -0.8, -1.2])],
    }
    if inline:
        batch[INLINE_OLD_LOG_PROBS_KEY] = [True]
        # The actor keeps a shape-compatible placeholder for pre-train
        # bookkeeping. The inline path must ignore it for loss and TIS.
        batch["log_probs"] = rollout_log_probs

    loss, metrics = loss_module.policy_loss_function(
        _policy_args(use_tis=use_tis),
        batch,
        current_log_probs,
        lambda tensor: tensor.mean(),
    )
    loss.backward()
    return loss.detach(), metrics, current_log_probs.grad.detach()


@pytest.mark.parametrize("use_tis", [False, True])
def test_inline_old_log_probs_matches_cached_current_policy_loss(monkeypatch, use_tis):
    cached_loss, cached_metrics, cached_grad = _run_policy_loss(
        monkeypatch,
        inline=False,
        use_tis=use_tis,
    )
    inline_loss, inline_metrics, inline_grad = _run_policy_loss(
        monkeypatch,
        inline=True,
        use_tis=use_tis,
    )

    assert torch.allclose(inline_loss, cached_loss, atol=0, rtol=0)
    assert inline_metrics.keys() == cached_metrics.keys()
    for key in inline_metrics:
        assert torch.allclose(inline_metrics[key], cached_metrics[key], atol=0, rtol=0), key
    assert torch.allclose(inline_grad, cached_grad, atol=0, rtol=0)
