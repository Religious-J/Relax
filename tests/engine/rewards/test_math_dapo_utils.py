# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.engine.rewards.math_dapo_utils import compute_score


def test_compute_score_accepts_plain_boxed_answer():
    result = compute_score("Reasoning.\nAnswer: \\boxed{6}", "6")

    assert result == {"score": 1.0, "acc": True, "pred": "6"}


def test_compute_score_accepts_bold_answer_label():
    result = compute_score("Reasoning.\n**Answer:** \\boxed{6}<|im_end|>", "6")

    assert result == {"score": 1.0, "acc": True, "pred": "6"}


def test_compute_score_accepts_bold_boxed_value():
    result = compute_score("Reasoning.\nAnswer: **\\boxed{6}**", "6")

    assert result == {"score": 1.0, "acc": True, "pred": "6"}


def test_compute_score_accepts_italic_answer_label():
    result = compute_score("Reasoning.\n_Answer:_ \\boxed{6}", "6")

    assert result == {"score": 1.0, "acc": True, "pred": "6"}


def test_compute_score_accepts_plain_unboxed_answer():
    result = compute_score("Reasoning.\nAnswer: 6", "6")

    assert result == {"score": 1.0, "acc": True, "pred": "6"}


def test_compute_score_uses_last_answer_and_last_boxed_value():
    result = compute_score("Answer: \\boxed{7}\nCorrection.\n**Answer:** first \\boxed{8}, finally \\boxed{6}", "6")

    assert result == {"score": 1.0, "acc": True, "pred": "6"}


def test_compute_score_preserves_internal_multiplication_marker():
    result = compute_score("Reasoning.\nAnswer: **\\boxed{2*3}**", "6")

    assert result == {"score": -1.0, "acc": False, "pred": "2*3"}


def test_compute_score_keeps_wrong_answer_negative():
    result = compute_score("Reasoning.\n**Answer:** \\boxed{7}", "6")

    assert result == {"score": -1.0, "acc": False, "pred": "7"}
