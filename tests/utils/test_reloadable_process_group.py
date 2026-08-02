# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils import reloadable_process_group as rpg


def test_deferred_destroy_overlaps_body_with_port_cooldown(monkeypatch):
    calls = []
    clock = iter([10.0, 10.75])

    def fake_destroy(*, post_destroy_delay):
        calls.append(("destroy", post_destroy_delay))
        return 20

    monkeypatch.setattr(rpg, "destroy_process_groups", fake_destroy)
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpg.time, "sleep", lambda delay: calls.append(("sleep", delay)))

    with rpg.destroy_process_groups_deferred(post_destroy_delay=2.0):
        calls.append(("body", None))

    assert calls[:2] == [("destroy", 0), ("body", None)]
    assert calls[2][0] == "sleep"
    assert calls[2][1] == pytest.approx(1.25)


def test_deferred_destroy_does_not_oversleep(monkeypatch):
    sleeps = []
    clock = iter([10.0, 12.5])

    monkeypatch.setattr(rpg, "destroy_process_groups", lambda *, post_destroy_delay: 20)
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpg.time, "sleep", sleeps.append)

    with rpg.destroy_process_groups_deferred(post_destroy_delay=2.0):
        pass

    assert sleeps == []


def test_deferred_destroy_skips_delay_when_no_group_was_destroyed(monkeypatch):
    sleeps = []

    monkeypatch.setattr(rpg, "destroy_process_groups", lambda *, post_destroy_delay: 0)
    monkeypatch.setattr(rpg.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(rpg.time, "sleep", sleeps.append)

    with rpg.destroy_process_groups_deferred(post_destroy_delay=2.0):
        pass

    assert sleeps == []


def test_deferred_destroy_rejects_negative_delay(monkeypatch):
    monkeypatch.setattr(
        rpg,
        "destroy_process_groups",
        lambda **_: pytest.fail("destroy must not run for an invalid delay"),
    )

    with pytest.raises(ValueError, match="non-negative"):
        with rpg.destroy_process_groups_deferred(post_destroy_delay=-0.1):
            pass


def test_deferred_destroy_preserves_body_error_after_finishing_cooldown(monkeypatch):
    sleeps = []
    clock = iter([10.0, 10.5])

    monkeypatch.setattr(rpg, "destroy_process_groups", lambda *, post_destroy_delay: 20)
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpg.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="pause failed"):
        with rpg.destroy_process_groups_deferred(post_destroy_delay=2.0):
            raise RuntimeError("pause failed")

    assert sleeps == pytest.approx([1.5])


def test_deferred_destroy_backend_skip_still_runs_body(monkeypatch):
    calls = []

    monkeypatch.setattr(rpg, "_should_skip_reload_and_destroy", lambda: True)
    monkeypatch.setattr(
        rpg.ReloadableProcessGroup,
        "destroy_process_groups",
        lambda **_: pytest.fail("backend skip must not destroy process groups"),
    )
    monkeypatch.setattr(rpg.time, "sleep", lambda _: pytest.fail("backend skip must not sleep"))

    with rpg.destroy_process_groups_deferred(post_destroy_delay=2.0):
        calls.append("body")

    assert calls == ["body"]
