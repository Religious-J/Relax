# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU coverage of conditioning reuse, collective decisions and telemetry."""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from relax.engine.sft import conditioning
from relax.models.generative_sft import ConditioningSpec
from relax.utils.conditioning_cache import build_key


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.conditioning_cache = None


@pytest.fixture
def runtime_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No model checkpoint or distributed cluster is needed by local tests."""
    monkeypatch.setattr(conditioning, "model_fingerprint", lambda _: "frozen-weights-v1")
    monkeypatch.setattr(conditioning.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("RELAX_CONDITION_PROFILE", "1")
    runtimes = []

    def make(*, enabled: bool = True, telemetry: bool = False, start_step: int = 0, num_steps: int = 2):
        index = len(runtimes)
        directory = tmp_path / f"run-{index}"
        monkeypatch.setenv("RELAX_CONDITION_CACHE", str(int(enabled)))
        monkeypatch.setenv("RELAX_CONDITION_CACHE_MAX_BYTES", str(4 * 1024**2))
        monkeypatch.setenv("RELAX_CONDITION_CACHE_DIR", str(tmp_path / "cache"))
        if telemetry:
            monkeypatch.setenv("RELAX_CONDITION_CACHE_RUN_DIR", str(directory))
        else:
            monkeypatch.delenv("RELAX_CONDITION_CACHE_RUN_DIR", raising=False)
        model = DummyModel()
        runtime = conditioning.ConditioningRuntime(
            model=model,
            model_path="unused-model-path",
            device=torch.device("cpu"),
            group=None,
            rank=0,
            world_size=1,
            namespace=f"test-run-{index}",
            config={"seed": 42},
            metadata_sha256="data-v1",
            global_batch_size=1,
            start_step=start_step,
            num_steps=num_steps,
        )
        runtimes.append(runtime)
        return runtime, model, directory

    yield make
    for runtime in reversed(runtimes):
        runtime.close(runtime._start_step, status="cancelled")


def _spec(*, cacheable: bool = True) -> ConditioningSpec:
    return ConditioningSpec("dummy.condition.v1", {"input": "same-content", "dtype": "bf16"}, cacheable)


def _restore(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if set(features) != {"value"}:
        raise ValueError("Unexpected conditioning fields")
    return {"condition": features["value"]}


def test_disabled_runtime_never_constructs_or_uses_cache(runtime_factory, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Disabled online profile must not construct cache storage")

    monkeypatch.setattr(conditioning, "ConditioningCache", forbidden)
    runtime, model, _ = runtime_factory(enabled=False)
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.arange(6, dtype=torch.bfloat16)}

    first = runtime.resolve(_spec(), compute, _restore)
    second = runtime.resolve(_spec(), compute, _restore)
    assert torch.equal(first["condition"], second["condition"])
    assert first["condition"].dtype == torch.bfloat16
    assert len(calls) == 2
    assert runtime.cache is None
    assert not (tmp_path / "cache").exists()
    assert model.conditioning_cache is runtime
    assert runtime._step["encoding_calls"] == 2
    assert runtime._step["effective_hit"] == 0


def test_cold_then_warm_returns_exact_bf16_without_reencoding(runtime_factory):
    runtime, _, _ = runtime_factory()
    expected = torch.tensor([0.0, -1.5, 2.25, 100.0], dtype=torch.bfloat16)
    calls = []

    def compute():
        calls.append(1)
        return {"value": expected.clone()}

    cold = runtime.resolve(_spec(), compute, _restore)
    warm = runtime.resolve(_spec(), compute, _restore)
    assert cold["condition"].dtype == warm["condition"].dtype == torch.bfloat16
    assert torch.equal(cold["condition"], expected)
    assert torch.equal(warm["condition"], expected)
    assert len(calls) == 1
    assert runtime._step["local_hit"] == runtime._step["effective_hit"] == 1
    assert runtime._step["encoding_calls"] == 1
    assert runtime._step["cache_read_bytes"] == expected.numel() * expected.element_size()


def test_cpu_hit_owns_storage_after_lease_closes(runtime_factory, monkeypatch):
    runtime, _, _ = runtime_factory()
    expected = torch.arange(8, dtype=torch.bfloat16)
    runtime.resolve(_spec(), lambda: {"value": expected}, _restore)
    lookup = runtime.cache.lookup
    borrowed_pointers = []

    def tracked_lookup(key):
        lease = lookup(key)
        borrowed_pointers.append(lease.data["value"].data_ptr())
        return lease

    monkeypatch.setattr(runtime.cache, "lookup", tracked_lookup)
    result = runtime.resolve(_spec(), lambda: pytest.fail("Expected a warm hit"), _restore)["condition"]
    assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
    assert result.dtype == expected.dtype
    assert result.data_ptr() != borrowed_pointers[0], "Returned CPU data must not borrow a closed mmap lease"
    runtime.close(0, status="cancelled")
    assert torch.equal(result, expected)


def _fake_cuda_events(runtime, monkeypatch, actions):
    runtime.device = torch.device("cuda:0")
    monkeypatch.setattr(runtime, "_resources", lambda **_kwargs: {})

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, stream=None):
            actions.append("event_record")

        def synchronize(self):
            actions.append("event_wait")

    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("Cache copies must not synchronize the entire device")

    monkeypatch.setattr(conditioning.torch.cuda, "Event", Event)
    monkeypatch.setattr(conditioning.torch.cuda, "current_stream", lambda device: "copy-stream")
    monkeypatch.setattr(conditioning.torch.cuda, "synchronize", forbidden_sync)


@pytest.mark.parametrize("failure", ["copy", "restore"])
def test_cuda_copy_failure_drains_event_before_lease_close_and_unanimous_fallback(
    runtime_factory, monkeypatch, failure
):
    runtime, _, _ = runtime_factory()
    actions = []
    _fake_cuda_events(runtime, monkeypatch, actions)
    expected = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)

    class CopyTensor:
        def __init__(self, fail=False):
            self.fail = fail

        def to(self, *_args, **_kwargs):
            actions.append("copy")
            if self.fail:
                raise RuntimeError("simulated partial H2D failure")
            return expected.clone()

        def numel(self):
            return expected.numel()

        def element_size(self):
            return expected.element_size()

    class Lease:
        data = {"value": CopyTensor(), "invalid_field": CopyTensor(fail=failure == "copy")}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            actions.append("lease_close")

    monkeypatch.setattr(runtime.cache, "lookup", lambda key: Lease())
    monkeypatch.setattr(runtime.cache, "put", lambda *_args: False)
    monkeypatch.setattr(runtime.cache, "invalidate", lambda key: actions.append("invalidate"))
    votes = []

    def vote(ready, group):
        actions.append("vote")
        votes.append(ready)
        return ready

    def compute():
        actions.append("compute")
        return {"value": expected.clone()}

    monkeypatch.setattr(conditioning, "unanimous_ready", vote)
    result = runtime.resolve(_spec(), compute, _restore)
    assert torch.equal(result["condition"], expected)
    assert votes == [False], "Copy/restore failure must reach the one shared decision before online compute"
    assert actions.index("event_record") < actions.index("event_wait") < actions.index("lease_close")
    assert (
        actions.index("lease_close") < actions.index("invalidate") < actions.index("vote") < actions.index("compute")
    )
    assert runtime._step["cache_errors"] == 1
    assert (
        runtime._step["condition_h2d_bytes"]
        == (1 if failure == "copy" else 2) * expected.numel() * expected.element_size()
    )
    assert runtime._step["h2d_enqueue_seconds"] >= 0
    assert runtime._step["h2d_wait_seconds"] >= 0


def test_optional_packed_copy_counts_actual_transfer_bytes_and_keeps_one_vote(runtime_factory, monkeypatch):
    runtime, _, _ = runtime_factory()
    actions, votes = [], []
    _fake_cuda_events(runtime, monkeypatch, actions)
    expected = torch.arange(4, dtype=torch.bfloat16)

    class Lease:
        def __enter__(self):
            return self

        def copy_to(self, device):
            assert device == torch.device("cuda:0")
            actions.append("packed_copy")
            return {"value": expected.clone()}, 4096

        def __exit__(self, *_args):
            actions.append("lease_close")

    def vote(ready, group):
        actions.append("vote")
        votes.append(ready)
        return ready

    monkeypatch.setattr(runtime.cache, "lookup", lambda key: Lease())
    monkeypatch.setattr(conditioning, "unanimous_ready", vote)
    result = runtime.resolve(_spec(), lambda: pytest.fail("Packed hit must skip online encoding"), _restore)
    assert torch.equal(result["condition"], expected)
    assert votes == [True]
    assert actions == ["packed_copy", "event_record", "event_wait", "lease_close", "vote"]
    assert runtime._step["condition_h2d_bytes"] == 4096
    assert runtime._step["cache_read_bytes"] == expected.numel() * expected.element_size()
    assert runtime._step["effective_hit"] == 1


def test_local_hit_with_missing_peer_forces_online_but_preserves_entry(runtime_factory, monkeypatch):
    runtime, _, _ = runtime_factory()
    expected = torch.arange(4, dtype=torch.bfloat16)
    calls = []

    def compute():
        calls.append(1)
        return {"value": expected.clone()}

    runtime.resolve(_spec(), compute, _restore)
    runtime._step.clear()
    votes = []

    def peer_missing(ready: bool, group: Any) -> bool:
        votes.append((ready, group))
        return False

    monkeypatch.setattr(conditioning, "unanimous_ready", peer_missing)
    actual = runtime.resolve(_spec(), compute, _restore)
    assert votes == [(True, None)]
    assert torch.equal(actual["condition"], expected)
    assert len(calls) == 2
    assert runtime._step["local_hit"] == runtime._step["forced_recompute"] == 1
    assert runtime._step["effective_hit"] == 0
    monkeypatch.setattr(conditioning, "unanimous_ready", lambda ready, group: ready)
    runtime.resolve(_spec(), compute, _restore)
    assert len(calls) == 2


def test_invalid_restore_schema_is_invalidated_and_rebuilt_online(runtime_factory, monkeypatch):
    runtime, _, _ = runtime_factory()
    spec = _spec()
    key = build_key({"weights": runtime.fingerprint, "schema": spec.name, "dependencies": spec.dependencies})
    assert runtime.cache.put(key, {"wrong_field": torch.ones(3, dtype=torch.bfloat16)})
    invalidated = []
    original_invalidate = runtime.cache.invalidate

    def invalidate(key_to_remove):
        invalidated.append(key_to_remove)
        return original_invalidate(key_to_remove)

    monkeypatch.setattr(runtime.cache, "invalidate", invalidate)
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.tensor([1, 2, 3], dtype=torch.bfloat16)}

    actual = runtime.resolve(spec, compute, _restore)
    assert torch.equal(actual["condition"], torch.tensor([1, 2, 3], dtype=torch.bfloat16))
    assert invalidated == [key]
    assert runtime._step["cache_errors"] == 1
    runtime.resolve(spec, compute, _restore)
    assert len(calls) == 1, "A rebuilt schema must be reusable on the next lookup"


def test_uncacheable_conditions_preserve_fp32_gradient_and_cleanup(runtime_factory, monkeypatch):
    runtime, model, _ = runtime_factory()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Trainable conditions must bypass cache reads and writes")

    monkeypatch.setattr(runtime.cache, "lookup", forbidden)
    monkeypatch.setattr(runtime.cache, "put", forbidden)
    inputs = torch.tensor([1.0, -3.0], dtype=torch.float32)
    result = runtime.resolve(_spec(cacheable=False), lambda: {"value": inputs * model.weight}, _restore)
    result["condition"].square().sum().backward()
    expected = 2 * model.weight.detach() * inputs.square().sum()
    assert result["condition"].requires_grad
    assert model.weight.grad.dtype == torch.float32
    assert torch.equal(model.weight.grad, expected)
    runtime.close(0, status="cancelled")
    assert model.conditioning_cache is None
    assert torch.equal(model.weight.grad, expected), "Cleanup must not clear the optimizer's FP32 gradients"


@pytest.mark.parametrize("mutation", ["copy", "unfreeze_refreeze", "replace"])
def test_conditioning_mutation_permanently_bypasses_stale_entries(runtime_factory, monkeypatch, mutation):
    runtime, model, directory = runtime_factory(telemetry=True, num_steps=3)
    model.weight.requires_grad_(False)

    def state():
        parameter = model.weight
        return {
            "frozen": not parameter.requires_grad,
            "parameters": {
                "prefix.weight": {
                    "identity": id(parameter),
                    "version": parameter._version,
                    "local_version": parameter._version,
                    "requires_grad": parameter.requires_grad,
                    "dtype": str(parameter.dtype),
                    "shape": tuple(parameter.shape),
                }
            },
        }

    model.conditioning_state = state
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.ones(3) * model.weight}

    runtime.begin_step()
    runtime.sample(0)
    runtime.resolve(_spec(), compute, _restore)
    runtime.end_step(0, local_samples=1, loss=0.25)
    runtime.begin_step()
    if mutation == "copy":
        with torch.no_grad():
            model.weight.copy_(torch.tensor(3.0))
    elif mutation == "replace":
        model.weight = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)
    else:
        model.weight.requires_grad_(True)
        runtime.sample(0)  # The training transition is observed before forward.
        with torch.no_grad():
            model.weight.add_(1)
        model.weight.requires_grad_(False)
    runtime.sample(0)
    first_reason = runtime._cache_guard_reason

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A changed encoder must neither load nor republish the old cache key")

    monkeypatch.setattr(runtime.cache, "lookup", forbidden)
    monkeypatch.setattr(runtime.cache, "put", forbidden)
    votes = []

    def vote(ready, group):
        votes.append(ready)
        return ready

    monkeypatch.setattr(conditioning, "unanimous_ready", vote)
    result = runtime.resolve(_spec(), compute, _restore)
    assert torch.equal(result["condition"], torch.ones(3) * 3)
    assert runtime._cache_guard_disabled and model.conditioning_reuse_disabled
    assert len(calls) == 2 and votes == [False]
    runtime.sample(0)
    runtime.resolve(_spec(), compute, _restore)
    assert runtime._cache_guard_reason == first_reason
    assert runtime._step["effective_hit"] == 0
    assert runtime._step["cache_errors"] == 0
    runtime.end_step(1, local_samples=1, loss=0.25)
    events = [json.loads(line) for line in (directory / "events-rank0.jsonl").read_text().splitlines()]
    last = [event for event in events if event["stage"] == "step"][-1]
    assert last["cache_guard_disabled"]
    assert last["cache_guard_reason"] == first_reason
    assert "prefix.weight" in first_reason


def test_initial_trainable_conditioning_state_disables_reuse(runtime_factory):
    runtime, model, _ = runtime_factory()
    model.conditioning_state = lambda: {
        "frozen": False,
        "parameters": {"prefix.weight": {"requires_grad": True}},
    }
    runtime.sample(0)
    assert runtime._cache_guard_disabled
    assert model.conditioning_reuse_disabled
    assert runtime._cache_guard_reason == "prefix.weight.requires_grad is true"


@pytest.mark.parametrize("failure", ["lookup", "put"])
def test_cache_io_errors_keep_online_result_and_downstream_fp32_gradient(runtime_factory, monkeypatch, failure):
    runtime, model, _ = runtime_factory()

    def fail(*_args, **_kwargs):
        raise OSError("simulated cache I/O failure")

    monkeypatch.setattr(runtime.cache, failure, fail)
    expected = torch.tensor([1.25, -0.5], dtype=torch.float32)
    result = runtime.resolve(_spec(), lambda: {"value": expected.clone()}, _restore)
    (result["condition"] * model.weight).sum().backward()
    assert torch.equal(result["condition"], expected)
    assert model.weight.grad.dtype == torch.float32
    assert torch.equal(model.weight.grad, expected.sum())
    assert runtime._step["cache_errors"] == 1


def test_force_online_populates_cache_without_skip(runtime_factory):
    runtime, _, _ = runtime_factory()
    calls = []

    def compute():
        calls.append(1)
        return {"value": torch.arange(3, dtype=torch.bfloat16)}

    runtime.force_online = True
    runtime.resolve(_spec(), compute, _restore)
    runtime.resolve(_spec(), compute, _restore)
    assert len(calls) == 2
    runtime.force_online = False
    runtime.resolve(_spec(), compute, _restore)
    assert len(calls) == 2


def test_step_jsonl_is_complete_delta_counted_and_close_is_idempotent(runtime_factory, monkeypatch):
    runtime, model, directory = runtime_factory(telemetry=True)
    monkeypatch.setattr(runtime, "_resources", lambda **_kwargs: {"cpu_rss_bytes": None, "shm_used_bytes": None})

    def compute():
        return {"value": torch.ones(2, dtype=torch.bfloat16)}

    for step in range(2):
        runtime.begin_step()
        runtime.sample(7)
        runtime.record_prepare(0.125)
        runtime.record_input_transfer(16)
        runtime.resolve(_spec(), compute, _restore)
        runtime.end_step(step, local_samples=1, loss=0.25)
    runtime.close(2)
    runtime.close(2)
    events = [json.loads(line) for line in (directory / "events-rank0.jsonl").read_text().splitlines()]
    assert [event["stage"] for event in events] == ["init", "step", "step", "close"]
    initial, cold, warm, closed = events
    assert initial["expected_steps"] == 2
    assert initial["model_fingerprint"] == "frozen-weights-v1"
    assert initial["metadata_sha256"] == "data-v1"
    assert cold["encoding_calls"] == 1 and cold["effective_hit"] == 0
    assert warm["encoding_calls"] == 0 and warm["effective_hit"] == 1
    assert cold["repeated_h2d_bytes"] == 0 and warm["repeated_h2d_bytes"] == 16
    assert cold["sample_ids"] == warm["sample_ids"] == [7]
    assert cold["global_samples"] == warm["global_samples"] == 1
    assert cold["local_samples"] == warm["local_samples"] == 1
    assert cold["wall_seconds"] > 0 and warm["wall_seconds"] > 0
    for row in (cold, warm):
        assert row["condition_prepare_seconds"] == 0.125
        assert all(row[field] >= 0 for field in ("h2d_enqueue_seconds", "h2d_wait_seconds", "cache_restore_seconds"))
    assert warm["h2d_wait_seconds"] == warm["condition_h2d_bytes"] == 0
    assert cold["cpu_rss_bytes"] is None and warm["shm_used_bytes"] is None
    assert closed["completed_steps"] == 2 and closed["status"] == "complete"
    assert closed["total_seconds"] >= 0 and closed["close_seconds"] >= 0
    assert model.conditioning_cache is None
    assert runtime._writer.closed


def test_cancelled_close_records_only_this_runs_completed_steps(runtime_factory):
    runtime, _, directory = runtime_factory(telemetry=True, start_step=4, num_steps=8)
    runtime.close(5, status="cancelled")
    initial, closed = [json.loads(line) for line in (directory / "events-rank0.jsonl").read_text().splitlines()]
    assert initial["start_step"] == 4 and initial["expected_steps"] == 4
    assert closed["completed_steps"] == 1 and closed["status"] == "cancelled"


def test_resource_sampling_uses_light_stats_between_periodic_first_and_last_details(runtime_factory, monkeypatch):
    runtime, _, directory = runtime_factory(telemetry=True, num_steps=18)
    calls = []

    def full():
        calls.append("full")
        return {"storage_scope": "filesystem", "resident_bytes": 32, "total_bytes": 64, "metadata_bytes": 32}

    def light():
        calls.append("light")
        return {
            "storage_scope": "index_accounting",
            "resident_bytes": 32,
            "total_bytes": None,
            "metadata_bytes": None,
            "leased_bytes": None,
        }

    monkeypatch.setattr(runtime.cache, "stats", full)
    monkeypatch.setattr(runtime.cache, "stats_light", light, raising=False)
    monkeypatch.setattr(conditioning, "_rss_bytes", lambda: 123)
    for step in range(18):
        runtime.begin_step()
        runtime.end_step(step, local_samples=1, loss=0.25)
    runtime.close(18)
    assert calls == ["full" if step in {0, 16, 17} else "light" for step in range(18)] + ["full"]
    events = [json.loads(line) for line in (directory / "events-rank0.jsonl").read_text().splitlines()]
    steps = [row for row in events if row["stage"] == "step"]
    assert all(row["cpu_rss_bytes"] == 123 and "shm_used_bytes" in row for row in steps)
    assert all(row["cache_bytes"] == 32 for row in steps)
    assert [row["step"] for row in steps if row["cache_stats_detailed"]] == [0, 16, 17]
    assert steps[1]["cache_storage_scope"] == "index_accounting"
    assert steps[1]["disk_bytes"] is None
    assert steps[1]["cache_stats"]["metadata_bytes"] is steps[1]["cache_stats"]["leased_bytes"] is None
    assert steps[16]["disk_bytes"] == 64
    assert events[-1]["cache_stats_detailed"] and events[-1]["stage"] == "close"


def test_missing_light_stats_falls_back_to_full_and_preserves_unknown_values(runtime_factory, monkeypatch):
    runtime, _, _ = runtime_factory()
    calls = []

    def full():
        calls.append(1)
        return {"resident_bytes": None, "total_bytes": None, "metadata_bytes": None}

    monkeypatch.setattr(runtime.cache, "stats_light", None, raising=False)
    monkeypatch.setattr(runtime.cache, "stats", full)
    resources = runtime._resources(detailed=False)
    assert calls == [1]
    assert resources["cache_stats_detailed"] and resources["cache_storage_scope"] == "filesystem"
    assert resources["cache_bytes"] is resources["disk_bytes"] is None
    with monkeypatch.context() as patch:
        patch.setattr(runtime.cache, "directory", Path("/dev/shm/relax-conditioning/test"))
        assert runtime._resources(detailed=False)["disk_bytes"] == 0


@pytest.mark.parametrize("scope", ["unavailable", "last_detailed_snapshot"])
def test_stale_or_unavailable_snapshots_do_not_claim_current_storage(runtime_factory, monkeypatch, scope):
    runtime, _, _ = runtime_factory()
    monkeypatch.setattr(
        runtime.cache,
        "stats_light",
        lambda: {
            "storage_scope": scope,
            "resident_bytes": 32,
            "total_bytes": 64,
        },
        raising=False,
    )
    resources = runtime._resources(detailed=False)
    assert resources["cache_storage_scope"] == scope
    assert resources["cache_bytes"] is resources["disk_bytes"] is None
    assert not resources["cache_stats_detailed"]


def test_resource_stats_failure_is_unknown_and_does_not_prevent_close(runtime_factory, monkeypatch):
    runtime, model, directory = runtime_factory(telemetry=True)

    def unavailable():
        raise OSError("simulated resource read failure")

    monkeypatch.setattr(runtime.cache, "stats_light", unavailable, raising=False)
    monkeypatch.setattr(runtime.cache, "stats", unavailable)
    resources = runtime._resources(detailed=False)
    assert resources["cache_storage_scope"] == "unavailable"
    assert resources["cache_bytes"] is resources["disk_bytes"] is None
    runtime.close(0, status="cancelled")
    assert model.conditioning_cache is None and runtime._writer.closed
    closed = json.loads((directory / "events-rank0.jsonl").read_text().splitlines()[-1])
    assert closed["status"] == "cancelled" and closed["cache_bytes"] is None


def test_missing_shared_memory_and_cpu_resources_are_unknown(runtime_factory, monkeypatch):
    runtime, _, _ = runtime_factory(enabled=False)

    def unavailable(*_args, **_kwargs):
        raise OSError("resource measurement unavailable")

    monkeypatch.setattr(conditioning.os, "statvfs", unavailable)
    monkeypatch.setattr(conditioning.Path, "read_text", unavailable)
    resources = runtime._resources()
    assert resources["shm_used_bytes"] is None
    assert resources["cpu_rss_bytes"] is None
    assert "gpu_allocated_bytes" not in resources, "CPU-only runs must not claim measured zero GPU memory"


def _gloo_ready_worker(rank: int, rendezvous: str, queue: Any) -> None:
    """Use real processes so mixed cache readiness cannot hide a rank
    mismatch."""
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))
    try:
        mixed = conditioning.unanimous_ready(rank == 0, group)
        all_ready = conditioning.unanimous_ready(True, group)
        none_ready = conditioning.unanimous_ready(False, group)
        queue.put((rank, mixed, all_ready, none_ready))
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo is required")
def test_real_gloo_group_requires_unanimous_readiness(tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_gloo_ready_worker, args=(rank, str(tmp_path / "rendezvous"), queue))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=25)
            assert not process.is_alive(), "A cache-readiness vote stranded a Gloo rank"
            assert process.exitcode == 0
        assert sorted(queue.get(timeout=2) for _ in processes) == [(0, False, True, False), (1, False, True, False)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        queue.close()
        queue.join_thread()
