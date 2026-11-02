# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import fcntl
import hashlib
import json
import multiprocessing
import os
import struct
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from relax.utils.conditioning_cache import ConditioningCache, build_key, fingerprint_file  # noqa: E402


def _cache(capacity: int, directory: str | Path, namespace: str) -> ConditioningCache:
    return ConditioningCache(capacity, directory, namespace, max_staging_bytes=max(1 << 20, 8 * capacity))


def _payload() -> dict:
    value = torch.arange(128, dtype=torch.bfloat16).reshape(8, 16)
    return {"prefix": (value, value[:, ::2]), "mask": [torch.tensor([True, False]), None], "count": 2}


def _assert_exact(actual, reference) -> None:
    if isinstance(reference, torch.Tensor):
        assert actual.dtype == reference.dtype
        assert actual.shape == reference.shape
        assert not actual.requires_grad
        assert torch.equal(actual.reshape(-1).view(torch.uint8), reference.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(reference, dict):
        assert actual.keys() == reference.keys()
        for name, value in reference.items():
            _assert_exact(actual[name], value)
    elif isinstance(reference, (list, tuple)):
        assert type(actual) is type(reference)
        assert len(actual) == len(reference)
        for item, value in zip(actual, reference, strict=True):
            _assert_exact(item, value)
    else:
        assert actual == reference


def _entry_size(directory: Path, payload: dict) -> int:
    with _cache(1 << 20, directory, "measure") as cache:
        assert cache.put("item", payload)
        return cache.stats()["resident_bytes"]


def _parallel_writer(directory: str, capacity: int, index: int, start, queue) -> None:
    try:
        with _cache(capacity, directory, "shared") as cache:
            if not start.wait(15):
                raise RuntimeError("Timed out waiting for writers")
            for number in range(8):
                key = f"{index}-{number}"
                tensor = torch.full((128,), index * 8 + number, dtype=torch.bfloat16)
                cache.put(key, {"value": tensor})
                lease = cache.lookup(key)
                if lease is not None:
                    with lease:
                        _assert_exact(lease.data, {"value": tensor})
                assert cache.stats()["resident_bytes"] <= capacity
        queue.put(None)
    except BaseException as error:
        queue.put(repr(error))


def _borrow_until_released(directory: str, capacity: int, ready, release, queue) -> None:
    try:
        with _cache(capacity, directory, "shared") as cache:
            lease = cache.lookup("kept")
            assert lease is not None
            with lease:
                ready.set()
                if not release.wait(15):
                    raise RuntimeError("Timed out holding the lease")
                _assert_exact(lease.data, _payload())
        queue.put(None)
    except BaseException as error:
        queue.put(repr(error))


def _abandoned_write(directory: str, capacity: int) -> None:
    cache = _cache(capacity, directory, "shared")
    with cache._locked():
        index = cache._index()
        index["reserved_bytes"] = 32
        cache._save_index(index)
        (cache.directory / "payload.pending").write_bytes(b"partial")
        os._exit(0)


def _abandoned_namespace(directory: str) -> None:
    cache = _cache(1 << 20, directory, "dead-run")
    assert cache.put("kept", _payload())
    lease = cache.lookup("kept")
    assert lease is not None
    # Process death must release both the client and payload locks without
    # relying on Python destructors or context-manager cleanup.
    os._exit(0)


def test_key_covers_content_versions_preprocessing_and_dtype(tmp_path: Path) -> None:
    image = tmp_path / "same-name.bin"
    image.write_bytes(b"first")
    dependencies = {"input": fingerprint_file(image), "encoder": "weights-v1", "resize": [32, 64], "dtype": "bf16"}
    key = build_key(dependencies)
    assert key == build_key(dict(reversed(list(dependencies.items()))))
    for field, value in (("encoder", "weights-v2"), ("resize", [64, 32]), ("dtype", "fp32")):
        assert build_key({**dependencies, field: value}) != key
    image.write_bytes(b"other")
    assert build_key({**dependencies, "input": fingerprint_file(image)}) != key
    with pytest.raises(ValueError):
        build_key({"invalid": float("nan")})
    with pytest.raises(TypeError):
        build_key({"invalid": {1: "not a string key"}})


def test_tensor_tree_roundtrip_is_bit_exact_and_put_owns_snapshot(tmp_path: Path) -> None:
    original = _payload()
    original["fp32"] = torch.tensor([float("nan"), float("inf"), -0.0, 1.25], requires_grad=True)
    with _cache(1 << 20, tmp_path, "roundtrip") as cache:
        assert cache.put("item", original)
        lease = cache.lookup("item")
        assert lease is not None
        with lease:
            _assert_exact(lease.data, original)
            assert cache.stats()["leased_bytes"] > 0
        assert lease.data is None
        original["prefix"][0].fill_(99)
        with cache.lookup("item") as second:
            assert second.data["prefix"][0][0, 0] == 0
        stats = cache.stats()
        assert stats["hits"] == 2 and stats["misses"] == 0
        assert stats["leased_bytes"] == 0
        assert stats["checksum_bytes"] == stats["read_bytes"]
        assert stats["bytes_reserved"] == stats["inflight_writes"] == 0


def test_lru_and_capacity_include_current_leases(tmp_path: Path) -> None:
    payload = _payload()
    size = _entry_size(tmp_path, payload)
    with _cache(size * 2, tmp_path, "lru") as cache:
        assert cache.put("a", payload) and cache.put("b", payload)
        with cache.lookup("a"):
            pass
        assert cache.put("c", payload)
        assert cache.lookup("b") is None
        with cache.lookup("a"), cache.lookup("c"):
            assert not cache.put("d", payload)
            assert not cache.clear()
            assert cache.stats()["leased_bytes"] == size * 2
        assert cache.put("d", payload)
        stats = cache.stats()
        assert stats["resident_bytes"] <= size * 2
        assert sum(path.stat().st_size for path in cache.directory.glob("*.safetensors")) <= size * 2
        assert stats["evictions"] == 2 and stats["bypasses"] == 1
        assert cache.clear()


@pytest.mark.parametrize("damage_tensor_bytes", [False, True])
def test_recovery_bills_leased_payload_actual_size_before_new_write(tmp_path: Path, damage_tensor_bytes: bool) -> None:
    payload = _payload()
    size = _entry_size(tmp_path, payload)
    with _cache(size * 2, tmp_path, "damaged-size") as cache:
        assert cache.put("a", payload) and cache.put("b", payload)
        digest = hashlib.sha256(b"a").hexdigest()
        with cache.lookup("a") as lease:
            path = cache.directory / "index.json"
            index = json.loads(path.read_text())
            index["entries"][digest]["size"] = 0
            if damage_tensor_bytes:
                # Otherwise-valid metadata also needs actual-size recovery.
                index["entries"][digest]["tensor_bytes"] = 0
            path.write_text(json.dumps(index))
            assert cache.put("c", payload)
            actual_bytes = sum(item.stat().st_size for item in cache.directory.glob("*.safetensors"))
            assert actual_bytes == size * 2
            recovered = json.loads(path.read_text())
            assert recovered["entries"][digest]["size"] == size
            assert recovered["entries"][digest]["invalid"] is True
            detailed = cache.stats()
            assert detailed["resident_bytes"] == actual_bytes
            assert detailed["leased_bytes"] == size
            _assert_exact(lease.data, payload)
        assert cache.lookup("a") is None  # Damaged metadata cannot authorize a cached read.
        assert cache.put("a", payload)
        with cache.lookup("a") as rebuilt:
            _assert_exact(rebuilt.data, payload)
        assert sum(item.stat().st_size for item in cache.directory.glob("*.safetensors")) <= size * 2


def test_disabled_and_oversized_entries_do_not_allocate_storage(tmp_path: Path) -> None:
    with _cache(0, tmp_path / "disabled", "run") as cache:
        assert cache.lookup("item") is None
        assert not cache.put("item", _payload())
        assert cache.stats()["misses"] == 1
        assert not (tmp_path / "disabled").exists()
    with _cache(64, tmp_path, "small") as cache:
        assert not cache.put("too big", _payload())
        assert cache.stats()["entries"] == 0
        assert not list(cache.directory.glob("*.pending"))


def test_staging_budget_is_checked_before_host_copies(tmp_path: Path, monkeypatch) -> None:
    original = torch.Tensor.to
    copies = []

    def record_copy(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    payload = _payload()
    monkeypatch.setattr(torch.Tensor, "to", record_copy)
    with ConditioningCache(1 << 20, tmp_path, "staging", max_staging_bytes=64) as cache:
        assert not cache.put("item", payload)
        assert not copies
        stats = cache.stats()
        assert stats["entries"] == stats["staging_reserved_bytes"] == 0
        assert stats["peak_staging_reserved_bytes"] == 0
        assert stats["max_payload_and_staging_bytes"] == (1 << 20) + 64


def test_invalidate_releases_only_selected_unborrowed_entry(tmp_path: Path) -> None:
    with _cache(1 << 20, tmp_path, "invalidate") as cache:
        assert cache.put("a", _payload())
        assert cache.put("b", _payload())
        assert not cache.put("a", _payload())  # True means a new write, not already present.
        with cache.lookup("a"):
            assert not cache.invalidate("a")
        assert cache.invalidate("a")
        assert cache.lookup("a") is None
        with cache.lookup("b") as lease:
            _assert_exact(lease.data, _payload())
        assert cache.put("a", _payload())


@pytest.mark.parametrize("corruption", ["bitflip", "truncate", "missing", "index"])
def test_missing_or_corrupt_entries_fall_back_and_can_rebuild(tmp_path: Path, corruption: str) -> None:
    with _cache(1 << 20, tmp_path, "corrupt") as cache:
        assert cache.put("item", _payload())
        path = next(cache.directory.glob("*.safetensors"))
        if corruption == "bitflip":
            content = bytearray(path.read_bytes())
            content[-1] ^= 1
            path.write_bytes(content)
        elif corruption == "truncate":
            path.write_bytes(b"broken")
        elif corruption == "missing":
            path.unlink()
        else:
            (cache.directory / "index.json").write_text("not json")
        assert cache.lookup("item") is None
        assert cache.put("item", _payload())
        with cache.lookup("item") as lease:
            _assert_exact(lease.data, _payload())
        if corruption != "missing":
            assert cache.stats()["corruptions"] >= 1


def test_borrow_and_cache_contexts_release_after_exceptions(tmp_path: Path) -> None:
    cache = _cache(1 << 20, tmp_path, "exceptions")
    with pytest.raises(RuntimeError, match="cancel"):
        with cache:
            assert cache.put("item", _payload())
            lease = cache.lookup("item")
            with lease:
                raise RuntimeError("cancel")
    assert lease.data is None
    assert not cache.directory.exists()
    cache.close()


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_partial_writes_release_reservations(tmp_path: Path, monkeypatch, error_type) -> None:
    original = Path.write_bytes

    def fail_payload(path: Path, data: bytes) -> int:
        if path.name == "payload.pending":
            original(path, data[: len(data) // 2])
            raise error_type("interrupted write")
        return original(path, data)

    with _cache(1 << 20, tmp_path, "write-failure") as cache:
        monkeypatch.setattr(Path, "write_bytes", fail_payload)
        if error_type is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                cache.put("item", _payload())
        else:
            assert not cache.put("item", _payload())
        assert not list(cache.directory.glob("*.pending"))
        index = json.loads((cache.directory / "index.json").read_text())
        assert index["reserved_bytes"] == 0 and not index["entries"]
        assert cache.stats()["peak_reserved_bytes"] > 0


def test_last_client_cleans_only_its_namespace_and_live_peer_survives(tmp_path: Path) -> None:
    first = _cache(1 << 20, tmp_path, "../run")
    second = _cache(1 << 20, tmp_path, "../run")
    other = _cache(1 << 20, tmp_path, "other")
    try:
        assert first.put("item", _payload()) and other.put("other", _payload())
        first.close()
        with second.lookup("item") as lease:
            _assert_exact(lease.data, _payload())
        second.close()
        assert not first.directory.exists()
        assert other.directory.exists()
    finally:
        first.close()
        second.close()
        other.close()
    assert list((tmp_path / "relax-conditioning-v1").iterdir()) == [tmp_path / "relax-conditioning-v1/.lock"]


def test_spawned_writers_publish_atomically_with_bounded_storage(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue, start = context.Queue(), context.Event()
    capacity = 4096
    with _cache(capacity, tmp_path, "shared") as cache:
        workers = [
            context.Process(target=_parallel_writer, args=(str(tmp_path), capacity, n, start, queue)) for n in range(3)
        ]
        for worker in workers:
            worker.start()
        start.set()
        try:
            assert [queue.get(timeout=60) for _ in workers] == [None] * len(workers)
        finally:
            for worker in workers:
                worker.join(timeout=15)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
                assert worker.exitcode == 0
        assert cache.stats()["resident_bytes"] <= capacity
        assert not list(cache.directory.glob("*.pending"))


def test_other_process_lease_prevents_eviction_and_dead_writer_is_reaped(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready, release, queue = context.Event(), context.Event(), context.Queue()
    size = _entry_size(tmp_path, _payload())
    with _cache(size, tmp_path, "shared") as cache:
        assert cache.put("kept", _payload())
        worker = context.Process(target=_borrow_until_released, args=(str(tmp_path), size, ready, release, queue))
        worker.start()
        try:
            assert ready.wait(30)
            assert cache.stats()["leased_bytes"] == size
            assert not cache.put("replacement", _payload())
            release.set()
            assert queue.get(timeout=30) is None
        finally:
            release.set()
            worker.join(timeout=15)
            if worker.is_alive():
                worker.kill()
                worker.join()
        assert worker.exitcode == 0
        assert cache.put("replacement", _payload())
        abandoned = context.Process(target=_abandoned_write, args=(str(tmp_path), size))
        abandoned.start()
        abandoned.join(timeout=30)
        if abandoned.is_alive():
            abandoned.kill()
            abandoned.join()
        assert abandoned.exitcode == 0
        assert cache.stats()["bytes_reserved"] == 0
        assert not list(cache.directory.glob("*.pending"))
        assert len(list(cache.directory.glob("client-*"))) == 1
        with cache.lookup("replacement") as lease:
            _assert_exact(lease.data, _payload())


def test_corrupt_index_cannot_escape_namespace(tmp_path: Path) -> None:
    outside = tmp_path / "keep.safetensors"
    outside.write_bytes(b"unrelated")
    with _cache(4096, tmp_path, "run") as cache:
        index = {"format": 1, "max_bytes": 4096, "entries": {"../../keep": {"size": 9}}}
        (cache.directory / "index.json").write_text(json.dumps(index))
        assert cache.lookup(hashlib.sha256(b"key").hexdigest()) is None
        assert cache.stats()["corruptions"] >= 1
    assert outside.read_bytes() == b"unrelated"


def test_hot_lookup_does_not_recover_or_probe_other_payloads(tmp_path: Path, monkeypatch) -> None:
    with _cache(1 << 20, tmp_path, "hot-index") as cache:
        assert cache.put("target", _payload()) and cache.put("other", _payload())
        target = cache.directory / f"{hashlib.sha256(b'target').hexdigest()}.safetensors"
        original_stat = Path.stat
        checksum_before = cache.stats()["checksum_bytes"]

        def fail_scan(*args, **kwargs):
            raise AssertionError("A hot hit must not scan the namespace")

        def target_stat(path, *args, **kwargs):
            if path.parent == cache.directory and path.suffix == ".safetensors":
                assert path == target
            return original_stat(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "glob", fail_scan)
            patch.setattr(Path, "iterdir", fail_scan)
            patch.setattr(Path, "is_file", fail_scan)
            patch.setattr(Path, "stat", target_stat)
            for _ in range(2):
                with cache.lookup("target") as lease:
                    _assert_exact(lease.data, _payload())
        assert cache.stats()["checksum_bytes"] - checksum_before == 2 * target.stat().st_size


def test_light_stats_accounts_index_without_claiming_lease_or_disk_sizes(tmp_path: Path, monkeypatch) -> None:
    cache = _cache(1 << 20, tmp_path, "light-stats")
    try:
        assert cache.put("item", _payload())
        with cache.lookup("item"):
            detailed = cache.stats()
            assert detailed["storage_scope"] == "filesystem"
            assert detailed["leased_bytes"] == detailed["resident_bytes"] > 0

            def fail_scan(*args, **kwargs):
                raise AssertionError("Light statistics must only read the index")

            with monkeypatch.context() as patch:
                patch.setattr(Path, "glob", fail_scan)
                patch.setattr(Path, "iterdir", fail_scan)
                patch.setattr(Path, "stat", fail_scan)
                patch.setattr(cache, "_storage", fail_scan)
                light = cache.stats_light()
            assert light["storage_scope"] == "index_accounting"
            for field in ("entries", "resident_bytes", "bytes_reserved", "staging_reserved_bytes", "inflight_writes"):
                assert light[field] == detailed[field]
            assert all(light[field] is None for field in ("leased_bytes", "metadata_bytes", "total_bytes"))
        assert cache.stats()["leased_bytes"] == 0
    finally:
        cache.close()
    closed = cache.stats_light()
    assert closed["storage_scope"] == "unavailable"
    assert all(closed[field] is None for field in cache._empty_storage())


@pytest.mark.parametrize("bad_size", [-1, "invalid", True])
def test_light_stats_repairs_corrupt_non_target_accounting(tmp_path: Path, bad_size) -> None:
    with _cache(1 << 20, tmp_path, "bad-accounting") as cache:
        assert cache.put("target", _payload()) and cache.put("other", _payload())
        path = cache.directory / "index.json"
        index = json.loads(path.read_text())
        index["entries"][hashlib.sha256(b"other").hexdigest()]["size"] = bad_size
        path.write_text(json.dumps(index))
        with cache.lookup("target") as lease:
            _assert_exact(lease.data, _payload())  # A hit validates only its own entry.
        stats = cache.stats_light()
        assert stats["storage_scope"] == "filesystem"
        assert stats["resident_bytes"] == stats["entries"] == 0
        assert stats["corruptions"] > 0 and stats["errors"] > 0
        assert cache.put("target", _payload())


@pytest.mark.parametrize("field,value", [("sha256", "bad"), ("tensor_bytes", -1), ("size", True), ("access", -1)])
def test_hot_lookup_rejects_invalid_target_metadata_and_rebuilds(tmp_path: Path, field: str, value) -> None:
    with _cache(1 << 20, tmp_path, "bad-target") as cache:
        assert cache.put("target", _payload())
        path = cache.directory / "index.json"
        index = json.loads(path.read_text())
        index["entries"][hashlib.sha256(b"target").hexdigest()][field] = value
        path.write_text(json.dumps(index))
        assert cache.lookup("target") is None
        assert cache.stats()["resident_bytes"] == 0
        assert cache.put("target", _payload())
        with cache.lookup("target") as lease:
            _assert_exact(lease.data, _payload())


def test_new_run_reclaims_dead_namespaces_but_preserves_live_and_borrowed(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    with _cache(1 << 20, tmp_path, "live-run") as live:
        assert live.put("kept", _payload())
        worker = context.Process(target=_abandoned_namespace, args=(str(tmp_path),))
        worker.start()
        worker.join(timeout=30)
        if worker.is_alive():
            worker.kill()
            worker.join()
        assert worker.exitcode == 0
        dead = live.directory.parent / build_key({"namespace": "dead-run"})
        assert dead.exists()
        payload = next(dead.glob("*.safetensors"))
        foreign = live.directory.parent / ("f" * 64)
        foreign.mkdir()
        (foreign / "keep").write_text("not owned by the conditioning cache")
        with payload.open("rb") as borrowed:
            fcntl.flock(borrowed.fileno(), fcntl.LOCK_SH)
            with _cache(1 << 20, tmp_path, "new-run-before-release"):
                assert dead.exists()
        with _cache(1 << 20, tmp_path, "new-run-after-release") as fresh:
            assert not dead.exists()
            assert fresh.stats()["reclaimed_namespaces"] == 1
            assert live.directory.exists()
            with live.lookup("kept") as lease:
                _assert_exact(lease.data, _payload())
            assert (foreign / "keep").read_text() == "not owned by the conditioning cache"


def test_packed_copy_is_one_transfer_and_owns_aligned_nested_storage(tmp_path: Path, monkeypatch) -> None:
    shared = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    payload = {
        "alias": [shared, shared.view(-1), shared[:, ::2]],
        "nested": (torch.empty((0, 3), dtype=torch.float16), None),
        "index": torch.arange(4, dtype=torch.int64),
        "mask": torch.ones(16, dtype=torch.bool),
        "none": None,
    }
    original, calls = torch.Tensor.to, []

    def record_copy(value, *args, **kwargs):
        calls.append((value.dtype, value.numel()))
        return original(value, *args, **kwargs)

    with _cache(1 << 20, tmp_path, "packed") as cache:
        assert cache.put("item", payload)
        with cache.lookup("item") as lease:
            size = lease._file_size - len(lease._file_header)
            with monkeypatch.context() as patch:
                patch.setattr(torch.Tensor, "to", record_copy)
                moved, transferred = lease.copy_to("cpu")
            assert calls == [(torch.uint8, size)] and transferred == size
            _assert_exact(moved, payload)
            assert (
                moved["alias"][0].untyped_storage().data_ptr() != lease.data["alias"][0].untyped_storage().data_ptr()
            )
            assert len(lease._transfer_sources) == 1
            assert all(offset % 16 == 0 for offset, length in lease._packed_layout().values() if length)
            with pytest.raises(RuntimeError, match="only one"):
                lease.copy_to("cpu")
        assert not lease._transfer_sources and not lease._tensors
    _assert_exact(moved, payload)  # Remains valid after the namespace and CPU maps are gone.


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_packed_copy_preserves_dense_dtype_bytes(tmp_path: Path, dtype) -> None:
    payload = {"value": torch.tensor([0, 1], dtype=dtype)}
    with _cache(1 << 20, tmp_path, "dtype") as cache:
        assert cache.put("item", payload)
        with cache.lookup("item") as lease:
            moved, size = lease.copy_to("cpu")
            assert size == payload["value"].numel() * payload["value"].element_size()
            _assert_exact(moved, payload)


def test_mixed_unaligned_layout_falls_back_with_exact_owned_tensors(tmp_path: Path) -> None:
    shared = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    payload = {
        "alias": [shared, shared.view(-1), shared[:, ::2]],
        "nested": (torch.tensor(1.25, dtype=torch.float64), torch.empty((0, 3), dtype=torch.float16)),
        "index": torch.arange(3, dtype=torch.int64),
        "mask": torch.tensor([True, False]),
        "none": None,
    }
    with _cache(1 << 20, tmp_path, "mixed-layout") as cache:
        assert cache.put("item", payload)
        with cache.lookup("item") as lease:
            assert lease._packed_layout() is None
            moved, size = lease.copy_to("cpu")
            assert size == sum(tensor.numel() * tensor.element_size() for tensor in lease._tensors.values())
            _assert_exact(moved, payload)
            assert (
                moved["alias"][0].untyped_storage().data_ptr() != lease.data["alias"][0].untyped_storage().data_ptr()
            )
        assert not lease._transfer_sources
    _assert_exact(moved, payload)


def test_copy_fallback_retains_sources_and_returns_real_tensor_bytes(tmp_path: Path, monkeypatch) -> None:
    with _cache(1 << 20, tmp_path, "fallback") as cache:
        assert cache.put("item", _payload())
        with cache.lookup("item") as lease:
            monkeypatch.setattr(lease, "_packed_layout", lambda: None)
            moved, size = lease.copy_to("cpu")
            assert size == sum(tensor.numel() * tensor.element_size() for tensor in lease._tensors.values())
            assert len(lease._transfer_sources) == len(lease._tensors)
            _assert_exact(moved, _payload())
            assert moved["prefix"][0].data_ptr() != lease.data["prefix"][0].data_ptr()
        assert not lease._transfer_sources
    _assert_exact(moved, _payload())


@pytest.mark.parametrize("field", ["shape", "dtype", "offsets", "size"])
def test_packed_layout_rejects_bad_tensor_headers(tmp_path: Path, field: str) -> None:
    with _cache(1 << 20, tmp_path, "bad-layout") as cache:
        assert cache.put("item", {"value": torch.arange(8, dtype=torch.int64)})
        with cache.lookup("item") as lease:
            header = json.loads(lease._file_header[8:])
            item = header["t0"]
            if field == "shape":
                item["shape"] = [9]
            elif field == "dtype":
                item["dtype"] = "F64"
            elif field == "offsets":
                item["data_offsets"] = [1, 65]
            else:
                item["data_offsets"] = [0, 63]
            encoded = json.dumps(header, separators=(",", ":")).encode()
            length = len(lease._file_header) - 8
            assert len(encoded) <= length
            changed = struct.pack("<Q", length) + encoded.ljust(length, b" ")
            with Path(lease._descriptor.name).open("r+b") as destination:
                destination.write(changed)
            lease._file_header = changed  # Exercise checks after the reader-level validation boundary.
            with pytest.raises(ValueError):
                lease.copy_to("cpu")
            assert not lease._transfer_sources


def test_copy_detects_header_changes_after_lookup(tmp_path: Path) -> None:
    with _cache(1 << 20, tmp_path, "changed-header") as cache:
        assert cache.put("item", _payload())
        with cache.lookup("item") as lease:
            with Path(lease._descriptor.name).open("r+b") as destination:
                destination.seek(8)
                destination.write(b"[")
            with pytest.raises(ValueError, match="changed"):
                lease.copy_to("cpu")
        assert cache.lookup("item") is None  # Every new lookup still verifies the full SHA256.


def test_failed_transfer_keeps_source_until_lease_close(tmp_path: Path, monkeypatch) -> None:
    def fail_copy(*args, **kwargs):
        raise RuntimeError("copy failed")

    with _cache(1 << 20, tmp_path, "failed-copy") as cache:
        assert cache.put("item", _payload())
        with cache.lookup("item") as lease:
            monkeypatch.setattr(torch.Tensor, "to", fail_copy)
            with pytest.raises(RuntimeError, match="copy failed"):
                lease.copy_to("cpu")
            assert len(lease._transfer_sources) == 1
        assert not lease._transfer_sources
