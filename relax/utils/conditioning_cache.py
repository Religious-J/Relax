# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded, node-local storage for deterministic frozen conditioning tensors.

Construct one client per worker after process startup. Workers sharing a run
namespace map the same immutable safetensors files on CPU. Lease.data is borrowed:
finish device copies before closing it and do not retain its CPU tensors later.
Put snapshots its input; callers must not mutate that input during the call.

The payload budget includes leased entries and temporary file reservations.
A separate staging budget bounds the single writer's CPU tensor/serialization
buffers before copying begins. JSON/client metadata is bounded by entries/live
workers and reported separately. The final client removes its namespace. One
stable lock remains per directory, avoiding split locks on namespace recreation.
After hard process termination, the next cache initialization reclaims abandoned
namespaces once all client and payload locks are free; no cleanup daemon runs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_FORMAT = 1
_OWNER = "relax-conditioning-v1"
_PACKED_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "C64": "complex64",
    "C128": "complex128",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
    "F8_E8M0": "float8_e8m0fnu",
}


def build_key(dependencies: Mapping[str, Any]) -> str:
    """Hash explicit content/version/preprocessing/dtype JSON dependencies.

    Paths should be represented by content fingerprints, not filenames. Model
    adapters own the completeness of this dependency declaration.
    """

    def validate(value: Any) -> None:
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Conditioning dependencies require string keys")
            for item in value.values():
                validate(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                validate(item)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError("Conditioning dependencies must contain JSON values")

    validate(dependencies)
    encoded = json.dumps(dependencies, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_file(path: str | Path) -> str:
    """Hash file contents in bounded chunks; call outside the hot path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: Any, tensors: dict[str, Any]) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Conditioning cache requires dense, unquantized tensors")
        name = f"t{len(tensors)}"
        tensors[name] = value
        return ["tensor", name]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Conditioning dictionaries require string keys")
        return ["dict", [[key, _flatten(item, tensors)] for key, item in value.items()]]
    if isinstance(value, (tuple, list)):
        return ["tuple" if isinstance(value, tuple) else "list", [_flatten(item, tensors) for item in value]]
    if value is None or isinstance(value, (str, int, float, bool)):
        return ["value", value]
    raise TypeError(f"Unsupported conditioning value: {type(value).__name__}")


def _restore(schema: Any, reader: Any, collected: dict[str, Any] | None = None) -> Any:
    kind, value = schema
    if kind == "tensor":
        tensor = reader[value] if isinstance(reader, dict) else reader.get_tensor(value)
        if collected is not None:
            collected[value] = tensor
        return tensor
    if kind == "dict":
        return {key: _restore(item, reader, collected) for key, item in value}
    if kind in ("list", "tuple"):
        items = [_restore(item, reader, collected) for item in value]
        return tuple(items) if kind == "tuple" else items
    if kind == "value":
        return value
    raise ValueError("Unknown conditioning tree node")


def _serialization_bound(tensors: dict[str, Any], schema: str, tensor_bytes: int) -> int:
    # Overestimate dtype strings and every offset's decimal length. Metadata is
    # already ASCII JSON, so Python/Rust JSON escaping cannot increase this.
    header: dict[str, Any] = {"__metadata__": {"format": str(_FORMAT), "tree": schema}}
    for name, value in tensors.items():
        header[name] = {"dtype": "X" * 32, "shape": list(value.shape), "data_offsets": [tensor_bytes, tensor_bytes]}
    header_bytes = len(json.dumps(header, separators=(",", ":"), allow_nan=False).encode())
    return tensor_bytes + 8 + (header_bytes + 7) // 8 * 8


def _valid_entry(key: str, entry: Any) -> bool:
    return (
        isinstance(key, str)
        and len(key) == 64
        and all(char in "0123456789abcdef" for char in key)
        and isinstance(entry, dict)
        and all(type(entry.get(field)) is int and entry[field] >= 0 for field in ("size", "access"))
        and isinstance(entry.get("sha256"), str)
        and (
            entry.get("invalid") is True
            or (
                type(entry.get("tensor_bytes")) is int
                and 0 <= entry["tensor_bytes"] <= entry["size"]
                and len(entry["sha256"]) == 64
                and all(char in "0123456789abcdef" for char in entry["sha256"])
            )
        )
    )


class ConditioningLease:
    """A borrowed CPU tensor tree protected from eviction until close()."""

    def __init__(
        self,
        cache: ConditioningCache,
        descriptor: Any,
        reader: Any,
        data: Any,
        schema: Any,
        tensors: dict[str, Any],
        metadata: dict[str, str],
    ) -> None:
        self._cache = weakref.ref(cache)
        self._descriptor, self._reader = descriptor, reader
        self.data = data
        self._schema, self._tensors, self._metadata = schema, tensors, metadata
        self._file_size, self._file_header = self._read_header()
        self._copy_started = False
        self._transfer_sources: list[Any] = []

    def _read_header(self) -> tuple[int, bytes]:
        descriptor = self._descriptor.fileno()
        size = os.fstat(descriptor).st_size
        prefix = os.pread(descriptor, 8, 0)
        if len(prefix) != 8:
            raise ValueError("Truncated conditioning header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= min(size - 8, 100 * 1024**2):
            raise ValueError("Invalid conditioning header size")
        header = os.pread(descriptor, length, 8)
        if len(header) != length:
            raise ValueError("Truncated conditioning metadata")
        return size, prefix + header

    def _packed_layout(self) -> dict[str, tuple[int, int]] | None:
        import torch

        if self._read_header() != (self._file_size, self._file_header):
            raise ValueError("Conditioning header changed during its lease")
        header = json.loads(self._file_header[8:])
        if not isinstance(header, dict) or header.get("__metadata__") != self._metadata:
            raise ValueError("Conditioning header metadata differs from the validated reader")
        if set(header) - {"__metadata__"} != set(self._tensors):
            raise ValueError("Conditioning header fields differ from its tensor tree")
        data_start, intervals, layout, compatible = len(self._file_header), [], {}, True
        for name, tensor in self._tensors.items():
            item = header[name]
            if not isinstance(item, dict):
                raise ValueError("Invalid conditioning tensor descriptor")
            shape, offsets = item.get("shape"), item.get("data_offsets")
            if (
                not isinstance(shape, list)
                or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
                or tuple(shape) != tuple(tensor.shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(type(offset) is int for offset in offsets)
            ):
                raise ValueError("Invalid conditioning tensor shape or offsets")
            begin, end = offsets
            if not 0 <= begin <= end <= self._file_size - data_start:
                raise ValueError("Conditioning tensor exceeds its payload")
            dtype = getattr(torch, _PACKED_DTYPES.get(item.get("dtype"), ""), None)
            if dtype is None:
                if self._reader.get_slice(name).get_dtype() != item.get("dtype"):
                    raise ValueError("Conditioning tensor dtype changed")
                compatible = False  # Valid future/sub-byte dtypes retain the ordinary copy path.
            elif dtype != tensor.dtype or end - begin != math.prod(shape) * tensor.element_size():
                raise ValueError("Conditioning tensor dtype or byte size differs from its validated view")
            if begin != end:
                # The destination buffer is allocator-aligned. Keep every view
                # aligned for vectorized consumers as well as dtype views.
                compatible &= begin % 16 == 0 and begin % tensor.element_size() == 0
                intervals.append((begin, end))
            layout[name] = (begin if begin != end else 0, end - begin)
        previous = 0
        for begin, end in sorted(intervals):
            if begin != previous:
                raise ValueError("Conditioning payload contains gaps or overlapping tensors")
            previous = end
        if previous != self._file_size - data_start:
            raise ValueError("Conditioning payload contains unreferenced bytes")
        return layout if compatible else None

    def copy_to(self, device: Any) -> tuple[Any, int]:
        """Copy once, retaining CPU DMA sources until the caller closes its
        lease.

        Dense aligned files use one byte-buffer transfer and typed views with
        unchanged bits/dtypes. Only tensor data is transferred; the validated
        safetensors header stays on the CPU. Valid unsupported layouts use
        independent tensor copies. CPU targets also receive new owned storage.
        The caller must finish asynchronous copies before close(), including
        when this method raises after enqueue.
        """
        import torch

        if self._descriptor is None:
            raise RuntimeError("Conditioning lease is closed")
        if self._copy_started:
            raise RuntimeError("A conditioning lease permits only one copy_to call")
        self._copy_started = True
        layout = self._packed_layout()
        if not self._tensors:
            return _restore(self._schema, {}), 0
        if layout is None:
            self._transfer_sources = list(self._tensors.values())
            moved = {
                name: value.to(device=device, copy=True, non_blocking=True) for name, value in self._tensors.items()
            }
            return _restore(self._schema, moved), sum(value.numel() * value.element_size() for value in moved.values())
        raw = torch.from_file(
            self._descriptor.name, shared=False, size=self._file_size, dtype=torch.uint8, device="cpu"
        )
        data = raw.narrow(0, len(self._file_header), self._file_size - len(self._file_header))
        self._transfer_sources = [data]
        packed = data.to(device=device, copy=True, non_blocking=True)
        moved = {
            name: packed.narrow(0, offset, length).view(self._tensors[name].dtype).reshape(self._tensors[name].shape)
            for name, (offset, length) in layout.items()
        }
        return _restore(self._schema, moved), data.numel()

    def __enter__(self) -> ConditioningLease:
        if self._descriptor is None:
            raise RuntimeError("Conditioning lease is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor is None:
            return
        self.data = None
        self._tensors.clear()
        self._transfer_sources.clear()
        self._schema, self._metadata, self._file_header = None, {}, b""
        try:
            self._reader.__exit__(None, None, None)
        finally:
            self._reader = None
            self._descriptor.close()  # process exit also releases this flock
            self._descriptor = None
            cache = self._cache()
            if cache is not None:
                cache._leases.discard(self)


class ConditioningCache:
    """Run-scoped, process-safe LRU cache; zero capacity disables all I/O.

    File/serialization failures are observable misses, allowing online
    encoding. Hashes verify the entire payload before each borrow. Stats
    operation counters are local; storage counters describe the namespace and
    must not be summed across ranks on the same node. No pickle or model-
    specific imports occur.
    """

    def __init__(
        self, max_bytes: int, directory: str | Path, namespace: str, *, max_staging_bytes: int | None = None
    ) -> None:
        if max_bytes < 0 or (max_staging_bytes is not None and max_staging_bytes < 0):
            raise ValueError("Cache capacities must be nonnegative")
        self.max_bytes = int(max_bytes)
        self.max_staging_bytes = self.max_bytes if max_staging_bytes is None else int(max_staging_bytes)
        self.directory = Path(directory) / "relax-conditioning-v1" / build_key({"namespace": namespace})
        self._mutex = threading.RLock()
        self._lock_file: Any = None
        self._client_file: Any = None
        self._client_path = self.directory / f"client-{uuid.uuid4().hex}"
        self._leases: set[ConditioningLease] = set()
        self._closed = False
        self._counters: dict[str, int | float] = dict.fromkeys(
            (
                "lookups",
                "hits",
                "misses",
                "writes",
                "evictions",
                "corruptions",
                "errors",
                "bypasses",
                "read_bytes",
                "write_bytes",
                "reused_tensor_bytes",
                "load_seconds",
                "save_seconds",
                "peak_reserved_bytes",
                "peak_temporary_host_bytes",
                "peak_staging_reserved_bytes",
                "checksum_bytes",
                "reclaimed_namespaces",
            ),
            0,
        )
        self._last_storage = self._empty_storage()
        if not max_bytes:
            return
        try:
            self.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._lock_file = (self.directory.parent / ".lock").open("a+b")
            with self._locked():
                self._scavenge_namespaces()
                self.directory.mkdir(exist_ok=True, mode=0o700)
                if not (self.directory / "owner.json").exists():
                    owner_pending = self.directory / "owner.pending"
                    owner_pending.write_text(json.dumps({"owner": _OWNER, "namespace": self.directory.name}))
                    os.replace(owner_pending, self.directory / "owner.json")
                self._reap_clients()
                index = self._index()
                if index["max_bytes"] != self.max_bytes or index["max_staging_bytes"] != self.max_staging_bytes:
                    raise ValueError("Workers sharing a conditioning namespace require identical capacity")
                self._client_file = self._client_path.open("x+b")
                fcntl.flock(self._client_file.fileno(), fcntl.LOCK_EX)
                self._save_index(index)
        except Exception as error:
            self._error("initialize", error)
            self.close()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._mutex:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _empty_storage() -> dict[str, int]:
        return dict.fromkeys(
            (
                "entries",
                "resident_bytes",
                "leased_bytes",
                "bytes_reserved",
                "staging_reserved_bytes",
                "inflight_writes",
                "metadata_bytes",
                "total_bytes",
            ),
            0,
        )

    def _error(self, operation: str, error: Exception) -> None:
        self._counters["errors"] += 1
        logger.warning("Conditioning cache %s failed; using online encoding: %s", operation, error)

    def _reap_clients(self, directory: Path | None = None) -> int:
        live = 0
        for path in (directory or self.directory).glob("client-*"):
            with path.open("rb") as descriptor:
                try:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    live += 1
                else:
                    path.unlink(missing_ok=True)
        return live

    def _scavenge_namespaces(self) -> None:
        """Reclaim abandoned runs while the stable directory lock is held.

        Registration uses this same lock, so no new client can join between
        checking client/payload locks and removal. An ownership marker and a
        hash-only directory name prevent adoption of unrelated directories.
        Payload checks additionally protect outstanding borrows even if a
        caller's client marker has been removed out of band.
        """
        for directory in self.directory.parent.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            if len(directory.name) != 64 or any(char not in "0123456789abcdef" for char in directory.name):
                continue
            try:
                owner = json.loads((directory / "owner.json").read_text())
                if owner != {"owner": _OWNER, "namespace": directory.name} or self._reap_clients(directory):
                    continue
                for path in directory.glob("*.safetensors"):
                    if path.is_symlink():
                        break
                    with path.open("rb") as descriptor:
                        try:
                            fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            break
                else:
                    shutil.rmtree(directory)
                    self._counters["reclaimed_namespaces"] += 1
            except FileNotFoundError:
                continue
            except (OSError, ValueError, TypeError) as error:
                self._error("scavenge", error)

    def _index(self, *, recover: bool = True) -> dict[str, Any]:
        index = {
            "format": _FORMAT,
            "max_bytes": self.max_bytes,
            "max_staging_bytes": self.max_staging_bytes,
            "entries": {},
            "reserved_bytes": 0,
            "staging_reserved_bytes": 0,
        }
        try:
            loaded = json.loads((self.directory / "index.json").read_text())
            if (
                not isinstance(loaded, dict)
                or loaded.get("format") != _FORMAT
                or not isinstance(loaded.get("entries"), dict)
                or not all(
                    type(loaded.get(field)) is int and loaded[field] >= 0
                    for field in ("max_bytes", "max_staging_bytes", "reserved_bytes", "staging_reserved_bytes")
                )
                or (recover and not all(_valid_entry(key, entry) for key, entry in loaded["entries"].items()))
            ):
                raise ValueError("Invalid conditioning cache index")
            index = loaded
        except FileNotFoundError:
            if not recover:
                raise
        except (ValueError, KeyError, TypeError) as error:
            self._counters["corruptions"] += 1
            if not recover:
                raise ValueError("Invalid conditioning cache index") from error
        if not recover:
            return index
        # A writer holds this lock throughout publication. Pending files seen
        # by another lock owner therefore belong to an exited writer.
        for name in ("payload.pending", "index.pending"):
            (self.directory / name).unlink(missing_ok=True)
        index["reserved_bytes"] = 0
        index["staging_reserved_bytes"] = 0
        entries = index["entries"]
        for key in list(entries):
            path = self.directory / f"{key}.safetensors"
            try:
                actual = path.lstat()
            except FileNotFoundError:
                del entries[key]
                continue
            if not stat.S_ISREG(actual.st_mode):
                path.unlink(missing_ok=True)
                del entries[key]
                self._counters["corruptions"] += 1
            elif actual.st_size != entries[key]["size"]:
                # A damaged index must never undercharge a still-borrowed
                # payload. Invalid leased files remain billed at actual size.
                entries[key].update(size=actual.st_size, invalid=True)
                self._counters["corruptions"] += 1
                self._remove(key, index)
        for path in self.directory.glob("*.safetensors"):
            if path.stem not in entries:
                actual = path.lstat()
                if not stat.S_ISREG(actual.st_mode):
                    path.unlink(missing_ok=True)
                    self._counters["corruptions"] += 1
                    continue
                entries[path.stem] = {"size": actual.st_size, "access": 0, "sha256": "", "invalid": True}
                self._remove(path.stem, index)
        return index

    def _save_index(self, index: dict[str, Any]) -> None:
        temporary = self.directory / "index.pending"
        try:
            temporary.write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))
            os.replace(temporary, self.directory / "index.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _remove(self, key: str, index: dict[str, Any]) -> bool:
        path = self.directory / f"{key}.safetensors"
        try:
            with path.open("rb") as descriptor:
                try:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                path.unlink()
        except FileNotFoundError:
            pass
        index["entries"].pop(key, None)
        return True

    def lookup(self, key: str) -> ConditioningLease | None:
        # Protect this client's lifetime while releasing the shared namespace
        # lock during hashing/mapping. Other workers can read concurrently.
        with self._mutex:
            return self._lookup(key)

    def _lookup(self, key: str) -> ConditioningLease | None:
        self._counters["lookups"] += 1
        started, hit = time.perf_counter(), False
        descriptor = reader = None
        digest, entry = "", None
        try:
            if self._closed or self._client_file is None:
                return None
            digest = hashlib.sha256(key.encode()).hexdigest()
            with self._locked():
                index = self._index(recover=False)
                entry = index["entries"].get(digest)
                if entry is None:
                    return None
                if not _valid_entry(digest, entry):
                    raise ValueError("Invalid conditioning cache entry")
                path = self.directory / f"{digest}.safetensors"
                if entry.get("invalid") or path.stat().st_size != entry["size"]:
                    raise ValueError("Invalid conditioning payload size")
                descriptor = path.open("rb")
                fcntl.flock(descriptor.fileno(), fcntl.LOCK_SH)
            self._counters["checksum_bytes"] += entry["size"]
            if fingerprint_file(path) != entry["sha256"]:
                raise ValueError("Conditioning payload checksum mismatch")
            from safetensors import safe_open

            reader = safe_open(str(path), framework="pt", device="cpu")
            reader.__enter__()
            metadata = reader.metadata() or {}
            if metadata.get("format") != str(_FORMAT):
                raise ValueError("Invalid conditioning payload format")
            schema, tensors = json.loads(metadata["tree"]), {}
            data = _restore(schema, reader, tensors)
            with self._locked():
                index = self._index(recover=False)
                current = index["entries"].get(digest)
                if not _valid_entry(digest, current) or current.get("invalid") or current["sha256"] != entry["sha256"]:
                    raise ValueError("Conditioning payload invalidated during read")
                current["access"] = time.time_ns()
                self._save_index(index)
            lease = ConditioningLease(self, descriptor, reader, data, schema, tensors, metadata)
            self._leases.add(lease)
            descriptor = reader = None
            self._counters["hits"] += 1
            self._counters["read_bytes"] += entry["size"]
            self._counters["reused_tensor_bytes"] += entry["tensor_bytes"]
            hit = True
            return lease
        except Exception as error:
            if reader is not None:
                reader.__exit__(None, None, None)
                reader = None
            try:
                with self._locked():
                    index = self._index()
                    if entry is not None:
                        if digest in index["entries"]:
                            index["entries"][digest]["invalid"] = True
                    if descriptor is not None:
                        descriptor.close()
                        descriptor = None
                    if entry is not None:
                        self._remove(digest, index)
                    self._save_index(index)
            except Exception as cleanup_error:
                self._error("invalidate", cleanup_error)
            if entry is not None:
                self._counters["corruptions"] += 1
            self._error("lookup", error)
        finally:
            if reader is not None:
                reader.__exit__(None, None, None)
            if descriptor is not None:
                descriptor.close()
            self._counters["load_seconds"] += time.perf_counter() - started
            if not hit:
                self._counters["misses"] += 1
        return None

    def put(self, key: str, tree: Any) -> bool:
        started = time.perf_counter()
        try:
            if self._closed or self._client_file is None:
                self._counters["bypasses"] += 1
                return False
            tensors: dict[str, Any] = {}
            schema = json.dumps(_flatten(tree, tensors), separators=(",", ":"), allow_nan=False)
            tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            if tensor_bytes + len(schema.encode()) > self.max_bytes:
                self._counters["bypasses"] += 1
                return False
            digest = hashlib.sha256(key.encode()).hexdigest()
            with self._locked():
                index = self._index()
                if digest in index["entries"]:
                    if not index["entries"][digest].get("invalid"):
                        return False
                    if not self._remove(digest, index):
                        self._counters["bypasses"] += 1
                        return False
                staging = 3 * tensor_bytes + 2 * _serialization_bound(tensors, schema, tensor_bytes)
                if staging > self.max_staging_bytes:
                    self._counters["bypasses"] += 1
                    return False
                index["staging_reserved_bytes"] = staging
                self._save_index(index)
                self._counters["peak_staging_reserved_bytes"] = max(
                    self._counters["peak_staging_reserved_bytes"], staging
                )
                try:
                    from safetensors.torch import save

                    cpu_tensors, storage_ids = {}, set()
                    for name, tensor in tensors.items():
                        value = tensor.detach().to(device="cpu").contiguous()
                        storage_id = value.untyped_storage().data_ptr()
                        if storage_id in storage_ids:
                            value = value.clone()
                        storage_ids.add(storage_id)
                        cpu_tensors[name] = value
                    payload = save(cpu_tensors, metadata={"format": str(_FORMAT), "tree": schema})
                    size = len(payload)
                    # Only one writer serializes per node. This upper bound includes
                    # its CPU staging tensors and serialized bytes, not caller-owned
                    # inputs or consumer-owned GPU copies.
                    self._counters["peak_temporary_host_bytes"] = max(
                        self._counters["peak_temporary_host_bytes"], tensor_bytes + size
                    )
                    if size > self.max_bytes:
                        self._counters["bypasses"] += 1
                        return False
                    required = sum(entry["size"] for entry in index["entries"].values()) + size
                    for victim in sorted(index["entries"], key=lambda item: index["entries"][item]["access"]):
                        if required <= self.max_bytes:
                            break
                        victim_size = index["entries"][victim]["size"]
                        if self._remove(victim, index):
                            required -= victim_size
                            self._counters["evictions"] += 1
                    if required > self.max_bytes:
                        self._save_index(index)
                        self._counters["bypasses"] += 1
                        return False
                    index["reserved_bytes"] = size
                    self._save_index(index)
                    self._counters["peak_reserved_bytes"] = max(self._counters["peak_reserved_bytes"], size)
                    temporary = self.directory / "payload.pending"
                    try:
                        temporary.write_bytes(payload)
                        os.replace(temporary, self.directory / f"{digest}.safetensors")
                        index["entries"][digest] = {
                            "size": size,
                            "tensor_bytes": tensor_bytes,
                            "access": time.time_ns(),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    finally:
                        temporary.unlink(missing_ok=True)
                        index["reserved_bytes"] = 0
                        self._save_index(index)
                    self._counters["writes"] += 1
                    self._counters["write_bytes"] += size
                    return True
                finally:
                    index["staging_reserved_bytes"] = 0
                    self._save_index(index)
        except Exception as error:
            self._error("write", error)
            return False
        finally:
            self._counters["save_seconds"] += time.perf_counter() - started

    def _storage(self, index: dict[str, Any]) -> dict[str, int]:
        result = self._empty_storage()
        result["entries"] = len(index["entries"])
        result["resident_bytes"] = sum(entry["size"] for entry in index["entries"].values())
        result["bytes_reserved"] = index["reserved_bytes"]
        result["staging_reserved_bytes"] = index["staging_reserved_bytes"]
        result["inflight_writes"] = int(index["staging_reserved_bytes"] > 0)
        for key, entry in index["entries"].items():
            with (self.directory / f"{key}.safetensors").open("rb") as descriptor:
                try:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result["leased_bytes"] += entry["size"]
        result["metadata_bytes"] = sum(
            path.stat().st_size
            for path in self.directory.iterdir()
            if path.suffix != ".safetensors" and path.name != "payload.pending"
        )
        result["total_bytes"] = result["resident_bytes"] + result["bytes_reserved"] + result["metadata_bytes"]
        return result

    def stats(self) -> dict[str, int | float | str]:
        scope = "last_detailed_snapshot"
        if not self._closed and self._client_file is not None:
            try:
                with self._locked():
                    self._reap_clients()
                    index = self._index()
                    self._save_index(index)
                    self._last_storage = self._storage(index)
                    scope = "filesystem"
            except Exception as error:
                self._error("stats", error)
        return {
            **self._counters,
            **self._last_storage,
            "storage_scope": scope,
            "max_bytes": self.max_bytes,
            "max_staging_bytes": self.max_staging_bytes,
            "max_payload_and_staging_bytes": self.max_bytes + self.max_staging_bytes,
        }

    def stats_light(self) -> dict[str, int | float | str | None]:
        """Read index accounting without probing payloads or recovering files.

        Missing/corrupt files can make accounted bytes differ from actual disk
        use until detailed stats/recovery. Lease and metadata sizes require a
        filesystem scan and are explicitly unknown here. Closed clients also
        report unknown accounting rather than a stale snapshot.
        """
        storage: dict[str, int | None] = dict.fromkeys(self._empty_storage(), None)
        scope = "unavailable"
        if not self._closed and self._client_file is not None:
            try:
                with self._locked():
                    index = self._index(recover=False)
                    if not all(
                        isinstance(entry, dict) and type(entry.get("size")) is int and entry["size"] >= 0
                        for entry in index["entries"].values()
                    ):
                        raise ValueError("Invalid conditioning cache accounting")
                    storage.update(
                        entries=len(index["entries"]),
                        resident_bytes=sum(entry["size"] for entry in index["entries"].values()),
                        bytes_reserved=index["reserved_bytes"],
                        staging_reserved_bytes=index["staging_reserved_bytes"],
                        inflight_writes=int(index["staging_reserved_bytes"] > 0),
                    )
                    scope = "index_accounting"
            except Exception as error:
                self._error("light stats", error)
                return self.stats()  # Detailed recovery validates every entry before summing.
        return {
            **self._counters,
            **storage,
            "storage_scope": scope,
            "max_bytes": self.max_bytes,
            "max_staging_bytes": self.max_staging_bytes,
            "max_payload_and_staging_bytes": self.max_bytes + self.max_staging_bytes,
        }

    def invalidate(self, key: str) -> bool:
        """Forget a bad model-level schema without disturbing unrelated
        entries.

        Returns False while any worker still borrows this entry; the caller
        should release its own lease before invalidating and recomputing.
        """
        if self._closed or self._client_file is None:
            return True
        try:
            with self._locked():
                index = self._index()
                digest = hashlib.sha256(key.encode()).hexdigest()
                if digest in index["entries"]:
                    index["entries"][digest]["invalid"] = True
                removed = self._remove(digest, index)
                self._save_index(index)
                return removed
        except Exception as error:
            self._error("invalidate", error)
            return False

    def clear(self) -> bool:
        if self._closed or self._client_file is None:
            return True
        try:
            with self._locked():
                index = self._index()
                for key in list(index["entries"]):
                    self._remove(key, index)
                self._save_index(index)
                return not index["entries"]
        except Exception as error:
            self._error("clear", error)
            return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._lock_file is not None:
                with self._locked():
                    for lease in list(self._leases):
                        lease.close()
                    if self._client_file is not None:
                        self._client_file.close()
                        self._client_file = None
                        self._client_path.unlink(missing_ok=True)
                        if not self._reap_clients():
                            shutil.rmtree(self.directory)
                            self._last_storage = self._empty_storage()
        except Exception as error:
            self._error("close", error)
        finally:
            self._closed = True
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

    def __enter__(self) -> ConditioningCache:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
