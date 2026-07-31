# RFC: FLA Chunkwise Context Parallel for GDN

- **Task**: 32 — FLA Chunkwise CP for GDN
- **Status**: Proposed
- **Branch**: `task32`
- **Date**: 2026-07-31
- **Related**: [redai-infra/Relax#172](https://github.com/redai-infra/Relax/issues/172)

---

## 1. Summary

When the GDN head count does not divide `TP × CP`, Relax currently falls back to gathering
the full projected sequence on every CP rank and running a redundant full-sequence GDN scan.
This is correct, but duplicates GDN compute and activations by `cp_size` and relies on
`--recompute-granularity full` to control memory use.

This RFC adopts **FLA chunkwise CP (KCP)**, available in FLA 0.4.2. Each rank keeps and scans
only its contiguous sequence partition. Cross-rank recurrence is resolved by all-gathering
compact state summaries `(h_ext, M)` and folding them locally, rather than passing state
sequentially from rank to rank.

The implementation selectively backports NVIDIA Megatron-LM's existing GDN chunkwise CP;
it does not design a new Relax kernel or communication algorithm. Relax integration must
also handle zig-zag THD layout conversion, per-micro-batch dynamic CP routing, packed-sequence
metadata caching, and convolution state across rank boundaries.

In v1, `auto` preserves the existing headwise/all-gather routing and `chunkwise` is explicit.
This allows the new path to be validated independently before it becomes an automatic choice.

Delivery order:

1. FLA/MCore compatibility backport.
2. Relax integration and runtime routing.
3. Correctness and performance report.

---

## 2. Background

### 2.1 Current execution paths

Relax's `_patch_gdn_for_dynamic_cp()` installs a dynamic-CP wrapper around
`GatedDeltaNet.forward`. It currently provides two paths:

| Path | Condition | Behavior |
|---|---|---|
| **Headwise CP** | CP=1, non-THD, or `num_key_heads % (TP × runtime_CP) == 0` | MCore `cp2hp` all-to-all converts sequence parallelism to head parallelism; each rank scans the full sequence for `1/CP` of the heads |
| **All-gather fallback** | heads do not divide `TP × runtime_CP` | gather the full sequence, run redundant convolution and GDN on every rank, then slice the local output |

Headwise CP is non-redundant because recurrent states are independent across heads. Its limit
is that a single head cannot be split further.

### 2.2 Headwise CP limit

Relax's `_patch_gdn_for_dynamic_cp()` selects headwise CP when:

```text
num_key_heads % (tp_size × runtime_cp_size) == 0
```

Headwise CP cannot split a head. For example, Qwen3.5-9B has 16 key heads, so geometries such
as TP=2/CP=16 cannot use it.

MCore's `_all_to_all_cp2hp` directly computes `h_out = h_in // world_size` without a complete
divisibility guard. Relax's wrapper check therefore also prevents invalid head partitioning.

### 2.3 Cost of the fallback path

The all-gather fallback performs:

1. `in_proj` on the CP-local sequence.
2. `_gdn_cp_gather_full`: `dist.all_gather` on `[s_local, b, C]` followed by zig-zag
   reassembly; backward returns gradients through `reduce_scatter(SUM)`.
3. Short convolution and GDN over the full sequence on every rank with
   `initial_state=None`.
4. `gdn_cp_slice` back to the local zig-zag shard, followed by `out_proj`.

Existing measurements for step 2 are:

| Environment | Input | CP | Dtype | All-gather latency | Effective bandwidth |
|---|---|---:|---|---:|---:|
| `relax-task06`, 2× A6000, PCIe | `[8192, 1, 12320]` | 2 | bf16 | **5.5–6.7 ms** | **36.6 GB/s** |

This cost is paid per GDN layer, per micro-batch, and per forward. With full activation
recompute, it is paid again during the recompute forward.

More importantly, convolution and GDN scan the full sequence redundantly on every rank.
Increasing CP reduces local attention and MLP work but **does not reduce GDN work on this
path**. Full-sequence activation duplication is also the main reason the current path
requires `--recompute-granularity full`.

Chunkwise CP still incurs layout all-to-all and an `(h_ext, M)` all-gather, so its
communication is not free. It removes the full-sequence gather and `CP×` redundant GDN
compute, while its summary size does not grow with sequence length.

### 2.4 Inherited constraints

The new path preserves the existing GDN/Relax constraints:

- training only; `inference_context` must be absent and upstream GDN does not support
  inference yet;
- deterministic GDN is incompatible with packed THD, so deterministic chunkwise CP is not
  supported in v1;
- host `cu_seqlens` must be reused to avoid device-to-host synchronization per layer or
  recompute;
- CP>1 continues to require `--calculate-per-token-loss`;
- FLA must be pinned to a reproducible 0.4.2 commit rather than a version range.

### 2.5 Why not pass state rank-to-rank

For state `S ∈ R^{K×V}`, the gated delta recurrence is:

$$S_t = (\alpha_t I - \beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top$$

Although `S` is small, sequentially passing it between ranks serializes all sequence
partitions. On Relax's zig-zag layout the dependency chain is especially long, so this is
not a viable replacement for the concurrent all-gather fallback.

---

## 3. Design

### 3.1 FLA chunkwise CP

FLA 0.4.2 represents each local partition as an affine state transition:

```text
S_out = M · S_in + h_ext
```

Each rank computes `(h_ext, M)` concurrently, all-gathers these summaries, and locally folds
the preceding ranks' transitions. Backward applies the corresponding reverse fold. No rank
waits for another rank's local scan.

The payload is independent of sequence length but not negligible. For fp32 summaries:

| TP-local value heads | K=V | Per-rank summary | Gathered at CP=4/8/16 |
|---:|---:|---:|---:|
| 16 | 128 | 2 MB | 8 / 16 / 32 MB |
| 32 | 128 | 4 MB | 16 / 32 / 64 MB |

Here `H` means the **TP-local value-head count after GQA expansion**. The final report must
measure this collective directly.

### 3.2 Upstream backport and layout conversion

Megatron-LM commit
[`5139086e`](https://github.com/NVIDIA/Megatron-LM/commit/5139086e7f1240f51ccdf97b244293dd133d6323)
already integrates FLA KCP. Relax will selectively backport it onto its pinned MCore tree.

The GDN path becomes:

1. Project rank-local hidden states.
2. Convert Megatron zig-zag THD layout to contiguous-time layout with all-to-all.
3. Build `FLACPContext` from global sequence boundaries and the runtime CP group.
4. Run rank-local convolution and GDN with FLA CP communication.
5. Convert back to zig-zag layout.
6. Run the rank-local output projection.

The conversion is required because a recurrent scan needs contiguous time ranges, whereas
Relax's zig-zag layout assigns two distant ranges to each rank.

### 3.3 Runtime modes

Relax exposes four modes:

| Mode | Behavior |
|---|---|
| `auto` (default) | Preserve current routing: `headwise` when divisible, otherwise `all_gather` |
| `chunkwise` | Explicit FLA KCP |
| `headwise` | Explicit existing MCore headwise CP |
| `all_gather` | Explicit existing Relax fallback |

`auto` does **not** select chunkwise in v1. This keeps the default behavior unchanged until
chunkwise has passed production validation.

MCore's static `linear_cp_mode` cannot express Relax's per-micro-batch dynamic CP decision.
The backported forward therefore accepts a call-level mode override (or exposes separately
callable headwise/chunkwise bodies). Relax passes the resolved mode as an argument and never
mutates shared config state.

| Relax mode | Static MCore mode | Runtime path |
|---|---|---|
| `auto` | `chunkwise` | wrapper chooses `headwise` or `all_gather` |
| `chunkwise` | `chunkwise` | `chunkwise` |
| `headwise` | `headwise` | `headwise` |
| `all_gather` | `chunkwise` | wrapper-owned `all_gather` |

The permissive static `chunkwise` value is necessary because static `headwise` construction
validates divisibility against `TP × static_CP`. Runtime validation must instead use the
resolved CP group size.

Existing Relax helper `_relax_gdn_cp_config_assert()` becomes obsolete under this mapping.
PR 2 must remove or update it and add `TransformerConfig` construction tests, rather than
leaving its head-count monkey patch active.

Explicit modes never silently fall back. The selected mode is immutable for a forward and
its recompute.

### 3.4 Collective agreement preflight

All ranks must validate routing before the first GDN collective. Once per micro-batch, they
all-gather a fixed-size integer record containing:

```text
(mode, group_size, global_layout_fingerprint, conv_width)
```

These invariant fields must match on every rank. `group_rank` is validated separately:

- locally: `0 <= group_rank < group_size`;
- collectively: gathered ranks must be exactly `{0, ..., group_size - 1}`.

The layout fingerprint describes global metadata only; it must not include rank-local routes.
Any mismatch raises on every rank before chunkwise/headwise collectives begin.

For dynamic CP, `PackedSeqParams.cp_group` and `local_cp_size` must either both be present or
both be absent, and their sizes must agree.

### 3.5 Metadata and packed THD

Routing metadata is built once per micro-batch, not once per GDN layer. A sidecar stores:

- host cumulative sequence lengths;
- runtime group identity, size, rank and device;
- forward/inverse zig-zag-to-contiguous routes;
- one `FLACPContext` per convolution width.

Its validity key is `(boundary fingerprint, group identity, CP size/rank, device)`. Both
`packed_seq_params` and `vlm_packed_seq_params` data paths must propagate the canonical host
boundaries. No process group or host copy may be created in a layer forward.

Required packed-THD invariants:

- global sequence boundaries are preserved across layout conversion;
- forward and inverse routes are exact inverses;
- recurrent and convolution state reset at each packed-sequence boundary;
- padding tokens are excluded from valid-token metrics;
- dynamic groups are created during initialization, never inside `forward`.

### 3.6 Convolution boundaries

FLA 0.4.2 exchanges the last `W-1` raw tokens of each rank's **flattened contiguous
partition**, constructs the next rank's convolution initial state, and sends `dh0` backward.
Sequence-boundary metadata controls how many received tokens are valid.

The relevant v1 edge condition is therefore:

```text
rank-local flattened partition length >= W - 1
```

It is **not** necessary for every individual packed-sequence fragment on a rank to have that
length. If an extreme test produces a whole rank partition shorter than `W-1`, v1 fails fast;
multi-hop halo gathering is out of scope.

Tests cover flat partition lengths `0, 1, W-2, W-1, W`, packed boundaries, first/last ranks,
and forward/backward (`dx`, `dw`) parity.

### 3.7 v1 restrictions

- Training only; no GDN inference.
- Non-deterministic chunkwise kernel only.
- No external `initial_state` or `output_final_state=True` in CP mode.
- THD batch dimension is 1; SBHD CP>1 is limited to micro-batch size 1.
- Key head dimension is at most 256.
- `chunkwise` cannot be combined with Relax's data-layout flag `--allgather-cp`.
- Packed dynamic-CP CUDA Graph capture is out of scope.

---

## 4. Delivery

### PR 1 — FLA/MCore compatibility

- Pin FLA 0.4.2 to commit
  [`ca910f8`](https://github.com/fla-org/flash-linear-attention/commit/ca910f88529565b28b6e16465258f2e239a02dc7).
- Selectively backport the KCP layout helpers, config validation, CP group resolution and GDN
  forward from `5139086e`.
- Add the call-level mode override required by dynamic CP.
- Preserve existing Relax MCore patch fixes.
- Include a file/hunk manifest because this is not a clean cherry-pick.
- Add import, CP=1 and headwise regression tests.

PR 1 changes no Relax runtime routing: `auto` still chooses headwise/all-gather.

### PR 2 — Relax integration

- Add `--gdn-cp-mode={auto,chunkwise,headwise,all_gather}`.
- Wire call-level routing, metadata sidecar and agreement preflight.
- Remove/update `_relax_gdn_cp_config_assert()`.
- Keep the legacy all-gather helpers as rollback paths.
- Update the Qwen3.5 recipe, tests and benchmark collection.

Likely scope includes:

```text
docker/Dockerfile
docker/patch/megatron/
relax/backends/megatron/{model,cp_utils,data}.py
relax/utils/arguments.py
scripts/training/sft/
tests/backends/megatron/
```

Changing `arguments.py` requires maintainer approval under the repository policy.

### Rollback

Switch the recipe from `chunkwise` to `headwise` where divisible, otherwise to
`all_gather` with full recompute. If necessary, restore the recorded Relax/image/MCore/FLA
baseline tuple after active jobs drain.

---

## 5. Validation

### 5.1 Correctness

Reference: CP=1 full-sequence GDN, with CP>1 `all_gather` as a migration cross-check.

Required coverage:

- forward output and input/parameter gradients;
- THD single/multiple sequences, padding and rank-boundary cuts;
- sequence lengths around GDN chunk boundaries (`63/64/65`, `127/128/129`);
- convolution flat-partition boundaries (`0/1/W-2/W-1/W`);
- TP/SP combinations and non-divisible synthetic head geometry;
- dynamic CP micro-batches with CP in `{1,2,4}`;
- all four modes and direct branch/collective assertions;
- one-rank metadata corruption producing a clean error instead of a hang;
- checkpoint save/resume and recompute parity.

Routing tests must observe the executed branch or collectives; numerical parity alone cannot
distinguish headwise from chunkwise.

Numerical gates use:

```text
NRMS(x) = RMS(candidate - reference) / (RMS(reference) + 1e-8)
```

| Tensor | Gate |
|---|---:|
| output, `dq`, `dk`, `dv`, `dg` | `< 2e-3` |
| `dbeta` | provisional `< 1e-2` |
| CP=1 refactor parity | `rtol=1e-4, atol=1e-4` |

The `dbeta` threshold, dtype, input distribution and `use_qk_l2norm_in_kernel` setting must
be frozen from measured target-geometry errors before implementation tests are finalized.

### 5.2 Performance

Frozen primary setup:

| Item | Value |
|---|---|
| Model | Qwen3.5-9B, pinned revision |
| Topology | TP=2, CP=4, PP=1, SP on |
| Hardware | 8× H100 80GB SXM, NVSwitch |
| Runs | 20 warmup + 200 measured steps |
| Seeds | 1234 / 1235 / 1236, paired |

Two required experiments:

1. **Fallback replacement:** forced `all_gather` versus forced `chunkwise`, same image and
   recompute setting. If the baseline OOMs, report it as a capacity result; do not fabricate
   a throughput ratio.
2. **Non-regression:** frozen main baseline, candidate-image forced `headwise`, and
   candidate-image `chunkwise`. This separates dependency changes from algorithm changes.

A reduced-head 8-GPU configuration (for example 4 key heads at TP=2/CP=4) verifies the real
non-divisible `auto → all_gather` route and explicit chunkwise correctness. A 32-GPU
Qwen3.5-9B test at TP=2/CP=16 is optional and must be reported separately if available.

Report effective non-padding tokens/s, step-time distribution, peak allocated/reserved
memory, `(h_ext, M)` communication time, GPU utilization, loss, gradient norm and exact
effective-token counts.

Gates:

- **Fallback replacement:** at least 10% throughput improvement with no more than 5% memory
  regression, **or** at least 10% memory reduction with no more than 5% throughput regression.
- **Headwise non-regression:** throughput and memory within a 5% non-inferiority margin.
- Effective-token counts must match exactly; loss/held-out quality must not regress beyond
  `0.01` nats/token.

Publish per-seed ratios and the geometric mean with a 95% CI on paired log-ratios. Also report
whether chunkwise permits removal of mandatory full recompute.

---

## 6. Risks and non-goals

| Risk | Mitigation |
|---|---|
| Static MCore mode conflicts with dynamic routing | Call-level override; never mutate shared config |
| Rank disagreement hangs the job | Fixed-size agreement preflight before GDN collectives |
| Incorrect packed routing leaks state | Exact layout tests and multi-sequence gradient parity |
| Per-layer CPU synchronization | Micro-batch metadata sidecar and profiler check |
| Summary collective is larger than expected | Measure payload and time directly |
| Dependency backport regresses existing paths | CP=1/headwise controls and selective-backport manifest |
| Whole rank partition is shorter than `W-1` | Fail fast in v1; multi-hop convolution halo is future work |

Non-goals for v1:

- full MCore/Megatron-Bridge upgrade;
- a new state-communication algorithm;
- GDN inference or deterministic chunkwise CP;
- CUDA Graph support for packed dynamic CP;
- optimizer-state resharding across CP sizes;
- removal of the existing all-gather fallback.

---

## 7. Maintainer decisions

1. Approve the selective backport of `5139086e` and the documented call-level mode override.
2. Approve the `--gdn-cp-mode` flag and `auto` remaining backward-compatible in v1.
3. Confirm the two-PR split and benchmark gates.
4. Decide whether a 32-GPU non-divisible production run is available; otherwise scope
   production performance claims to the single-node topology.
