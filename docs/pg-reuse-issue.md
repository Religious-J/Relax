Status: Proposed  
Author: @ldemon2333, @Religious-J  
Task tracker: [Task 23 — Colocate 纯文本性能](https://github.com/redai-infra/Relax/issues/86)  
Relax baseline: [0694cd536aa9bddb9cd78452585e7f56feebd6a2](https://github.com/redai-infra/Relax/commit/0694cd536aa9bddb9cd78452585e7f56feebd6a2)

# [Performance] Reuse training process groups across TMS pause for immediate weight synchronization

## 背景

在 actor 与 rollout 共卡、并开启训练卸载的场景中，每个训练 step 都需要执行以下状态切换：

```text
actor training
  → sleep/offload training model
  → update weights to SGLang
  → rollout
  → wake up training model
```

当前实现为了保证 NCCL communicator 不在 TMS pause 期间持有失效资源，会在 `sleep()` 中销毁 Megatron 训练 process groups，并在 `update_weights()` 前重新创建。

基线路径为：

```text
sleep()
  → destroy 20 training process groups
  → wait 2 s for NCCL socket release
  → torch_memory_saver.pause()

update_weights()
  → reload 20 training process groups
  → export weights to SGLang
  → destroy training process groups again

wake_up()
  → torch_memory_saver.resume()
  → reload training process groups
```

Profile 和运行日志显示，这组 destroy/reload 操作位于权重同步的关键路径上。它不改变传输的数据量，因此属于纯框架生命周期开销。

## 优化思路

在受支持的共卡权重同步场景中，训练 process groups 只需要跨越一个很短的窗口：

```text
sleep() → update_weights()
```

因此可以让训练 PG 在 TMS pause 后继续存活，直接用于紧接着发生的权重导出。权重同步结束后仍然销毁 PG，不让 communicator 常驻整个 rollout。

优化后的生命周期为：

```text
sleep()
  → verify all ranks can preserve PG
  → Gloo barrier
  → torch_memory_saver.pause()
  → validate preserved communicators

update_weights()
  → reuse live training process groups
  → export weights to SGLang
  → destroy process groups once
  → defer/overlap the NCCL release cooldown

rollout
  → process groups remain destroyed

wake_up()
  → torch_memory_saver.resume()
  → reload training process groups
```

该方案减少了每 step 一次完整的训练 PG destroy/reload 循环，并将最后的 NCCL cooldown 移出权重同步关键路径。

需要强调的是：这不是永久保留 NCCL process groups。PG 只在 `sleep()` 到紧接着的 `update_weights()` 之间复用，权重同步完成后仍会销毁。

## 实现

Prototype commit: [f98aa2bab591674b896183fff6b9f037bc0cbb2c](https://github.com/Religious-J/Relax/commit/f98aa2bab591674b896183fff6b9f037bc0cbb2c)

### 1. 显式 PG 生命周期状态

增加以下状态：

```text
PG_ACTIVE
PG_PAUSED_LIVE
PG_DESTROYED
```

用于区分：

- PG 正常工作；
- TMS 已 pause，但 PG 仍然可用；
- PG 已销毁，需要重新创建。

### 2. 保守的能力检查

只有同时满足以下条件才允许复用：

- actor role；
- actor/rollout colocate；
- `offload_train`；
- per-step rollout；
- Torch Memory Saver 已启用；
- 使用 `UpdateWeightFromTensor` 本地 IPC 权重同步；
- weights backuper 已启用；
- pipeline parallel size 为 1；
- 非 critic；
- 非 fully-async；
- 所有 reloadable process groups 当前均处于 active 状态；
- 所有训练 rank 对复用决策达成一致。

实验通过运行时开关控制：

```text
--preserve-train-process-groups-for-weight-sync
```

该开关只用于 A/B 验证，不改变训练或 rollout 参数。

### 3. pause 后 communicator canary

首次进入 PG reuse 路径时，会在 TMS pause 后对实际使用的非平凡通信组运行 `all_reduce` canary：

- TP；
- DP；
- DP+CP；
- CP；
- EP；
- ETP。

每个 communicator 都验证：

```text
all_reduce(ones) == process group world size
```

随后通过 Gloo control group 汇总所有 rank 的执行结果。

如果任意 communicator 失败：

```text
disable PG preservation
→ destroy preserved groups
→ fall back to the original destroy/reload lifecycle
```

### 4. 异常回收

如果复用 PG 后的权重同步抛出异常，会立即销毁仍存活的训练 communicator，避免将半失效 PG 泄漏到 rollout 阶段。

## 测试配置

硬件与软件：

- 8 × NVIDIA H20，97,871 MiB/GPU；
- PyTorch 2.11.0+cu129；
- CUDA 12.9；
- NCCL 2.28.9；
- A/B 实验使用同一 PG reuse worktree，唯一运行时差异为是否启用复用开关。

模型与数据：

```text
Model: /workspace/models/Qwen3-30B-A3B
Data:  /root/data/dapo-math-17k/dapo-math-17k.jsonl
```

训练入口：

```bash
bash scripts/training/text/run-qwen3-30B-A3B-8xgpu.sh
```

主要拓扑：

- actor/rollout colocate on 8 GPUs；
- training TP=4、CP=2、EP=8；
- DeepEP flex dispatcher；
- SGLang TP=8；
- global batch size 256；
- rollout batch size 32；
- 8 samples per prompt；
- CPU optimizer offload。

A/B 两组使用相同代码、模型、数据及训练参数，训练脚本 SHA256 均为：

```text
1b6d15d7d08ef996907b689e8bed3508fa9b4f26fa8705addc4835d39185316d
```

唯一差异：

```bash
# Baseline
ENABLE_TRAIN_PG_REUSE=0 \
bash scripts/training/text/run-qwen3-30B-A3B-8xgpu.sh

# Optimized
ENABLE_TRAIN_PG_REUSE=1 \
bash scripts/training/text/run-qwen3-30B-A3B-8xgpu.sh
```

两组均运行至 warmup step 和第一个 steady-state step 完成。

Focused tests：

```bash
pytest -q \
  tests/utils/test_reloadable_process_group.py \
  tests/backends/megatron/test_actor_pg_lifecycle.py
```

结果：`9 passed`。

## 性能结果

### Steady step 1

| Metric | Baseline | PG reuse | Delta |
|---|---:|---:|---:|
| `sleep_time` | 12.046 s | 10.042 s | **-2.003 s / -16.63%** |
| `update_weights_time` | 9.090 s | 6.156 s | **-2.934 s / -32.27%** |
| `wake_up_time` | 2.371 s | 1.895 s | -0.476 s / -20.10% |
| Lifecycle switch total | 23.507 s | 18.094 s | **-5.413 s / -23.03%** |
| `step_time` | 412.579 s | 403.170 s | -2.28% |
| `step_token_per_s` | 4150.01 | 4183.82 | **+0.81%** |
| `step_resp_token_per_s` | 4056.16 | 4087.78 | **+0.78%** |
| Actor train tokens | 1,712,206 | 1,686,793 | -1.48% |
| Peak GPU memory | 75,473 MiB | 75,477 MiB | +4 MiB |

### 收益归因

可直接归因的收益是生命周期阶段减少的：

```text
5.413 / 412.579 = 1.31% of baseline step
```

其中：

- `sleep_time` 减少约 2 秒，主要来自跳过 pause 前的 destroy 和固定 NCCL cooldown；
- `update_weights_time` 减少约 2.93 秒，主要来自跳过 20 个训练 PG 的重新创建，并将同步后的 cooldown 移出关键路径；
- 权重转换、权重读取和 HTTP shard 传输逻辑没有变化。

完整 step time 减少 2.28%，但不能全部归因于本优化，因为 optimized rollout 最终产生的 actor tokens 少了 1.48%。因此目前更可靠的结论是：

```text
直接生命周期收益：约 5.41 秒/step，约占整步 1.31%
观察到的吞吐收益：约 +0.8%，仅作为方向性结果
```

## 正确性验证

完成的验证包括：

- pause 后 TP/DP/CP/EP communicator canary 通过；
- 日志确认进入：

```text
[weight-sync-pg] reusing live training process groups for weight export
```

- baseline 和 optimized 都完成 warmup 与 steady-state 训练；
- loss、gradient norm、TIS 和 mismatch metrics 均为有限值；
- 没有 process-group reload failure；
- 没有 weight-version mismatch；
- 没有 traceback 或 CUDA OOM；
- 权重同步后可以正常进入下一轮 rollout；
- focused lifecycle tests：`9 passed`。

测试覆盖：

- 只有立即权重同步窗口允许 PG reuse；
- canary 失败自动退回原有生命周期；
- 权重同步异常时销毁 preserved PG；
- 所有 rank 必须对复用决策达成一致；
- CP/EP 拓扑被纳入 canary；
- reloadable PG registry 必须全部处于 active 状态。

由于两组 rollout 独立采样，不能要求输出 token、loss 或 log-prob bitwise identical。这里验证的是 PG reuse 不改变训练计算路径和权重内容，并且完整训练、同步和下一轮 rollout 均能成功执行。

## 局限性

1. 当前只有每组一次 steady-state run，尚不足以给出稳定的 PR 级 E2E 收益结论。

2. 优化只适用于短窗口、本地 IPC、colocate 权重同步。critic、fully-async、PP>1 和不支持的 weight updater 会自动回退。

3. 虽然峰值显存基本不变，但 PG reuse 会在 pause 后短时间保留 NCCL native resources。实验中 steady paused-window 的物理显存从约 `8.86–8.87 GiB` 上升到 `12.45–14.24 GiB`。该数据来自单次 A/B，仍需重复确认，但说明这项优化会减少 SGLang 恢复时的瞬时显存余量。

4. 代码复杂度高于单纯重叠 2 秒 NCCL cooldown：需要生命周期状态机、拓扑 canary、全 rank 共识和异常回收。

## 结论

PG reuse 是有效的框架级次要优化：

- 避免每 step 一次不必要的训练 PG destroy/reload；
- 状态切换阶段减少约 5.41 秒；
- 对当前 30B workload 的直接整步贡献约 1.31%；
- 观察到约 +0.8% 吞吐提升；
- 不改变训练参数或权重同步内容。

但它的适用范围较窄，且会增加 pause 窗口的 native GPU memory 占用。建议将其作为独立实验或次级性能贡献，不适合作为 Task 23 的主要性能 PR。正式提交前应至少补充 2–3 组重复 A/B，并重点验证暂停窗口的显存余量。
