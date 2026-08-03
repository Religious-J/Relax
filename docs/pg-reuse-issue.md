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
