# Relax Issue #86 / Task 23 性能优化报告

> 本模板用于记录 Qwen3-4B、DAPO Math、4×NVIDIA RTX PRO 5000 72GB
> 环境下的 Colocate 性能优化实验。基线部分已经填入，优化实验部分在运行后补充。

## 1. 结论摘要

- 优化目标：降低端到端 step time，提高 rollout/response throughput。
- Baseline 配置：单个 TP4 SGLang rollout engine，Actor TP2，4 卡 Colocate。
- Baseline 稳定 step time：`91.11 s`。
- Baseline response throughput：`535.21 tok/s`。
- Baseline 主要瓶颈：rollout/wait，占稳定 step time 的约 `75.3%`。
- 优化方案：`<!-- 例如：将 rollout-num-gpus-per-engine 从 4 改为 2 -->`
- 优化后稳定 step time：`<!-- 待填 -->`
- Step time 改善：`<!-- 待填 --> %`
- 优化后 response throughput：`<!-- 待填 -->`
- Response throughput 改善：`<!-- 待填 --> %`
- 正确性和稳定性结论：`<!-- 待填 -->`

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| Host | `pro5k4` |
| GPU | 4×NVIDIA RTX PRO 5000 72GB Blackwell |
| Repository | `/root/Relax` |
| Git commit | `b9da9f818fca70ff311009d4213be196acfa426e` |
| Model | `/workspace/models/Qwen3-4B` |
| Dataset | `/root/data/dapo-math-17k/dapo-math-17k.jsonl` |
| Training algorithm | GRPO / DAPO reward |
| Actor parallelism | TP2, PP1, CP1 |
| Baseline rollout parallelism | 1×TP4 SGLang engine |
| Colocate | Enabled |
| GPU monitor interval | Approximately 1 second |

## 3. 公平性约束

Baseline 和优化实验必须保持以下项目不变：

- 相同 Git commit 和 reward 修复。
- 相同模型、数据集和 prompt shuffle/seed 设置。
- 相同 rollout batch size、samples per prompt 和 global batch size。
- 相同最大响应长度和采样参数。
- 相同 Actor 并行策略和训练参数。
- 每个配置至少运行两次。
- 统计时排除 warm-up step 0，使用 step 1–5。
- 每轮运行保存完整训练日志、GPU CSV 和运行元信息。

只允许改变：

```text
<!-- 填写本次优化明确改变的参数 -->
```

## 4. Reward 正确性修复

原始 reward parser 会把以下正确答案解析成 `pred="**6"`：

```text
**Answer:** \boxed{6}<|im_end|>
```

修复后，parser 会先从最终 `Answer:` 行中提取最后一个完整的
`\boxed{...}`，再进行答案归一化。

修改文件：

- `relax/engine/rewards/math_dapo_utils.py`
- `tests/engine/rewards/test_math_dapo_utils.py`

验证结果：

- `tests/engine/rewards`：48 passed。
- 新增定向回归测试：8 passed。
- 原始故障样例：`score=1.0, pred=6`。
- `py_compile`：passed。
- `git diff --check`：passed。

## 5. Baseline 参数

| 参数 | 值 |
|---|---:|
| Number of GPUs | 4 |
| Number of rollout steps | 6 |
| Rollout batch size | 4 |
| Samples per prompt | 2 |
| Samples per step | 8 |
| Global batch size | 8 |
| Maximum response length | 8192 |
| Actor tensor parallel size | 2 |
| Rollout GPUs per engine | 4 |
| Number of rollout engines | 1 |
| Max tokens per GPU | 10240 |
| Log-probs max tokens per GPU | 30720 |
| SGLang static memory fraction | 0.8 |
| Learning rate | 1e-6 |

Baseline 脚本：

```text
/root/Relax/scripts/training/text/run-qwen3-4B-4xgpu-pro5000-baseline.sh
```

## 6. Baseline 重复实验

稳定指标均为 step 1–5 的均值。

| Run | Ray Job | Wall time | Step time | Wait time | Train time | Response tok/s | Actor tok/s | Peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 1 | `raysubmit_T2Nm1Qjjbs1E74dx` | 727 s | 91.27 s | 68.69 s | 22.58 s | 536.58 | 2507.08 | 65035 MiB |
| Baseline 2 | `raysubmit_LZgAKYFYF3S3RHPd` | 733 s | 90.95 s | 68.55 s | 22.41 s | 533.84 | 2505.69 | 64675 MiB |
| Mean | — | 730 s | 91.11 s | 68.62 s | 22.49 s | 535.21 | 2506.39 | 64855 MiB |
| Run difference | — | 0.82% | 0.35% | 0.21% | 0.76% | 0.51% | 0.06% | 0.56% |

两次 baseline 的性能指标波动较小，可以作为后续优化实验的对照。

### 6.1 Baseline 质量指标

| Run | Raw reward mean | Positive rewards | Non-zero gradient steps | OOM/Traceback |
|---|---:|---:|---:|---:|
| Baseline 1 | -0.5833 | 10/48 | 6/6 | 0 |
| Baseline 2 | -0.5417 | 11/48 | 4/6 | 0 |
| Combined | -0.5625 | 21/96 | 10/12 | 0 |

零梯度 step 的原因是对应 GRPO prompt 组内奖励相同，不代表 reward parser
失效。日志中的正确 Markdown 答案均能得到正奖励。

### 6.2 Baseline GPU 指标

| 指标 | Baseline 1 | Baseline 2 | Mean |
|---|---:|---:|---:|
| Overall GPU utilization | 69.19% | 69.12% | 69.16% |
| Utilization while memory >50 GiB | 94.35% | 94.19% | 94.27% |
| Stable rollout-window utilization | 96.70% | 97.81% | 97.26% |
| Stable actor-window utilization | 75.40% | 75.02% | 75.21% |
| Peak memory | 65035 MiB | 64675 MiB | 64855 MiB |
| Peak temperature | 62°C | 61°C | 62°C max |

### 6.3 Baseline 阶段占比

| 阶段 | 平均耗时 | 占稳定 step time |
|---|---:|---:|
| Reported rollout/wait path | 68.62 s | 75.3% |
| Training path | 22.49 s | 24.7% |
| Sleep + update weights + wake-up | 9.19 s | 10.1% |

`Sleep + update weights + wake-up` 位于整体流水线内部，是 wait path 的组成部分，
不能再次与 step time 相加。

## 7. 优化方案

### 7.1 问题分析

<!--
说明为什么选择该优化：
- 哪个阶段最慢？
- 哪个指标证明它是瓶颈？
- 为什么该改动可能有效？
- 预期风险是什么？
-->

### 7.2 代码或参数改动

```diff
- <!-- baseline -->
+ <!-- optimized -->
```

建议的第一个候选：

```diff
- --rollout-num-gpus-per-engine 4
+ --rollout-num-gpus-per-engine 2
```

该配置会从一个 TP4 rollout engine 改为两个 TP2 rollout engines。

### 7.3 预期效果

- `<!-- rollout/wait time 预期变化 -->`
- `<!-- response throughput 预期变化 -->`
- `<!-- 通信量或并发能力预期变化 -->`
- `<!-- 显存和稳定性风险 -->`

## 8. 优化实验结果

稳定指标均使用 step 1–5。

| Run | Ray Job | Step time | Wait time | Train time | Response tok/s | Actor tok/s | Peak memory | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Optimized 1 | `<!-- job id -->` | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 2 | `<!-- job id -->` | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Mean | — | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | — |

### 8.1 每步数据

| Run | Step | Raw reward | Response length | Step time | Wait time | Train time | Response tok/s | Grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimized 1 | 0 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 1 | 1 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 1 | 2 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 1 | 3 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 1 | 4 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 1 | 5 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |

第二次优化运行可复制以上六行。

### 8.2 GPU 数据

| Run | Overall util | Active util | Rollout util | Actor-window util | Peak memory | Peak temp |
|---|---:|---:|---:|---:|---:|---:|
| Optimized 1 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 2 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |

### 8.3 正确性和稳定性

| Run | Positive rewards | Non-zero gradient steps | NaN/Inf | OOM | Traceback | Residual processes |
|---|---:|---:|---:|---:|---:|---:|
| Optimized 1 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| Optimized 2 | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |

## 9. Baseline 与优化结果对比

| Metric | Baseline mean | Optimized mean | Change | Better? |
|---|---:|---:|---:|---|
| Stable step time | 91.11 s | <!-- --> | <!-- -->% | <!-- --> |
| Rollout/wait time | 68.62 s | <!-- --> | <!-- -->% | <!-- --> |
| Train time | 22.49 s | <!-- --> | <!-- -->% | <!-- --> |
| Response throughput | 535.21 tok/s | <!-- --> | <!-- -->% | <!-- --> |
| Actor throughput | 2506.39 tok/s | <!-- --> | <!-- -->% | <!-- --> |
| Peak memory | 65035 MiB max | <!-- --> | <!-- --> | <!-- --> |

计算公式：

```text
Step time improvement (%) =
    (baseline_step_time - optimized_step_time) / baseline_step_time * 100

Throughput improvement (%) =
    (optimized_throughput - baseline_throughput) / baseline_throughput * 100
```

## 10. 结果判断

### 10.1 性能门槛

- [ ] 两次优化运行都成功完成。
- [ ] Stable step time 相对 baseline 有明确改善。
- [ ] Response throughput 没有下降，或下降原因有合理解释。
- [ ] 改善幅度明显大于 baseline 的 run-to-run 波动。

### 10.2 质量门槛

- [ ] Reward parser 回归测试通过。
- [ ] 正确 Markdown 答案能够得到正奖励。
- [ ] 无 NaN、Inf、OOM 和未处理异常。
- [ ] 输出长度、batch size 和采样参数与 baseline 一致。
- [ ] 训练结束后无活跃 GPU/Ray/SGLang 残留进程。

## 11. 最终结论

<!--
建议使用以下格式：

优化方案将 ______ 从 ______ 调整为 ______。
两次重复实验的 stable step time 从 baseline 的 91.11 s 降至 ______ s，
改善 ______%；response throughput 从 535.21 tok/s 提升至 ______ tok/s，
改善 ______%。实验未出现 ______，质量指标 ______。
因此建议 / 不建议合入该优化。
-->

## 12. 复现命令

Baseline：

```bash
cd /root/Relax
NUM_GPUS=4 \
MODEL_DIR=/workspace/models \
DATA_DIR=/root/data \
NUM_ROLLOUT=6 \
bash scripts/training/text/run-qwen3-4B-4xgpu-pro5000-baseline.sh
```

Optimized：

```bash
<!-- 填写优化脚本或完整命令 -->
```

## 13. 实验产物

Baseline 1：

```text
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-15:43:33.log
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-15:43:33-launcher.log
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-15:43:33-gpu.csv
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-15:43:33-meta.txt
```

Baseline 2：

```text
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-16:38:27.log
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-16:38:27-launcher.log
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-16:38:27-gpu.csv
/root/Relax/log/qwen3-4b-GRPO-gpu4-pro5000-baseline-2026-07-28-16:38:27-meta.txt
```

Optimized：

```text
<!-- 填写优化实验日志、GPU CSV 和元信息路径 -->
```
