# OPSD 算法对齐说明

本文件分三部分：

1. **Part 1 — 算法规范**：以 `method.md` 为准；同时标注它和 `paper.md` 的关系。
2. **Part 2 — 当前实现状态**：代码现在做了什么、为什么、和 paper / method 的偏差点。
3. **Part 3 — 工程改进与 opt-in 旋钮**：项目自有扩展，默认行为对齐 paper / method，扩展全部 opt-in。

历史 BUG.md 的全部条目（截断 / w_entropy=0 / KL<0 / mode-collapse 苗头 / lr-pg=0）均已在代码层面修复，下游影响详见 Part 2。

---

## Part 1 — 算法规范（Source of truth: `method.md` + `paper.md`）

> `paper.md` 的 OPSD 是**单条 teacher 路径**：teacher 直接看 ground-truth `y*`。
> `method.md` 在此基础上做了扩展：用 *学生自己跑出来的正确轨迹* 当作 privileged 信息，并允许多条 mixture。
> 即 `method.md` ⊇ `paper.md`：paper 的训练循环可看作 method 在 `N=1`、`τ_k = y*` 时的特例。

### 1.1 总流程（method.md §10，本仓 K+1 改造）

对每个 prompt `x`：

1. **On-policy rollout**：从旧策略 `π_{θ_old}(·|x)` 采样 **K+1** 条轨迹（`opsd_k+1` 条）。
   - 1 条作为 student `y`；剩下 K 条进 candidate pool。
   - 这样 `y` **不会出现在 candidate pool 中**，杜绝"student 恰好等于某条 τ_k"导致的 mixture 退化（详见 Part 2 §B.2）。
2. **成功筛选（method.md §3）**：`B_x = { τ_j ∈ pool | R(x, τ_j) = 1 }`；空则 fallback 到 GT（`--opsd-fallback-to-gt`）。
3. **质量打分（method.md §4）**：
   ```
   s(τ) = 1 − η_l · Len(τ)/L_max − η_f · FormatPenalty(τ) + η_c · Conf(τ)
   Conf(τ) = (1/|τ|) Σ_t log π_T(τ_t | x, τ_<t)
   ```
   保留 `B_x ← TopK(B_x, s, K_b)`。
4. **多样性选择（method.md §5）**：k-center greedy；距离推荐 token-level JSD：
   `d(τ_i, τ_j) = (1/T) Σ_t JSD(q_i^t, q_j^t)`。选出 `N` 条 → `P_x`。
5. **Teacher 前向（method.md §6）**：
   ```
   q_k^t = π_T(· | x, τ_k+, y_<t),   p^t = π_θ(· | x, y_<t)
   ```
   `π_T` 由 `paper.md §4.1` 锁定为 **initial policy**（frozen），不随训练更新。
6. **Mixture 权重（method.md §7）**：
   ```
   Δ_k^t = KL(q_k^t ‖ p^t),   h_k^t = H(q_k^t),   g_k^t = (1/(N−1)) Σ_{j≠k} JSD(q_k^t, q_j^t)
   w_k^t = softmax_k(−β·Δ_k^t − γ·h_k^t + ρ·g_k^t)
   ```
   纯 raw 分布上做 softmax，**method.md 没有温度项**。
7. **Mixture teacher（method.md §8）**：`q_mix^t = Σ_k w_k^t · q_k^t`。
8. **蒸馏目标（method.md §9 / paper Eq.8）**：
   ```
   L_distill = (1/T) Σ_t KL(q_mix^t ‖ p^t)             (主损失，forward KL，全词表)
   L_total   = L_distill + α_RKL · Σ_t KL(p^t ‖ q_mix^t)        (可选 RKL 辅助，α_RKL ≪ 1)
   ```
   `paper.md §3.2` + Figure 4：**per-position KL clamp `τ=0.05`** 对长训练稳定性至关重要（官方 `--jsd_token_clip 0.05`）。

### 1.2 Paper Table 6 推荐超参

| 项 | 值 | 来源 |
|---|---|---|
| Learning rate | `5e-6` | Table 6 |
| Effective batch size | `32` | Table 6 |
| Max completion length | `1024`（数学推理可放宽到 4k–8k，paper 是短链场景） | Table 6 |
| Generations per prompt | `1`（paper 简化为 single-y）；本仓 K+1 = 17 | Table 6 + 本仓扩展 |
| Sampling temperature | `1.1` | Table 6 |
| Top-p / Top-k | `0.95 / 20` | OPSD 官方训练脚本 |
| KL clip τ | `0.05` | Paper §3.2 / `run_opsd_*.sh --jsd_token_clip 0.05` |
| Training steps | `100` | Table 6 |
| LoRA r / α | `64 / 128` | Table 6（slime 走全微调，不用 LoRA） |

### 1.3 Method.md §13 推荐区间

| 项 | 范围 | 含义 |
|---|---|---|
| K | 8–32 | 候选池大小 |
| N | 2–4 | 多样性选中条数 |
| K_b | 8–16 | 质量预筛 top-K_b |
| β（KL 权） | 0.5–2 | mixture-weight 公式中 Δ 的系数 |
| γ（entropy 权） | 0.1–1 | mixture-weight 公式中 h 的系数 |
| ρ（diversity 权） | 0.1–1 | mixture-weight 公式中 g 的系数 |
| λ_distill | 0.05–0.5 | "L_RL + λ·L_distill" 框架下的蒸馏权；纯蒸馏时取 1.0 |

---

## Part 2 — 当前实现状态（Implementation Status）

### A. 与 paper.md / method.md 的核心对齐项

| 项 | 状态 | 默认值 | 备注 |
|---|---|---|---|
| 主损失 = `α·KL(q_mix‖p_θ)` | ✅ | `α=1.0` | `plugin.py:loss_function`。`--opsd-mix-with-policy-loss` 默认 False（无 GRPO 杂交） |
| frozen initial-policy teacher | ✅ | `--opsd-freeze-teacher=True` | `plugin.py:before_train_step_hook` 路径 B：训练前快照 → swap-back |
| K+1 rollout，1 学生 + K 候选 | ✅ | `opsd_k+1` | `rollout.py:generate_rollout`；学生不再出现在 candidate pool |
| Mixture-weight 无温度 | ✅ | `--opsd-temperature=1.0` | `mixture_weights` 里 `/ T` 默认 T=1.0 等价 raw |
| Per-position KL clamp `τ=0.05` | ✅ | `--opsd-jsd-token-clip=0.05` | 对应 paper `--jsd_token_clip 0.05`；post sum-over-vocab，保证非负 |
| Per-(pos,vocab) clip 默认关 | ✅ | `--opsd-pointwise-kl-clip=None` | 该 clip 单边性会让 KL 走负（BUG.md #3），保留 opt-in 但不默认 |
| Forward-KL 主，RKL 可选 | ✅ | `--opsd-rkl-weight=0.0` | `_VocabParallelRKLDiv` 实现，paper `α_RKL ≪ 1` |
| token_jsd 多样性距离 | ✅ | `--opsd-diversity-metric=token_jsd` | method.md §5 推荐 |
| Quality 三项 + Conf 归一 | ✅ + 🔧 | conf-norm=`rank` | method.md §4 公式；Conf 跨候选归一 [0,1]（详见 §C） |

### B. BUG.md 的修复（按编号）

| BUG | 问题 | 根因 | 修复 |
|---|---|---|---|
| #1 | 99% rollout 截断、reward 全 0、训练无信号 | 1.7B math 推理需要 >1024 token，paper Table 6 的 1024 不够 | `scripts/run_qwen3_*_opsd.sh`：`--rollout-max-response-len` 提到 8192 |
| #2 | `train/opsd_w_entropy ≡ 0`，mixture 看似一直 collapse | `distillation.py:392` 操作符优先级：`-(w*log_w).sum(0).clamp(min=0)` 解析为 `-((w*log_w).sum(0).clamp(min=0))`，先把负的 sum clamp 到 0，再取负得 -0.0 | 加显式括号：`(-(w * log_w).sum(0)).clamp(min=0)` |
| #3 | `train/opsd_kl` 单调走负 | `_VocabParallelKLDiv` 里 per-(pos,vocab) clip 是单边的（只压正项 ≥ τ，不动负项），可让 sum-over-vocab 变负 | (a) 默认值对调：`opsd-pointwise-kl-clip` `0.05 → None`，`opsd-jsd-token-clip` `None → 0.05`（paper-faithful per-position clamp）；(b) 加 `metrics["opsd_kl_clamped"] = max(0, kl)` 镜像监控 |
| #4 | mode collapse 苗头（student log_probs 越来越自信） | #3 让 reverse 推力被削弱 | 期望 #3 修好后缓解；监控 `train/opsd_kl` 持续 ≥ 0 即可 |
| #5 | `train/lr-pg_0/1 = 0` | 读的是 `opt_param_scheduler.get_lr(...)` 计划值，不是 optimizer 实际生效值 | `model.py:731` 改读 `param_group["lr"]` |
| 衍生 | "学生 = privileged trace" 退化（mixture one-hot） | 旧代码学生从 K 中随机挑，且 candidate pool 也是 K，K=16 全对组里 100% 重叠 | **K+1 设计**：rollout 17 条，1 学生 + 16 候选，从结构上隔开 |

### C. 仍是项目自有约定（method.md 未严格规定，写在这里以免后人翻代码）

| 约定 | 位置 | 理由 |
|---|---|---|
| Conf 跨候选 `rank` 归一化 | `distillation.py:add_conf` | method.md §4 只给 `Conf = mean log-prob`，没指 numerical scaling。raw Conf 在 [-5,-1] nats 量纲会盖过 Len/Format（[0,1] 量纲），让默认 `η_c=0.5` 失去"权重"语义。`rank → [0,1]` 让三项共享数轴。可用 `--opsd-quality-conf-norm raw` 还原字面行为 |
| 顶 K=512 词表近似 mixture 权重 | `distillation.py:mixture_weights` | full-vocab 在 N=4 / T=8k 下显存爆。method.md 没限定，工程妥协。可调 `--opsd-weight-top-k` |
| GT-fallback 当 candidate 全错 | `rollout.py:_collect_privileged` | method.md §3 说"can fallback to GT"，没限定形式；本仓拿 label 直接 encode 当 1 条 trace |

---

## Part 3 — 工程改进与 opt-in 旋钮

| 改进 | 文件 | 触发方式 | 默认 |
|---|---|---|---|
| Hybrid（蒸馏 + GRPO PG） | `plugin.py` | `--opsd-mix-with-policy-loss` | False（paper 是替代 GRPO，不是叠加） |
| RKL 辅助项 | `distillation.py:_VocabParallelRKLDiv` | `--opsd-rkl-weight <α_RKL>` | 0.0（关） |
| Mixture-weight 温度软化 | `distillation.py:mixture_weights` | `--opsd-temperature <T>` | 1.0（raw） |
| Per-(pos,vocab) KL clip（已知有副作用） | `distillation.py:_VocabParallelKLDiv` | `--opsd-pointwise-kl-clip <τ>` | None（关） |
| `unigram_jsd` 提前多样性选择，省 q-forward | `selection.py`、`distillation.py:prepare_teacher_outputs` | `--opsd-diversity-metric unigram_jsd` | `token_jsd` |
| Conf 归一化模式 | `distillation.py:add_conf` | `--opsd-quality-conf-norm rank/zscore/minmax/raw` | `rank` |
| 顶 K 词表近似（mixture / JSD） | `distillation.py`、`selection.py` | `--opsd-weight-top-k`、`--opsd-diversity-top-k` | 512 / 128 |
| Frozen-teacher 关闭（学生即 teacher） | `plugin.py` | `--no-opsd-freeze-teacher` | True |
| 外部教师（不再是 OPSD，是普通 KD） | `plugin.py` + slime `--use-opd` | `--no-opsd-freeze-teacher --use-opd --opd-type megatron --opd-teacher-load <path>` | 不启用 |

### 与 slime 内置 `--use-opd` 的关系

slime 自带的 `--use-opd / --opd-kl-coef` 实现的是 **paper §3.2 末尾的 alternative objective (Eq.9)**——在采样 token 上做 PG-flavored 修正，把 `reverse_kl = log π_S − log π_T` 当 advantage 修正项 (`apply_opd_kl_to_advantages`)。它和本 plugin 走的 main objective 不一样：

- slime `--use-opd`：单 teacher、采样 token 级 PG，每 token 一个标量信号。
- 本 plugin：N teacher mixture、**full-vocab** forward-KL，每 token 在全词表上稠密。

teacher 装载基础设施可共用（同一个 `"teacher"` weight tag）；语义边界保留在 `--opsd-freeze-teacher` 这个旋钮上。

---

## Part 4 — 健康监控（TB / stdout 速查）

| 指标 | 健康范围 | 异常含义 |
|---|---|---|
| `rollout/raw_reward` | 0.3–0.9 | 太低 → student 太弱 / 数据太难 / 截断；太高 → group 全对，mixture 失多样性 |
| `rollout/truncated_ratio` | < 0.1 | 高 → response_len 设小了，参考 BUG.md #1 |
| `rollout/zero_std/count_0+count_1` | < `rollout-batch-size` | == batch 说明所有 group std=0，OPSD 输入退化 |
| `train/opsd_kl` | ≥ 0 单调向稳定值 | 走负 → BUG.md #3，确认 `--opsd-pointwise-kl-clip` 未误开 |
| `train/opsd_kl_clamped` | == `opsd_kl` | 不等 → 上一行违例时的镜像 |
| `train/opsd_w_entropy` | 0.3–0.8 | 0 → mixture 在 one-hot；1 → 完全均匀，β 太小或诊断 bug 重现；调 `--opsd-kl-weight` ±2× |
| `train/lr-pg_0/1` | == `--lr` | 0 → BUG.md #5 修了；若仍为 0，说明 `param_group["lr"]` 未由 scheduler 写入 |
| `train/grad_norm` | 0.1–1.0 | 太大 → KL clip 没生效或 RKL 过强；太小 → 训练信号弱 |

---

## TL;DR

- **默认行为**：method.md §6-9 主路径 + paper §3.2 per-position KL clip τ=0.05。frozen initial-policy teacher，full-vocab forward-KL，token_jsd 多样性，`rank` 归一 Conf。
- **Rollout 设计**：每 prompt 跑 K+1 条，1 当 student、K 当 candidate pool（学生不在池中）。
- **超参**：脚本 paper Table 6 对齐（lr 5e-6 / batch 32 / temp 1.1 / top-p 0.95 / top-k 20）；`--rollout-max-response-len` 因数学推理需要从 1024 调高到 8192。
- **项目改进**：全部 opt-in，详见 Part 3。
- **Bug 修复**：BUG.md 全部条目落代码，详见 Part 2 §B。
