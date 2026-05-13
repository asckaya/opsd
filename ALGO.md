# OPSD 算法对齐说明

本文件分两部分:

1. **Part 1 — 算法规范**:method spec + paper 算法定义,本仓的 K+1 扩展。
2. **Part 2 — REF**:paper 关键公式、表格、实验细节(原文摘录,供查证)。

---

## Part 1 — 算法规范

> Paper OPSD 是**单条 teacher 路径**:teacher 直接看 ground-truth `y*`。
> 本仓在此基础上做 Diverse Self-Privileged 扩展:用 *学生自己跑出来的正确轨迹* 当作 privileged 信息,允许多条 mixture。
> 即 method ⊇ paper:paper 训练循环是 method 在 `N=1`、`τ_k = y*` 时的特例。

### 1.1 总流程

对每个 prompt `x`:

1. **On-policy rollout**:从旧策略 `π_{θ_old}(·|x)` 采样 **K+1** 条轨迹(`opsd_k+1` 条)。
   - 1 条作为 student `y`;剩下 K 条进 candidate pool。
   - 学生 **不在 candidate pool 中**,杜绝"student 恰好等于某条 τ_k"导致的 mixture 退化。
2. **成功筛选**:`B_x = { τ_j ∈ pool | R(x, τ_j) = 1 }`;空则 fallback 到 GT(`--opsd-fallback-to-gt`)。
3. **质量打分**:
   ```
   s(τ) = 1 − η_l · Len(τ)/L_max − η_f · FormatPenalty(τ) + η_c · Conf(τ)
   Conf(τ) = (1/|τ|) Σ_t log π_T(τ_t | x, τ_<t)
   ```
   保留 `B_x ← TopK(B_x, s, K_b)`。
4. **多样性选择**:k-center greedy;距离推荐 token-level JSD:
   `d(τ_i, τ_j) = (1/T) Σ_t JSD(q_i^t, q_j^t)`。选出 `N` 条 → `P_x`。
   - 候选距离另一种轻量备选:`d_text = 1 − cos(embed(τ_i), embed(τ_j))`(本仓未实现)。
5. **Teacher 前向**:
   ```
   q_k^t = π_T(· | x, τ_k+, y_<t),   p^t = π_θ(· | x, y_<t)
   ```
   `π_T` 锁定为 **initial policy**(frozen),不随训练更新。
6. **Mixture 权重**:
   ```
   Δ_k^t = KL(q_k^t ‖ p^t),   h_k^t = H(q_k^t),   g_k^t = (1/(N−1)) Σ_{j≠k} JSD(q_k^t, q_j^t)
   w_k^t = softmax_k(−β·Δ_k^t − γ·h_k^t + ρ·g_k^t)
   ```
   纯 raw 分布上做 softmax,没有温度项。
7. **Mixture teacher**:`q_mix^t = Σ_k w_k^t · q_k^t`。
8. **蒸馏目标**:
   ```
   L_distill = (1/T) Σ_t KL(q_mix^t ‖ p^t)             (主损失,forward KL,全词表)
   L_total   = L_distill + α_RKL · Σ_t KL(p^t ‖ q_mix^t)        (可选 RKL 辅助,α_RKL ≪ 1)
   ```
   **per-position KL clamp `τ=0.05`** 对长训练稳定性至关重要(官方 `--jsd_token_clip 0.05`)。

### 1.2 Paper Table 6 推荐超参

| 项 | 值 | 来源 |
|---|---|---|
| Learning rate | `5e-6` | Table 6 |
| Effective batch size | `32` | Table 6 |
| Max completion length | `1024`(数学推理可放宽到 4k–8k,paper 是短链场景) | Table 6 |
| Generations per prompt | `1`(paper 简化为 single-y);本仓 K+1 = 17 | Table 6 + 本仓扩展 |
| Sampling temperature | `1.1` | Table 6 |
| Top-p / Top-k | `0.95 / 20` | OPSD 官方训练脚本 |
| KL clip τ | `0.05` | `run_opsd_*.sh --jsd_token_clip 0.05` |
| Training steps | `100` | Table 6 |
| LoRA r / α | `64 / 128` | Table 6(slime 走全微调,不用 LoRA) |

### 1.3 推荐区间

| 项 | 范围 | 含义 |
|---|---|---|
| K | 8–32 | 候选池大小 |
| N | 2–4 | 多样性选中条数 |
| K_b | 8–16 | 质量预筛 top-K_b |
| β(KL 权) | 0.5–2 | mixture-weight 公式中 Δ 的系数 |
| γ(entropy 权) | 0.1–1 | mixture-weight 公式中 h 的系数 |
| ρ(diversity 权) | 0.1–1 | mixture-weight 公式中 g 的系数 |
| λ_distill | 0.05–0.5 | "L_RL + λ·L_distill" 框架下的蒸馏权;纯蒸馏时取 1.0 |

### 1.4 关键优势 / 风险对照

**优势**:① 完全 on-policy;② teacher-free(同模型双角色,无需更大老师);③ 利用 student 自有成功经验;④ dense token-level 学习信号;⑤ 多路径 reasoning 蒸馏。

**风险与对策**:

| 风险 | 对策 |
|---|---|
| 成功轨迹噪声 | 质量评分 `s(τ)` + TopK_b 预筛 |
| Teacher 与 student 太接近(无信号) | mixture 权重里 diversity 项 `+ρ·g` 推开 |
| Teacher 与 student 太远(KL 爆) | mixture 权重里 KL 项 `−β·Δ` 拉回 + per-position clip τ=0.05 |
| Mode collapse | 主损失用 forward KL(RKL 仅 opt-in) |

---

## Part 2 — REF(paper 关键摘录)

> 摘自 OPSD paper(Zhao et al., 2025),用于公式 / 数值查证。本仓默认行为以 method 扩展为准,本节仅作 single-teacher OPSD 的权威参考。

### 2.1 Algorithm 1

```
Require: ReasoningdatasetS = {(x_i, y*_i)}_{i=1..N}; language model p_θ; divergence D (e.g., JSD_β)
1: Let p_S(·|x) and p_T(·|x, y*) be the same model p_θ under different conditioning.
2: while not converged do
3:   Sample a minibatch B ⊂ S
4:   for all (x, y*) ∈ B do
5:     Sample on-policy response ŷ ∼ p_S(·|x)
6:     Compute the token-wise divergence along the student rollout:
        ℓ(x, y*) = D(p_T ‖ p_S)(ŷ|x)
                 = (1/|ŷ|) Σ_{n=1..|ŷ|} D( p_T(·|ŷ_<n, x, y*) ‖ p_S(·|ŷ_<n, x) )
7:   L_OPSD(θ) = (1/|B|) Σ_{(x,y*)∈B} ℓ(x, y*); update θ
```

### 2.2 关键公式

**Eq.6 — Trajectory-averaged token-wise divergence**(paper §3.2)

```
D(p_T ‖ p_S)(ŷ|x) = (1/|ŷ|) Σ_{n=1..|ŷ|} D( p_T(·|x, y*, ŷ_<n) ‖ p_S(·|x, ŷ_<n) )
```

**Eq.7 — Jensen-Shannon divergence with mixing weight β ∈ [0,1]**:

```
JSD_β(p_T ‖ p_S) = β · D_KL(p_T ‖ m) + (1−β) · D_KL(p_S ‖ m),    m = β·p_T + (1−β)·p_S
```

Paper 默认 `β = 0.5`,做过 fwd-KL / rev-KL / JSD_{0.5} 三者消融:JSD_{0.5} 最稳定,reverse-KL 与 JSD_{0.5} 在某些设置下"limited or negative"(paper §4.1)。

**Eq.8 — On-policy distillation expectation form**:

```
L(θ) = E_{(x,y*)∼S} [ E_{ŷ∼p_S(·|x)} [ D(p_T ‖ p_S)(ŷ|x) ] ]
```

梯度只回流 student logits(`stop_grad` on teacher)。

**Eq.9 — Sampled-token policy-gradient alternative**(paper §3.2 末尾):

```
A_n(x, ŷ) = log p_T(ŷ_n | x, y*, ŷ_<n) − log p_S(ŷ_n | x, ŷ_<n)         (advantage)

L(θ) = − E_{(x,y*)∼S} E_{ŷ∼p_S(·|x)} [ (1/|ŷ|) Σ_n A_n(x,ŷ) · log p_S(ŷ_n | x, ŷ_<n) ]
```

`A_n` 对 θ 视为常数;这是 slime `--use-opd` 实现的目标。和本 plugin 主路径(Eq.8)互斥。

### 2.3 Per-token pointwise divergence clipping(paper §3.2)

观察到 token-level divergence 高度偏斜:少数 stylistic tokens(maybe / however / therefore …)的 KL 远高于 math tokens。Paper 定义 per-(position, vocab) 贡献:

```
ℓ^{(f)}_{n,v} = p_T(v|·) · f( p_S(v|·) / p_T(v|·) )

D_clip(p_T ‖ p_S) = (1/|ŷ|) Σ_n Σ_v min( ℓ^{(f)}_{n,v}, τ )
```

Paper 用 `τ = 0.05`,**未对 τ 做调参**(原文:"We didn't conduct tuning for the clipping parameter τ, optimizing this hyperparameter may yield further performance gains within the same 100-step budget for larger models.")。

⚠️ 本仓不直接用这个 per-(pos,vocab) clip(单边裁剪让 sum-over-vocab 走负)。改用 paper 官方训练脚本 `--jsd_token_clip 0.05`(per-position clamp post sum-over-vocab),对应 `--opsd-jsd-token-clip`。

### 2.4 Table 6 — Training Configuration for GRPO and OPSD

| Parameter | GRPO | OPSD |
|---|---|---|
| LearningRate | 5e-6 | 5e-6 |
| EffectiveBatchSize | 32 | 32 |
| LoRA Rank (r) | 64 | 64 |
| LoRA Alpha (α) | 128 | 128 |
| LoRA TargetModules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | (同 GRPO) |
| MaxCompletionLength | 16,000 | **1024** |
| Generations per Prompt | 8 | **1** |
| Sampling Temperature | 1.2 | **1.1** |
| KL Coefficient (β) | 0.0 | – |
| Training Steps | 500 | 100 |

实验环境:8× A100 / H100,gradient checkpointing,FlashAttention2,AdamW,bfloat16。
> 注:OPSD MaxCompletionLength=1024 是 paper 短链场景;本仓数学推理任务实测需 ≥8192。

### 2.5 Table 7 — SFT Baseline Configuration

| Parameter | Value |
|---|---|
| LearningRate | 5e-6 |
| EffectiveBatchSize | 32 |
| LoRA r / α | 64 / 128 |
| MaxSequenceLength | 16000 |
| TrainingSteps | 100 |

### 2.6 Table 8 — Evaluation Parameters

| Parameter | Value |
|---|---|
| MaxNewTokens | 38912 |
| ThinkingMode | Enabled |
| Top-p | 0.95 |
| Top-k | -1 |
| Min-p | 0.0 |
| PresencePenalty | 0.0 |
| SamplesperPrompt | 12 |
| Temperature | 1.0 |

> Qwen3 官方推荐(paper §4):temperature=1.0,max_gen=38k。本仓 eval 配置以此为基准。

### 2.7 Thinking-Mode 选择(paper §4.3.1 / Appendix C)

Paper 在 Qwen3-1.7B/4B/8B 上扫了 student/teacher 的 ThinkingMode(TM-on/off)组合,结论:
- **TM-off student / TM-on teacher** 给出最大的 math-token KL 信号(Qwen3-1.7B: Math KL=0.14,显著高于其它三种组合)。
- Paper 主实验采用 **TM-off student + TM-on teacher**。

| Student / Teacher | Qwen3-1.7B (Style / Math / Other) | Qwen3-4B | Qwen3-8B |
|---|---|---|---|
| TM-off / TM-off | 0.68 / 0.12 / 0.11 | 0.61 / 0.06 / 0.10 | 0.56 / 0.05 / 0.11 |
| TM-on  / TM-off | 0.51 / 0.10 / 0.17 | 0.41 / 0.05 / 0.18 | 0.33 / 0.05 / 0.15 |
| TM-on  / TM-on  | 0.51 / 0.09 / 0.08 | 0.50 / 0.04 / 0.09 | 0.42 / 0.04 / 0.08 |
| **TM-off / TM-on** | **0.85 / 0.14 / 0.25** | **0.92 / 0.10 / 0.29** | **0.79 / 0.06 / 0.25** |

### 2.8 Token category 定义(paper Appendix C)

- **Style tokens**:maybe, perhaps, probably, possibly, let, okay, hmm, wait, because, since, so, thus, hence, therefore, but, however, although, though, yet, or, alternatively, instead, otherwise, actually, really, just, simply, basically, very, quite, pretty, rather, fairly, now, then, next, first, second, finally, try, see, check, note, recall, think, idea, strategy, approach, method, way, would, could, should, might, can, huge, large, big, small, tiny, interesting, tricky, complex, simple.
- **Math tokens**:exponential, exponent, power, base, logarithm, log, ln, compare, less, equal, larger, smaller, greater, factor, prime, divisible, equation, expression, formula, inequality, rational, irrational, real, integer, coefficient, variable, constant, sum, product, difference, quotient, fraction, denominator, numerator, root, square, cube, nth, maximum, minimum, optimize, bound.

### 2.9 引用

```
@article{zhao2025opsd,
  title  = {On-Policy Self-Distillation for Large Language Models},
  author = {Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  year   = {2025},
  url    = {https://github.com/siyan-zhao/OPSD}
}
```
