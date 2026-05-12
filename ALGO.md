# OPSD 算法对齐说明

本文件分两部分：

1. **预期算法**（Part 1）——以 `method.md` 为准，并标注它和 `paper.md` 的关系。
2. **当前实现状态**（Part 2）——所有原本的未对齐项现在的修复情况；项目自有改进保留为 **opt-in**，默认行为与 method.md 对齐、超参与 paper.md 对齐。

历史版本记录的"未对齐项"已按用户要求全部修复，详见 Part 2 的"修复状态"列。

---

## Part 1 ─ 预期算法（Expected Algorithm）

> `paper.md` 的 OPSD 是单条 teacher 路径：teacher 直接看 ground-truth `y*`。
> `method.md` 在此基础上做了扩展：用 *学生自己跑出来的正确轨迹* 当作 privileged 信息，并允许多条混合。
> 也就是说：`method.md` ⊇ `paper.md`。`paper.md` 的训练循环可以看作 `method.md` 在 `N=1`、用 GT 当 privileged trace 时的特例。

### 1.1 总流程（method.md §10）

对每个 prompt `x`：

1. **On-policy rollout（§2）**：从旧策略 `π_{θ_old}(·|x)` 采样 `K` 条轨迹。
2. **成功筛选（§3）**：`B_x = { τ_j | R(x, τ_j) = 1 }`；空则可 fallback 到 GT。
3. **质量打分（§4）**：
   ```
   s(τ) = 1 − η_l · Len(τ)/L_max − η_f · FormatPenalty(τ) + η_c · Conf(τ)
   Conf(τ) = (1/|τ|) Σ_t log π_T(τ_t | x, τ_<t)
   ```
   保留 `B_x ← TopK(B_x, s, K_b)`。
4. **多样性选择（§5）**：k-center greedy；距离推荐 token-level JSD：
   `d_dist(τ_i,τ_j) = (1/T) Σ_t JSD(q_i^t, q_j^t)`。
5. **学生采样**：复用 rollout 中的某一条轨迹作为 `y`。
6. **Teacher 前向（§6）**：
   ```
   q_k^t = π_T(· | x, τ_k+, y_<t),    p^t = π_θ(· | x, y_<t)
   ```
   `π_T` 由 `paper.md §4.1` 锁定为 **initial policy**（frozen），不随训练更新。
7. **Teacher 混合权重（§7）**：
   ```
   Δ_k^t = KL(q_k^t ‖ p^t),   h_k^t = H(q_k^t),   g_k^t = (1/(N−1)) Σ_{j≠k} JSD(q_k^t, q_j^t)
   w_k^t ∝ exp(−β·Δ_k^t − γ·h_k^t + ρ·g_k^t)     ← softmax_k
   ```
8. **Mixture teacher（§8）**：`q_mix^t = Σ_k w_k^t · q_k^t`。
9. **蒸馏目标（§9）**：
   ```
   L_distill = (1/T) Σ_t KL(q_mix^t ‖ p^t)             (主损失，forward KL)
   L_total   = L_distill + α_RKL · Σ_t KL(p^t ‖ q_mix^t),    α_RKL ≪ 1   (可选 RKL 辅助)
   ```
   `paper.md §3.2` + Figure 4：**逐 (位置, 词表项) pointwise clipping** 对长训练稳定性至关重要。
   官方源码（`../OPSD/opsd_trainer.py:464`）实现是 `jsd.clamp(max=τ)`，等价于
   ```
   ℓ_{n,v} = q(v) · (log q(v) − log p(v))     ← forward KL 的 per-(pos,vocab) 项
   D_clip = (1/|ŷ|) Σ_n Σ_v min(ℓ_{n,v}, τ)
   ```
   `f-散度` 的写法 `ℓ = p_T·f(p_S/p_T)` 在 `f(u)=−log u` 时也等价于上式。

### 1.2 推荐超参（含来源）

| 来源 | 项 | 值 |
|---|---|---|
| method.md §13 | K（rollout 数） | 8–32 |
| method.md §13 | N（多样性选中） | 2–4 |
| method.md §13 | K_b（质量预筛） | 8–16 |
| method.md §13 | β（KL 权） | 0.5–2 |
| method.md §13 | γ（entropy 权） | 0.1–1 |
| method.md §13 | ρ（diversity 权） | 0.1–1 |
| method.md §13 | λ_distill（蒸馏权 α） | 0.05–0.5（在 L = L_RL + α·L_distill 框架下；纯蒸馏可取 1.0） |
| paper.md Table 6 | Learning rate | 5e-6 |
| paper.md Table 6 | Effective batch size | 32 |
| paper.md Table 6 | LoRA r / α | 64 / 128（论文用 LoRA；slime Megatron 路径走全微调） |
| paper.md Table 6 | Max completion length | 1024 |
| paper.md Table 6 | Generations per prompt | 1（paper 的简化版） |
| paper.md Table 6 | Sampling temperature | 1.1 |
| paper.md Table 6 | Training steps | 100 |
| paper.md Table 5 | 推荐风格 | **TM-off 学生 / TM-on 教师**（math token KL 信号最强） |
| paper.md Table 8 | Eval：MaxNewTokens / TM / TopP / TopK / Temp | 38912 / on / 0.95 / −1 / 1.0 |
| `../OPSD` 源码脚本 | per-(pos,vocab) clip τ | 0.05（`jsd_token_clip` in `run_opsd_1b.sh`） |
| `../OPSD` 源码脚本 | top_p / top_k for sampling | 0.95 / 20 |

---

## Part 2 ─ 当前实现状态（Implementation Status）

> 文件位置：`slime_plugins/opsd/`、`slime/utils/arguments.py`、`scripts/run_qwen3_1.7B_opsd*.sh`。
> 状态符号：✅ 已修复且默认对齐 / 🔧 已修复但保留 opt-in 旋钮 / 📌 文档/约定明确，无代码变更。

### A. 关键语义偏差 — 已修复

| 项 | 状态 | 修复 |
|---|---|---|
| A.1 OPSD loss 与 GRPO PG-loss 杂交 | ✅ + 🔧 | `plugin.py:loss_function` 默认是 **纯** `α·KL(q_mix‖p_θ)`（method.md §9 / paper Eq 8）。新增 `--opsd-mix-with-policy-loss`（默认 `False`）保留旧 hybrid 行为以做消融。`opsd_alpha` 现在严格是 method.md §9 的 `L_distill` 系数。 |
| A.2 默认不开 per-(pos,vocab) KL clip | ✅ | `--opsd-pointwise-kl-clip` 默认从 `None` 改成 **`0.05`**（与官方 `../OPSD/scripts/run_opsd_*.sh --jsd_token_clip 0.05` 一致；paper §3.2 Fig.4 强调长训练必需）。传 `<= 0`（如 `-1`）显式关闭，由 `arguments.py` 的后处理统一规整成 `None`。 |
| A.3 mixture-weight 计算有温度软化 | 📌 | `opsd_temperature` 默认 1.0，纯 raw 分布。文档（README / argparse help）现已写明"只对 Δ/h/g 的 softmax 起作用，不影响最终 KL 损失"。 |
| A.4 Conf 默认 rank 归一化 | 🔧（保留） | `--opsd-quality-conf-norm` 默认仍然是 **`rank`**。理由：method.md §4 只给公式，没说 Conf 的 numerical scaling；raw Conf（mean log-prob，-5..-1 nats）会盖过 `Len/L_max` 和 `Format`（都在 [0, 1]）一个量纲，让默认 `η_c = 0.5` 失去"权重"语义。`rank` 把它压到 [0, 1] 与其它项共享数轴。这也解释了为什么 method.md §13 推荐超参表 *不* 列 η_c——它的 scale 由这里的归一化方案决定。`raw` 是 opt-in：纯字面照搬 method.md 公式时用。 |
| A.5 method.md §9 的 RKL 辅助项缺失 | ✅ | 新增 `--opsd-rkl-weight`（默认 `0.0`）。打开后跑一个 vocab-parallel 的 `KL(p_θ‖q_mix)` 全 vocab 反向（`distillation.py:_VocabParallelRKLDiv`），加到总 loss 上并日志为 `opsd_rkl_loss`。method.md 推荐 `α_RKL ≪ 1`，所以默认关闭。 |

### B. paper.md Table 6 超参 — 脚本已对齐

修改 `scripts/run_qwen3_1.7B_opsd.sh` 与 `scripts/run_qwen3_1.7B_opsd_split.sh`：

| 项 | 老值 | 新值（paper Table 6） |
|---|---|---|
| `--lr` | `1e-6` | **`5e-6`** |
| `--global-batch-size` | `128` | **`32`** |
| `--rollout-max-response-len` | `2048` | **`1024`** |
| `--rollout-temperature` | `1.0` | **`1.1`** |
| `--rollout-top-p` | （未设） | **`0.95`** |
| `--rollout-top-k` | （未设） | **`20`** |
| `--opsd-diversity-metric` | `unigram_jsd`（脚本覆盖） | （删除覆盖，走默认 `token_jsd`） |
| `--opsd-temperature` | `1.0`（脚本显式传） | （删除，复用默认 1.0） |

`LoRA r/α`：slime 的 Megatron 路径用全微调，不走 LoRA。LR `5e-6` 对全微调可能偏大，但保持与 paper 一致；若需要更稳可在脚本里再降一档，但这就脱离 paper 复现轨了。

### C. method.md §13 范围 — 全部满足

| 项 | 脚本现值 | method.md §13 范围 | 备注 |
|---|---|---|---|
| K | 16 | 8–32 | ✓ |
| N | 4 | 2–4 | ✓ |
| K_b | 8 | 8–16 | ✓ |
| β (`opsd_kl_weight`) | 1.0 | 0.5–2 | ✓ |
| γ (`opsd_entropy_weight`) | 0.5 | 0.1–1 | ✓ |
| ρ (`opsd_diversity_weight`) | 0.5 | 0.1–1 | ✓ |
| `opsd_alpha`（纯蒸馏 loss 系数） | 1.0 | — | method.md §13 的 0.05–0.5 是在"L_RL + α·L_distill"框架下；现在我们是纯蒸馏，1.0 即"原 loss"，等价。 |

### D. 项目自有的工程改进 — 保留为 opt-in

| 改进 | 实现位置 | 触发方式 |
|---|---|---|
| `unigram_jsd` 多样性距离（提前 diversity 选择以减少 q-forward） | `selection.py:pairwise_unigram_jsd`、`distillation.py:prepare_teacher_outputs` | `--opsd-diversity-metric unigram_jsd`（默认 `token_jsd`） |
| Conf 跨候选归一化（`rank` 默认）；`zscore` / `minmax` / `raw` 可选 | `distillation.py:add_conf` | `--opsd-quality-conf-norm <mode>`（默认 `rank`） |
| `_VocabParallelKLDiv` 的 per-(pos,vocab) clip | `distillation.py` | `--opsd-pointwise-kl-clip <τ>`（默认 `0.05`） |
| sum-后 token-level KL clamp（与 paper 的 pointwise clip 不同的一个钝刀防御） | `distillation.py:distillation_loss` | `--opsd-jsd-token-clip <c>`（默认关闭） |
| 顶 K 词表近似计算 Δ/h/g、JSD 距离 | `distillation.py:mixture_weights`、`selection.py:_seq_jsd` | `--opsd-weight-top-k`、`--opsd-diversity-top-k`（保留默认） |
| Hybrid 法：在纯蒸馏之外叠加 GRPO PG-loss | `plugin.py` | `--opsd-mix-with-policy-loss`（默认 `False`） |
| RKL 辅助项 `α_RKL · KL(p_θ‖q_mix)` | `distillation.py:_VocabParallelRKLDiv` | `--opsd-rkl-weight <α_RKL>`（默认 0.0） |
| Mixture-weight 温度软化 | `distillation.py:mixture_weights` | `--opsd-temperature <T>`（默认 1.0，等价 raw） |

### E. 文档/接口收尾

- `slime_plugins/opsd/README.md`：超参表已更新到新默认值；Performance notes 中关于"unigram_jsd 是脚本默认"的说法已改正；新增"External teacher"一节，说明 `--no-opsd-freeze-teacher --use-opd --opd-type megatron --opd-teacher-load <path>` 接外部教师的用法（这不再是 OPSD，是普通 KD，用于明确边界）。
- `slime_plugins/opsd/{selection.py, distillation.py, rollout.py}`、README：把残留的 `metho.md` 链接改成 `method.md`（项目里实际文件名）。
- `slime_plugins/opsd/PLAN_FROZEN_TEACHER.md` 仍残留 `metho.md` 字样，是历史 plan 文档，未触碰。

---

## 与 slime 内置 `--use-opd` 的关系（解释）

slime 自带的 `--use-opd / --opd-kl-coef` 走的是 **paper §3.2 末尾的 Alternative objective (Eq 9)**——采样 token 的 policy-gradient 形态，把 `reverse_kl = log π_S − log π_T` 当 advantage 修正项 (`apply_opd_kl_to_advantages`)。它和我们的 plugin 走的 main objective 不一样：

- slime `--use-opd`：单 teacher、采样 token 级 PG 信号，一个 token 一个标量 reward。
- 本 plugin：N teacher mixture、**full-vocab** forward-KL 信号，每个 token 在全词表上稠密。

它们的 teacher 加载基础设施可以共用：本 plugin `actor.py:111-118` 已经允许 `--no-opsd-freeze-teacher --use-opd --opd-type megatron --opd-teacher-load <path>` 把外部 checkpoint 装进 `"teacher"` tag。但语义上："外部教师 + mixture full-vocab KL" 不再是 OPSD 自蒸馏，是普通 KD 的一个变体——按需自取。

---

## TL;DR

- **默认行为**：method.md / paper.md main objective 对齐——纯 full-vocab forward-KL 蒸馏，frozen 初始策略 teacher，per-(pos,vocab) clip τ=0.05，token_jsd 多样性，raw Conf。
- **超参**：脚本完全按 paper Table 6（LR 5e-6 / batch 32 / max-len 1024 / temp 1.1 / top-p 0.95 / top-k 20）。
- **项目自有改进**：全部保留为 opt-in 旋钮，默认关闭，详见 Part 2 §D。
