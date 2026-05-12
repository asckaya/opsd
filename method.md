# Diverse Self-Privileged OPSD（详细算法）

## 1. 问题定义

给定输入问题：
x ∈ 𝒳

学生模型（待优化）：
π_θ(y | x)

目标：
利用 student 自身成功推理轨迹作为 privileged information，
进行 on-policy self-distillation。

---

## 2. On-policy 采样

从旧策略采样 K 条轨迹：

τ_j ~ π_{θ_old}(· | x),   j = 1...K

每条轨迹：

τ_j = (r_j, a_j)

---

## 3. 成功轨迹筛选

定义验证函数：

R(x, τ_j) = 1[Verify(a_j, a*) = 1]

构造成功集合：

B_x = { τ_j | R(x, τ_j) = 1 }

若 B_x 为空：
- 跳过
或
- 使用 GT solution 作为 fallback

---

## 4. 质量评分

定义：

s(τ) =
    1
    - η_l · Len(τ)/L_max
    - η_f · FormatPenalty(τ)
    + η_c · Conf(τ)

其中：

Conf(τ) =
    (1/|τ|) Σ_t log π_{θ_T}(τ_t | x, τ_<t)

保留：

B_x ← TopK(B_x, s(τ), K_b)

---

## 5. 多样性选择（核心）

目标：

P_x = argmax_{|S|=N} Σ_{i≠j} d(τ_i, τ_j)

### 距离定义

#### 方法1（轻量）
d_text(τ_i, τ_j) = 1 - cos(embed(τ_i), embed(τ_j))

#### 方法2（推荐）
d_dist(τ_i, τ_j) =
    (1/T) Σ_t JSD(q_i^t, q_j^t)

---

### 近似算法

k-center greedy：

1. τ_1 = argmax s(τ)
2. τ_{k+1} = argmax min distance

---

## 6. Teacher 构造

对每条 privileged trace：

q_k^t =
    π_{θ_T}(· | x, τ_k^+, y_<t)

student：

p_θ^t =
    π_θ(· | x, y_<t)

---

## 7. Teacher 权重（关键设计）

定义：

Δ_k^t = KL(q_k^t || p_θ^t)

h_k^t = H(q_k^t)

g_k^t = (1/(N-1)) Σ JSD(q_k^t, q_j^t)

---

权重：

w_k^t ∝ exp(
    -β Δ_k^t
    -γ h_k^t
    +ρ g_k^t
)

归一化：

w_k^t = softmax_k(...)

---

## 8. Mixture Teacher

q_mix^t =
    Σ_{k=1}^N w_k^t q_k^t

---

## 9. 蒸馏目标

### 主损失（推荐）

L_distill =
    (1/T) Σ_t KL(q_mix^t || p_θ^t)

等价：

L =
    - Σ_t Σ_v q_mix^t(v) log p_θ^t(v)

---

### 可选（辅助）

Reverse KL：

L_RKL =
    Σ KL(p_θ^t || q_mix^t)

最终：

L_total =
    L_distill + α L_RKL

α ≪ 1

---

## 10. 完整训练流程

For each x:

1. 采样 τ_j
2. 构造 B_x
3. 质量筛选
4. 多样性选择 → P_x
5. 采样 student rollout y
6. 对每个 token t：
    - 计算 q_k^t
    - 计算权重 w_k^t
    - 得到 q_mix^t
7. 计算 KL loss
8. 更新 θ

---

## 11. 关键优势

- 完全 on-policy
- teacher-free（无需更大模型）
- 利用 student 成功经验
- dense token-level learning
- 多路径 reasoning 蒸馏

---

## 12. 风险与对策

### 风险1：成功轨迹噪声
→ 质量评分 + TopK

### 风险2：teacher 太接近
→ 加 diversity term

### 风险3：teacher 太远
→ 加 KL 惩罚

### 风险4：mode collapse
→ 主用 forward KL

---

## 13. 推荐超参数

| 参数 | 推荐 |
|------|------|
| K | 8–32 |
| N | 2–4 |
| K_b | 8–16 |
| β | 0.5–2 |
| γ | 0.1–1 |
| ρ | 0.1–1 |
| λ_distill | 0.05–0.5 |

---

## 14. 一句话总结

student 成功轨迹  
→ 多样性特权信息  
→ 多 teacher mixture  
→ token-level KL 蒸馏
