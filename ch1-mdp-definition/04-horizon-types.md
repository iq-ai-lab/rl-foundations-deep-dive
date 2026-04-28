# 04. Finite-Horizon vs Infinite-Horizon vs Average Reward

## 🎯 핵심 질문

- Finite-horizon, discounted infinite-horizon, average-reward MDP 의 수학적 차이는 무엇인가?
- 왜 $\gamma < 1$ 에서 $\gamma = 1$ 로 바뀌면 이론이 완전히 달라지는가?
- Average-reward 의 정의는 무엇이고, 왜 Cesaro 극한을 사용하는가?
- Finite-horizon MDP 에서 value 는 왜 time-dependent $V_t(s)$ 인가?
- Blackwell optimality 와 canonical form 은 무엇인가?

---

## 🔍 왜 이 분류가 필수인가

대부분의 교재는 discounted infinite-horizon MDP 만 다룹니다. 하지만 현실의 문제는:

- **Finite-horizon**: 고정된 기한이 있는 프로젝트 (3개월 계획, 시험 준비)
- **Discounted**: 먼 미래의 reward 를 적게 봄 (경제학, 금융)
- **Average-reward**: 장기 평균 효율 (manufacturing, network routing — 무한 운영)

각각의 **optimal policy 구조가 다릅니다:**

```
Finite-horizon       → V_t(s) depends on time remaining
Discounted           → V(s) only depends on state (stationary optimal)
Average-reward       → V(s) + h(s) (potential-based value) 더 복잡
```

또한 **$\gamma < 1$ 이라는 가정이 깨지는 순간** ($\gamma = 1$), Bellman contraction 이 성립하지 않습니다. 이를 해결하는 수학적 도구들이 다릅니다.

---

## 📐 수학적 선행 조건

- **Ch1-01~03**: MDP 정의, Markov property, policy types
- **Ch2**: Bellman equation (특히 contraction property)
- **Analysis**: Limit superior, Cesaro convergence, limsup/liminf
- **(Optional) Probability Theory**: Concentration inequality

---

## 📖 직관적 이해

### Horizon 에 따른 의사결정 구조

```
Finite-horizon (T steps remaining):
  t=0     t=1     ...     t=T-1   t=T
  s₀ ─→  s₁ ─→  ...      sₜ ─→  STOP
  Time matters! 남은 시간이 결정에 영향
  
Infinite-horizon discounted (γ < 1):
  s₀ ─→  s₁ ─→  s₂ ─→  ...
  Time doesn't matter (stationary), 
  but far future discounted
  
Average-reward (long-run average):
  t=1      t=2      ...    t→∞
  lim (1/T) Σ R_t
  Time-scale neutral, long-run efficiency
```

### 왜 $\gamma = 1$ 이 문제인가?

Discounted case: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k} \leq \frac{R_{\max}}{1-\gamma}$ (bounded, $\gamma < 1$)

$\gamma = 1$ case: $G_t = \sum_{k=0}^{\infty} R_{t+k}$ (unbounded, diverges if $R > 0$)

→ Value function 자체가 무한대로 갈 수 있음 → Bellman 의 $\gamma$-contraction 이 성립하지 않음

**해결책**: average-reward 로 정의 하면 다시 bounded 문제로 변환 가능.

### Cesaro Sum 의 역할

```
Path: R₁, R₂, R₃, ..., Rₜ, ...

Discounted: G = Σ γ^k R_k
  (weights: 1, γ, γ², ...) → exponentially decaying

Cesaro average: J = lim_{T→∞} (1/T) Σ_{t=1}^T R_t
  (weights: 1/T, 1/T, ..., 1/T) → uniform (democratic)
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Finite-Horizon MDP

MDP $(\mathcal{S}, \mathcal{A}, P, R, T, \rho_0)$ 여기서:
- $T < \infty$ (terminal time)
- Trajectory: $(s_0, a_0, \ldots, s_{T-1}, a_{T-1}, s_T)$ (정확히 $T+1$ states)
- Reward: $\sum_{t=0}^{T-1} R_t(s_t, a_t)$ (or with discount $\gamma$)

**Value function**: Time-dependent

$$V_t^\pi(s) := \mathbb{E}\left[\sum_{k=t}^{T-1} \gamma^{k-t} R_k(s_k, a_k) \,\Big|\, s_t = s\right]$$

특별히 $V_T^\pi(s) = 0$ (terminal, no more steps).

**Bellman**:
$$V_t^\pi(s) = \mathbb{E}[R_t(s, a) + \gamma V_{t+1}^\pi(s') | s, a \sim \pi(\cdot | t, s)]$$

$t$-dependent!

### 정의 4.2 — Discounted Infinite-Horizon MDP

MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ where $\gamma \in [0, 1)$.

**Return**:
$$G_t := \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$

**Value function** (time-independent):
$$V^\pi(s) := \mathbb{E}[G_t | s_t = s] = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k} \,\Big|\, s_t = s\right]$$

**Bellman**:
$$V^\pi(s) = \mathbb{E}[R(s, a) + \gamma V^\pi(s') | s, a \sim \pi]$$

(No $t$ dependence)

### 정의 4.3 — Average-Reward MDP

MDP $(\mathcal{S}, \mathcal{A}, P, R, \rho_0)$ (no discount).

**Average return**:
$$J^\pi(\rho_0) := \limsup_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[R(s_t, a_t) | \pi, \rho_0]$$

(Cesaro/Chisini average)

**Stationary-average value**: 존재한다면,
$$J^\pi = \limsup_{T \to \infty} (1/T) \mathbb{E}[\sum_t R_t]$$

이때 **value decomposition**:
$$V^\pi(s) = J^\pi + h^\pi(s)$$

where $J^\pi$ = average per-step reward, $h^\pi(s)$ = potential/bias term.

---

## 🔬 정리와 증명

### 정리 4.1 — Finite-Horizon 의 최적 정책은 Time-Dependent Markovian

Finite-horizon MDP, $\gamma \leq 1$ 임의:

**Optimal value**:
$$V_t^*(s) = \max_a [R_t(s, a) + \gamma \mathbb{E}[V_{t+1}^*(s') | s, a]]$$

with $V_T^*(s) = 0$.

**Backward induction**으로 unique solution. Greedy:
$$\pi_t^*(s) = \arg\max_a [R_t(s, a) + \gamma \mathbb{E}[V_{t+1}^*(s') | s, a]]$$

는 optimal deterministic policy.

**증명**: 
Dynamic programming principle (optimality principle). $t=T-1$ 부터 backward:
- $V_{T-1}^*(s) = \max_a R_{T-1}(s, a)$ (greedy at terminal)
- Inductively: assuming $V_{t+1}^*$ optimal, $V_t^*$ follows from Bellman

$\square$

### 정리 4.2 — Discounted $\gamma < 1$ 의 필요충분성

$|R| \leq R_{\max}$ bounded reward 하에서:

$$\text{Bellman operator } T^* \text{ is } \gamma\text{-contraction}$$
$$\Leftrightarrow \gamma \in [0, 1)$$

**증명** ($\Rightarrow$): Banach fixed point theorem 적용 위해 $\gamma < 1$ 필수 (Ch2 참고).

**증명** ($\Leftarrow$): $\gamma = 1$ 이면 $T^* V = \max_a [\cdots]$ 는 contraction 아님. 예: 상수 보상 reward $R \equiv 1$ 면 $V^*(s) = \infty$ (diverges).

$\square$

### 정리 4.3 — Average-Reward 에서의 Blackwell Optimality

Policy $\pi$ 가 **Blackwell optimal** $\Leftrightarrow$ $\pi$ 가 모든 $\gamma \in [0, 1)$ 에 충분히 가까운 값에 대해 $\gamma$-discounted optimal.

**Theorem (Blackwell 1962)**: Finite MDP 에서:
1. Blackwell optimal policy 존재
2. 그것은 average-reward optimal 도 됨
3. 역으로 average-reward optimal 중 일부는 Blackwell optimal

**증명 sketch**: $\gamma \to 1$ 극한에서:
$$V^\gamma(s) = \frac{1}{1-\gamma} [J^\pi + (1-\gamma) h^\pi(s) + O((1-\gamma)^2)]$$

Blackwell optimality = 모든 $\gamma$ 에서 best → average-reward term $J^\pi$ 먼저, ties break by $h^\pi$ term.

$\square$

### 정리 4.4 — Canonical Form for Average-Reward DP

Finite MDP, average-reward setting에서:

$$h^\pi(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} (R(s_t, a_t) - J^\pi(\pi)) \,\Big|\, s_0 = s, \pi\right]$$

**Value iteration**:
$$V_{n+1}(s) = \max_a [R(s, a) - \beta_n + \mathbb{E}[V_n(s') | s, a]]$$

where $\beta_n = \max_s V_n(s)$ (normalization, prevents drift).

**증명**: Centered reward $(R - \beta)$ 로 변환 → bounded effective reward → convergence to potential $h^\pi$.

$\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Finite-Horizon: Backward Induction

```python
import numpy as np

# 3-state, 2-action, T=5 finite MDP
n_states, n_actions, T = 3, 2, 5

# Time-dependent reward
R = np.random.randn(T, n_states, n_actions)
R = np.clip(R, -1, 5)  # Make positive for clarity

# Time-dependent transition (simple: mostly stay)
P = np.zeros((T, n_states, n_actions, n_states))
for t in range(T):
    for s in range(n_states):
        for a in range(n_actions):
            probs = [0.7 if i == s else 0.15 for i in range(n_states)]
            P[t, s, a, :] = probs

gamma = 0.9

# Backward induction
V = np.zeros((T+1, n_states))
V[T, :] = 0  # Terminal value

for t in range(T-1, -1, -1):
    Q = R[t] + gamma * np.einsum('san,n->sa', P[t], V[t+1])
    V[t] = Q.max(axis=1)
    print(f"t={t}: V[{t}] = {V[t]}")

print(f"\n✓ Finite-horizon value computed by backward induction")
print(f"  V[0] (optimal from start) = {V[0]}")
```

### 실험 2 — Discounted Infinite-Horizon vs γ=1 Divergence

```python
# Demonstrate why γ < 1 is necessary

# Simple 2-state MDP
n_states = 2
P = np.array([
    [[1.0, 0.0],    # state 0: always stay
     [1.0, 0.0]],
    [[0.0, 1.0],    # state 1: always go to 0
     [0.0, 1.0]]
])
R = np.array([
    [1.0, 1.0],     # R(0, a) = 1
    [1.0, 1.0]      # R(1, a) = 1
])

gammas = [0.5, 0.9, 0.99, 0.999]  # Approaching 1

for gamma in gammas:
    V = np.zeros(n_states)
    for iteration in range(2000):
        Q = R + gamma * np.einsum('san,n->sa', P, V)
        V_new = Q.max(axis=1)
        if np.linalg.norm(V_new - V) < 1e-10:
            break
        V = V_new
    
    max_v = V.max()
    print(f"γ = {gamma:6.3f}:  V_max = {max_v:10.4f},  R_max/(1-γ) = {1.0/(1-gamma):10.4f}")

# Try γ = 1 (should diverge or behave strangely)
gamma = 1.0
V = np.zeros(n_states)
for iteration in range(20):
    Q = R + gamma * np.einsum('san,n->sa', P, V)  # 0*∞ issue avoided but still problem
    V_new = Q.max(axis=1)
    print(f"Iteration {iteration}: V = {V_new}")
    if V_new.max() > 1000:
        print("  Diverging! γ=1 breaks contraction.")
        break
    V = V_new

print(f"\n✓ γ < 1 keeps V bounded, γ=1 diverges")
```

### 실험 3 — Average-Reward: Cesaro Convergence

```python
# Generate a trajectory and compute average reward

# 5-state ergodic MDP
n_states = 5
n_actions = 2
np.random.seed(42)

P = np.random.rand(n_states, n_actions, n_states)
P /= P.sum(axis=2, keepdims=True)
R = np.random.randn(n_states, n_actions)

# Random policy
pi = np.ones((n_states, n_actions)) / n_actions

# Simulate long trajectory
s = 0
trajectory_rewards = []
for t in range(10000):
    a = np.random.choice(n_actions, p=pi[s])
    reward = R[s, a]
    trajectory_rewards.append(reward)
    s = np.random.choice(n_states, p=P[s, a])

# Cesaro average
rewards = np.array(trajectory_rewards)
cumsum = np.cumsum(rewards)
cesaro_avg = cumsum / np.arange(1, len(cumsum) + 1)

# Plot convergence
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# First 1000 steps
ax1.plot(cesaro_avg[:1000], alpha=0.7)
ax1.set_xlabel('Time step t')
ax1.set_ylabel('(1/t) Σ R_k')
ax1.set_title('Cesaro Average Convergence (first 1000 steps)')
ax1.grid(True, alpha=0.3)

# All 10000 steps
ax2.plot(cesaro_avg, alpha=0.7)
ax2.set_xlabel('Time step t')
ax2.set_ylabel('(1/t) Σ R_k')
ax2.set_title('Cesaro Average (all 10000 steps)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=cesaro_avg[-1], color='r', linestyle='--', label=f'Limit ≈ {cesaro_avg[-1]:.4f}')
ax2.legend()

plt.tight_layout()
# plt.savefig('cesaro_convergence.png')
# plt.show()

print(f"Average reward (Cesaro): {cesaro_avg[-1]:.6f}")
print(f"Standard error: {cesaro_avg.std() / np.sqrt(len(cesaro_avg)):.6f}")
print(f"✓ Cesaro average converges to J^π")
```

### 실험 4 — Finite vs Infinite Optimality 비교

```python
# Compare finite-horizon policy with infinite-horizon policy

n_s, n_a = 4, 2
gamma = 0.95
T = 100  # Finite horizon

np.random.seed(123)
P = np.random.rand(n_s, n_a, n_s)
P /= P.sum(axis=2, keepdims=True)
R = np.random.randn(n_s, n_a)

# 1. Finite-horizon optimal (time-dependent)
V_fin = np.zeros((T+1, n_s))
for t in range(T-1, -1, -1):
    Q = R + gamma * np.einsum('san,n->sa', P, V_fin[t+1])
    V_fin[t] = Q.max(axis=1)

# 2. Infinite-horizon optimal (time-independent)
V_inf = np.zeros(n_s)
for _ in range(1000):
    Q_inf = R + gamma * np.einsum('san,n->sa', P, V_inf)
    V_inf_new = Q_inf.max(axis=1)
    if np.linalg.norm(V_inf_new - V_inf) < 1e-10:
        break
    V_inf = V_inf_new

print(f"Finite-horizon T={T} vs Infinite-horizon (γ={gamma}):")
print(f"\nAt t=0 (beginning of finite horizon):")
print(f"  V_finite[0] = {V_fin[0]}")
print(f"  V_infinite  = {V_inf}")
print(f"  Difference: {np.abs(V_fin[0] - V_inf).max():.6f}")

print(f"\nAt t={T//2} (middle of finite horizon):")
print(f"  V_finite[{T//2}] = {V_fin[T//2]}")
print(f"  Difference from V_inf: {np.abs(V_fin[T//2] - V_inf).max():.6f}")

print(f"\n✓ As T→∞, finite-horizon approaches infinite-horizon")
```

---

## 🔗 후속 레포와의 연결

- **Ch2 Bellman Equation**: Discounted 의 contraction property 유도는 $\gamma \in [0, 1)$ 근거
- **Ch3 Value Iteration**: 수렴 rate $O(\gamma^k)$ 는 $\gamma < 1$ 에 의존
- **Advanced RL**: Average-reward 알고리즘 (differential reward) 은 특별 취급 (Ch4)
- **Model-Free RL**: TD, Q-learning 도 discounted (대부분), average-reward variant 별도

---

## ⚖️ 가정과 한계

| Horizon type | 가정 | 한계 | 대응 |
|--------------|------|------|------|
| Finite | $T < \infty$ fixed | 정확히 언제 끝나는가? | Episodic task definition |
| Discounted | $\gamma < 1$ | 먼 미래 무시 (myopic) | $\gamma \to 1$ 극한 분석 (Ch4) |
| Average-reward | $J^\pi$ 존재 (ergodic) | Non-ergodic chains? | Communicating class 분해 필요 |

---

## 📌 핵심 정리

| MDP Type | Return | Value | Bellman | Optimal Policy |
|----------|--------|-------|---------|-----------------|
| Finite-horizon | $\sum_{t=0}^{T-1} \gamma^t R_t$ | $V_t(s)$ time-dependent | $V_t = T_t V_{t+1}$ | Backward induction |
| Discounted | $\sum_{t=0}^{\infty} \gamma^t R_t$ | $V(s)$ stationary | $V = T^* V$ | VI/PI |
| Average-reward | $\lim (1/T)\sum R_t$ | $V(s) = J + h(s)$ | Canonical form | Blackwell optimal |

**Key constraint**: $\gamma \in [0, 1)$ $\Rightarrow$ bounded convergence. $\gamma = 1$ $\Rightarrow$ need average-reward or episodic.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Finite-horizon MDP 에서 $T=3$ 일 때, 2-state system:

| $s$ | $a$ | $R(s, a)$ | $P(s' \| s, a)$ |
|-----|-----|-----------|-----------------|
| 0 | 0 | 1 | $[0.5, 0.5]$ |
| 0 | 1 | 2 | $[0.0, 1.0]$ |
| 1 | 0 | 0 | $[1.0, 0.0]$ |
| 1 | 1 | 3 | $[0.5, 0.5]$ |

Backward induction 으로 $V_0(0), V_0(1)$ 을 구하시오. ($\gamma = 1$)

<details>
<summary>해설</summary>

**Terminal**: $V_3(s) = 0$ for all $s$.

**t=2**:
$$V_2(s) = \max_a R(s, a)$$
- $V_2(0) = \max(1, 2) = 2$
- $V_2(1) = \max(0, 3) = 3$

**t=1**:
$$V_1(s) = \max_a [R(s, a) + \sum_s P(s' | s, a) V_2(s')]$$
- State 0: 
  - Action 0: $1 + 0.5 \cdot 2 + 0.5 \cdot 3 = 1 + 2.5 = 3.5$
  - Action 1: $2 + 1.0 \cdot 3 = 5$
  - Max: $V_1(0) = 5$
- State 1:
  - Action 0: $0 + 1.0 \cdot 2 = 2$
  - Action 1: $3 + 0.5 \cdot 2 + 0.5 \cdot 3 = 3 + 2.5 = 5.5$
  - Max: $V_1(1) = 5.5$

**t=0**:
$$V_0(s) = \max_a [R(s, a) + V_1(s')]$$
- State 0:
  - Action 0: $1 + 0.5 \cdot 5 + 0.5 \cdot 5.5 = 1 + 5.25 = 6.25$
  - Action 1: $2 + 1.0 \cdot 5.5 = 7.5$
  - Max: $V_0(0) = 7.5$
- State 1:
  - Action 0: $0 + 1.0 \cdot 5 = 5$
  - Action 1: $3 + 0.5 \cdot 5 + 0.5 \cdot 5.5 = 3 + 5.25 = 8.25$
  - Max: $V_0(1) = 8.25$

**Answer**: $V_0(0) = 7.5, V_0(1) = 8.25$. $\square$

</details>

**문제 2** (심화): Discounted infinite-horizon 과 finite-horizon 의 관계를 설명하시오. $T \to \infty$ 에서:
$$\lim_{T \to \infty} V_0^{\text{(finite-T)}}(s) = V^\infty(s)?$$

<details>
<summary>해설</summary>

**Yes, under conditions:**

**Claim**: $|R| \leq R_{\max}$, $\gamma < 1$, bounded state space 이면:
$$\lim_{T \to \infty} V_0^{(T)}(s) = V^\infty(s)$$

**Proof idea**:
- Finite-horizon: $V_0^{(T)} = \max_a [R(s, a) + \gamma \mathbb{E}[V_1^{(T)}(s') | s, a]]$
- As $T \to \infty$, tail discount $\gamma^T \to 0$ (since $\gamma < 1$)
- Tail effect vanishes → approaches infinite-horizon value
- Convergence rate: $|V_0^{(T)} - V^\infty| \leq O(\gamma^T)$

**Example**: If $R = 1$ everywhere, $\gamma = 0.9$:
- Infinite-horizon: $V = 1 + 0.9 V$ → $V = 10$
- Finite T: $V_0^{(T)} = 1 + 0.9(1 + 0.9(1 + \cdots))$ up to $T$ steps = $1 \cdot (1 - 0.9^T)/(1 - 0.9) = 10(1 - 0.9^T)$
- As $T \to \infty$: $10(1 - 0) = 10$ ✓

**Conclusion**: Finite-horizon is **special case** of infinite-horizon when T is large. $\square$

</details>

**문제 3** (논문 비평): Howard & Veinott (1966) 의 canonical form 은 average-reward DP 의 핵심이다. 그들이 왜 value decomposition $V(s) = J + h(s)$ 를 도입했는가? Modern policy gradient (PG) 에서도 이런 decomposition 이 나타나는가?

<details>
<summary>해설</summary>

**Canonical form 의 역할**:
$$V^\pi(s) = J^\pi + h^\pi(s)$$

where:
- $J^\pi = \lim_T (1/T) \sum_t R_t$ = average per-step reward
- $h^\pi(s) = $ "advantage relative to average" (potential-based bias)

**왜 도입?**
1. Average-reward value 가 unbounded → 분해로 bounded parts 분리
2. Policy comparison: 우선 $J^\pi$ 비교 (main), $h^\pi$ 로 ties break
3. Value iteration 에서: $h$ 에만 수렴하면 됨 (main value 빼고)

**Modern RL 에서의 연계**:

Policy Gradient theorem 에서:
$$\nabla_\theta J(\pi_\theta) \propto \mathbb{E}[A^\pi(s, a) \nabla_\theta \log \pi(a | s; \theta)]$$

where $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ = advantage.

이것이 **baseline-corrected** advantage → 본질적으로 Howard 의 decomposition 과 동일 구조!

**결론**: 
- Average-reward: $V = J + h$ decomposition 필수
- Policy Gradient: $A = Q - V$ baseline correction 동일 개념
- 현대 RL 의 baseline 사용 가 Howard 로부터 유래

$\square$

</details>

---

[◀ 이전: 03. Policy 의 종류와 Stationary Policy 충분성](./03-policy-types.md) | [📚 README](../README.md) | [다음 ▶: 05. POMDP 와 Belief State](./05-pomdp.md)
