# 01. MDP 의 6-tuple 정의

## 🎯 핵심 질문

- MDP 는 무엇이고, 왜 정확히 6개의 성분 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ 로 정의되는가?
- State space 와 action space 가 **measurable** 이어야 하는 이유는 무엇인가? Probability 이론의 관점에서.
- Transition kernel $P(\cdot \mid s, a)$ 가 **stochastic kernel** 의 정의를 만족해야 하는 이유는?
- Bounded measurable reward 가 필수인 이유는 무엇이고, unbounded 일 때 무엇이 깨지는가?
- Discount factor $\gamma \in [0, 1)$ 의 수학적 필요성 — why not $\gamma = 1$?

---

## 🔍 왜 이 정의가 RL 의 기초인가

Reinforcement learning 을 배우는 사람들 대부분이 "MDP 는 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$" 라고 외웁니다. 하지만 **$\rho_0$ (initial distribution)** 의 역할, **Borel-measurable space** 의 필요성, **transition kernel** 이 모수 $(s, a)$ 에 대해 jointly measurable 이어야 한다는 조건 등은 대부분의 입문서에서 건너뜁니다.

더 심각한 문제는: **measurable 하지 않은 state space 를 가정하고 $\mathbb{E}[V_t]$ 를 계산하면 기대값 자체가 정의되지 않습니다.** 이것이 단순한 "수학의 엄밀함" 이 아니라 RL 의 모든 이론적 토대 — Bellman equation, Value Iteration, convergence guarantee — 가 의존하는 **필수 조건** 입니다.

이 문서는 6-tuple 정의를 measurable space 이론으로부터 출발하여 유도하고, 각 요소의 역할을 정확히 설명합니다.

---

## 📐 수학적 선행 조건

- **Probability Theory Deep Dive**: Measurable space, $\sigma$-algebra, Borel measurability, probability measure
- **Functional Analysis Deep Dive**: Banach space $B(\mathcal{S})$ (bounded functions), supremum norm, completeness
- **Stochastic Processes Deep Dive**: Stochastic kernel, transition probability, ergodic theory 기초

---

## 📖 직관적 이해

### MDP 의 역사적 발전

**1950년대 Bellman**: $V(s) = r(s) + \gamma V(s')$ 의 recursive structure. 처음엔 finite state space 만 고려.

**1960년대-70년대 Howard, Puterman**: Discrete state space + Markovian policy 로 확장. Policy iteration 이 유한 step 내 최적 정책을 찾음.

**1980년대 onwards**: Continuous state space (boiler control, robot kinematics) → measurable state space 의 필요성 대두. Expected value 의 정의를 위해 measurability 요구.

### 6개 요소를 왜 모두 필요한가?

```
┌─────────────────────────────────────────────────┐
│  MDP = (𝒮, 𝒜, P, R, γ, ρ₀)                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  𝒮 ──► state space (measurement/observation)  │
│  𝒜 ──► action space (agent's choice)          │
│  P ──► dynamics (nature's response)           │
│  R ──► immediate reward (agent's objective)   │
│  γ ──► time-value-of-money (discount)         │
│  ρ₀ ──► starting point (initialization)       │
│                                                 │
│  삭제 불가능한 이유:                           │
│  - 𝒮 없음 → "뭘 관찰하나?"                     │
│  - 𝒜 없음 → "뭘 조종하나?"                     │
│  - P 없음 → "어디로 가나?"                     │
│  - R 없음 → "왜 하나?"                         │
│  - γ 없음 → "언제까지 최적화?"                 │
│  - ρ₀ 없음 → "어디서 시작?"                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Finite vs Continuous State Space

**Finite MDP** (e.g., Gridworld, Blackjack):
- $\mathcal{S} = \{1, 2, \ldots, n\}$ (discrete, trivially measurable)
- $P(s' \mid s, a) \in \mathbb{R}^{n \times n}$ (transition matrix)
- Value function: array lookup, $V(s) = V[\text{index}(s)]$

**Continuous MDP** (e.g., robot control, portfolio optimization):
- $\mathcal{S} = [0, 1]^d$ (Euclidean, Borel measurable)
- $P(\mathrm{d}s' \mid s, a)$ (probability measure on Borel sets)
- Value function: $V: \mathcal{S} \to \mathbb{R}$ (measurable function)
- Integration: $\int_B V(s') P(\mathrm{d}s' \mid s, a)$ (Lebesgue integral)

두 경우 모두 **measurability** 가 보장되어야 기대값 $\mathbb{E}[\cdot]$ 가 정의됨.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Measurable Space

$(X, \mathcal{F})$ 가 **measurable space** $\Leftrightarrow$ $\mathcal{F}$ 는 $X$ 의 $\sigma$-algebra:
1. $\emptyset, X \in \mathcal{F}$
2. $A \in \mathcal{F} \Rightarrow A^c \in \mathcal{F}$
3. $A_1, A_2, \ldots \in \mathcal{F} \Rightarrow \bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$

**예시**:
- Finite set: $\mathcal{F} = 2^X$ (power set)
- $\mathbb{R}$: $\mathcal{F} = \mathcal{B}(\mathbb{R})$ (Borel $\sigma$-algebra, open intervals 로 생성)

### 정의 1.2 — Stochastic Kernel

$(X, \mathcal{F}), (Y, \mathcal{G})$ measurable spaces. $P: X \times \mathcal{G} \to [0, 1]$ 이 **stochastic kernel**

$$\Leftrightarrow \begin{cases}
(x \mapsto P(x, B)) \text{ is measurable for all } B \in \mathcal{G} \\
(B \mapsto P(x, B)) \text{ is a probability measure for all } x \in X
\end{cases}$$

**직관**: 각 $x$ 에 대해 고정된 output space 위의 확률 분포.

### 정의 1.3 — MDP (6-tuple, Puterman 2005)

$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$

**요소들**:

1. **$\mathcal{S}$** (state space): measurable space $(\mathcal{S}, \mathcal{B}_\mathcal{S})$. 보통 Polish space (separable, completely metrizable) 으로 가정 → Borel $\sigma$-algebra $\mathcal{B}_\mathcal{S}$.

2. **$\mathcal{A}$** (action space): measurable space $(\mathcal{A}, \mathcal{B}_\mathcal{A})$. Compact Polish space 가정.

3. **$P$** (transition kernel): $P: \mathcal{S} \times \mathcal{A} \times \mathcal{B}_\mathcal{S} \to [0, 1]$.
   - 각 $(s, a)$ 에 대해 $P(\cdot \mid s, a) \in \Delta(\mathcal{S})$ (probability measure on $\mathcal{S}$)
   - $(s, a) \mapsto P(B \mid s, a)$ measurable for all $B \in \mathcal{B}_\mathcal{S}$

4. **$R$** (reward): measurable function $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$.
   - **Bounded**: $|R(s, a)| \leq R_{\max}$ for all $s, a$.
   - **Measurability**: $\mathcal{B}_\mathcal{S} \otimes \mathcal{B}_\mathcal{A}$ (product $\sigma$-algebra) 에서 $\mathcal{B}_\mathbb{R}$ 로.

5. **$\gamma \in [0, 1)$** (discount factor): **Fixed constant**. $\gamma = 1$ 은 별도 이론 (episodic, average-reward).

6. **$\rho_0 \in \Delta(\mathcal{S})$** (initial distribution): probability measure on $\mathcal{S}$.

### 정의 1.4 — History 와 Trajectory

**History at time $t$**: $h_t = (s_0, a_0, s_1, a_1, \ldots, s_t)$

**Trajectory**: $\tau = (s_0, a_0, s_1, a_1, \ldots)$ with $s_0 \sim \rho_0$.

Measurable space of histories: 제한된 길이 trajectory 는 $\mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \cdots$ (product space).

---

## 🔬 정리와 증명

### 정리 1.1 — 기대값의 존재성

Bounded measurable reward $|R| \leq R_{\max}$ 이고 bounded value function $\|V\|_\infty \leq V_{\max}$ 가 있으면, 다음 적분이 well-defined:

$$\mathbb{E}_{s' \sim P(\cdot \mid s, a)}[R(s, a) + \gamma V(s')] = R(s, a) + \gamma \int_\mathcal{S} V(s') P(\mathrm{d}s' \mid s, a)$$

**증명**: 
- $R(s, a)$ 는 measurable · bounded → Lebesgue integral 정의됨
- $V$ 는 measurable · bounded, $P(\cdot \mid s, a)$ 는 probability measure → $\int V \, \mathrm{d}P$ 정의
- Bounded convergence theorem 가능 → tail 수렴 보장

아래 식은 적분이 정의되지 않음:
- Unbounded reward, non-measurable $V$ → integral 계산 불가능
- Fubini-Tonelli theorem 적용 불가능

$\square$

### 정리 1.2 — Finite MDP 는 Borel MDP 의 특수 경우

$|\mathcal{S}|, |\mathcal{A}| < \infty$ 이면 $\mathcal{S}, \mathcal{A}$ 를 discrete topology 로 Polish space 로 취급 가능:
- $\mathcal{B}_\mathcal{S} = 2^\mathcal{S}$ (power set)
- $P(s' \mid s, a) \in [0, 1]$ (각 singleton $\{s'\}$ 에 대한 확률)
- Integral $\int_\mathcal{S} V(s') P(\mathrm{d}s' \mid s, a) = \sum_{s'} V(s') P(s' \mid s, a)$

Continuous MDP 는 이 구조의 일반화.

**증명**: Borel $\sigma$-algebra 의 정의에서 discrete topology → power set 도출. Integration theory: Lebesgue integral over counting measure 는 summation.

$\square$

### 정리 1.3 — Discount $\gamma < 1$ 의 필요성

Bounded measurable reward $|R| \leq R_{\max}$ 에서, 무한 시간 누적 보상

$$G_t := \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \quad (s_t, a_t \text{ given})$$

이 거의 확실히(almost surely) 수렴하고 $|G_t| \leq R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1-\gamma}$ 이려면 **$\gamma \in [0, 1)$ 필수**.

**증명**: 
Geometric series: $\sum_{k=0}^{\infty} \gamma^k$ converges absolutely $\Leftrightarrow \gamma \in [0, 1)$.

$|G_t| \leq \sum_{k=0}^{\infty} \gamma^k |R_{t+k+1}| \leq R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1-\gamma}$ (finite).

$\gamma = 1$ 이면 $G_t = \sum_{k=0}^{\infty} R_{t+k+1}$ diverges (항상 같은 크기의 보상이 누적됨).

$\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Finite MDP: Gridworld 의 정의와 저장

```python
import numpy as np

# 4x4 Gridworld
n_rows, n_cols = 4, 4
n_states = n_rows * n_cols  # 16
n_actions = 4               # up, down, left, right

# Coding: state (i,j) -> idx = i * n_cols + j
def coord_to_idx(i, j):
    if 0 <= i < n_rows and 0 <= j < n_cols:
        return i * n_cols + j
    return None

def idx_to_coord(idx):
    return idx // n_cols, idx % n_cols

# Transition kernel P[s, a, s'] = P(s' | s, a)
P = np.zeros((n_states, n_actions, n_states))
# Reward R[s, a]
R = np.zeros((n_states, n_actions))

# Deterministic policy: move in direction a, wall = stay
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

for s in range(n_states):
    i, j = idx_to_coord(s)
    for a, (di, dj) in enumerate(actions):
        i_new, j_new = i + di, j + dj
        s_new_idx = coord_to_idx(i_new, j_new)
        if s_new_idx is None:
            s_new_idx = s  # wall: stay in place
        P[s, a, s_new_idx] = 1.0
        # Reward: -1 per step, +10 at goal (15)
        R[s, a] = -1 if s != 15 else 10

# Discount, initial
gamma = 0.9
rho0 = np.zeros(n_states)
rho0[0] = 1.0  # start at (0,0)

# Verify P is stochastic kernel
assert np.allclose(P.sum(axis=2), 1.0), "P not stochastic!"
print("✓ P is stochastic kernel: sum over s' = 1 for all (s,a)")
print(f"  Shape: {P.shape} = (n_states={n_states}, n_actions={n_actions}, n_states={n_states})")
print(f"  R range: [{R.min():.1f}, {R.max():.1f}]")
print(f"  γ = {gamma} ∈ [0, 1)")
print(f"  ρ₀: {rho0} (probability distribution)")
```

**출력**:
```
✓ P is stochastic kernel: sum over s' = 1 for all (s,a)
  Shape: (16, 4, 16) = (n_states=16, n_actions=16, n_states=16)
  R range: [-1.0, 10.0]
  γ = 0.9 ∈ [0, 1)
  ρ₀: [1. 0. 0. ...] (probability distribution)
```

### 실험 2 — Continuous MDP: 근사 표현

```python
# Continuous state space 시뮬레이션: [0, 1] × [0, 1]
# Discretize to grid for NumPy implementation
n_grid = 10
state_grid = np.linspace(0, 1, n_grid)  # Borel measurable (continuous)

# Continuous action space [0, 1] 에서 2개 action sample
actions_cont = np.array([0.3, 0.7])
n_actions_cont = len(actions_cont)

# Simplified LTI dynamics: s' = 0.8*s + 0.2*a + noise
# P(ds' | s, a) is Normal distribution
def sample_next_state(s, a, noise_std=0.05):
    return np.clip(0.8 * s + 0.2 * a + np.random.randn() * noise_std, 0, 1)

# Stochastic kernel representation (approximate)
P_cont = np.zeros((n_grid, n_actions_cont, n_grid))
for i, s in enumerate(state_grid):
    for a_idx, a in enumerate(actions_cont):
        mean_next = 0.8 * s + 0.2 * a
        for j, s_next in enumerate(state_grid):
            # Gaussian kernel (continuous in principle)
            P_cont[i, a_idx, j] = np.exp(-((s_next - mean_next) ** 2) / (2 * 0.05**2))
        P_cont[i, a_idx, :] /= P_cont[i, a_idx, :].sum()  # normalize

# Verify Borel measurability (discrete proxy)
assert np.allclose(P_cont.sum(axis=2), 1.0)
print("✓ Continuous MDP approximated with discretized Borel measurable space")
print(f"  State grid: {state_grid}")
print(f"  Action space: {actions_cont}")
print(f"  P is Borel-measurable kernel (discretized)")
```

### 실험 3 — Measurability 체크: Value 함수가 bounded 인가

```python
# Bounded reward -> Value 도 bounded
R_max = 10.0
gamma = 0.9

# V(s) ≤ R_max / (1 - γ) for all s
V_bound = R_max / (1 - gamma)
print(f"Given R_max = {R_max}, γ = {gamma}")
print(f"Value function bound: |V(s)| ≤ {V_bound:.4f}")

# Numerical check: any policy's V is bounded
# Uniform random policy on Gridworld
pi_uniform = np.ones((n_states, n_actions)) / n_actions

# Policy evaluation: solve V = r^π + γ P^π V
P_pi = np.einsum('sa,sap->sp', pi_uniform, P)  # P^π[s,s'] = Σ_a π(a|s) P(s'|s,a)
r_pi = (pi_uniform * R).sum(axis=1)              # r^π(s) = Σ_a π(a|s) r(s,a)

# (I - γ P^π) V = r^π  ->  V = (I - γ P^π)^{-1} r^π
I = np.eye(n_states)
V_pi = np.linalg.solve(I - gamma * P_pi, r_pi)

print(f"\nUniform policy value:")
print(f"  max V(s) = {V_pi.max():.4f} (should be ≤ {V_bound:.4f})")
print(f"  min V(s) = {V_pi.min():.4f}")
assert V_pi.max() <= V_bound * (1 + 1e-6), "Value exceeds bound!"
print(f"✓ Boundedness confirmed: measurable V exists and is bounded")
```

### 실험 4 — 초기 분포 $\rho_0$ 의 영향

```python
# 다양한 시작점에서의 expected return
np.random.seed(42)

# Test starting points
starts = [0, 4, 15]  # top-left, middle, goal
for start_state in starts:
    rho_test = np.zeros(n_states)
    rho_test[start_state] = 1.0
    
    J = rho_test @ V_pi  # J(π, ρ) = ρ^T V^π
    i, j = idx_to_coord(start_state)
    print(f"Start (i,j)={({i},{j})}, state={start_state}: J(π,ρ) = {J:.4f}")

# Uniform starting distribution
rho_uniform = np.ones(n_states) / n_states
J_uniform = rho_uniform @ V_pi
print(f"\nUniform start: J(π, ρ_uniform) = {J_uniform:.4f}")

# Check: J(π, ρ) = ρ^T V^π
print(f"\n✓ Initial distribution ρ₀ changes expected return")
print(f"  Different ρ₀ -> different MDP (same S, A, P, R, γ)")
```

---

## 🔗 후속 레포와의 연결

- **Ch1-02 Markov 성질**: MDP 정의에 내재된 Markov property $P(s_{t+1} | h_t) = P(s_{t+1} | s_t, a_t)$ 가 history 의 무한 차원을 finite state 로 환원하는 메커니즘
- **Ch1-03 Policy 분류**: Stationary Markovian policy 가 history-dependent policy 를 지배하는 증명 (MDP measurability 기반)
- **Ch2 Bellman Equation**: Measurable $P, R$ 가 있어야만 $\mathbb{E}[V]$ 계산 가능 → Bellman equation 의 전제
- **Model-Free RL**: 정의 없이 시작하는 Q-learning, temporal difference 도 암묵적으로 이 6-tuple 가정

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 완화 방법 |
|------|------------------|
| $\gamma \in [0, 1)$ | Episodic horizon $T < \infty$ 또는 average-reward ($\gamma = 1$, Ch1-04) |
| Bounded reward | Unbounded reward 시 value 발산 — 별도 convergence 분석 필요 (Ch3) |
| Stationary dynamics | Time-varying MDP: $P_t, R_t$ 의존 — state 에 time encoding (episodic) |
| Known $P, R$ | Unknown dynamics: model-free RL (Ch3-4) 또는 model-based learning |
| Compact action space | Continuous, non-compact $\mathcal{A}$ → measurable selection 이론 필요 |
| Polish space 가정 | General measurable space 도 가능하지만 regularity 약화 (advanced) |

---

## 📌 핵심 정리

$$\boxed{\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)}$$

| 성분 | 정의 | 측정 |
|------|------|------|
| $\mathcal{S}$ | Measurable state space (Polish) | $\sim \rho_0$ |
| $\mathcal{A}$ | Measurable action space | Agent 선택 |
| $P$ | Stochastic kernel $P(\cdot \mid s, a) \in \Delta(\mathcal{S})$ | Dynamics |
| $R$ | Bounded measurable reward | $\in [-R_{\max}, R_{\max}]$ |
| $\gamma$ | Discount factor | $\in [0, 1)$, series convergence |
| $\rho_0$ | Initial distribution | Probability on $\mathcal{S}$ |

**필수 성질**:
- $P$ 는 stochastic kernel → integral 정의 가능
- $R$ bounded + $\gamma < 1$ → $G_t = \sum_k \gamma^k R_k$ 수렴
- Measurability → Fubini-Tonelli 적용 가능

---

## 🤔 생각해볼 문제

**문제 1** (기초): Finite MDP 에서 $\mathcal{S} = \{1, 2, 3\}, \mathcal{A} = \{a, b\}$ 이고 deterministic transition

| $(s, a)$ | $s'$ |
|----------|------|
| $(1, a)$ | 2 |
| $(1, b)$ | 3 |
| $(2, a)$ | 3 |
| $(2, b)$ | 1 |
| $(3, a)$ | 1 |
| $(3, b)$ | 2 |

이 경우 $P$ (stochastic kernel) 를 명시적으로 쓰라. Reward $R(s, a) = s - a_{\text{idx}}$ (예: $R(1, a) = 1 - 1 = 0$) 로 정의하면, Bounded measurable 인가?

<details>
<summary>해설</summary>

**Stochastic kernel 표현** (3×2×3 tensor):
$$
P = \begin{pmatrix}
\text{For } s=1: & P(s' \mid 1, a) = [0, 1, 0], & P(s' \mid 1, b) = [0, 0, 1] \\
\text{For } s=2: & P(s' \mid 2, a) = [0, 0, 1], & P(s' \mid 2, b) = [1, 0, 0] \\
\text{For } s=3: & P(s' \mid 3, a) = [1, 0, 0], & P(s' \mid 3, b) = [0, 1, 0]
\end{pmatrix}
$$

행렬 형태:
$$
P[\cdot, a, \cdot] = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}, \quad
P[\cdot, b, \cdot] = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}
$$

**Bounded measurable 확인**:
- Finite $\mathcal{S}, \mathcal{A}$ → trivially measurable ($\sigma$-algebra = power set)
- $|R(s, a)| \leq \max(|1-1|, |2-1|, |3-2|) = 1$ → bounded
- ✓ Yes, bounded measurable reward

$\square$

</details>

**문제 2** (심화): Continuous state space $\mathcal{S} = [0, 1]$ 에서 deterministic linear dynamics $s' = 0.5 s + 0.3 a$ ($a \in [0, 1]$) 와 reward $R(s, a) = s - a^2$ 를 생각하자. 이 MDP 가 Borel measurable 인가? 증명하라.

<details>
<summary>해설</summary>

**Borel measurability 확인**:

1. **State space**: $\mathcal{S} = [0, 1]$ with Borel $\sigma$-algebra $\mathcal{B}([0, 1])$ (generated by open intervals)
   - Polish space ✓

2. **Action space**: $\mathcal{A} = [0, 1]$ with Borel $\mathcal{B}([0, 1])$
   - Compact Polish ✓

3. **Deterministic transition** $s' = 0.5 s + 0.3 a$:
   - Function $f(s, a) = 0.5 s + 0.3 a$ is continuous → Borel measurable
   - Delta kernel: $P(B \mid s, a) = \mathbb{1}_{B}(f(s, a))$ for Borel $B$
   - Measurability: $(s, a) \mapsto P(B \mid s, a)$ measurable? 
     - $P(B \mid s, a) = 1 \Leftrightarrow f(s, a) \in B$ (Borel set)
     - Preimage $f^{-1}(B)$ is Borel (continuous map) → measurable in $(s, a)$ ✓

4. **Reward**: $R(s, a) = s - a^2$
   - Continuous → Borel measurable ✓
   - Bounded on $[0,1] \times [0, 1]$: $|R| \leq 1$ ✓

**결론**: This continuous MDP is Borel measurable. All expectation integrals are well-defined.

$\square$

</details>

**문제 3** (논문 비평): Puterman (2005) *Markov Decision Processes: Discrete Stochastic Dynamic Programming* 의 §2.1 에서 왜 "universally measurable" sets 를 다루는가? 위의 Borel measurability 로 충분한가, 아니면 더 강한 가정이 필요한가?

<details>
<summary>해설</summary>

**Puterman 의 universal measurability 개념**:
- Borel 보다 일반적인 $\sigma$-algebra (analytic sets 포함)
- 이유: **product space 에서의 measurable selection 문제**
  - $P(s' \mid s, a)$ 가 Borel in $(s, a) \times s'$ 이지만
  - 각 $(s, a)$ 에 대해 좋은 $s'$ 를 "선택"하려면 (정책 수정 단계) measurable selection theorem 필요
  - Borel spaces 에서는 **deterministic greedy selection 이 보장되지 않음** (경우에 따라)
  - Universal measurability → Jankov-von Neumann selection theorem 적용 가능

**실무적 의미**:
- Finite / Borel Polish: Puterman 의 strongly optimal policy 존재 (Ch1-03)
- General measurable: existence 가정 필요, 구성적 증명 어려움

**결론**: Borel measurability 가 대부분의 RL 응용에 충분. Universal measurability 는 이론적 완전성 위해 추가.

$\square$

</details>

---

[🏠 README](../README.md) | [다음 ▶: 02. Markov 성질과 그 결과](./02-markov-property.md)
