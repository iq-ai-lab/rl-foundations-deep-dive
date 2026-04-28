# 05. POMDP 와 Belief State

## 🎯 핵심 질문

- POMDP (Partially Observable MDP) 는 MDP 와 어떻게 다른가? 왜 필요한가?
- Observation $o_t$ 와 hidden state $s_t$ 의 구분은 무엇이고, 이것이 왜 문제를 복잡하게 하는가?
- Belief state $b(s) = \Pr(s_t = s | h_t^o)$ 는 무엇이고, POMDP 를 "full MDP" 로 변환할 수 있는가?
- Bayes update rule 로 belief 를 어떻게 갱신하는가?
- Continuous belief space 의 계산 부담은 무엇인가?

---

## 🔍 왜 이것이 필수인가

현실의 대부분의 문제는 **partially observable** 입니다:

- Robot: camera 로 방의 일부만 봄 (뒤는 모름)
- 의료: 환자의 숨겨진 질병 상태 (진단 검사로만 부분 관찰)
- Trading: 경쟁자의 의도 모름 (가격 움직임만 봄)
- Game: 상대 카드 모름 (표면 행동만 봄)

MDP 에서는 **$s_t$ 완전 관찰** 가정. 하지만 현실 대부분은 **$o_t$ 부분 관찰** 만 가능.

이 문제를 푸는 표준 방법이 **belief state (belief MDP)** 로 변환하는 것입니다:

$$\text{POMDP} \xrightarrow{\text{belief state}} \text{Full MDP}$$

책의 이 챕터는 이 변환과 그 의미를 정확히 설명합니다.

---

## 📐 수학적 선행 조건

- **Ch1-01~04**: MDP 정의, Markov property, policy types, horizon types
- **Probability Theory Deep Dive**: Bayes theorem, conditional probability, belief update
- **Measure Theory**: Probability simplex $\Delta(\mathcal{S})$, measurability of belief mappings

---

## 📖 직관적 이해

### MDP vs POMDP

```
MDP (Fully Observable):
  Hidden state s_t: [0, 1, 0, 0] (one-hot, 4 states) ← Agent KNOWS
  Observation o_t: [0, 1, 0, 0] ← Perfect
  Agent sees: state 1 clearly
  
POMDP (Partially Observable):
  Hidden state s_t: [0, 1, 0, 0] ← Agent DOESN'T know
  Observation o_t: Noisy / limited measurement
  Agent sees: probability distribution over states
  Belief: b(s) = Pr(s_t = s | history of observations)
```

### Information Evolution

```
Timeline:
t=0: b₀(s) = initial belief
         ↓ (takes action a₀)
Transition: s_t → s_{t+1} (hidden)
         ↓ (gets observation o_{t+1})
Update belief: b_{t+1}(s) = Bayes(o_{t+1}, b_t)
         ↓ (updates belief)
```

### 왜 Belief State 가 MDP 인가?

Belief state 는 **current state** 로 취급 가능:
- **State**: $b(s)$ = probability distribution on states
- **Action**: agent 선택 $a$
- **Transition**: Bayes update → next belief $b'$
- **Observation**: Markovian (현재 belief 만 needed for next belief update)

따라서 belief space 에서 MDP 를 푸는 것 = original POMDP 푸는 것.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — POMDP (7-tuple)

$\mathcal{M}_{\text{POMDP}} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P, O, R, \gamma)$

**요소**:
- $\mathcal{S}$: Hidden state space (agent 모르는 실제 상태)
- $\mathcal{A}$: Action space
- $\mathcal{O}$: Observation space (agent 관찰 가능)
- $P(s' | s, a)$: Transition kernel (MDP 와 동일)
- $O(o | s')$: **Observation model** — $\Pr(\text{observe } o | \text{true state } s')$
- $R(s, a)$: Reward (hidden state 기반, observation 아님)
- $\gamma \in [0, 1)$: Discount factor

**관찰 그래프**:
```
Hidden:     s_t  ──[P(·|s,a)]──>  s_{t+1}
                                      │
Observed:                            [O(o|s')]
                                      ↓
                                      o_{t+1}
Agent's knowledge:  history h_t^o = (o_0, a_0, ..., o_t)
```

### 정의 5.2 — Belief State

**History** (observable):
$$h_t^o = (o_0, a_0, o_1, a_1, \ldots, o_{t-1}, a_{t-1}, o_t) \in (\mathcal{O} \times \mathcal{A})^t \times \mathcal{O}$$

**Belief state** at time $t$:
$$b_t(s) := \Pr(s_t = s \,|\, h_t^o, \text{initial } b_0, \text{dynamics } P, O) \in \Delta(\mathcal{S})$$

$\Delta(\mathcal{S})$ = probability simplex (확률 분포 집합).

**Belief space**:
$$\mathcal{B} = \Delta(\mathcal{S}) = \left\{ b: \mathcal{S} \to [0, 1], \sum_s b(s) = 1 \right\}$$

(Continuous, high-dimensional)

### 정의 5.3 — Belief Update (Bayes)

Given belief $b_t(s)$, action $a_t$, observation $o_{t+1}$:

$$b_{t+1}(s') = \frac{O(o_{t+1} | s') \sum_s P(s' | s, a_t) b_t(s)}{Z}$$

where $Z = \sum_{s'} O(o_{t+1} | s') \sum_s P(s' | s, a_t) b_t(s)$ (normalization).

**Intuition**: Numerator = likelihood (obs model) × transition × prior belief. Denominator = normalization to probability.

### 정의 5.4 — Belief MDP

Transform POMDP → belief-space MDP:

$$\mathcal{M}_{\text{belief}} = (\Delta(\mathcal{S}), \mathcal{A}, P_b, R_b, \gamma)$$

where:
- **State** = belief $b \in \Delta(\mathcal{S})$
- **Transition**:
  $$P_b(b' | b, a) = \sum_{o \in \mathcal{O}} O(o | \text{state estimated from } b) \cdot [\text{Bayes}]$$
  (probabilistic over observation, deterministic update given observation)
- **Reward**:
  $$R_b(b, a) = \sum_s b(s) R(s, a)$$
  (expected reward under belief)

---

## 🔬 정리와 증명

### 정리 5.1 — Belief MDP 는 (Full) MDP

Belief-space 에서 정의된 $\mathcal{M}_{\text{belief}}$ 는 **full observable MDP** (모든 Ch1-01 조건 만족).

**증명**:

**Step 1 — Measurability**: $\Delta(\mathcal{S})$ 는 Polish space (if $\mathcal{S}$ Polish) → Borel measurable.

**Step 2 — Stochastic kernel**: 
$$P_b(b' | b, a) = \int_\mathcal{O} O(o | s') \left[\sum_s P(s' | s, a) b(s)\right] \mathrm{d}o \quad (\text{marginalized})$$

is stochastic kernel (sum to 1, measurable).

**Step 3 — Bounded reward**:
$$R_b(b, a) = \mathbb{E}_s[R(s, a) \,|\, b] \leq R_{\max}$$ (bounded)

**Conclusion**: $\mathcal{M}_{\text{belief}}$ satisfies full MDP definition. All Ch2-3 theory applies!

$\square$

### 따름 정리 5.2 — POMDP 최적 정책은 Belief-based

Original POMDP 의 optimal policy:
$$\pi^*(a | h_t^o) = \pi_b^*(a | b_t)$$

즉, **belief state $b_t$ 만 필요** — history 전체 아님.

**증명**: Belief-MDP 는 full MDP → Ch1-02 (Markov property) 적용 → belief $b_t$ 충분.

**직관**: History 의 모든 정보는 이미 belief $b(s)$ 에 포함됨 (Bayesian posterior).

$\square$

### 정리 5.3 — Bayes Update 의 수학적 구조

Belief update operator:
$$\tau(b, a, o) := \text{Bayes}(o, b, a) = \frac{O(o | \cdot) \circ T_a(b)}{Z}$$

is **well-defined measurable map** $\Delta(\mathcal{S}) \times \mathcal{A} \times \mathcal{O} \to \Delta(\mathcal{S})$.

**증명**: 
- $T_a(b) := \mathbb{E}_s[P(\cdot | s, a)]|_b$ = transition operator on beliefs
- $O(o | \cdot)$ = likelihood (measurable in state)
- Bayes = pointwise multiply (likelihood · prior) then normalize
- Normalization 존재 iff $O(o | T_a(b))$ 가 positive-measure (almost surely true in well-specified models)

$\square$

### 정리 5.4 — Information Gain in Belief Update

선택적: observation $o$ 가 belief 를 감소시키는 정도:

$$\mathcal{H}(b \,|\, a, o) := -\sum_{s'} b'(s') \log b'(s')$$

is lower than $\mathcal{H}(b)$ in expectation (assuming non-degenerate observations).

(Information-theoretic proof: KL divergence reduction)

---

## 💻 NumPy 구현 검증

### 실험 1 — Tiger Problem (Kassaras 예제)

```python
import numpy as np

# Tiger Problem: 2 doors, tiger hidden behind one
# State: 0=tiger_left, 1=tiger_right
# Action: 0=listen, 1=open_left, 2=open_right
# Observation: 0=hear_left, 1=hear_right (when listening)

n_states = 2
n_actions = 3
n_obs = 2

# Transition: tiger stays (deterministic)
P = np.zeros((n_states, n_actions, n_states))
for s in range(n_states):
    for a in range(n_actions):
        P[s, a, s] = 1.0

# Observation model (when listening a=0)
# If tiger left (s=0): 90% hear left, 10% hear right
O = np.zeros((n_obs, n_states))
O[0, 0] = 0.9  # hear_left | tiger_left
O[1, 0] = 0.1
O[0, 1] = 0.1  # hear_left | tiger_right
O[1, 1] = 0.9

# Initial belief: uniform
b = np.array([0.5, 0.5])

print("Tiger Problem - Belief Update Example")
print(f"Initial belief: {b}")

# Simulate: take action 0 (listen), get observation 0 (hear left)
a = 0
o = 0

# Bayes update
numerator = O[o, :] * b
Z = numerator.sum()
b_new = numerator / Z

print(f"Action: listen, Observation: hear_left")
print(f"P(tiger_left | hear_left) = {b_new[0]:.4f}")
print(f"P(tiger_right | hear_left) = {b_new[1]:.4f}")
print(f"✓ Belief concentrated on tiger_left (90% > 50%)")

# Another listen
o = 0
numerator = O[o, :] * b_new
Z = numerator.sum()
b_new_2 = numerator / Z
print(f"\nAfter second listen (hear left again):")
print(f"P(tiger_left) = {b_new_2[0]:.4f}")
print(f"✓ Belief further concentrated")
```

### 실험 2 — Belief Space (2-state system)

```python
# Visualize belief space for 2-state system
# Belief = [b(s=0), b(s=1)], constraint b(0) + b(1) = 1
# 1D manifold (simplex)

import matplotlib.pyplot as plt

n_states = 2
beliefs = []
for i in range(101):
    b1 = i / 100.0
    b0 = 1.0 - b1
    beliefs.append([b0, b1])

beliefs = np.array(beliefs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 1D representation
ax1.plot(beliefs[:, 0], color='blue', label='P(state=0)')
ax1.plot(beliefs[:, 1], color='orange', label='P(state=1)')
ax1.fill_between(range(len(beliefs)), beliefs[:, 0], alpha=0.3, color='blue')
ax1.set_xlabel('Belief index')
ax1.set_ylabel('Probability')
ax1.set_title('Belief Space (2-state): 1D simplex')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2D representation (but constrained to line b_0 + b_1 = 1)
ax2.scatter(beliefs[:, 0], beliefs[:, 1], s=10, alpha=0.6)
ax2.plot(beliefs[:, 0], beliefs[:, 1], 'b-', label='Belief simplex')
ax2.set_xlabel('P(state=0)')
ax2.set_ylabel('P(state=1)')
ax2.set_title('Belief Space in 2D (constraint line)')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('belief_space_2state.png')
# plt.show()

print(f"✓ Belief space for 2-state = 1D simplex")
print(f"  General n-state = (n-1)-dimensional simplex")
print(f"  Continuous! (uncountably infinite beliefs)")
```

### 실험 3 — Belief MDP 의 Value Iteration

```python
# POMDP 를 belief-MDP 로 변환하고 VI 실행

# Simple 2-state POMDP
n_s = 2
n_a = 2
n_o = 2

# Transition (identity, states don't change for simplicity)
P = np.zeros((n_s, n_a, n_s))
P[0, 0, 0] = 0.6
P[0, 0, 1] = 0.4
P[1, 0, 0] = 0.3
P[1, 0, 1] = 0.7
P[0, 1, 0] = 0.5
P[0, 1, 1] = 0.5
P[1, 1, 0] = 0.5
P[1, 1, 1] = 0.5

# Observation model
# O[o, s] = Pr(o | s)
O = np.array([
    [0.8, 0.2],  # obs 0: likely from state 0
    [0.2, 0.8]   # obs 1: likely from state 1
])

# Reward
R = np.array([
    [1.0, 0.0],  # state 0: reward 1 for action 0
    [0.0, 1.0]   # state 1: reward 1 for action 1
])

gamma = 0.95

def belief_update(b, a, o, P, O):
    """Bayes update: b' = Bayes(o, b, a)"""
    # Transition: b_pred[s'] = sum_s P(s'|s,a) b(s)
    b_pred = np.einsum('sas,s->s', P[:, a, :], b)
    # Likelihood weighting
    numerator = O[o, :] * b_pred
    Z = numerator.sum()
    return numerator / Z if Z > 0 else b_pred

def belief_reward(b, a, R):
    """Expected reward under belief"""
    return np.einsum('s,sa->a', b, R[..., a:a+1]).sum()

# Discretize belief space (for simplicity)
n_belief_grid = 11
belief_grid = []
for i in range(n_belief_grid):
    b0 = i / (n_belief_grid - 1)
    b1 = 1.0 - b0
    belief_grid.append([b0, b1])
belief_grid = np.array(belief_grid)

# Value iteration on belief grid
V_belief = np.zeros(len(belief_grid))

for iteration in range(50):
    V_new = np.zeros(len(belief_grid))
    
    for b_idx, b in enumerate(belief_grid):
        # For each belief, compute max over actions
        q_vals = []
        for a in range(n_a):
            # Expected Q(b, a) over observations
            q_a = belief_reward(b, a, R)
            # Add expected future value (approximated)
            for o in range(n_o):
                b_next = belief_update(b, a, o, P, O)
                # Find closest belief in grid
                dists = np.linalg.norm(belief_grid - b_next, axis=1)
                closest_idx = dists.argmin()
                q_a += gamma * V_belief[closest_idx] / n_o
            q_vals.append(q_a)
        V_new[b_idx] = max(q_vals)
    
    if np.linalg.norm(V_new - V_belief) < 1e-6:
        print(f"Converged at iteration {iteration}")
        break
    V_belief = V_new

print(f"Value function on belief space:")
for i, b in enumerate(belief_grid):
    print(f"  b=({b[0]:.2f}, {b[1]:.2f}): V(b) = {V_belief[i]:.4f}")

print(f"✓ Value Iteration on continuous belief space (discretized grid)")
```

### 실험 4 — Information Gain (Entropy Reduction)

```python
from scipy.stats import entropy

# Measure how observation reduces uncertainty

# Initial belief: uncertain
b_init = np.array([0.5, 0.5])
H_init = entropy(b_init)

# After observation (from Tiger example)
O = np.array([[0.9, 0.1], [0.1, 0.9]])
b_after = np.array([0.9, 0.1])
H_after = entropy(b_after)

print(f"Information Gain Example:")
print(f"Initial belief: {b_init} → Entropy: {H_init:.4f} bits")
print(f"After observation: {b_after} → Entropy: {H_after:.4f} bits")
print(f"Information gain: {H_init - H_after:.4f} bits")
print(f"✓ Observation reduced uncertainty (entropy decreased)")

# Comparison: different observations
b_bad = np.array([0.55, 0.45])
H_bad = entropy(b_bad)
print(f"\nWeak observation: {b_bad} → Entropy: {H_bad:.4f} bits")
print(f"Weak info gain: {H_init - H_bad:.4f} bits")
print(f"✓ Good observations provide more information")
```

---

## 🔗 후속 레포와의 연결

- **Ch2 Bellman Equation (Belief-MDP)**: POMDP 의 belief-MDP 변환은 Bellman 을 belief space 에 적용 가능
- **Ch3 Model-Free RL**: Belief-based Q-learning 은 continuous state space RL 의 근초
- **Advanced RL**: Approximate belief tracking (filtering) — Deep RL 에서 RNN/LSTM 으로 학습
- **Model-Based RL**: POMDP 풀기 위해 particle filter, Kalman filter 등 사용

---

## ⚖️ 가정과 한계

| 가정 | 한계 | 대응 |
|------|------|------|
| Known $O(o \| s)$ | Observation model 모름 | Learn $O$ from data (system identification) |
| Finite $\mathcal{S}, \mathcal{O}$ | Continuous state/observation | Belief filtering (Kalman, particle) |
| Markovian observation | Non-Markovian sensor (memory) | Augment state with sensor history |
| Discrete belief space | Continuous belief | Function approximation (NN for belief) |
| $\gamma < 1$ | Average-reward POMDP | Special theory (Ch4) |

---

## 📌 핵심 정리

$$\boxed{\text{POMDP} \xrightarrow{\text{belief-MDP}}(\Delta(\mathcal{S}), \mathcal{A}, P_b, R_b, \gamma)}$$

| 개념 | 정의 |
|------|------|
| Belief state | $b(s) = \Pr(s \| h^o_t)$ ∈ $\Delta(\mathcal{S})$ |
| Belief update | $b' = \frac{O(o \| \cdot) T_a(b)}{Z}$ (Bayes) |
| Belief MDP | Fully observable MDP on belief space |
| Optimality | $\pi^*(a \| h^o) = \pi_b^*(a \| b)$ |
| Complexity | Belief space continuous & high-dim |

**Key insight**: POMDP = complex information problem, belief-MDP = equivalent standard MDP (but in continuous state space).

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-state, 2-observation POMDP 에서:
- $\mathcal{S} = \{s_0, s_1\}$
- $\mathcal{O} = \{o_0, o_1\}$
- $O(o_0 | s_0) = 0.9, O(o_0 | s_1) = 0.2$ (observation model given)

Initial belief: $b = [0.6, 0.4]$ (60% likely state 0).

Action $a$ 을 취하고 observation $o_0$ 을 받으면, updated belief 는?

<details>
<summary>해설</summary>

Bayes rule 적용 (정상화된 transition 무시, deterministic 가정):

$$b'(s) = \frac{O(o_0 | s) \cdot b(s)}{\sum_{s'} O(o_0 | s') \cdot b(s')}$$

계산:
- Numerator: $[O(o_0 | s_0) \cdot b(s_0), O(o_0 | s_1) \cdot b(s_1)]$
  - $[0.9 \cdot 0.6, 0.2 \cdot 0.4] = [0.54, 0.08]$
- Normalization: $Z = 0.54 + 0.08 = 0.62$
- **Result**: $b'(s_0) = 0.54 / 0.62 = 0.871, b'(s_1) = 0.08 / 0.62 = 0.129$

**Answer**: $b' \approx [0.87, 0.13]$. ✓ Observation $o_0$ 이 $s_0$ 을 더 likely 하게 함.

$\square$

</details>

**문제 2** (심화): Belief space $\Delta(\mathcal{S})$ 가 $(n-1)$-dimensional simplex 라는 것을 증명하시오. 그리고 이것이 continuous state space RL 을 어렵게 하는 이유를 설명하시오.

<details>
<summary>해설</summary>

**Proof that $\Delta(\mathcal{S})$ is $(n-1)$-dimensional**:

$\Delta(\mathcal{S}) = \{ b \in \mathbb{R}_+^n : \sum_{i=1}^n b_i = 1 \}$

Constraint (hyperplane) 하나: $\sum b_i = 1$ 이므로 degrees of freedom = $n - 1$.

Explicit parametrization: $(b_1, \ldots, b_{n-1}) \in [0, 1]^{n-1}$, $b_n = 1 - \sum_{i<n} b_i$.

따라서 $(n-1)$-dimensional manifold. ✓

**Why this complicates RL**:

1. **State space explosion**: $n$-state POMDP → $(n-1)$-dimensional continuous state space for belief
   - $n=10$ states → 9-dim belief space
   - $n=1000$ states → 999-dim belief space!
   - Value function $V: \mathbb{R}^{999} \to \mathbb{R}$ 를 근사하기는 극도로 어려움

2. **Function approximation needed**: $V(b)$ to be parameterized, but $b$ is high-dimensional, constrained
   - Neural network? Exponential samples 필요 (curse of dimensionality)
   - Linear FA? Limited expressiveness

3. **Computational cost**: Belief update itself 가 $O(n^2)$ per step (matrix-vector multiplication)

**Conclusion**: POMDP solving is computationally intractable for large state spaces without approximation. Ch3+ Modern RL 에서 RNN/transformer 로 implicit belief tracking 하는 이유.

$\square$

</details>

**문제 3** (논문 비평): Kaelbling, Littman & Cassandra (1998) "Planning and Acting in Partially Observable Stochastic Domains" 에서 그들은 왜 **belief-state MDP** 변환을 핵심 기여로 본 것인가? 이것이 POMDP 이론에서 혁신적이었던 이유는?

<details>
<summary>해설</summary>

**Before KLC 1998**: 
- POMDP 는 매우 어려운 문제 → solution methods 거의 없음 또는 heuristic
- Value function 정의가 명확하지 않음 (history-dependent?)

**KLC 의 핵심 기여**:

$$\text{POMDP problem} = \text{MDP 문제} \text{ (in belief space)}$$

이것의 의미:
1. **Conceptual unification**: POMDP 는 "다른 종류의 문제" 가 아님 → MDP 의 **state space 를 extends** 한 것
2. **Algorithm portability**: 모든 MDP algorithm (VI, PI, DP) 가 belief space 에서 즉시 적용 가능
3. **Theoretical framework**: Bellman optimality, contraction properties 모두 그대로 사용 가능

**Why revolutionary?**:
- Prior: POMDP = special, intractable, need custom algorithms
- KLC: POMDP = standard MDP with extended state space → apply 40 years of DP theory!

**Modern consequence**:
- Deep RL 에서 recurrent networks (LSTM, GRU) → implicit belief state 학습
- Belief-as-state 개념이 modern partially observable RL 의 foundation

**결론**: KLC 의 belief-MDP reduction 은 POMDP 를 이론적·실무적으로 accessible 하게 만든 landmark contribution. $\square$

</details>

---

[◀ 이전: 04. Finite-Horizon vs Infinite-Horizon vs Average Reward](./04-horizon-types.md) | [📚 README](../README.md) | [다음 ▶: Ch2-01. Discounted Return 의 정의와 수렴](../ch2-bellman-expectation/01-discounted-return.md)
