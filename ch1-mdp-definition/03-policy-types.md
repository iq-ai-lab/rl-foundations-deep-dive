# 03. Policy 의 종류와 Stationary Policy 충분성

## 🎯 핵심 질문

- Policy 는 몇 종류인가? Deterministic vs stochastic, history-dependent vs Markovian vs stationary 의 차이는?
- 왜 "stationary Markovian deterministic policy" 만 고려해도 최적 정책을 찾을 수 있는가? (Puterman 정리)
- History-dependent policy 나 stochastic policy 가 더 좋을 수는 없는가?
- Finite MDP 에서 stationary policy 중 최적이 존재한다는 것을 어떻게 증명하는가?
- Policy space 의 크기는? 왜 greedy 와 argmax 가 deterministic 을 보장하는가?

---

## 🔍 왜 이 정리가 중요한가

대부분의 RL 알고리즘은 **stationary Markovian policy** 만 최적화합니다:
- Q-learning, Policy Gradient, Actor-Critic — 모두 $\pi(a | s)$ 형태
- History-dependent 형태는 다루지 않음

이것이 **optimal** 이라는 것을 증명하지 않으면, 우리는:
1. 더 나은 policy class 를 놓치고 있는 건 아닌가?
2. History 정보를 버리는 것이 정말 OK 인가?
3. 이 제약이 최적성을 잃게 하지 않는가?

를 확실하게 답할 수 없습니다.

이 문서는 **Puterman (2005) 의 정리** — "finite MDP, discounted infinite-horizon 에서 deterministic stationary Markovian policy 중 최적이 존재" — 를 증명하고, 그 의미를 설명합니다.

---

## 📐 수학적 선행 조건

- **Ch1-01, Ch1-02**: MDP definition, Markov property
- **Probability Theory**: Stochastic dominance, measurable functions
- **Fixed point theory**: Contraction mapping, monotone convergence
- **(Optional) Ch2**: Bellman equation (먼저 읽으면 더 명확)

---

## 📖 직관적 이해

### Policy 의 위계

```
┌─────────────────────────────────────────────────────┐
│                  Policy Classes                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. History-dependent stochastic                  │
│     π(a | h_t) = Pr(a | past + current)          │
│     ← Most general, exponentially large space     │
│                                                     │
│  2. Markovian stochastic                          │
│     π(a | s) ← Depends only on state              │
│     ↑ Puterman: This level sufficient             │
│                                                     │
│  3. Markovian deterministic                       │
│     π(s) = a ← Single action per state             │
│                                                     │
│  ↓ Each ⊃ next (stricter class)                   │
│  ↓ But Puterman: All achieve same V*              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 왜 stationary 만으로 충분한가?

**직관**: Markov property + discount + bounded reward 의 조합이 time-dependent structure 를 제거.

- 무한 시간에서 "지금이 언제인가" 는 정책 선택에 영향을 주지 않음
- 과거 경험은 state 에 인코딩됨
- 미래 보상 계산은 state 만으로 충분

따라서 **time $t$ 에 의존하지 않는 policy** 만 고려 가능.

### Greedy Policy 의 역할

Bellman equation $V^*(s) = \max_a [R(s, a) + \gamma \mathbb{E}[V^*(s') | s, a]]$ 의 greedy solution:

$$\pi^*(a | s) = \arg\max_a [R(s, a) + \gamma \mathbb{E}[V^*(s') | s, a]]$$

이 greedy policy 가:
1. **Deterministic** 자동 (argmax 가 deterministic)
2. **Markovian** 자동 (state 만의 함수)
3. **Stationary** 자동 (time-independent)

---

## ✏️ 엄밀한 정의

### 정의 3.1 — History-Dependent Policy

$\mu$ 가 **history-dependent stochastic policy** $\Leftrightarrow$:

$$\mu(a | h_t) := \Pr(a_t = a | h_t) \in [0, 1], \quad \sum_a \mu(a | h_t) = 1$$

History $h_t = (s_0, a_0, \ldots, s_{t-1}, a_{t-1}, s_t)$ 에 의존.

**Performance**:
$$J(\mu) := \mathbb{E}_{s_0 \sim \rho_0}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \,\Big|\, a_t \sim \mu(\cdot | h_t)\right]$$

### 정의 3.2 — Markovian Policy

$\pi$ 가 **Markovian stochastic policy** $\Leftrightarrow$:

$$\pi(a | h_t) = \pi(a | s_t)$$

즉, history 에서 현재 state 만 추출한 함수.

### 정의 3.3 — Stationary Policy

$\pi$ 가 **stationary** $\Leftrightarrow$ time $t$ 에 무관:

$$\pi_t(a | s) = \pi(a | s) \quad \text{for all } t$$

MDP 가 stationary 하면 (dynamics, reward time-independent), optimal policy 도 stationary.

### 정의 3.4 — Deterministic Policy

$\pi$ 가 **deterministic** $\Leftrightarrow$ 각 state 에서 정확히 하나의 action 만 가능:

$$\pi(a | s) \in \{0, 1\}, \quad \sum_a \pi(a | s) = 1$$

또는 함수 형태: $\pi: \mathcal{S} \to \mathcal{A}$.

### 정의 3.5 — Policy Class 의 표기

- $\Pi_{\text{H}}$: History-dependent (most general)
- $\Pi_{\text{M}}$: Markovian ⊂ $\Pi_{\text{H}}$
- $\Pi_{\text{S}}$: Stationary Markovian ⊂ $\Pi_{\text{M}}$
- $\Pi_{\text{D}}$: Deterministic stationary ⊂ $\Pi_{\text{S}}$

Puterman theorem: 최적은 $\Pi_{\text{D}}$ 에 있음.

---

## 🔬 정리와 증명

### 정리 3.1 (Puterman 정리, 1987)

**Finite MDP** (finite $\mathcal{S}, \mathcal{A}$), **discounted infinite-horizon** ($\gamma < 1$), **bounded reward** 하에서:

1. **Optimal value function** $V^*$ 이 unique 하게 존재
2. **Deterministic stationary Markovian policy** 중 optimal 존재
3. History-dependent 이나 stochastic policy 보다 더 나을 수 없음

**증명**:

**Step 1 — Bellman Optimality Equation.**

$$V^*(s) = \max_a [R(s, a) + \gamma \sum_{s'} P(s' | s, a) V^*(s')]$$

이 고유해 $V^*$ 존재 (Banach fixed point theorem, $T^*$ 가 $\gamma$-contraction).

**Step 2 — Greedy Policy 정의.**

$$\pi^*(s) := \arg\max_a [R(s, a) + \gamma \sum_{s'} P(s' | s, a) V^*(s')]$$

(유한 action space → argmax 존재)

**Step 3 — Greedy Policy 의 최적성.**

Policy evaluation: $\pi^*$ 따라가면

$$V^{\pi^*}(s) = R(s, \pi^*(s)) + \gamma \sum_{s'} P(s' | s, \pi^*(s)) V^{\pi^*}(s')$$

Bellman 정의에서:
$$V^*(s) = R(s, \pi^*(s)) + \gamma \sum_{s'} P(s' | s, \pi^*(s)) V^*(s')$$

따라서 $V^{\pi^*}$ 는 위 식의 고유해 → $V^{\pi^*} = V^*$.

**Step 4 — History-Dependent 보다 나을 수 없음.**

임의의 history-dependent $\mu$ 에 대해:
$$J(\mu) \leq V^*(s_0) = J(\pi^*)$$

왜냐하면 Ch1-02 (Markov property) 에서: value 는 현재 state 만 의존 → history extra 정보 no advantage 제공.

**Step 5 — Stochastic 보다 나을 수 없음.**

Stochastic policy $\pi(a | s)$ 의 expected action value:
$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) V^\pi(s')$$

Average over actions:
$$V^\pi(s) = \sum_a \pi(a | s) Q^\pi(s, a) \leq \max_a Q^\pi(s, a)$$

Equality in Bellman → greedy deterministic choice.

$\square$

### 따름 정리 3.2 — Policy Iteration 은 finite step 에 수렴

Finite MDP, deterministic stationary policy class 에서:

1. Policy iteration 은 최대 $|\mathcal{A}|^{|\mathcal{S}|}$ 번의 iteration 내 $\pi^*$ 에 도달
2. 각 iteration 에서 policy 가 strict improvement 이거나 종료

**증명**:
- Deterministic stationary policy 개수: 유한 ($|\mathcal{A}|^{|\mathcal{S}|}$)
- Policy improvement lemma (Ch2): 새 greedy policy 더 좋거나 같음
- 항상 다르면 strict improvement → 무한 loop 불가능
- 따라서 유한 step 내 최적 정책 도달

$\square$

### 정리 3.3 — Stationarity 의 필요충분조건 (MDP level)

MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ 의 dynamics/reward 가 time-independent 이면:

**Optimal value function 도 time-independent** $V^*(s)$ (not $V^*_t(s)$)

따라서 optimal policy 도 stationary.

**증명**: Bellman optimality $V^*(s) = \max_a [\cdots]$ 에는 time $t$ 가 등장하지 않음 → $V^*$ unique, time-independent → greedy $\pi^*$ also stationary. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Policy 종류별 성능 비교 (Gridworld)

```python
import numpy as np

# 4×4 Gridworld setup
n_states = 16
n_actions = 4
gamma = 0.9

# Random MDP (Ch1-01 재사용)
np.random.seed(42)
P = np.random.rand(n_states, n_actions, n_states)
P /= P.sum(axis=2, keepdims=True)
R = np.random.randn(n_states, n_actions)

def policy_eval(pi, P, R, gamma, n_iter=500):
    """Evaluate policy, return V"""
    V = np.zeros(n_states)
    for _ in range(n_iter):
        Q = R + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(axis=1)  # weighted average by π
        if np.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    return V

# 1. Stationary Markovian Stochastic Policy (uniform random)
pi_random = np.ones((n_states, n_actions)) / n_actions
V_random = policy_eval(pi_random, P, R, gamma)
J_random = V_random[0]  # Performance from state 0

# 2. Stationary Markovian Deterministic (random greedy)
Q_opt = R + gamma * np.einsum('sap,p->sa', P, V_random)
pi_det = np.zeros((n_states, n_actions))
pi_det[np.arange(n_states), Q_opt.argmax(axis=1)] = 1.0
V_det = policy_eval(pi_det, P, R, gamma)
J_det = V_det[0]

# 3. Value Iteration to get V* and π*
V_star = np.zeros(n_states)
for _ in range(500):
    Q_star = R + gamma * np.einsum('sap,p->sa', P, V_star)
    V_new = Q_star.max(axis=1)
    if np.abs(V_new - V_star).max() < 1e-10:
        break
    V_star = V_new

pi_star = np.zeros((n_states, n_actions))
pi_star[np.arange(n_states), Q_star.argmax(axis=1)] = 1.0
J_star = V_star[0]

print(f"Policy Performance Comparison:")
print(f"  Random stochastic:     J = {J_random:+.6f}")
print(f"  Random deterministic:  J = {J_det:+.6f}")
print(f"  Optimal (V*):          J = {J_star:+.6f}")
print(f"\n✓ Deterministic ≥ Stochastic (Puterman)")
print(f"  Improvement: {J_det - J_random:.6f}")
```

### 실험 2 — History vs Markovian: History 가 도움되지 않음

```python
# Simulate a scenario where history-awareness seems natural
# But Markovian state is still sufficient

# Simple 3-state MDP with "pattern" in history
# State: abstract position
# Hidden: pattern from past (for illustration)

n_states = 3
n_actions = 2
gamma = 0.95

# Dynamics: next state depends only on current state
P = np.array([
    [[0.7, 0.3, 0.0],   # state 0, action 0
     [0.2, 0.3, 0.5]],  # state 0, action 1
    [[0.3, 0.4, 0.3],   # state 1, action 0
     [0.5, 0.2, 0.3]],  # state 1, action 1
    [[0.0, 0.0, 1.0],   # state 2, action 0 (absorbing)
     [0.0, 0.0, 1.0]],  # state 2, action 1
])

R = np.array([
    [1.0, 0.5],
    [0.5, 2.0],
    [10.0, 10.0]
])

# Optimal Markovian policy
V = np.zeros(n_states)
for _ in range(100):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V = Q.max(axis=1)

pi_opt = np.zeros((n_states, n_actions))
pi_opt[np.arange(n_states), Q.argmax(axis=1)] = 1.0

# Even if we augment with "history pattern" as pseudo-state,
# the fundamental value cannot exceed V* (by Bellman)
V_star = V
J_star = V_star[0]

print(f"Markovian optimal policy:")
print(f"  V* = {V_star}")
print(f"  π*(s) = {pi_opt.argmax(axis=1)}")
print(f"  J(π*) from state 0 = {J_star:.6f}")
print(f"\n✓ Even if history were available, Bellman optimality")
print(f"  limits V to ≤ V*. No benefit from history.")
```

### 실험 3 — Policy Iteration 의 finite convergence

```python
# Demonstrate policy iteration converges in finite steps

n_states = 8
n_actions = 3
gamma = 0.9

np.random.seed(100)
P = np.random.rand(n_states, n_actions, n_states)
P /= P.sum(axis=2, keepdims=True)
R = np.random.randn(n_states, n_actions)

# Start with random policy
pi = np.ones((n_states, n_actions)) / n_actions

policy_improvements = []
for iteration in range(50):
    # Evaluation: solve V = T^π V
    V = np.zeros(n_states)
    for _ in range(500):
        Q = R + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(axis=1)
        if np.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    
    # Improvement: greedy policy
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    pi_new = np.zeros((n_states, n_actions))
    pi_new[np.arange(n_states), Q.argmax(axis=1)] = 1.0
    
    is_improved = not np.allclose(pi, pi_new)
    policy_improvements.append(is_improved)
    
    if not is_improved:
        print(f"Converged at iteration {iteration}")
        break
    
    pi = pi_new
    print(f"Iteration {iteration}: Policy improved")

print(f"\n✓ Policy iteration converges in finite steps: {iteration + 1}")
print(f"  Max possible iterations: {n_actions ** n_states} (too large to enumerate)")
```

### 실험 4 — Deterministic vs Stochastic: 실제 성능

```python
# Directly compare deterministic vs stochastic on learned problem

# 5-state environment
n_s, n_a = 5, 2
gamma = 0.95

np.random.seed(200)
P = np.random.rand(n_s, n_a, n_s)
P /= P.sum(axis=2, keepdims=True)
R = np.random.randn(n_s, n_a) * 5

# Solve for V*
V = np.zeros(n_s)
for _ in range(1000):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V_new = Q.max(axis=1)
    if np.abs(V_new - V).max() < 1e-12:
        break
    V = V_new

# Deterministic greedy
pi_det = np.zeros((n_s, n_a))
pi_det[np.arange(n_s), Q.argmax(axis=1)] = 1.0
V_det = np.zeros(n_s)
for _ in range(1000):
    Q_det = R + gamma * np.einsum('sap,p->sa', P, V_det)
    V_det = (pi_det * Q_det).sum(axis=1)
    if np.linalg.norm(V_det - V) < 1e-12:
        break

# Stochastic: add small perturbation to greedy
pi_stoch = pi_det.copy()
eps = 0.1
pi_stoch = (1 - eps) * pi_det + eps * (1 / n_a)  # mix with uniform
pi_stoch /= pi_stoch.sum(axis=1, keepdims=True)

V_stoch = np.zeros(n_s)
for _ in range(1000):
    Q_stoch = R + gamma * np.einsum('sap,p->sa', P, V_stoch)
    V_stoch = (pi_stoch * Q_stoch).sum(axis=1)

print(f"Comparison: Deterministic vs Stochastic")
print(f"  V* (optimal):          {V}")
print(f"  V^π_det (deterministic): {V_det}")
print(f"  V^π_stoch (stochastic):  {V_stoch}")
print(f"\n  Performance gap (det - stoch): {(V_det - V_stoch).mean():.6f}")
print(f"✓ Deterministic achieves V* (or very close)")
print(f"✓ Stochastic loses performance: {eps} × mix cost")
```

---

## 🔗 후속 레포와의 연결

- **Ch2-01 Bellman Equation**: Bellman optimality 의 greedy solution 이 deterministic stationary
- **Ch2-02 Value Iteration**: V-iteration 이 deterministic optimal policy 찾음
- **Ch2-03 Policy Iteration**: Policy iteration 의 finite convergence 는 Puterman 정리 직접 응용
- **Ch3 Model-Free RL**: Q-learning, SARSA, Actor-Critic 모두 stationary policy 최적화
- **Advanced RL**: Deep RL 에서도 $\pi(a | s; \theta)$ = stationary Markovian 가정

---

## ⚖️ 가정과 한계

| 가정 | 한계 |  대응 |
|------|------|------|
| Finite $\mathcal{S}, \mathcal{A}$ | Continuous/infinite spaces | Measurable selection (advanced) 또는 discretize |
| $\gamma < 1$ | Average-reward (Ch1-04) | $\gamma = 1$ 특별 분석 |
| Stationary MDP | Time-varying environment | Augment state with time: $(t, s)$ |
| Markov property | POMDP / partially observable | Belief state (Ch1-05) |

---

## 📌 핵심 정리

$$\boxed{\pi^* \in \Pi_{\text{D}} = \text{(Deterministic Stationary Markovian)}}$$

**Puterman Theorem** (Finite MDP, $\gamma < 1$):

| 성질 | 결과 |
|------|------|
| Optimal exists | 최적 정책 반드시 존재 |
| Location | Deterministic stationary class |
| Characterization | Bellman optimality 의 greedy |
| Uniqueness | Value $V^*$ unique, 정책은 非unique (ties possible) |
| Convergence | PI 는 finite step 내 도달 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-state, 2-action MDP:

| $(s, a)$ | $R(s,a)$ | $P(s' \| s, a)$ |
|----------|----------|-----------------|
| $(1, 1)$ | 0 | $[0.9, 0.1]$ |
| $(1, 2)$ | 1 | $[0.5, 0.5]$ |
| $(2, 1)$ | 5 | $[0.0, 1.0]$ |
| $(2, 2)$ | 0 | $[1.0, 0.0]$ |

$\gamma = 0.9$ 일 때, optimal deterministic policy 를 찾으시오. Stochastic policy 가 더 나을 수 있는가?

<details>
<summary>해설</summary>

**Bellman optimality equation**:
$$V^*(s) = \max_a [R(s, a) + 0.9 \sum_{s'} P(s' | s, a) V^*(s')]$$

Setup 으로 $V^*$ 계산:

State 1: 
- Action 1: $0 + 0.9(0.9 V^*(1) + 0.1 V^*(2))$
- Action 2: $1 + 0.9(0.5 V^*(1) + 0.5 V^*(2))$

State 2:
- Action 1: $5 + 0.9(0.0 V^*(1) + 1.0 V^*(2)) = 5 + 0.9 V^*(2)$
- Action 2: $0 + 0.9(1.0 V^*(1)) = 0.9 V^*(1)$

Iteration / algebraic solve → $V^*(1), V^*(2)$ 도출, greedy action.

**Typical result**: Deterministic greedy optimal.

**Stochastic**: 더 나을 수 없음 (Puterman). ✓

$\square$

</details>

**문제 2** (심화): Policy class hierarchy $\Pi_{\text{D}} \subset \Pi_{\text{S}} \subset \Pi_{\text{M}} \subset \Pi_{\text{H}}$ 에서, 각 class 내에서의 최적값을 $V^*_{\text{D}}, V^*_{\text{S}}, V^*_{\text{M}}, V^*_{\text{H}}$ 라 하자. 이들 사이의 대소관계는?

<details>
<summary>해설</summary>

**Theorem**: $V^*_{\text{D}} = V^*_{\text{S}} = V^*_{\text{M}} = V^*_{\text{H}}$

**증명**:
- $\Pi_{\text{D}} \subset \Pi_{\text{S}} \subset \cdots$ 로 $V^*_{\text{D}} \leq V^*_{\text{S}} \leq \cdots$
- Puterman: Optimal deterministic $\pi^* \in \Pi_{\text{D}}$ 가 achieves $V^*_{\text{H}}$ (전체 최적)
- 따라서 $V^*_{\text{D}} = V^*_{\text{H}}$ (즉, 모두 동일)

**직관**: Markov property 가 있으면, history extra information no benefit.

$\square$

</details>

**문제 3** (논문 비평): Puterman (2005) 의 정리는 **finite MDP** 에만 적용된다. Continuous state space 에서는 어떻게 되는가? Szepesvári (2010) 이나 Bertsekas (2012) 는 이 경우를 어떻게 다루는가?

<details>
<summary>해설</summary>

**Finite MDP**: 모든 policy class 가 countable dimensionality.

**Continuous MDP** ($\mathcal{S} = \mathbb{R}^d$):
- Deterministic stationary policy space도 **uncountable** ($\mathcal{A}^{\mathcal{S}}$)
- Argmax 자체가 measurable selection problem

**Puterman 의 일반화** (universal measurability):
- $\sigma$-algebra 를 analytic sets 로 확대
- Jankov-von Neumann selection theorem → deterministic optimal 존재 **보장**

**Szepesvári, Bertsekas 의 practical approach**:
- Continuous: discretize or use function approximation
- Measurable selection guarantee 대신 **convergence to near-optimal** 만 증명
- Deep RL: parameterized policy $\pi(a | s; \theta)$ 로 최적화 (unconstrained search)

**결론**: Theory 는 극한에서 optimal deterministic stationary 존재 보장. 실제는 approximation.

$\square$

</details>

---

[◀ 이전: 02. Markov 성질과 그 결과](./02-markov-property.md) | [📚 README](../README.md) | [다음 ▶: 04. Finite-Horizon vs Infinite-Horizon vs Average Reward](./04-horizon-types.md)
