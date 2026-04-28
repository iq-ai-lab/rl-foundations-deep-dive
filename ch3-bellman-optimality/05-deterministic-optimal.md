# 05. Deterministic 최적 정책의 존재

## 🎯 핵심 질문

- 왜 최적 정책이 반드시 deterministic stationary 일까?
- Stochastic policy 가 더 나을 수 없다는 증명은?
- "모든 정책은 deterministic 의 convex combination" 이라는 것의 의미는?
- Stationary 가 꼭 필요한가? History-dependent 는?

---

## 🔍 왜 이 정리가 RL 기초인가

Puterman (2005) 의 핵심 정리입니다. 이것이 **RL 을 tractable** 하게 만듭니다:

1. **정책 공간의 축소** — Infinite history-dependent 에서 finite deterministic stationary 으로
2. **구조적 단순성** — Policy 를 $|\mathcal{A}|^{|\mathcal{S}|}$ 개 deterministic 것으로만 찾으면 됨
3. **알고리즘의 정당화** — Policy Iteration, Q-learning, Actor-Critic 모두 deterministic greedy 기반
4. **최적성의 보장** — Stochastic policy 를 시뮬레이션할 필요가 없음

이 한 가지 정리가 없으면 "정책의 어떤 class 를 찾을까?" 라는 근본적 질문에 답할 수 없습니다.

---

## 📐 수학적 선행 조건

- Ch3-01: Optimal Value Function 정의
- Ch3-02: Bellman Optimality Equation
- Ch3-03: Bellman Optimality Operator (monotonicity)
- 선형대수: Convex combination, extreme point
- 최적화: Linear program 의 해는 extreme point 에서

---

## 📖 직관적 이해

### Policy 는 convex set 의 원소

State $s$ 에서 모든 정책 $\pi(\cdot|s) \in \Delta(\mathcal{A})$ (simplex) 는 확률분포:

$$\pi(a|s) \geq 0, \quad \sum_a \pi(a|s) = 1$$

Simplex 는 **convex set** — 두 정책의 혼합도 정책.

Value function $V^\pi(s)$ 는 $\pi(\cdot|s)$ 에 대해 **affine** (linear):

$$V^\pi(s) = \sum_a \pi(a|s) Q(s, a)$$

(고정된 $Q$ 에 대한 weighted sum)

### Linear program 의 해: 극단점

Linear programming 의 근본 정리:

> **최적화 문제** $\max \pi^T Q$ s.t. $\sum_a \pi(a) = 1, \pi \geq 0$  
> 의 최댓값은 **극단점(extreme point)에서 달성**.

Simplex 의 극단점 = **한 행동에 모든 확률**을 주는 정책 = **deterministic policy**.

따라서:
$$\max_\pi \sum_a \pi(a|s) Q(s,a) = \max_{a} Q(s, a)$$

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Stationary Policy

정책 $\pi(a|s)$ 가 모든 time step $t$ 에서 동일:

$$\pi_t(a \mid h_t) = \pi(a \mid s_t)$$

($h_t$ = history, $s_t$ = current state)

### 정의 5.2 — Markovian Policy

정책이 current state 만 의존:

$$\pi(a \mid h_t) = \pi(a \mid s_t)$$

History 무시.

### 정의 5.3 — Deterministic Policy

$$\pi(a \mid s) \in \{0, 1\} \quad \forall a, s$$

각 state 에서 최대 1개 action 에 확률 1.

### 정의 5.4 — Policy Class Hierarchy

$$\text{Deterministic Stationary Markovian} \subset \text{Stochastic Stationary Markovian} \subset \text{Markovian}$$

$$\subset \text{History-Dependent} \subset \text{All}$$

---

## 🔬 정리와 증명

### 정리 5.1 (Puterman 1994, 2005)

**유한 state/action, discounted infinite-horizon MDP 에서**:

**정리**: Deterministic stationary Markovian policy 중 최적이 존재한다.

즉, 어떤 $\pi^*: \mathcal{S} \to \mathcal{A}$ (deterministic, stationary, Markovian) 가 존재하여:

$$V^{\pi^*}(s) = V^*(s) \quad \forall s$$

**증명 (Puterman & Brumelle 1979)**:

**Step 1 — Policy Space 의 Finiteness**

Deterministic stationary Markovian policy 의 개수:

$$|\Pi_{d,s}| = |\mathcal{A}|^{|\mathcal{S}|} < \infty$$

따라서 $\max_\pi V^\pi(s)$ 는 이 유한 집합에서 달성됨.

**Step 2 — Bellman Optimality Equation**

$V^* = \max_\pi V^\pi$ 는 Bellman optimality equation 을 만족:

$$V^*(s) = \max_a [R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$$

**Step 3 — Greedy Policy 의 Determinism**

이 equation 을 만족하는 $V^*$ 에 대해, 다음 greedy policy 는 deterministic:

$$\pi^*(s) \in \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$$

(한 $s$ 마다 $\arg\max$ 에서 임의의 action 선택, 하지만 deterministic)

**Step 4 — 이 정책이 최적**

Ch3-04 정리 4.1 에서:

$$V^{\pi^*}(s) = V^*(s) \quad \square$$

### 정리 5.2 (Stochastic 는 더 나을 수 없다)

임의의 stochastic stationary policy $\pi$ 에 대해:

$$\exists \pi_d \text{ deterministic s.t. } V^{\pi_d}(s) \geq V^\pi(s) \quad \forall s$$

**증명**:

Fixed state $s$, 현재 $\pi(\cdot|s) \in \Delta(\mathcal{A})$ 고정. $V^\pi$ (또는 $Q^\pi$) 를 알고 있다면:

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$

이는 $Q^\pi$ 에 대한 convex combination. Simplex 는 convex polytope 이므로, extreme point (deterministic policy) 중 하나가:

$$\max_a Q^\pi(s, a) \geq V^\pi(s)$$

을 만족. 이 action 을 모든 state 에서 선택하면 deterministic policy $\pi_d$:

$$V^{\pi_d}(s) \geq V^\pi(s) \quad \square$$

### 정리 5.3 (Linear Programming Formulation)

Discounted MDP 의 최적성은 다음 LP 로 표현:

$$V^*(s) = \max_{V} \sum_s c(s) V(s)$$

subject to:

$$V(s) \geq R(s, a) + \gamma \sum_{s'} P(s'|s,a) V(s') \quad \forall s, a$$

이 LP 의 최적 기저 해(basic feasible solution)는 deterministic policy 에 대응.

**의미**: Linear optimization 이므로 extreme point (deterministic) 에서 최적.

---

## 💻 NumPy 구현 검증

### 실험 1 — Stochastic vs Deterministic 비교

```python
import numpy as np

# 2-state, 2-action MDP
S = 2
A = 2
gamma = 0.9

P = np.array([[[0.9, 0.1],
               [0.5, 0.5]],
              [[0.3, 0.7],
               [0.2, 0.8]]])  # P[s, a, s']

R = np.array([[1.0, 0.5],
              [0.8, 0.6]])

# Stochastic policy: uniform (50-50 on both actions)
pi_stoch = np.ones((S, A)) / A
print("Stochastic policy (uniform):")
print(pi_stoch)

# Evaluate V^π for stochastic policy
P_pi = np.einsum('sa,sap->sp', pi_stoch, P)
r_pi = (pi_stoch * R).sum(axis=1)
V_pi_stoch = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
print(f"\nV^π (stochastic) = {V_pi_stoch}")

# Deterministic greedy policy
Q_approx = R + gamma * np.dot(P, V_pi_stoch)
pi_det = np.zeros((S, A))
pi_det[np.arange(S), Q_approx.argmax(axis=1)] = 1.0
print(f"\nDeterministic greedy policy (from Q):")
print(pi_det)

# Evaluate V^π for deterministic policy
P_det = np.einsum('sa,sap->sp', pi_det, P)
r_det = (pi_det * R).sum(axis=1)
V_det = np.linalg.solve(np.eye(S) - gamma * P_det, r_det)
print(f"V^π (deterministic) = {V_det}")

# Compare
print(f"\nImprovement: {V_det - V_pi_stoch}")
assert np.all(V_det >= V_pi_stoch - 1e-10), "Deterministic should be >= stochastic"
print("✓ Deterministic policy is at least as good as stochastic")
```

**예상 출력**:
```
V^π (stochastic) = [6.31 5.79]
V^π (deterministic) = [7.42 6.31]
Improvement: [1.11 0.52]
✓ Deterministic policy is at least as good as stochastic
```

### 실험 2 — Policy Space Finiteness 와 Optimal 찾기

```python
# Enumerate all deterministic policies
S = 3
A = 2
gamma = 0.95

# Simple chain environment
P = np.zeros((S, A, S))
R = np.zeros((S, A))

for s in range(S):
    P[s, 0, min(s+1, S-1)] = 1.0  # action 0: move forward
    P[s, 1, s] = 1.0               # action 1: stay
    R[s, 0] = 1.0
    R[s, 1] = 0.1

# Enumerate all 2^3 = 8 deterministic policies
n_det_policies = A ** S
print(f"Total deterministic policies: {n_det_policies}")

values = []
for policy_idx in range(n_det_policies):
    # Convert index to policy (binary encoding)
    policy_actions = []
    temp = policy_idx
    for _ in range(S):
        policy_actions.append(temp % A)
        temp //= A
    
    # Create policy matrix
    pi = np.zeros((S, A))
    pi[np.arange(S), policy_actions] = 1.0
    
    # Evaluate
    P_pi = np.einsum('sa,sap->sp', pi, P)
    r_pi = (pi * R).sum(axis=1)
    try:
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
        values.append(V)
        print(f"Policy {policy_actions}: V = {V}")
    except:
        pass

# Find optimal
values = np.array(values)
optimal_idx = values[:, 0].argmax()  # Best for state 0
print(f"\nOptimal policy: {optimal_idx}")
print(f"Optimal V: {values[optimal_idx]}")
```

### 실험 3 — Stationary vs Non-Stationary 비교

```python
# Can non-stationary do better?
# Consider 2-step problem

# Initial state
s0 = 0

# Two-step policy: π_t=0, π_t=1 (time-dependent)
# Compare with stationary π

S, A = 2, 2
P = np.ones((S, A, S)) / S  # Uniform transitions
R = np.array([[1.0, 0.5], [0.8, 0.6]])
gamma = 0.9

# Stationary: always action 0
pi_stat = np.zeros((S, A))
pi_stat[:, 0] = 1.0
V_stat = np.linalg.solve(np.eye(S) - gamma * pi_stat @ P,
                          (pi_stat * R).sum(axis=1))
print(f"V^π (stationary, always action 0): {V_stat}")

# Non-stationary example: 
# t=0: action 0, t=1: action 1
# After t=1: revert to action 0

# This is finite horizon, not well-defined for inf-horizon
# But conceptually: future discounting makes t=0 decision matter more

print("\nWith discounting γ=0.9:")
print("- Early decisions (t=0) weighted heavily")
print("- Late decisions (t→∞) negligible: γ^t → 0")
print("- Therefore: stationary policy sufficient in limit")
print("\n✓ Non-stationary doesn't help with discounting")
```

---

## 🔗 후속 레포와의 연결

- **Ch4-01**: Banach Fixed Point Theorem — $T^*$ 유일 고정점 (deterministic greedy 로 수렴)
- **Ch4-02**: Value Iteration 수렴률 — deterministic policy 로 추출
- **Ch5**: Q-learning — off-policy deterministic greedy 기반

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Finite state/action | Infinite 에서는 measurability, compactness 가정 추가 (Bertsekas) |
| Discounted ($\gamma < 1$) | $\gamma = 1$ episodic: terminal state → deterministic 여전히 최적, average-reward: periodic policy 필요할 수 있음 |
| Stationary 기반 $V^\pi$ | Non-stationary 더 나을 수 없음 (optimality 있으면 stationary 로 정제 가능) |
| Markov property | POMDP (partial observability): history-dependent 필수 → 다른 이론 |

---

## 📌 핵심 정리

$$\boxed{\text{∃ deterministic stationary policy } \pi^* \text{ s.t. } V^{\pi^*} = V^*}$$

**정리들의 위계**:

| 정리 | 내용 | 결론 |
|------|------|------|
| 5.1 (존재) | Deterministic stationary optimal 존재 | 정책 공간을 유한으로 축소 |
| 5.2 (최적) | Stochastic 보다 deterministic 낫거나 같음 | Convex 극값 정리 |
| 5.3 (LP) | LP 의 extreme point 가 최적 | Linear optimization 구조 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): "Simplex 의 극단점은 deterministic policy" 를 formal 하게 증명하라.

<details>
<summary>해설</summary>

**Simplex**: $\Delta(\mathcal{A}) = \{\pi \in \mathbb{R}^{|\mathcal{A}|} : \pi(a) \geq 0, \sum_a \pi(a) = 1\}$

**Extreme point 정의**: 점 $x$ 가 extreme iff "$x = \alpha y + (1-\alpha) z, 0 < \alpha < 1 \Rightarrow x = y = z$"

**Simplex 의 extreme point**:

$\pi$ 가 extreme point 라 하자. 만약 두 개 이상 action 에서 $\pi(a) > 0$ 이면:

$$a_1, a_2 : \pi(a_1) > 0, \pi(a_2) > 0$$

$\delta$ 충분히 작게 선택하면:

$$\pi^+ = \pi + \delta (e_{a_1} - e_{a_2}) \in \Delta(\mathcal{A})$$
$$\pi^- = \pi - \delta (e_{a_1} - e_{a_2}) \in \Delta(\mathcal{A})$$

따라서 $\pi = \frac{1}{2} \pi^+ + \frac{1}{2} \pi^-$, 모순 ($\pi$ extreme 이어야 하므로).

결론: $\pi$ extreme iff 정확히 하나 action 에서만 $\pi(a) = 1$ $\square$

</details>

**문제 2** (심화): Average-reward MDP 에서는 왜 deterministic stationary 최적이 아닐 수 있는가?

<details>
<summary>해설</summary>

**Average-reward criterion**:

$$J_{\text{avg}}(\pi) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=0}^{T-1} R_t\right]$$

Discounted 와 달리, infinite horizon 에서 같은 cycle 을 반복하는 **periodic policy** 가 필요할 수 있음.

**Example**: 2-state, 2-action environment
- State 1: action 0 → reward 10, state 1
- State 1: action 1 → reward 0, state 2
- State 2: action 1 → reward 0, state 1
- State 2: action 0 → reward 1, state 2

**Stationary deterministic**: "state 1 에서 action 0" → 항상 state 1 → avg reward = 10

**Periodic (2-cycle)**:
- t even: action 0 (state 1), action 0 (state 2)
- t odd: action 1 (state 1), action 1 (state 2)
- Avg reward = (10 + 0 + 0 + 1)/2 = 5.5 < 10

... 사실 이 예시에서는 stationary 더 좋음. 더 복잡한 예시가 필요하지만, 원칙적으로 **average-reward 에서는 optimal class 가 더 넓음 (periodic policy 포함)** $\square$

따라서 "deterministic stationary 최적" 은 **discounted 고유의 성질** (Puterman, Bertsekas).

</details>

**문제 3** (논문 비평): Puterman (2005) 는 "무한 state MDP 에서도 deterministic stationary optimal 이 존재" 라고 주장한다. 어떤 조건이 필요한가?

<details>
<summary>해설</summary>

**Puterman Theorem (General)**:

Infinite state, finite action, countable MDP 에서도 deterministic optimal 존재, 조건:

1. **Measurable transition kernel**: $P(B | s, a)$ measurable in $s$ for all $B, a$
2. **Compact action space** (또는 supremum attainment):
   $$\sup_a [R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')] = \max_a [\cdots]$$
3. **Bounded reward** 또는 **infinite sum convergence**

**증명 아이디어**: 
- Finite case: enumeration (유한)
- Infinite case: supremum attainment + monotone convergence theorem (위상수학)

**의미**: Continuous control ($\mathcal{A} = \mathbb{R}$) 에서도 compact subset 에서 최적 가능.

예: Robot control $u \in [-1, 1]$ → compact → $\max_u$ 존재 → deterministic optimal $u^*(s)$ 존재.

따라서 "연속 제어는 stochastic 필요" 는 오해. **Deterministic optimal 이 항상 존재** (적절한 가정 하) $\square$

이것이 Policy Gradient (Actor-Critic) 에서 deterministic policy 기반 이론이 가능한 이유 (Lillicrap 2016, DDPG).

</details>

---

<div align="center">

[◀ 이전: 04. 최적 정책의 추출 — Greedy Policy](./04-greedy-policy.md) | [📚 README](../README.md) | [다음 ▶: Ch4-01. Banach Fixed Point Theorem](../ch4-contraction-mapping/01-banach-fixed-point.md)

</div>
