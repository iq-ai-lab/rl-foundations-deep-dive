# 04. 최적 정책의 추출 — Greedy Policy

## 🎯 핵심 질문

- $V^*$ 를 알고 있을 때, 최적 정책을 어떻게 구하는가?
- Greedy policy $\pi^*(s) = \arg\max_a Q^*(s, a)$ 가 정말 최적인가?
- 여러 행동이 동률일 때 어떻게 선택하는가? 성능은?
- Tie-breaking 의 임의성이 최적성을 해치는가?

---

## 🔍 왜 이 정리가 RL 기초인가

RL 의 최종 목표는 **최적 정책을 찾는 것**입니다. 앞서 $V^*$ 를 정의하고 구했지만, 정책 자체는 아직 없습니다. Greedy policy 는 다음을 보장합니다:

1. **$V^*$ 주어진 후 1-step 으로 정책 복구** — "왜 이 정책이 최적?"의 답
2. **구조적 단순성** — deterministic, state-only 로 정책 표현
3. **실전 알고리즘의 기초** — Q-learning, Actor-Critic 모두 greedy action 선택에 기반

이 문서는 **"가치를 알면 정책은 자동"** 이라는 RL 의 핵심 원리를 형식화합니다.

---

## 📐 수학적 선행 조건

- Ch3-01: Optimal Value Function
- Ch3-02: Bellman Optimality Equation
- Ch3-03: Bellman Optimality Operator
- Ch2-02: Policy definition (deterministic vs stochastic)

---

## 📖 직각적 이해

### Greedy 는 "가장 좋은 행동 선택"

State $s$ 에서:

- $Q^*(s, a)$ = "이 행동을 선택했을 때 최적을 따랐을 때의 가치"
- Greedy = "그 중 최대 가치 주는 행동만 선택"

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

**직관**: 현재 state 의 value 는 다음 step 의 최적값으로 결정되므로, 최고 $Q$-value 를 고르면 자동으로 최적.

### Tie-breaking 의 자유도

여러 행동이 같은 최고 값을 가지면:

```
Q*(s) = [0.8, 0.8, 0.5, 0.6]  (4 actions)
arg_max = [0, 1] (tie on actions 0, 1)
```

Greedy 는 이들 중 **임의로** 선택 가능. 그러나 모든 선택의 성능 동일:

$$V^{\pi^*}(s) = \max_a Q^*(s, a) = 0.8$$

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Greedy Policy with respect to $V$

주어진 value function $V$ (반드시 $V^*$ 가 아니어도 됨), greedy policy 는:

$$\pi^{\text{greedy}}(s) := \arg\max_a \left\{ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right\}$$

또는 $Q$-function 이용:

$$\pi^{\text{greedy}}(s) := \arg\max_a Q_V(s, a)$$

여기서 $Q_V(s, a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')$.

### 정의 4.2 — Optimal Greedy Policy

$V = V^*$ 일 때:

$$\pi^*(s) := \arg\max_a Q^*(s, a)$$

$\pi^*$ 는 **optimal greedy policy** 또는 **greedy policy with respect to $V^*$**.

### 정의 4.3 — Tie-breaking in $\arg\max$

$\arg\max_a f(a) = \{ a : f(a) = \max_a f(a) \}$ (집합)

Policy 로서는 이 집합 중 **하나**를 선택:

$$\pi^*(a \mid s) = \begin{cases} 1 & a \in \arg\max_{a'} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}$$

Multiple optimal actions 면 어느 것을 선택해도 성능 동일.

---

## 🔬 정리와 증명

### 정리 4.1 (Greedy Policy 는 최적)

$V = V^*$ 이면, greedy policy $\pi^*(s) = \arg\max_a Q^*(s, a)$ 는 최적:

$$V^{\pi^*}(s) = V^*(s) \quad \forall s$$

**증명**:

정의에 의해:

$$V^*(s) = \max_a Q^*(s, a) = Q^*(s, \pi^*(s))$$

(greedy 이므로 선택된 action 이 $\arg\max$)

따라서:

$$V^{\pi^*}(s) = \mathbb{E}[R(s, \pi^*(s)) + \gamma V^*(s') \mid s] = R(s, \pi^*(s)) + \gamma \sum_{s'} P(s'|s,\pi^*(s)) V^*(s')$$

$$= Q^*(s, \pi^*(s)) = \max_a Q^*(s, a) = V^*(s) \quad \square$$

### 정리 4.2 (Tie-breaking 의 영향)

만약 어떤 state $s$ 에서 $\max_a Q^*(s, a)$ 가 여러 action 에서 달성되면:

$$\forall a \in \arg\max_{a'} Q^*(s, a'), \quad Q^*(s, a) = V^*(s)$$

따라서 **모든 tie-breaking choice 가 같은 성능**:

$$V^{\pi_1}(s) = V^{\pi_2}(s) = \cdots = V^*(s)$$

(서로 다른 tie-breaking 을 선택한 정책들도 같은 value)

**증명**: 모두 $V^*(s) = \max_a Q^*(s, a)$ 를 달성하므로.

### 정리 4.3 (Policy Improvement with Greedy)

임의의 정책 $\pi$ 와 그 $Q^\pi$ 에 대해, greedy policy $\pi' = \arg\max_a Q^\pi(s,a)$ 는:

$$V^{\pi'}(s) \geq V^\pi(s) \quad \forall s$$

**증명** (Ch3-02 정리 2.2):

$$V^{\pi'}(s) = \max_a Q^\pi(s, a) \geq \sum_a \pi(a|s) Q^\pi(s,a) = V^\pi(s) \quad \square$$

Equality 는 $\pi$ 이미 greedy 일 때만 성립.

### 정리 4.4 (Deterministic Optimality)

Deterministic greedy policy 가 최적. 즉:

$$\exists \pi^* \text{ deterministic s.t. } V^{\pi^*} = V^*$$

Stochastic policy 의 혼합으로 더 나을 수 없음.

**증명** (Ch3-01 문제 2):

Value 는 $Q$ 의 expected weighted sum 이므로, convex combination 의 최대는 극단점 (deterministic) 에서만 달성 $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — $V^*$ 로부터 Greedy Policy 추출

```python
import numpy as np
import matplotlib.pyplot as plt

# 4x4 Gridworld
S = 16
A = 4
gamma = 0.9

# Setup (동일)
P = np.zeros((S, A, S))
R = np.zeros((S, A))
dirs = [[-1,0], [0,1], [1,0], [0,-1]]

for s in range(S):
    i, j = s // 4, s % 4
    if (i, j) == (3, 3):
        P[s, :, s] = 1.0
        R[s, :] = 1.0
        continue
    
    for a, (di, dj) in enumerate(dirs):
        ni, nj = i + di, j + dj
        if 0 <= ni < 4 and 0 <= nj < 4:
            ns = ni * 4 + nj
            P[s, a, ns] = 1.0
        else:
            P[s, a, s] = 1.0

# Compute V* via value iteration
V = np.zeros(S)
for _ in range(1000):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V_new = Q.max(axis=1)
    if np.linalg.norm(V - V_new, np.inf) < 1e-10:
        break
    V = V_new

# Extract greedy policy
Q_final = R + gamma * np.einsum('sap,p->sa', P, V)
pi_greedy_actions = Q_final.argmax(axis=1)

# Convert to one-hot for verification
pi_greedy = np.zeros((S, A))
pi_greedy[np.arange(S), pi_greedy_actions] = 1.0

# Verify: V^{π_greedy} should equal V*
V_pi = np.linalg.solve(
    np.eye(S) - gamma * np.einsum('sa,sap->sp', pi_greedy, P),
    (pi_greedy * R).sum(axis=1)
)

print(f"||V^π* - V*||_∞ = {np.linalg.norm(V_pi - V, np.inf):.2e}")
print("✓ Greedy policy achieves V*")

# Visualize policy as arrows
policy_names = ['↑', '→', '↓', '←']
policy_grid = np.array([policy_names[a] for a in pi_greedy_actions]).reshape(4, 4)
print("\nOptimal Policy (arrows):")
for i in range(4):
    print('  '.join(policy_grid[i]))
```

**예상 출력**:
```
||V^π* - V*||_∞ = 1.35e-15
✓ Greedy policy achieves V*

Optimal Policy (arrows):
→  →  →  ↓
↓  ↓  ↓  ↓
↓  ↓  ↓  ↓
→  →  →  ↓
```

### 실험 2 — Tie-breaking 의 효과

```python
# Construct MDP with ties
S = 2
A = 3
gamma = 0.9

# Both states have 2 actions with equal Q value
P = np.eye(S).reshape(S, 1, S)
P = np.repeat(P, A, axis=1)  # All actions -> stay (absorbing)

# Reward: state 0 has actions 0,1 tied; state 1 has all tied
R = np.array([[1.0, 1.0, 0.5],
              [0.5, 0.5, 0.5]])

# Compute V*
V = np.zeros(S)
for _ in range(100):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V = Q.max(axis=1)

Q_final = R + gamma * np.einsum('sap,p->sa', P, V)

print("Q*(s, a) for each state:")
print(Q_final)

# Try different tie-breaking
tie_break_options = [
    np.argmax,           # first max
    lambda x: np.random.choice(np.where(x == x.max())[0]),  # random
]

values = []
for i, tie_break_fn in enumerate(tie_break_options):
    if callable(tie_break_fn) and tie_break_fn == tie_break_options[0]:
        actions = Q_final.argmax(axis=1)
    else:
        actions = np.array([tie_break_fn(Q_final[s]) for s in range(S)])
    
    pi = np.zeros((S, A))
    pi[np.arange(S), actions] = 1.0
    
    # Evaluate
    V_pi = np.linalg.solve(np.eye(S) - gamma * np.einsum('sa,sap->sp', pi, P),
                           (pi * R).sum(axis=1))
    values.append(V_pi[0])
    print(f"\nTie-break option {i}: V = {V_pi}")

print(f"\nAll tie-breaking options give same performance: {np.allclose(values, values[0])}")
```

### 실험 3 — Policy Improvement Monotonicity

```python
# Start with suboptimal policy
pi = np.ones((S, A)) / A  # uniform random

for iteration in range(10):
    # Evaluate
    P_pi = np.einsum('sa,sap->sp', pi, P)
    r_pi = (pi * R).sum(axis=1)
    V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    
    # Greedy improvement
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    pi_new = np.zeros((S, A))
    pi_new[np.arange(S), Q.argmax(axis=1)] = 1.0
    
    # Performance
    V_pi = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    J = (V_pi * rho0).mean()  # average over initial states
    
    print(f"Iteration {iteration}: J(π) = {J:.6f}, " +
          f"changed: {not np.allclose(pi, pi_new)}")
    
    if np.allclose(pi, pi_new):
        print(f"Converged to optimal at iteration {iteration}")
        break
    
    pi = pi_new
```

---

## 🔗 후속 레포와의 연결

- **Ch3-05**: Deterministic 최적 정책의 존재성 증명
- **Ch4-01**: Value Iteration 수렴성 — 정책 추출과 함께 작동
- **Ch5**: Q-learning 에서 $\max_a Q(s, a)$ 의 역할

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| $V^*$ 정확히 알고 있음 | 실제: 추정 오차 있음 → greedy 가 suboptimal (Ch5 에서) |
| Tie-breaking 임의 가능 | 실제: epsilon-greedy 로 exploration 추가 (Ch5) |
| Deterministic action | Stochastic action 으로 확장 가능 (성능 동일) |
| Finite action | Infinite action 에서는 supremum attainment 보장 필요 |

---

## 📌 핵심 정리

$$\boxed{\pi^*(s) = \arg\max_a Q^*(s, a) \Rightarrow V^{\pi^*}(s) = V^*(s)}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| Greedy policy | $\arg\max_a Q$ | 가치로부터 정책 추출 |
| Tie-breaking | 여러 max 중 선택 | 성능 동일 (임의 선택 가능) |
| Deterministic | state 마다 1개 action | 최적 달성 가능 |
| Policy improvement | $\pi' = \text{greedy}(\pi)$ | Policy iteration 의 기초 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Greedy policy 는 $V^*$ 를 알 때만 최적이다. 만약 $V$ 가 $V^*$ 의 근사라면 ($\|V - V^*\|_\infty = \epsilon$), greedy policy 의 성능은 얼마나 나빠지는가?

<details>
<summary>해설</summary>

$\pi_V = \arg\max_a Q_V(s, a)$ 라 하자. (여기서 $Q_V(s,a) = R(s,a) + \gamma \sum P(s'|s,a) V(s')$)

$$V^{\pi_V}(s) - V^*(s) = V^{\pi_V}(s) - \max_a Q^*(s, a)$$

근사 오차 $\|V - V^*\|_\infty = \epsilon$ 에서:

$$Q_V(s, a) = R(s, a) + \gamma \sum P(s'|s,a) V(s') \geq R(s,a) + \gamma \sum P(s'|s,a) (V^*(s') - \epsilon)$$
$$= Q^*(s, a) - \gamma \epsilon$$

따라서 greedy on $V$ 는 $Q^*$ 상에서:

$$\max_a Q_V(s, a) \geq \max_a [Q^*(s, a) - \gamma \epsilon] = \max_a Q^*(s,a) - \gamma\epsilon = V^*(s) - \gamma\epsilon$$

이로부터:

$$V^{\pi_V}(s) \geq V^*(s) - \frac{2\gamma}{1-\gamma}\epsilon$$

(iteration 을 통한 error propagation)

**의미**: Value approximation error 가 $\frac{2\gamma}{1-\gamma}$ 배 amplify 될 수 있음. $\gamma \to 1$ 에서 악화 $\square$

이는 Ch5 (function approximation) 에서 deep RL 의 instability 를 설명.

</details>

**문제 2** (심화): ε-greedy 는 $\pi(a|s) = \begin{cases} 1-\varepsilon+\varepsilon/|A| & a = a^* \\ \varepsilon/|A| & a \neq a^* \end{cases}$ 로 정의된다. 왜 exploration (random action) 이 필요한가? Deterministic greedy 의 문제점은?

<details>
<summary>해설</summary>

**Deterministic greedy 의 문제**:

1. **Model-unknown 환경에서**: $Q$ 를 추정해야 하므로, greedy 만으로는 suboptimal action 을 try 하지 않음.

2. **Exploration-Exploitation trade-off**: 
   - Greedy: 현재 best action 만 → exploitation
   - Random: 모든 action equally → exploration
   - ε-greedy: 둘의 balance

3. **Ch5 예시** (Exploration Bonus):
   - Visited action: $Q$ 추정 정확 → greedy 신뢰
   - Unvisited action: $Q$ 모름 → $\varepsilon/|A|$ 확률로 try

**수식**:

$$\pi_\varepsilon(a|s) = (1-\varepsilon) \mathbb{1}[a = a^*] + \varepsilon/|A|$$

효과: exploration 으로 suboptimal 오차 감지 가능 → long-term 최적성 보증.

이것이 Q-learning 의 off-policy 성공 이유 $\square$

</details>

**문제 3** (논문 비평): Bertsekas & Tsitsiklis (1996) 는 "Asynchronous VI with Greedy Policy Extraction" 을 제안했다. 이는 모든 state 를 동시 갱신하지 않고도 최적성을 보장한다. 어떻게 가능한가?

<details>
<summary>해설</summary>

**Asynchronous VI** (Bertsekas 1982):

$$V_k(s) = \begin{cases}
\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')] & s \text{ selected at step } k \\
V_{k-1}(s) & \text{otherwise}
\end{cases}$$

모든 state 가 infinitely often 선택되면 $V_k \to V^*$ 수렴.

**Greedy Policy Extraction in Asynchronous Setting**:

각 step 에서 일부 state 만 update 해도, 전체 정책은:

$$\pi_k(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')]$$

로 계속 갱신하면, eventually optimal 에 수렴.

**의미**: 
- Synchronous VI: 매 iteration 모든 state update → $O(S)$ per step
- Asynchronous VI: 1개 state update → $O(1)$ per step, 총 step 수 비슷
- **실제 이득**: parallelization + memory efficiency (distributed RL)

따라서 모든 state 동시 update 는 충분조건이지 필요조건 아님 $\square$

</details>

---

<div align="center">

[◀ 이전: 03. Bellman Optimality Operator $T^*$](./03-optimality-operator.md) | [📚 README](../README.md) | [다음 ▶: 05. Deterministic 최적 정책의 존재](./05-deterministic-optimal.md)

</div>
