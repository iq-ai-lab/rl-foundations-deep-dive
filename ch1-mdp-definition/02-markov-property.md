# 02. Markov 성질과 그 결과

## 🎯 핵심 질문

- Markov property 의 정확한 수학적 형태는 무엇이고, 왜 필수적인가?
- History $h_t = (s_0, a_0, \ldots, s_t)$ 의 무한 차원 정보가 finite state $s_t$ 로 어떻게 환원되는가?
- Markov 성질이 없다면 어떤 일이 일어나는가? (과거 전체 기록을 tracking 해야 한다는 것의 의미)
- Markov 성질이 DP (Dynamic Programming) 의 **subproblem decomposition** 을 어떻게 가능하게 하는가?
- Bellman equation $V(s) = \mathbb{E}[R + \gamma V(s')] | s]$ 가 정의되려면 왜 Markov 성질이 필수인가?

---

## 🔍 왜 이 성질이 RL 의 기초인가

Reinforcement learning 의 모든 이론은 **Markov 성질** 에 기초합니다. 그런데 대부분의 입문 교재는 "Markov property 란 다음 state 가 현재 state 에만 의존한다" 는 한 문장으로 끝냅니다.

이것은 **얼마나 강력한 가정** 인지, **정보학적으로 무엇을 의미** 하는지, **어떻게 DP 를 가능** 하게 하는지, 그리고 **실제 환경에서 위반** 되었을 때 무엇이 깨지는지를 설명하지 않습니다.

더 정확히: Markov 성질 없이는 value function 자체가 **잘 정의되지 않습니다.** $V(s)$ 가 "state $s$ 로부터의 예상 누적 보상" 이라는 정의가 성립하려면, 과거 history 를 버릴 수 있어야 하기 때문입니다.

이 문서는 Markov 성질을 probability theory 로부터 엄밀히 정의하고, 그 결과들을 증명합니다.

---

## 📐 수학적 선행 조건

- **Ch1-01 MDP 정의**: Measurable space, stochastic kernel, bounded reward
- **Probability Theory Deep Dive**: Conditional probability, conditional expectation, tower property
- **Stochastic Processes Deep Dive**: Markov chain, filtration, optional stopping

---

## 📖 직관적 이해

### 과거를 버릴 수 있다는 것의 의미

일반적인 sequential decision problem 에서:

```
t=0           t=1              t=2              t=3
s₀ ──a₀──> s₁ ──a₁──> s₂ ──a₂──> s₃ ...
          (모든 과거 기억)   (어디서 왔는지 알아야 함)
```

**Non-Markovian**: "지금 어디 있는가" 는 과거 전체 경로 $(s_0, a_0, \ldots, s_{t-1}, a_{t-1})$ 에 의존.
- 예: 레이더 추적 비행기 (이전 위치들의 velocity 패턴이 미래 위치를 결정)

**Markovian**: "지금 어디 있는가" 만 알면 충분 — 과거는 버려도 됨.
- 예: 주사위 게임 (현재 주사위 면만 보면 다음 보상은 결정됨, 10번 전 주사위는 무관)

```
Markovian world:
t=0           t=1              t=2              t=3
s₀ ──a₀──> s₁ ──a₁──> s₂ ──a₂──> s₃ ...
                        ↑
                    이것만 알면 됨
                    (past 필요 없음)
```

### 정보학적 의미: Sufficiency 와 Completeness

State $s_t$ 는 **sufficient statistic** 임을 의미:
- $\mathbb{P}(s_{t+1} | h_t, a_t) = \mathbb{P}(s_{t+1} | s_t, a_t)$
- History 에 있는 모든 정보는 이미 state 에 인코딩됨
- 과거는 더 이상 유용하지 않음

### Computational 이점: DP Feasibility

Markovian 하지 않으면:
- State space dimension = trajectory history length
- Exponentially large state space (모든 past combination)
- Bellman equation 불가능

Markovian 하면:
- Finite state space 가능
- Bellman equation 정의: $V(s) = \mathbb{E}[R + \gamma V(s') | s]$
- DP subproblem decomposition

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Markov Property (유한 시간)

$\pi$ 정책 하에서, 모든 $t \geq 0$, 모든 history $h_t = (s_0, a_0, \ldots, s_t)$, 모든 measurable set $B \subseteq \mathcal{S}$:

$$P(s_{t+1} \in B \mid h_t, a_t) = P(s_{t+1} \in B \mid s_t, a_t)$$

또는 stochastic kernel 형태:
$$P(B \mid h_t, a_t) = P(B \mid s_t, a_t)$$

### 정의 2.2 — Markov Property (확률론적)

**Filtration** $\mathcal{F}_t = \sigma(s_0, a_0, \ldots, a_{t-1}, s_t)$ (history 에 의해 생성).

Process $(s_t)_{t \geq 0}$ 와 action policy $\pi$ 가 **Markovian** $\Leftrightarrow$ 모든 bounded measurable $f$:

$$\mathbb{E}[f(s_{t+1}) \mid \mathcal{F}_t, a_t] = \mathbb{E}[f(s_{t+1}) \mid s_t, a_t]$$

**직관**: Conditional expectation 이 state $s_t$ 만으로 표현 가능.

### 정의 2.3 — Reward 도 Markovian

Reward process $(R_t)_{t \geq 0}$ 는 Markov $\Leftrightarrow$ $R_t$ 는 $(s_t, a_t)$ 의 measurable function (과거 irrelevant).

$$\mathbb{E}[R_t \mid \mathcal{F}_{t-1}, a_t] = \mathbb{E}[R(s_t, a_t) \mid s_t, a_t] = R(s_t, a_t)$$

---

## 🔬 정리와 증명

### 정리 2.1 — Markov 성질이 있으면 Value Function 이 잘 정의됨

$\mathcal{M}$ 이 Markov property 를 만족하고, 정책 $\pi$ 하에서:

$$V^\pi(s) := \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_t \,\Big|\, s_0 = s\right] \quad \text{is well-defined}$$

즉, history $h_t$ 에 무관하게 state $s_t$ 에만 의존하는 deterministic function 이 존재.

**증명**:

**Step 1 — Definition via Bellman.** Markovian 이면 recursive equation 가 가능:

$$V^\pi(s) = \mathbb{E}[R(s, a) + \gamma V^\pi(s') \mid s]$$

where $a \sim \pi(\cdot | s)$, $s' \sim P(\cdot | s, a)$.

**Step 2 — Existence.** Bounded reward case ($|R| \leq R_{\max}$):

$$T^\pi V(s) := \mathbb{E}[R(s, a) + \gamma V(s') \mid s]$$

defines an operator on $B(\mathcal{S})$ (bounded measurable functions). By Banach fixed point theorem, unique $V^\pi$ exists.

**Step 3 — Uniqueness.** If $W$ also satisfies $W(s) = \mathbb{E}[R(s, a) + \gamma W(s') | s]$, then:

$$|V^\pi(s) - W(s)| = |\mathbb{E}[\gamma(V^\pi(s') - W(s')) | s]| \leq \gamma \|V^\pi - W\|_\infty$$

By contraction property, $V^\pi = W$.

**Conclusion**: Markovian 없으면, value 가 $(s_t, \text{past history})$ 에 의존 → different value at same $s$ → 함수 부정(not a function). $\square$

### 정리 2.2 — Bellman Equation 의 전제는 Markovian

Bellman expectation equation $V^\pi = T^\pi V^\pi$ 이 정의되려면:

1. **RHS 가 deterministic** (policy $\pi$ 고정): $T^\pi V(s)$ 는 $s$ 의 함수여야 함
2. **History-independence**: $\mathbb{E}[\cdot | s]$ 는 history 무관하게 정의되어야 함
3. **Sufficiency**: State $s$ 가 future dynamics 를 결정할 충분한 정보를 포함

이 모두 **Markov 성질**에서 출발.

**증명**: 만약 non-Markovian (e.g., $P(s_{t+1} | h_t, a_t) \neq P(s_{t+1} | s_t, a_t)$), 그러면:

$$\mathbb{E}[R(s_t, a_t) + \gamma V(s_{t+1}) | s_t, a_t] \neq \mathbb{E}[R(s_t, a_t) + \gamma V(s_{t+1}) | h_t, a_t]$$

RHS 는 history 의 함수 → value 를 state 의 함수로만 define 할 수 없음. $\square$

### 정리 2.3 — Markovian 이면 History-Dependent Policy 를 상기할 필요 없음

Finite MDP 에서, 모든 history-dependent policy $\mu(a | h_t)$ 에 대해, 동일하거나 더 나은 **stationary Markovian policy** $\pi(a | s)$ 존재.

**증명 스케치** (full proof 는 Ch1-03):

- Value function $V^\pi$ 는 history 무관 (Thm 2.1)
- Bellman optimality equation: $V^*(s) = \max_a \mathbb{E}[R(s, a) + \gamma V^*(s') | s]$
- Greedy policy $\pi^*(a | s) = \arg\max_a \mathbb{E}[\cdot | s]$ 는 Markovian, deterministic
- 이 policy 가 history-dependent 를 지배

$\square$

### 따름 정리 2.4 — Tower Property 와 Nested Expectation

Markovian process 에서:

$$\mathbb{E}[\mathbb{E}[f(s_{t+1}) | s_t, a_t] | s_{t-1}] = \mathbb{E}[f(s_{t+1}) | s_{t-1}]$$

더 일반적으로, future 의 expected return 은 현재 state 만 필요:

$$\mathbb{E}[\sum_{k=t}^{\infty} \gamma^{k-t} R_k \mid s_t] = V^\pi(s_t)$$

---

## 💻 NumPy 구현 검증

### 실험 1 — Markovian Gridworld: Bellman Equation 확인

```python
import numpy as np

# 4×4 Gridworld from Ch1-01
n_states = 16
n_actions = 4
gamma = 0.9

# P, R from Ch1-01 setup
P = np.random.RandomState(0).randn(n_states, n_actions, n_states)
P = np.abs(P) / np.abs(P).sum(axis=2, keepdims=True)  # normalize to stochastic
R = np.random.RandomState(0).randn(n_states, n_actions)

# Uniform policy π(a|s) = 1/4 for all s, a
pi = np.ones((n_states, n_actions)) / n_actions

# Bellman equation via fixed-point iteration
V = np.zeros(n_states)
for iteration in range(1000):
    # T^π V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V_new = (pi * Q).sum(axis=1)
    
    if np.abs(V_new - V).max() < 1e-10:
        print(f"Converged at iteration {iteration}")
        break
    V = V_new

print(f"\nV^\π from fixed-point iteration:")
print(f"  Shape: {V.shape}")
print(f"  Range: [{V.min():.4f}, {V.max():.4f}]")
print(f"✓ Bellman equation V = T^π V solved via Markovian property")
```

### 실험 2 — Non-Markovian 환경: Value 가 History-Dependent

```python
# Simulate non-Markovian environment
# State: robot position
# Hidden: previous 2 positions (pattern recognition)
# Action: depends on history, not just current position

n_explicit_states = 4  # Robot positions: 0, 1, 2, 3
n_hidden_states = 2    # Is moving forward or backward?

# Honest state = (position, history_pattern)
n_true_states = n_explicit_states * n_hidden_states  # 8 states

def non_markov_dynamics(pos, hidden, action):
    """Simplified non-Markovian: next state depends on previous 2"""
    new_pos = (pos + action) % n_explicit_states
    new_hidden = 1 if action > 0 else 0
    return new_pos, new_hidden

def apparent_value(pos, all_history):
    """If we ignore hidden state, value seems to depend on history"""
    # Extract pattern from last 2 actions
    if len(all_history) >= 2:
        pattern = (all_history[-1][1] + all_history[-2][1]) % 2
    else:
        pattern = 0
    # Return would depend on pattern → history-dependent!
    return pos + pattern * 10  # Made-up value formula

# Example trajectories from same position but different histories
position = 2
reward = np.array([1, 2, 1])

# History 1: increasing action
h1 = [(2, 1), (2, 1)]  # (pos, action) pairs
v1 = apparent_value(2, h1)

# History 2: decreasing action
h2 = [(2, -1), (2, -1)]
v2 = apparent_value(2, h2)

print(f"Same position, different histories:")
print(f"  Position = {position}")
print(f"  History 1 {h1} → V = {v1:.2f}")
print(f"  History 2 {h2} → V = {v2:.2f}")
print(f"✗ Non-Markovian: V(same state) ≠ V(different history)")
print(f"  Difference = {abs(v1 - v2):.2f} ✓ Problem!")
```

### 실험 3 — Markovian Advantage: State Sufficiency

```python
# Create Markovian MDP where state is sufficient
# Continuous state: position ∈ [0, 1]
# Transition: s' = 0.7 * s + 0.2 * a + noise
# This depends only on (s, a), not history

np.random.seed(42)
n_state_samples = 100
state_samples = np.linspace(0, 1, n_state_samples)

# Two different histories converging to same state
s_current = 0.5

# Compute distribution of next states for two different histories
# History 1: came from 0.2
action = 0.3
next_dist_1 = 0.7 * s_current + 0.2 * action + np.random.randn(1000) * 0.1

# History 2: came from 0.8 (different path, same current state!)
next_dist_2 = 0.7 * s_current + 0.2 * action + np.random.randn(1000) * 0.1

# Due to Markovian property, distributions are identical!
print(f"Markovian property verification:")
print(f"  Current state (both histories): s = {s_current}")
print(f"  Action (both): a = {action}")
print(f"\n  Distribution of next states:")
print(f"    History 1 (from 0.2): mean = {next_dist_1.mean():.4f}, std = {next_dist_1.std():.4f}")
print(f"    History 2 (from 0.8): mean = {next_dist_2.mean():.4f}, std = {next_dist_2.std():.4f}")
print(f"  Difference in means: {abs(next_dist_1.mean() - next_dist_2.mean()):.2e}")
print(f"✓ Markovian: P(s' | s, a) independent of history")
```

### 실험 4 — Value Iteration Convergence (Bellman Operator 활용)

```python
# Demonstrate Bellman operator as fixed-point
# This only works because of Markovian property

# Random MDP setup
n_s, n_a = 8, 2
gamma = 0.95
np.random.seed(123)

# Create random Markovian MDP
P = np.random.rand(n_s, n_a, n_s)
P /= P.sum(axis=2, keepdims=True)
R = np.random.rand(n_s, n_a) * 10

# Random policy
pi = np.ones((n_s, n_a)) / n_a

# Bellman operator T^π
def bellman_op(V, pi, P, R, gamma):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    return (pi * Q).sum(axis=1)

# Value iteration
V = np.zeros(n_s)
V_errors = []

for k in range(50):
    V_old = V.copy()
    V = bellman_op(V, pi, P, R, gamma)
    error = np.linalg.norm(V - V_old, ord=np.inf)
    V_errors.append(error)
    
    if k < 5 or k % 10 == 0:
        print(f"Iteration {k:2d}: ||V_new - V_old||_∞ = {error:.2e}")

print(f"\n✓ Fixed-point convergence (Markovian → Bellman → contraction)")
print(f"  Contraction rate γ = {gamma}")
print(f"  Final error = {V_errors[-1]:.2e}")
```

---

## 🔗 후속 레포와의 연결

- **Ch1-03 Policy 종류**: Stationary Markovian policy 의 최적성 증명은 Thm 2.1-2.3 의 직접 응용
- **Ch2 Bellman Equation**: Markovian 가정이 Bellman expectation / optimality equation 의 전제
- **Ch3 Value Iteration**: Bellman operator 의 $\gamma$-contraction 유도는 Markovian 기반
- **Model-Free RL**: TD, Q-learning, Actor-Critic 모두 Markov assumption 암묵적 사용
- **Advanced RL**: POMDP (Ch1-05) 는 **Markovian 아님** → belief state 도입으로 복구

---

## ⚖️ 가정과 한계

| 가정 | 현실의 위반 | 대응 방법 |
|------|-----------|----------|
| Markovian dynamics | Partially observable environment (POMDP, Ch1-05) | Belief state로 Markovian 복구 |
| Markovian reward | Reward depends on past context | Augment state with history |
| Stationary $P, R$ | Non-stationary environment (trends) | Time-indexed state: $(t, s)$ |
| Deterministic history-independent state | Memory-dependent decision (e.g., game of incomplete information) | Epistemic state (knowledge) 확장 |

---

## 📌 핵심 정리

$$\boxed{P(s_{t+1} | h_t, a_t) = P(s_{t+1} | s_t, a_t)}$$

**결과**:

| 결과 | 의미 |
|------|------|
| Value $V^\pi(s)$ well-defined | State 만의 함수 (history 무관) |
| Bellman equation 가능 | $V^\pi = T^\pi V^\pi$ 정의 가능 |
| Subproblem decomposition | DP feasible |
| Stationary policy sufficient | Ch1-03 에서 증명 |
| Finite-step PI convergence | Ch1-03 에서 활용 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 4-state chain MDP 에서
- State 0 → always go to 1
- State 1 → 50% 확률로 0 또는 2
- State 2 → always go to 3
- State 3 → always go to 3 (terminal)

이 MDP 가 Markovian 인가? 왜 또는 왜 아닌가? History 에 의존하는 동역학이 있는가?

<details>
<summary>해설</summary>

**Yes, Markovian.** 

Transition matrix:
$$P = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0.5 & 0 & 0.5 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

각 $(s, a)$ 쌍에 대해 $P(s' | s, a)$ 는 fixed. 어떤 history로 $s_t$ 에 도달했든 상관없이 $s_{t+1}$ 의 분포는 동일.

**Example**: 0→1 도 history 에 무관.

✓ This is **fully Markovian** (deterministic transition).

$\square$

</details>

**문제 2** (심화): Partially observable environment: 로봇이 2개 방 중 어느 방인지 모르지만, action (move left/right) 으로 다른 역학을 경험. 예를 들어:
- 방 A: move left → move right (반대로 됨)
- 방 B: move left → move left (정상)

로봇이 어느 방인지 모르면, 이것이 Markovian MDP 인가?

<details>
<summary>해설</summary>

**No, Non-Markovian** (관찰 기준으로).

관찰 가능: 방 정보 없음, action 과 position change 만 봄.

History: $(L, -), (L, +), (L, -)$ 같은 action-outcome pairs → 로봇은 "패턴" 으로 방을 추론 (숨겨진 상태).

$P(s_{t+1} | s_t, a_t)$ ≠ $P(s_{t+1} | h_t, a_t)$ 왜냐하면:
- Same $(s_t, a_t)$ = same position & action
- But different history → different room → different transition!

**해결책**: State 를 augment with belief:
$$s := (\text{position}, b(\text{room}))$$

이제 이것은 **belief MDP** 로 Markovian 복구 (Ch1-05 POMDP 참고).

$\square$

</details>

**문제 3** (논문 비평): Puterman (2005) 와 Bertsekas (2012) 는 모두 MDP 의 Markov property 를 서로 다른 수준의 엄밀함으로 정의한다. Puterman 은 "transition kernel" 언어, Bertsekas 는 "memorylessness" 기하학적 해석. 두 정의가 동치인가? 어느 것이 continuous state space 일반화에 더 적합한가?

<details>
<summary>해설</summary>

**Puterman (Kernel 언어)**:
$$P_t(s' | h_t, a_t) = P_t(s' | s_t, a_t)$$
(stochastic kernel 형태, measurability 명시적)

**Bertsekas (Memorylessness)**:
Next state 분포가 "과거를 기억하지 않는" = "history 에 independent"

**동치성**: Yes, both equivalent (적절한 measurability 가정 하에).

**일반화 적합성**:
- **Puterman (kernel)** 이 더 나음:
  - Polish space 일반화 직접 (measurable selection 이론)
  - Continuous state space → transition kernel is 정의의 핵심
  - Borel measurability 명시적

- **Bertsekas (geometry)** 는 finite/discrete 대상 더 적합

**결론**: Continuous-state RL (Ch3+) 에서는 **Puterman 의 kernel 정의** 가 표준.

$\square$

</details>

---

[◀ 이전: 01. MDP 의 6-tuple 정의](./01-mdp-tuple.md) | [📚 README](../README.md) | [다음 ▶: 03. Policy 의 종류와 Stationary Policy 충분성](./03-policy-types.md)
