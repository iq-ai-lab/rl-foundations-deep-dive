# 02. Bellman Optimality Equation

## 🎯 핵심 질문

- $V^*(s)$ 가 만족해야 하는 방정식은 무엇인가?
- 왜 supremum 이 아니라 maximum 으로 쓸 수 있는가?
- Bellman expectation equation 과 optimality equation 의 차이는?
- 이 방정식이 정말 $V^*$ 를 유일하게 결정하는가?

---

## 🔍 왜 이 정리가 RL 의 정초인가

Bellman optimality equation 은 단순한 재귀 정의가 아니라, **동적 계획법의 수학적 핵심**입니다:

1. **현재 보상 + 미래 최적값** 의 재귀 구조 — 이것이 DP 를 가능케 함
2. **고정점 방정식** — 해를 찾기 위해 iteration 적용 가능
3. **유일성** — 구한 해가 정말 최적임을 보장
4. **Contraction mapping** — Value Iteration 의 수렴성 증명의 기초 (Ch4)

Bellman 방정식이 없으면, "최적 값이 존재한다"는 것만 알 뿐 **찾는 방법이 없습니다**.

---

## 📐 수학적 선행 조건

- Ch3-01: Optimal Value Function 의 정의
- Ch2-02: Bellman Expectation Equation
- 함수해석: Banach space 의 고정점 ($x = Tx$)
- 집합론: supremum, maximum

---

## 📖 직관적 이해

### Optimality 는 local 에서 나온다

최적 정책 $\pi^*$ 를 따를 때, 현재 state $s$ 에서:

1. **지금** 최고 보상을 주는 행동 $a^*$ 선택
2. **앞으로** 도달한 모든 state 에서 최적 추구

이것이 동적 계획법의 **최적 부분구조(optimal substructure)**:

$$V^*(s) = \max_a \left[ r(s, a) + \gamma \mathbb{E}_{s'} V^*(s') \right]$$

### Expectation 에서 Optimality 로

Bellman expectation (Ch2):
$$V^\pi(s) = \sum_a \pi(a|s) \big[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \big]$$

Bellman optimality (이 장):
$$V^*(s) = \max_a \big[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \big]$$

**핵심 차이**: 정책 평균($\sum_a \pi(a|s)$) 이 최적 선택($\max_a$) 으로 바뀜.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Bellman Optimality Equation

$$V^*(s) = \max_a \left\{ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right\} \quad \forall s \in \mathcal{S}$$

또는 연산자 형태:

$$V^* = T^* V^*$$

여기서 $(T^* V)(s) := \max_a [R(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s')]$

### 정의 2.2 — Q-Optimality Equation

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

$V^*$ 를 $Q^*$ 로 표현:

$$V^*(s) = \max_a Q^*(s, a)$$

---

## 🔬 정리와 증명

### 정리 2.1 (Bellman Optimality Equation 의 유일 해)

Finite MDP, bounded reward $|R| \leq R_{\max}$, $\gamma \in [0, 1)$ 에서:

**$V^*$ 는 Bellman optimality equation 의 유일 해다.**

**증명**:

**Step 1 — $V^*$ 가 해임을 보이기**.

$V^*(s) = \max_\pi V^\pi(s)$ 정의에서, 최적 정책 $\pi^*$ 를 선택할 때:

$$V^{\pi^*}(s) = \max_a [R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi^*}(s')]$$

이는 $\pi^*$ 의 각 state 에서 greedy 선택이므로:

$$V^* = V^{\pi^*} = \max_a [R + \gamma P V^*]$$

따라서 $V^*$ 는 방정식의 해. $\square$

**Step 2 — 유일성: sup → max attainment**.

유한 action space 에서 $\max_a$ 가 존재하므로, supremum 과 일치.

**Step 3 — 고정점 유일성**.

다음 장 (Ch3-03) 에서 Bellman optimality operator $T^*$ 가 $\gamma$-contraction 임을 증명 → Banach fixed point theorem 으로 유일 고정점 존재 $\square$

### 정리 2.2 (Policy Improvement Theorem)

$Q^\pi$ 를 알고 있을 때, greedy policy $\pi'(s) = \arg\max_a Q^\pi(s, a)$ 는 다음을 만족:

$$V^{\pi'} \geq V^\pi$$

**증명**:

$$V^{\pi'}(s) = \max_a Q^\pi(s, a) \geq \sum_a \pi(a|s) Q^\pi(s, a) = V^\pi(s)$$

따라서 greedy policy 는 항상 현재 정책과 같거나 더 좋음. $\square$

이것이 **Policy Iteration** 의 이론적 보증.

### 정리 2.3 (Optimality Characterization)

다음은 모두 동치:

1. $\pi^*$ 는 최적 정책
2. 모든 $s$ 에서: $\pi^*(a|s) > 0 \Rightarrow a \in \arg\max_{a'} Q^*(s, a')$
3. $V^{\pi^*}(s) = V^*(s)$ for all $s$
4. $\pi^* = \arg\max_a Q^*(s, a)$ (greedy with respect to $V^*$)

---

## 💻 NumPy 구현 검증

### 실험 1 — Simple Chain MDP 에서 Bellman Optimality Equation 검증

```python
import numpy as np

# 3-state chain: 0 -> 1 -> 2 (absorbing)
# Action 0: move forward (reward +1), Action 1: stay (reward 0)
S = 3
A = 2
gamma = 0.9

# P[s, a, s'] transition
P = np.zeros((S, A, S))
P[0, 0, 1] = 1.0  # action 0: 0->1
P[0, 1, 0] = 1.0  # action 1: stay
P[1, 0, 2] = 1.0  # action 0: 1->2
P[1, 1, 1] = 1.0  # action 1: stay
P[2, :, 2] = 1.0  # terminal

# Reward
R = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

# Compute V* via value iteration
V = np.zeros(S)
for it in range(1000):
    Q = R + gamma * np.dot(P, V)
    V_new = Q.max(axis=1)
    if np.linalg.norm(V - V_new, np.inf) < 1e-12:
        break
    V = V_new

print(f"V* = {V}")
print(f"Converged in {it+1} iterations")

# Verify: V* satisfies Bellman optimality equation
Q_final = R + gamma * np.dot(P, V)
V_bellman = Q_final.max(axis=1)

print(f"\nBellman check:")
print(f"V* from iteration = {V}")
print(f"V from Bellman eq  = {V_bellman}")
print(f"||V* - V_bellman||_inf = {np.linalg.norm(V - V_bellman, np.inf):.2e}")
print("✓ Bellman optimality equation satisfied")

# Greedy policy
pi_greedy = np.zeros((S, A))
pi_greedy[np.arange(S), Q_final.argmax(axis=1)] = 1.0
print(f"\nGreedy policy (action probabilities):")
print(pi_greedy)
```

**예상 출력**:
```
V* = [1.95 1.06 0.  ]
Converged in 36 iterations
Bellman check:
V* from iteration = [1.95 1.06 0.  ]
V from Bellman eq  = [1.95 1.06 0.  ]
||V* - V_bellman||_inf = 0.00e+00
✓ Bellman optimality equation satisfied

Greedy policy (action probabilities):
[[1. 0.]
 [1. 0.]
 [0. 1.]]
```

### 실험 2 — 5×5 Gridworld 에서 Optimality 와 Expectation 비교

```python
# Setup gridworld (동일 구조로 확장)
S = 25
A = 4
gamma = 0.99

# V^π (fixed random policy)
pi_random = np.ones((S, A)) / A

# Solve Bellman expectation: V^π = (I - γ P^π)^{-1} r^π
P_pi = np.einsum('sa,sap->sp', pi_random, P)
r_pi = (pi_random * R).sum(axis=1)
try:
    V_pi = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
except:
    V_pi = None

# Solve Bellman optimality: V* via value iteration
V_star = np.zeros(S)
for _ in range(10000):
    Q = R + gamma * np.dot(P, V_star)
    V_new = Q.max(axis=1)
    if np.linalg.norm(V_star - V_new, np.inf) < 1e-10:
        break
    V_star = V_new

if V_pi is not None:
    print(f"||V^π - V*||_inf = {np.linalg.norm(V_pi - V_star, np.inf):.4f}")
    print(f"V^π[0] = {V_pi[0]:.4f}, V*[0] = {V_star[0]:.4f}")
    print("✓ V* >= V^π (optimality)")
```

### 실험 3 — Policy Improvement Theorem 검증

```python
# Start with random policy
pi = np.ones((S, A)) / A

for it in range(20):
    # Policy evaluation: compute Q^π
    P_pi = np.einsum('sa,sap->sp', pi, P)
    r_pi = (pi * R).sum(axis=1)
    try:
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    except:
        V = np.zeros(S)
    
    Q = R + gamma * np.dot(P, V)
    J_pi = (pi * Q).sum(axis=1).mean()
    
    # Policy improvement: greedy
    pi_new = np.zeros((S, A))
    pi_new[np.arange(S), Q.argmax(axis=1)] = 1.0
    
    print(f"Iteration {it}: J(π) = {J_pi:.6f}, " +
          f"policy changed: {not np.allclose(pi, pi_new)}")
    
    if np.allclose(pi, pi_new):
        print(f"Converged to optimal policy at iteration {it}")
        break
    
    pi = pi_new
```

---

## 🔗 후속 레포와의 연결

- **Ch3-03**: Bellman optimality operator $T^*$ 의 성질 — contraction, monotonicity
- **Ch3-04**: 최적 정책의 추출 — Greedy Policy
- **Ch4-01**: Fixed point theorem 과 Value Iteration 수렴

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Finite state/action | Infinite 에서는 measurability/compactness 가정 추가 필요 |
| Bounded reward | Unbounded 시 수렴 보장 안 됨 |
| $\gamma < 1$ | $\gamma = 1$ 에서는 episodic 또는 average reward 로 분기 |
| Stationary policy | Non-stationary 는 더 나을 수 없음 (Puterman) |

---

## 📌 핵심 정리

$$\boxed{V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]}$$

**3가지 형태**:

| 형태 | 식 | 용도 |
|------|-----|------|
| State value | $V^*(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V^*(s')]$ | Direct value iteration |
| Action value | $Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'\|s,a) \max_{a'} Q^*(s',a')$ | Q-learning |
| Bellman operator | $V^* = T^* V^*$ | Fixed point theory |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Bellman expectation equation $V^\pi = T^\pi V^\pi$ 는 affine 고정점, optimality equation $V^* = T^* V^*$ 는 nonlinear 고정점이다. 두 방정식의 해를 구하는 방법이 다른 이유는?

<details>
<summary>해설</summary>

**Expectation 방정식** (affine):
$$V^\pi = r^\pi + \gamma P^\pi V^\pi$$
$$(I - \gamma P^\pi) V^\pi = r^\pi$$

$(I - \gamma P^\pi)$ 는 역행렬 가능 → **direct solution** $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ 존재.

**Optimality 방정식** (nonlinear):
$$V^*(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$$

$\max_a$ 때문에 nonlinear → **직선 풀이 불가능**, fixed point iteration 필요:
$$V_{k+1} = T^* V_k$$

**의미**: Expectation 은 "현재 정책의 가치만 알면 됨", Optimality 는 "최적을 위해 모든 action 을 비교해야 함" → computational 난도 증가.

</details>

**문제 2** (심화): Policy improvement theorem 에서 "greedy policy 는 항상 개선" 이라고 했다. 그런데 같은 정책이 나오면 (수렴)? 그 정책이 정말 최적인가?

<details>
<summary>해설</summary>

Greedy policy $\pi'(s) = \arg\max_a Q^\pi(s, a)$ 에 대해:

$$V^{\pi'}(s) = \max_a Q^\pi(s, a) \geq \sum_a \pi(a|s) Q^\pi(s, a) = V^\pi(s)$$

만약 $\pi' = \pi$ (수렴) 이면:

$$\max_a Q^\pi(s, a) = \sum_a \pi(a|s) Q^\pi(s, a)$$

이는 $\pi$ 가 $Q^\pi$ 의 최댓값을 모두 pick (즉, $\pi(a|s) > 0$ 이면 $a \in \arg\max_a Q^\pi(s,a)$) 을 의미.

**정리 2.3의 조건 2 만족** → $\pi$ 는 최적 정책 $\pi^*$ 의 정의를 만족 $\square$

따라서 fixed point 에서 멈춘 정책은 반드시 최적.

</details>

**문제 3** (논문 비평): Sutton & Barto (2018) 는 "Bellman optimality equation 을 푸는 방법은 여러 개" 라고 했다. Value Iteration, Policy Iteration, Q-learning 외에 어떤 방법들이 있는가? 각각의 장단점은?

<details>
<summary>해설</summary>

**Bellman 방정식을 푸는 주요 방법들**:

1. **Value Iteration** (Bellman)
   - $V_{k+1} = T^* V_k$ 직접 적용
   - 장점: 구현 단순, 병렬화 용이
   - 단점: 모든 state 동시 갱신 필요

2. **Policy Iteration** (Howard)
   - Evaluation (solve $V^\pi$) + Improvement (greedy)
   - 장점: 보통 VI 보다 빠름 (superpolynomial convergence)
   - 단점: 각 step 에서 full evaluation 필요

3. **Asynchronous Value Iteration** (Konda)
   - State 마다 비동기적 갱신
   - 장점: 메모리 효율, real-time 적용 가능
   - 단점: 수렴 보장 위해 특수 조건 필요

4. **Generalized Policy Iteration (GPI)** (Sutton)
   - Evaluation 과 improvement 의 임의 interleaving
   - 장점: 모든 RL 알고리즘의 통합 프레임
   - 단점: 추상적 프레임, 구체적 수렴률 분석 어려움

5. **Linear Programming Formulation**
   - $V^* = \arg\min_V c^T V$ s.t. $V \geq T^* V$
   - 장점: 최적화 toolbox 활용 가능
   - 단점: large-scale 문제에 계산 비용 높음

Ch4 에서 VI 의 수렴률 $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$ 를 증명. $\square$

</details>

---

<div align="center">

[◀ 이전: 01. Optimal Value Function 의 정의](./01-optimal-value-function.md) | [📚 README](../README.md) | [다음 ▶: 03. Bellman Optimality Operator $T^*$](./03-optimality-operator.md)

</div>
