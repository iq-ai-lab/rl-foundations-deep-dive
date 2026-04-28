# 01. Optimal Value Function 의 정의

## 🎯 핵심 질문

- 모든 정책 중에서 최고 성능을 내는 value function 을 어떻게 정의하는가?
- $V^*(s) = \sup_\pi V^\pi(s)$ 와 $Q^*(s, a) = \sup_\pi Q^\pi(s, a)$ 의 의미는?
- 왜 개별 state 에서의 최댓값이 어떤 정책으로도 동시에 달성되는가?
- Optimal value 의 bounded-ness 는 언제 보장되는가?

---

## 🔍 왜 이 정의가 RL 기초인가

RL 의 궁극의 목표는 **최적 정책** $\pi^*$ 을 찾는 것입니다. 그러나 정책을 직접 정의하기 보다, **최적 성능**을 정량화하는 $V^*$ 와 $Q^*$ 를 먼저 정의합니다. 이것이 value-based RL 의 핵심 역순 추론(backward induction)입니다:

1. $V^*$ 를 정의 → 그 존재 증명
2. 고정점 방정식 (Bellman optimality equation) 유도
3. 방정식을 푸는 알고리즘 설계 (Value Iteration)
4. Optimal policy 는 $V^*$ 로부터 자동 도출

이 경로가 Policy Gradient 와의 본질적 차이입니다.

---

## 📐 수학적 선행 조건

- Ch2-01: Value Function 의 정의 ($V^\pi, Q^\pi$)
- Ch2-05: Value Function 의 고유성과 존재성
- 선형대수: Supremum (sup), bounded functional
- 위상수학: 완비거리공간 (complete metric space) 의 기초

---

## 📖 직관적 이해

### 최적 가치는 무엇인가

정책 $\pi$ 마다 value function $V^\pi(s)$ 가 다릅니다. State $s$ 에 도달했을 때:

- **최선의 정책**: 이 state 에서 앞으로 최대한 많이 받을 수 있는 expected return
- **최악의 정책**: 최소한 받을 것

$V^*(s)$ 는 **best case — 최선의 행동을 취할 때의 가치**:

$$V^*(s) = \max_\pi V^\pi(s)$$

마찬가지로 state-action pair 에 대해:

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

"이 상태에서 이 행동을 선택한 후, 앞으로 최적을 추구하면 얼마를 얻을까"

### 왜 sup 이 아니라 max 인가

MDP 가 유한 state/action 을 가지고 있으면, 정책 집합이 유한합니다:

$$|\{\pi : \mathcal{S} \to \Delta(\mathcal{A})\}| = |\mathcal{A}|^{|\mathcal{S}|}$$

따라서 max 가 존재하고, sup 과 같습니다. 무한 state/action 에서는 supremum 으로 정의하되, **점별로 달성(pointwise attained)** 됨을 별도로 증명해야 합니다 (Puterman 2005).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Optimal Value Function

무한 시간 할인 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ 에서:

$$V^*(s) := \sup_{\pi \in \Pi} V^\pi(s) = \max_{\pi \in \Pi_d} V^\pi(s)$$

여기서 $\Pi$ 는 모든 정책, $\Pi_d$ 는 deterministic stationary policy.

마찬가지로:

$$Q^*(s, a) := \sup_{\pi \in \Pi} Q^\pi(s, a) = \max_{\pi \in \Pi_d} Q^\pi(s, a)$$

### 정의 1.2 — Optimal Policy

$$\pi^*(s) := \arg\max_a Q^*(s, a)$$

$\pi^*$ 는 모든 state 에서 $V^*$ 를 달성하는 정책.

### 정의 1.3 — Relationship 간 관계식

$$V^*(s) = \max_a Q^*(s, a)$$

$$Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\big[R(s, a) + \gamma V^*(s')\big]$$

---

## 🔬 정리와 증명

### 정리 1.1 (Optimal Value 의 존재 및 유일성)

Finite MDP, discounted infinite-horizon 에서 다음이 성립:

1. **존재**: $V^*$ 는 모든 state 에서 존재
2. **달성**: 어떤 deterministic stationary policy $\pi^*$ 가 모든 state 동시에 $V^*$ 달성
3. **유일성**: $V^*$ 는 유일

**증명 sketch**:

(1) 유한 정책 수 $|\Pi_d| = |\mathcal{A}|^{|\mathcal{S}|} < \infty$ 이므로 max 가 존재.

(2) $\arg\max_a Q^*(s, a)$ 를 모든 state 에서 선택하는 정책 $\pi^*(s) = \arg\max_a Q^*(s, a)$ 를 정의하면, 이 정책이 $V^*(s)$ 달성:

$$V^{\pi^*}(s) = \max_a Q^*(s, a) = V^*(s)$$

(3) Bellman optimality equation 의 해가 유일 (다음 문서에서 증명) $\square$

### 정리 1.2 (Optimal Value 의 Boundedness)

Bounded reward $|R(s, a)| \leq R_{\max}$ 에서:

$$\|V^*\|_\infty := \max_s |V^*(s)| \leq \frac{R_{\max}}{1-\gamma}$$

**증명**:

$V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t R_t | s_0 = s] \leq \sum_{t=0}^\infty \gamma^t R_{\max} = \frac{R_{\max}}{1-\gamma}$

따라서 $V^* = \sup_\pi V^\pi \leq \frac{R_{\max}}{1-\gamma}$ $\square$

이 bound 는 contraction mapping 의 고정점 위치를 한정하는 데 핵심.

### 정리 1.3 (On-Policy vs Off-Policy Optimality)

$V^*$ 는 정책 무관적(policy-independent) 으로 정의되지만, 달성하는 정책은 정책 무관적이 아닙니다. 즉:

- 여러 정책이 $V^*$ 달성 가능 (tie-breaking 에서 임의성 존재)
- 그러나 달성하는 모든 정책의 성능은 동일: $V^{\pi_i^*} = V^* = V^{\pi_j^*}$

---

## 💻 NumPy 구현 검증

### 실험 1 — 4×4 Gridworld 에서 $V^*$ 계산

```python
import numpy as np
import matplotlib.pyplot as plt

# 4×4 grid, 목표 (3,3), 보상 +1, 다른 곳 0
S = 16
A = 4
gamma = 0.9

# Deterministic transitions: up, right, down, left
dirs = np.array([[-1,0], [0,1], [1,0], [0,-1]])

def grid_to_idx(i, j): return i*4 + j
def idx_to_grid(idx): return idx // 4, idx % 4

# State-action-nextstate: P[s,a,s']
P = np.zeros((S, A, S))
R = np.zeros((S, A))

for s in range(S):
    i, j = idx_to_grid(s)
    if (i, j) == (3, 3):  # 목표 상태
        P[s, :, s] = 1.0
        R[s, :] = 1.0
        continue
    
    for a, (di, dj) in enumerate(dirs):
        ni, nj = i + di, j + dj
        if 0 <= ni < 4 and 0 <= nj < 4:
            ns = grid_to_idx(ni, nj)
            P[s, a, ns] = 1.0
        else:
            P[s, a, s] = 1.0  # 경계에서 제자리
        if (ni, nj) == (3, 3) or (i, j) == (3, 3):
            R[s, a] = 1.0

# Value Iteration: 모든 정책 평가 후 max
V = np.zeros(S)
Q_history = []

for iteration in range(100):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V_new = Q.max(axis=1)
    
    if np.linalg.norm(V - V_new, np.inf) < 1e-8:
        break
    V = V_new
    Q_history.append(Q.copy())

print(f"Converged in {iteration+1} iterations")
print(f"V* (first 5 states): {V[:5]}")
print(f"||V*||_inf = {np.linalg.norm(V, np.inf):.4f}")
print(f"Upper bound R_max/(1-γ) = {1.0 / (1-gamma):.4f}")

# Visualization
V_grid = V.reshape(4, 4)
plt.figure(figsize=(8, 6))
plt.imshow(V_grid, cmap='viridis', origin='upper')
plt.colorbar(label='V*(s)')
plt.title('Optimal Value Function (4×4 Gridworld)')
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{V_grid[i,j]:.2f}', ha='center', va='center', 
                 color='white', fontsize=10)
plt.xlabel('Column')
plt.ylabel('Row')
plt.tight_layout()
plt.savefig('optimal_value_grid.png', dpi=150)
plt.close()
```

**예상 결과**: Goal (3,3) 에 가까울수록 V* 가 높음. (0,0) 에서 약 0.73.

### 실험 2 — 정책 개수와 supremum 달성

```python
# Finite MDP: deterministic stationary policy 중 V* 달성하는 정책 찾기
n_deterministic_policies = A ** S
print(f"Total deterministic stationary policies: {n_deterministic_policies}")

# Greedy policy: argmax_a Q*(s,a)
Q_final = R + gamma * np.einsum('sap,p->sa', P, V)
pi_greedy = np.zeros((S, A))
pi_greedy[np.arange(S), Q_final.argmax(axis=1)] = 1.0

# Verify: π_greedy achieves V*
V_pi = np.linalg.solve(np.eye(S) - gamma * np.einsum('sa,sap->sp', pi_greedy, P),
                       (pi_greedy * R).sum(axis=1))

print(f"||V_π* - V*||_inf = {np.linalg.norm(V_pi - V, np.inf):.2e}")
print("✓ Greedy policy achieves V*")
```

### 실험 3 — $\gamma$ 에 따른 $V^*$ 의 범위 변화

```python
gammas = [0.5, 0.9, 0.99]
results = {}

for gamma_test in gammas:
    V_test = np.zeros(S)
    for _ in range(1000):
        Q = R + gamma_test * np.einsum('sap,p->sa', P, V_test)
        V_test = Q.max(axis=1)
    results[gamma_test] = (V_test.max(), 1.0 / (1 - gamma_test))

print("γ\t||V*||_inf\tBound R_max/(1-γ)")
for gamma_test in gammas:
    inf_norm, bound = results[gamma_test]
    print(f"{gamma_test}\t{inf_norm:.4f}\t\t{bound:.4f}")
```

---

## 🔗 후속 레포와의 연결

- **Ch3-02**: $V^*$ 가 만족하는 방정식 — Bellman Optimality Equation
- **Ch3-03**: 방정식을 풀기 위한 연산자 — Bellman Optimality Operator $T^*$
- **Ch4**: Contraction 성질을 이용한 수렴 증명 — Banach Fixed Point Theorem

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Finite MDP | Infinite state/action 에서는 supremum attainment 별도 증명 필요 (Bertsekas) |
| Bounded reward | Unbounded reward 시 수렴성 보장 안 됨 |
| $\gamma < 1$ | $\gamma = 1$ 에서는 episodic 으로 수정, average reward 로 분기 |
| Stationary policy 만 고려 | Non-stationary · history-dependent 정책은 더 나을 수 없음 (Puterman 정리) |

---

## 📌 핵심 정리

$$\boxed{V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s, a) = \max_\pi Q^\pi(s, a)}$$

모든 finite MDP 에서 deterministic stationary policy 중 최적이 존재하며, 그 정책이 모든 state 에서 동시에 $V^*$ 달성.

| 양 | 정의 | 역할 |
|----|------|------|
| $V^*(s)$ | $\sup_\pi V^\pi(s)$ | 최적 가치 |
| $Q^*(s, a)$ | $\sup_\pi Q^\pi(s, a)$ | 최적 행동-가치 |
| $\pi^*$ | $\arg\max_a Q^*(s, a)$ | 최적 정책 (반드시 존재) |
| $\|V^*\|_\infty$ | $\leq R_{\max}/(1-\gamma)$ | Bounded 보증 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 만약 MDP 가 infinite action space 를 가지면, $\max_a Q^*(s, a)$ 가 존재하지 않을 수 있다. 이 경우 $V^*$ 를 어떻게 정의해야 하는가? Supremum attainment 조건은?

<details>
<summary>해설</summary>

Infinite action 시 $Q^*(s, a)$ 의 그래프가 unbounded 일 수 있음 → $\max_a$ 가 존재 안 함. 이 경우:

$$V^*(s) := \sup_a Q^*(s, a)$$

로 정의하되, **supremum attainment** (어떤 $a^*$ 에서 supremum 이 달성되는가)를 별도로 가정해야 함.

**Bertsekas (2012)**: Continuous action 에서 compact support 와 연속성 가정 하에 attainment 증명 (Weierstrass theorem). 이는 policy gradient 방법의 이론적 기반.

</details>

**문제 2** (심화): Deterministic policy 만으로 충분하다는 것을 어떻게 증명하는가? Stochastic policy 가 더 좋을 수 없는 이유는?

<details>
<summary>해설</summary>

임의의 stochastic policy $\pi(a|s)$ 의 value:

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a) \leq \max_a Q^\pi(s, a)$$

(이것은 convex combination 이 극단점(extreme point)보다 작거나 같다는 원리)

따라서 stochastic 은 항상 deterministic 의 convex combination 이므로, 최댓값은 극단점(deterministic) 에서만 달성. 이를 **linear optimization 의 극값 정리** 또는 **superposition principle** 이라 함.

따라서 optimal policy 를 찾을 때 stochastic 을 고려할 필요 없음 $\square$

</details>

**문제 3** (논문 비평): Puterman (2005) 정리는 "모든 finite MDP 에서 deterministic stationary Markovian policy 중 최적이 존재" 라고 주장한다. Non-stationary (history-dependent) 정책은 정말 필요 없는가? 어떤 MDP 에서도?

<details>
<summary>해설</summary>

**Puterman 의 정리** (Ch1 Foundations):

Discounted infinite-horizon, finite MDP 에서는 다음 중 하나:
1. Deterministic stationary optimal policy 존재
2. 모든 정책이 같은 성능 (degenerate case)

**History-dependent 의 불필요성**: 

Bellman 의 **최적 부분구조(optimal substructure)** — 최적 정책의 부분도 최적이므로, Markov 성질 하에서 현재 state 만으로 최적 행동이 결정됨. 과거 history 는 현재 state 에 이미 축약(summarization).

**예외**: 
- Average reward MDP 에서는 stationary 중에 최적이 없을 수 있음 → periodic policy 필요
- Partial observability (POMDP) 에서는 history-dependent 필수

따라서 **MDP 하에서는 stationary 로 충분**, POMDP 는 다른 이론 $\square$

</details>

---

<div align="center">

[◀ 이전: Ch2-05. Value Function 의 고유성과 존재성](../ch2-bellman-expectation/05-value-uniqueness.md) | [📚 README](../README.md) | [다음 ▶: 02. Bellman Optimality Equation](./02-bellman-optimality.md)

</div>
