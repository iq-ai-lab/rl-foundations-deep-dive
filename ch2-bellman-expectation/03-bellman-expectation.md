# 03. Bellman Expectation Equation 유도

## 🎯 핵심 질문

- Bellman expectation equation 을 정의 (recursive 하지 않은 것) 에서 어떻게 엄밀히 유도하는가?
- 왜 이 방정식을 "expectation" 이라 부르고, "optimality" 와 구별하는가?
- $V^\pi$ 와 $Q^\pi$ 에 대한 두 가지 recursive form 의 관계는?
- Tower property 와 Markov 성질이 유도에서 정확히 어떤 역할을 하는가?
- 고정점 표기법 $V^\pi = T^\pi V^\pi$ 는 무엇을 의미하는가?

---

## 🔍 왜 이 정리가 RL 의 정초인가

**Value function 의 정의**는 Ch2-02 에서 **기대값** 으로 주어집니다:

$$V^\pi(s) = \mathbb{E}[G_t | S_t = s]$$

그러나 이것은 **직접 계산할 수 없습니다** — 모든 가능한 trajectory 의 기대값을 다 구해야 하기 때문입니다.

**Bellman expectation equation** 은 이 정의를 **재귀적 형태** 로 변환합니다:

$$V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

**왜 혁명적인가?**

1. **계산 가능성**: 한 상태의 가치를 "이웃 상태의 가치" 로 표현 → iteration 가능
2. **Dynamic Programming 의 씨앗**: 무한 합 → 재귀 → 유한 계산
3. **모든 RL 알고리즘의 기반**: Value Iteration, Policy Iteration, 심지어 Deep Q-Network 도 이 방정식으로부터 파생

---

## 📐 수학적 선행 조건

- **Ch2-01, Ch2-02**: Discounted return, Value functions, One-step Bellman decomposition
- 확률론: Tower property, Markov 성질
- Ch1-02: MDP 의 Markov 성질과 transition kernel
- 선형대수: 벡터 표기, matrix form

---

## 📖 직관적 이해

### 유도의 핵심 아이디어

```
G_t = R_{t+1} + γ G_{t+1}    (재귀 정의, Ch2-01)
      ↓
E[G_t | s] = E[R_{t+1} + γ G_{t+1} | s]  (양변 기대값)
      ↓
V^π(s) = E[R_{t+1} | s] + γ E[G_{t+1} | s]  (선형성)
      ↓
      = E[R_{t+1} | s] + γ E[V^π(s') | s]  (정의)
      ↓
      = Σ_a π(a|s) R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')
                                        ↑
                                      재귀!
```

### 그림: Bellman Backup 의 의미

```
상태 s 에서 한 스텝:

        s
       /|\
      / | \
    a₁  a₂  a₃
     |   |   |
    s'  s'  s'  (transition)
     |   |   |
    V  V  V    (미래 가치)

V^π(s) = π(a₁|s)·[R(s,a₁) + γV(s')] 
       + π(a₂|s)·[R(s,a₂) + γV(s')]
       + π(a₃|s)·[R(s,a₃) + γV(s')]
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Bellman Expectation Equation for V

정책 $\pi$ 에 대한 상태-값 함수 $V^\pi: \mathcal{S} \to \mathbb{R}$ 는 다음 **고정점 방정식 (fixed point equation)** 의 유일한 해:

$$(T^\pi V)(s) := \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right]$$

$$V^\pi = T^\pi V^\pi$$

또는 상태별로:

$$V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

### 정의 3.2 — Bellman Expectation Equation for Q

$Q^\pi$ 도 유사하게:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s', a')$$

### 정의 3.3 — Operator Form (다음 장 예비)

선형 연산자 $T^\pi: B(\mathcal{S}) \to B(\mathcal{S})$ (bounded functions on $\mathcal{S}$):

$$(T^\pi V)(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right]$$

Bellman equation: $V^\pi$ 는 $T^\pi$ 의 고정점 (fixed point).

---

## 🔬 정리와 증명

### 정리 3.1 (Bellman Expectation Equation — State-Value Form)

정책 $\pi$ 에서 모든 상태 $s \in \mathcal{S}$ 에 대해:

$$\boxed{V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]}$$

**증명**:

**Step 1** — Return 의 재귀 분해 (Ch2-01, 정리 1.5):

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**Step 2** — 양변에 $\mathbb{E}_\pi[\cdot | S_t = s]$ 적용:

$$\mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]$$

**Step 3** — 좌변은 정의에 의해 $V^\pi(s)$:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]$$

**Step 4** — 우변을 선형성으로 분리:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s]$$

**Step 5** — 첫 항 계산 (MDP 의 정의):

$$\mathbb{E}_\pi[R_{t+1} | S_t = s] = \sum_a \pi(a|s) R(s, a) =: r^\pi(s)$$

(정책으로 행동 선택, 각 행동별 보상의 기대값)

**Step 6** — 둘째 항 계산 (Tower property + Markov 성질):

$$\mathbb{E}_\pi[G_{t+1} | S_t = s] = \mathbb{E}_\pi[\mathbb{E}_\pi[G_{t+1} | S_{t+1}] | S_t = s]$$

(Tower property: $\mathbb{E}[\mathbb{E}[X|Y]|Z] = \mathbb{E}[X|Z]$ when $Z \subseteq Y$ 를 의미하는 정보)

내부 기대값 $\mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] = V^\pi(s')$ 에 의해:

$$\mathbb{E}_\pi[G_{t+1} | S_t = s] = \mathbb{E}[V^\pi(S_{t+1}) | S_t = s]$$

**Step 7** — Markov 성질 (현재 행동 $A_t$ 의존성):

$$\mathbb{E}[V^\pi(S_{t+1}) | S_t = s] = \sum_a \pi(a|s) \mathbb{E}[V^\pi(S_{t+1}) | S_t = s, A_t = a]$$

$S_{t+1}$ 의 분포가 $(s, a)$ 에 의존:

$$\mathbb{E}[V^\pi(S_{t+1}) | S_t = s, A_t = a] = \sum_{s'} P(s'|s,a) V^\pi(s')$$

**Step 8** — 결합:

$$V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right] \quad \square$$

### 정리 3.2 (Bellman Expectation Equation — Action-Value Form)

모든 상태-행동 쌍 $(s, a)$ 에 대해:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s', a')$$

**증명** (스케치):

One-step Bellman decomposition (Ch2-02, 정리 2.2):

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

$V^\pi$ 를 정리 3.1 의 $Q$ 형태로 표현:

$$V^\pi(s') = \sum_{a'} \pi(a'|s') Q^\pi(s', a')$$

대입:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s', a') \quad \square$$

### 정리 3.3 (Vector-Matrix Form)

상태 수가 유한 $|\mathcal{S}| = n$, 정책 $\pi$ 고정일 때, $\mathbf{V} \in \mathbb{R}^n$ (각 상태의 값 벡터):

$$\mathbf{V}^\pi = \mathbf{r}^\pi + \gamma P^\pi \mathbf{V}^\pi$$

여기서:
- $\mathbf{r}^\pi(s) := \sum_a \pi(a|s) R(s, a)$ (policy-weighted reward vector)
- $P^\pi(s'|s) := \sum_a \pi(a|s) P(s'|s, a)$ (state transition matrix under policy)

**증명**: 정리 3.1 을 벡터로 쓴 것 $\square$.

### 정리 3.4 (관계: V 와 Q 의 순환)

세 식:
1. $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$ (전확률, Ch2-02)
2. $Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$ (one-step lookahead, Ch2-02)
3. $V^\pi(s) = \sum_a \pi(a|s) [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')]$ (Bellman, 위)

**관계**: 1 + 2 = 3.

**증명**:

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$

(식 1 대입)

$$= \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

(식 2 이용)

$= $ 식 3 $\quad \square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Bellman 방정식 직접 풀기 (Tabular)

```python
import numpy as np
from scipy.linalg import solve

# 4x4 Gridworld (상태 0~15)
S = 16
A = 4
gamma = 0.9

# Build P (deterministic transitions), R
def build_gridworld():
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    
    for s in range(S):
        row, col = s // 4, s % 4
        actions = [
            ((row - 1) % 4, col),     # up
            ((row + 1) % 4, col),     # down
            (row, (col - 1) % 4),     # left
            (row, (col + 1) % 4)      # right
        ]
        for a, (nr, nc) in enumerate(actions):
            next_s = nr * 4 + nc
            P[s, a, next_s] = 1.0
            R[s, a] = -1.0 if next_s != 15 else 10.0
    
    return P, R

P, R = build_gridworld()

# Uniform policy
pi = np.ones((S, A)) / A

# Method 1: Bellman equation 을 선형방정식으로 풀기
# V = r^π + γ P^π V
# (I - γ P^π) V = r^π

# P^π(s'|s) = Σ_a π(a|s) P(s'|s,a)
P_pi = np.einsum('sa,sap->sp', pi, P)  # shape (S, S)

# r^π(s) = Σ_a π(a|s) R(s,a)
r_pi = np.einsum('sa,sa->s', pi, R)  # shape (S,)

# Solve (I - γ P^π) V = r^π
A_matrix = np.eye(S) - gamma * P_pi
V_linear = solve(A_matrix, r_pi)

print("Method 1: Solve linear system (I - γP^π)V = r^π")
print("V^π shape:", V_linear.shape)
print("V^π at states 0, 7, 15:", V_linear[[0, 7, 15]])
print("Grid:\n", V_linear.reshape(4, 4).round(2))
```

### 실험 2 — Value Iteration (Bellman backup)

```python
# Method 2: Iterative Bellman backup
# V_{k+1} = T^π V_k = r^π + γ P^π V_k

def bellman_backup(V, pi, P, R, gamma):
    """Single Bellman backup: V_new = T^π V"""
    # Q(s,a) = R(s,a) + γ Σ P(s'|s,a) V(s')
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    # V(s) = Σ π(a|s) Q(s,a)
    V_new = np.einsum('sa,sa->s', pi, Q)
    return V_new

V = np.zeros(S)
V_hist = [V.copy()]

for k in range(100):
    V = bellman_backup(V, pi, P, R, gamma)
    V_hist.append(V.copy())
    
    # Convergence check
    if k > 0 and np.abs(V - V_hist[k-1]).max() < 1e-6:
        print(f"Converged at iteration {k}")
        break

V_iter = V
print("\nMethod 2: Iterative Bellman backup")
print("V^π at states 0, 7, 15:", V_iter[[0, 7, 15]])
print("Grid:\n", V_iter.reshape(4, 4).round(2))

# Error between two methods
print(f"\nDifference between linear solve and iteration:")
print(f"  Max |V_linear - V_iter| = {np.abs(V_linear - V_iter).max():.2e}")
```

### 실험 3 — Q 함수와의 관계 검증

```python
# Calculate Q from V
Q = R + gamma * np.einsum('sap,p->sa', P, V_iter)

# Verify: V(s) = Σ π(a|s) Q(s,a)
V_from_Q = np.einsum('sa,sa->s', pi, Q)
print("\nVerify Theorem 3.4.1: V = Σ π(a|s) Q(s,a)")
print(f"  Max error: {np.abs(V_iter - V_from_Q).max():.2e}  ✓")

# Verify: Q(s,a) = R(s,a) + γ Σ P(s'|s,a) V(s')
Q_check = R + gamma * np.einsum('sap,p->sa', P, V_iter)
print("\nVerify Theorem 3.4.2: Q = R + γ ΣP V")
print(f"  Max error: {np.abs(Q - Q_check).max():.2e}  ✓")

# Bellman equation verification
V_bell = np.zeros(S)
for s in range(S):
    V_bell[s] = np.sum(pi[s] * (R[s] + gamma * P[s] @ V_iter))

print("\nVerify Theorem 3.1: V(s) = Σ_a π(a|s)[R(s,a) + γΣ P V]")
print(f"  Max error: {np.abs(V_iter - V_bell).max():.2e}  ✓")
```

### 실험 4 — 정책 변화에 따른 Bellman 변화

```python
import matplotlib.pyplot as plt

# 여러 정책의 value 비교
gammas = [0.5, 0.9, 0.99]
policies = []
values = []

# Greedy toward goal, Uniform, Random-ish
policy_names = []

# 1. Greedy toward (3,3)
pi_greedy = np.zeros((S, A))
for s in range(S):
    row, col = s // 4, s % 4
    # Prefer right and down
    if col < 3:
        pi_greedy[s, 3] += 0.7  # right
    if row < 3:
        pi_greedy[s, 1] += 0.7  # down
    pi_greedy = np.maximum(pi_greedy, 0.05)  # minimum probability
    pi_greedy = pi_greedy / pi_greedy.sum(axis=1, keepdims=True)

policy_names.append('Greedy toward goal')
policies.append(pi_greedy)

# 2. Uniform
pi_uniform = np.ones((S, A)) / A
policy_names.append('Uniform random')
policies.append(pi_uniform)

# Compute values for each policy and gamma
fig, axes = plt.subplots(len(gammas), len(policies), figsize=(10, 9))

for i, g in enumerate(gammas):
    for j, (pi, name) in enumerate(zip(policies, policy_names)):
        # Solve for V
        P_pi = np.einsum('sa,sap->sp', pi, P)
        r_pi = np.einsum('sa,sa->s', pi, R)
        A_matrix = np.eye(S) - g * P_pi
        V = solve(A_matrix, r_pi)
        
        ax = axes[i, j]
        im = ax.imshow(V.reshape(4, 4), cmap='RdYlGn')
        ax.set_title(f'{name}\nγ={g}')
        plt.colorbar(im, ax=ax)

plt.suptitle('Value Functions for Different Policies and Discount Factors', y=1.00)
plt.tight_layout()
plt.savefig('bellman_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 후속 레포와의 연결

- **Ch2-04 (Bellman Operator)**: 연산자 $T^\pi$ 로 정식화 → 고정점 정리 적용
- **Ch3 (Optimality)**: Bellman optimality equation $V^* = T^* V^*$ (greedy max 포함)
- **Ch4 (Contraction)**: $T^\pi$ 가 $\gamma$-contraction 임을 증명
- **Ch5 (Value Iteration)**: $V_{k+1} = T^\pi V_k$ 반복이 $V^\pi$ 로 수렴
- **Ch5 (Policy Iteration)**: Policy evaluation 에서 Bellman 방정식을 푸는 것

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 한계 | 대응 |
|------|------|------|------|
| Finite MDP | $\|\mathcal{S}\|, \|\mathcal{A}\|$ 유한 | Continuous 상태 | Function approximation (Ch7) |
| Stationary policy | $\pi$ 시간 불변 | Non-stationary 전략 | 표준 MDP 밖 |
| Discount $\gamma < 1$ | Return 수렴 필요 | $\gamma = 1$ 무한합 | Episodic 변형 |
| Acyclic 계산 불필요 | Loop, cycle 허용 | Computational issue | Iteration 필수 |

---

## 📌 핵심 정리

$$\boxed{V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]}$$

**벡터 형태**: $(I - \gamma P^\pi) \mathbf{V}^\pi = \mathbf{r}^\pi$

| 요소 | 정의 | 역할 |
|------|------|------|
| $r^\pi(s)$ | $\sum_a \pi(a\|s) R(s,a)$ | 한 스텝 기대 보상 |
| $P^\pi(s'\|s)$ | $\sum_a \pi(a\|s) P(s'\|s,a)$ | Policy-induced transition |
| $T^\pi$ | 우측 연산자 | Bellman operator |
| **고정점** | $V^\pi = T^\pi V^\pi$ | 해를 특징지음 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 3.1 의 증명에서 Step 6 의 "Tower property" 를 다시 보자. $\mathbb{E}_\pi[G_{t+1} | S_t = s]$ 를 계산할 때, 왜 먼저 $S_{t+1}$ 로 조건부하는가?

<details>
<summary>해설</summary>

**Tower property의 형태**:
$$\mathbb{E}[X | Z] = \mathbb{E}[\mathbb{E}[X | Y] | Z]$$
(단, $Z \subseteq Y$ in terms of information)

**우리의 경우**:
- $X = G_{t+1}$ (미래의 누적 보상)
- $Y = S_{t+1}$ (다음 상태)
- $Z = S_t$ (현재 상태)

$$\mathbb{E}_\pi[G_{t+1} | S_t = s] = \mathbb{E}[\mathbb{E}_\pi[G_{t+1} | S_{t+1}] | S_t = s]$$

**이유**: $G_{t+1}$ 은 $S_{t+1}$ 의 값에만 의존하지만, $S_{t+1}$ 은 $S_t$ 와 $A_t$ (의 확률 분포) 에 의존.

따라서 $S_t$ 에서의 조건부 기대값을 구하려면:
1. 먼저 $S_{t+1}$ 주어졌을 때의 $G_{t+1}$ 기대값 = $V^\pi(S_{t+1})$
2. 그 다음 $S_t$ 주어졌을 때 $S_{t+1}$ 분포에서의 기대값

이것이 우리가 나중에 transition matrix $P$ 가 등장하는 이유. $\square$

</details>

**문제 2** (심화): 정리 3.3 에서 벡터-행렬 형태 $(I - \gamma P^\pi) \mathbf{V}^\pi = \mathbf{r}^\pi$ 가 주어진다. 이 선형방정식이 항상 **유일한 해** 를 가지는가? $(I - \gamma P^\pi)$ 가 가역인 조건은?

<details>
<summary>해설</summary>

**가역 조건**: $\det(I - \gamma P^\pi) \neq 0$

**정리**: $P^\pi$ 가 stochastic matrix (모든 행의 합이 1) 이고 $\gamma < 1$ 이면, $(I - \gamma P^\pi)$ 는 항상 가역.

**증명**:
- $P^\pi$ stochastic ⇒ spectral radius $\rho(P^\pi) \leq 1$
- 따라서 $\rho(\gamma P^\pi) \leq \gamma < 1$
- 모든 eigenvalue $\lambda$ 에 대해 $\gamma |\lambda| < 1$ ⇒ $1 - \gamma \lambda \neq 0$
- 즉, $(I - \gamma P^\pi)$ 의 모든 eigenvalue 가 0이 아님 ⇒ 가역

**폐쇄 해 (Closed form)**:
$$(I - \gamma P^\pi)^{-1} = \sum_{k=0}^\infty (\gamma P^\pi)^k$$

(Neumann series, Ch2-05 에서 자세히)

**$\gamma = 1$ 일 때**: $\rho(P^\pi) = 1$ 가능 (episodic 아니면 문제)

→ 이것이 **$\gamma < 1$ 이 수학적으로 필수** 인 이유. $\square$

</details>

**問題 3** (논文 비평): Bellman (1957) 의 원논문에서는 "optimality equation" 을 중심으로 다루고, "expectation equation" 은 부수적 결과로 취급한다. 반면 Sutton & Barto (2018) 는 expectation 을 먼저 가르치고 optimality 를 특수한 경우로 다룬다. 교육적 관점에서 어느 순서가 더 자연스러운가?

<details>
<summary>해설</summary>

**Bellman (1957) 의 순서**:
1. Optimality principle: "최적 정책은 suboptimality-free 한 substructure 가진다"
2. Optimality equation 유도
3. Value iteration 으로 풀기

**Sutton & Barto (2018) 의 순서**:
1. Policy evaluation (expectation equation)
2. Policy improvement (greedy)
3. 반복하면 optimality 수렴

**비교**:

| 관점 | Bellman (원리) | Sutton (실용) |
|------|---|---|
| 이론적 elegance | 고 (optimality principle 에서 시작) | 낮음 |
| 알고리즘 동기 | 어딘가 나타남 | 직관적 (평가 → 개선 반복) |
| 구현 순서 | Optimality equation 풀기 | Policy iteration 구현 |
| 입문자 관점 | "왜 greedy 가 최적?" 후속 질문 | "어느 정책이 나은가?" 의 연속성 |

**의견**: 
- **수학적 엄밀성**: Bellman (원리 먼저)
- **실전 구현**: Sutton (평가 먼저)
- **이 레포**: Expectation 먼저 (가치 함수 정의), 나중에 optimality → **Sutton 쪽에 가까움** 

이유는 "가치 함수의 존재성과 계산 방법" 을 먼저 확립하는 것이, optimality 를 정의하기 전에 필요하기 때문. $\square$

</details>

---

<div align="center">

[◀ 이전: 02. State-Value 와 Action-Value Function](./02-value-functions.md) | [📚 README](../README.md) | [다음 ▶: 04. Operator 표기법](./04-bellman-operator.md)

</div>
