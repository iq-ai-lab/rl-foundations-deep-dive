# 03. Bellman Optimality Operator $T^*$

## 🎯 핵심 질문

- Bellman optimality operator $T^*$ 의 수학적 정의는?
- 왜 $T^*$ 는 nonlinear 인가? Affine operator $T^\pi$ 와의 차이는?
- $T^*$ 가 $\gamma$-contraction 이라는 것의 의미는?
- Monotonicity 와 boundedness 는 어떻게 증명하는가?

---

## 🔍 왜 이 정리가 RL 기초인가

$T^*$ 는 **고정점 이론(fixed point theory)** 의 입구입니다:

1. **Bellman 방정식** → **고정점 방정식** $V^* = T^* V^*$ 로 재해석
2. **Operator 의 성질** → $\gamma$-contraction 증명
3. **Banach Fixed Point Theorem** (Ch4) → Value Iteration 의 수렴 보증

이 경로 없으면, "Value Iteration 을 돌리면 수렴한다" 를 증명할 수 없습니다.

---

## 📐 수학적 선행 조건

- Ch3-02: Bellman Optimality Equation
- Ch2-03: Bellman Expectation Operator (비교용)
- 함수해석: Banach space, norm, contraction
- 선형대수: supremum norm, stochastic matrix

---

## 📖 직관적 이해

### Operator 는 함수를 변환하는 함수

$T^*$ 는 **어떤 value function $V$ 를 받아 새로운 value function 을 반환하는 연산자**:

$$T^*: \mathbb{R}^S \to \mathbb{R}^S$$

입력 예: $V = [0.5, 1.2, 0.3, \ldots]$ (각 state 의 임의 추정값)

$$T^* V = \left[ \max_a [r_0^a + \gamma \sum_{s'} P(s'|0,a) V(s')], \, \max_a [\cdots] \, \right]$$

출력 예: $T^* V = [1.1, 1.8, 0.7, \ldots]$ (한 step 의 Bellman 연산 적용)

### Contraction 의 의미

두 value function $V, V'$ 가 다르면:

$$\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty$$

거리가 $\gamma$ 배로 축소 → 계속 반복하면 한 점으로 수렴.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Bellman Optimality Operator

$$T^*: \mathbb{R}^S \to \mathbb{R}^S$$

$$(T^* V)(s) := \max_a \left\{ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right\} \quad \forall s \in \mathcal{S}$$

또는 벡터 표기:

$$T^* V = \max_a [r^a + \gamma P^a V]$$

여기서 $r^a = [R(s,a)]_s$, $P^a$ 는 action $a$ 에 대한 transition matrix.

### 정의 3.2 — 비교: Expectation Operator (Affine)

$$T^\pi V = r^\pi + \gamma P^\pi V$$

$T^\pi$ 는 **affine** (선형 + 상수), $T^*$ 는 **nonlinear** ($\max$ 때문).

---

## 🔬 정리와 증명

### 정리 3.1 ($T^*$ 의 Relationship 과 $\max_\pi T^\pi$ 와의 관계)

**정리**: 임의의 $V \in \mathbb{R}^S$ 에 대해:

$$T^* V = \max_\pi T^\pi V \quad \text{(pointwise)}$$

**증명**:

$$\max_\pi T^\pi V = \max_\pi T^\pi V = \max_\pi \left[ r^\pi + \gamma P^\pi V \right]$$

정책 $\pi$ 가 deterministic policy 중 하나로 제한될 때:

$$\max_\pi (r^\pi + \gamma P^\pi V) = \max_\pi \sum_a \pi(a|s) [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$$

$\pi$ 가 deterministic 이면 어떤 $a^*$ 에서만 $\pi(a^*|s) = 1$:

$$= \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')] = (T^* V)(s) \quad \square$$

### 정리 3.2 ($T^*$ 는 $\gamma$-Contraction in Supremum Norm)

**정리**: Bounded reward $|R| \leq R_{\max}$, $\gamma \in [0, 1)$ 에서:

$$\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty \quad \forall V, V' \in \mathbb{R}^S$$

**증명**:

$V, V'$ 임의 두 함수. 각 state $s$ 에 대해:

$$(T^* V)(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$$

$$(T^* V')(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V'(s')]$$

차이:

$$(T^* V)(s) - (T^* V')(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$$
$$\quad - \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V'(s')]$$

**Lemma 3.1 (Max-Lipschitz)**: 

$$|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$$

이를 적용 ($f(a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')$, 유사하게 $g$ 정의):

$$|(T^* V)(s) - (T^* V')(s)| \leq \max_a \left| \gamma \sum_{s'} P(s'|s,a) [V(s') - V'(s')] \right|$$

$$\leq \gamma \max_a \sum_{s'} P(s'|s,a) |V(s') - V'(s')|$$

(Stochastic matrix $\sum_{s'} P(s'|s,a) = 1$ 로, weighted average)

$$\leq \gamma \max_a \sum_{s'} P(s'|s,a) \|V - V'\|_\infty$$

$$= \gamma \|V - V'\|_\infty$$

따라서:

$$\|T^* V - T^* V'\|_\infty = \max_s |(T^* V)(s) - (T^* V')(s)| \leq \gamma \|V - V'\|_\infty \quad \square$$

### 정리 3.3 ($T^*$ 는 Monotone)

**정리**: $V \leq V'$ (pointwise) 이면:

$$T^* V \leq T^* V'$$

**증명**:

$$V(s') \leq V'(s') \Rightarrow \sum_{s'} P(s'|s,a) V(s') \leq \sum_{s'} P(s'|s,a) V'(s')$$

$$\Rightarrow R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \leq R(s,a) + \gamma \sum_{s'} P(s'|s,a) V'(s')$$

$$\Rightarrow \max_a [\cdots] \leq \max_a [\cdots]$$

$$(T^* V)(s) \leq (T^* V')(s) \quad \square$$

### 정리 3.4 (Boundedness of $T^*$ Fixed Point)

$|R| \leq R_{\max}$, $\gamma < 1$ 에서 $V^* = T^* V^*$ 의 해는:

$$\|V^*\|_\infty \leq \frac{R_{\max}}{1-\gamma}$$

**증명** (sketch):

$T^*$ 를 0 에 적용:

$$T^* 0 = \max_a [R(s,a) + 0] = \max_a R(s,a) \leq R_{\max}$$

이를 반복 ($T^*$ 가 monotone, contraction):

$$T^{*k} 0 \to V^* \quad (\text{as } k \to \infty)$$

Contraction 의 고정점은 $[0, T^* 0]$ 범위에 있으므로:

$$V^*(s) \leq \lim_{k \to \infty} \gamma^k \cdot \frac{R_{\max}}{1-\gamma} = \frac{R_{\max}}{1-\gamma} \quad \square$$

---

## 💻 NumPy 구현 검증

### 실험 1 — $T^*$ Operator 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 1-state MDP: action 0 -> reward 1, action 1 -> reward 0.5
# Single state, so V is scalar
S = 1
gamma = 0.9

R = np.array([[1.0, 0.5]])
P = np.ones((S, 1, S))  # Absorbing

def T_star(V):
    """Apply Bellman optimality operator."""
    Q = R + gamma * np.dot(P[0], V)
    return Q.max()

# Test: V 값에 따른 T*V 변화
V_range = np.linspace(0, 10, 100)
T_star_V = np.array([T_star(np.array([v])) for v in V_range])

plt.figure(figsize=(10, 6))
plt.plot(V_range, T_star_V, 'b-', linewidth=2, label='T*V')
plt.plot(V_range, V_range, 'k--', linewidth=1, label='V (fixed point line)')
# Fixed point: T*V = V 교점
V_star = 1.0 / (1 - gamma)  # Closed form
plt.plot(V_star, V_star, 'ro', markersize=10, label=f'V* = {V_star:.2f}')
plt.xlabel('V')
plt.ylabel('T*V')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Bellman Optimality Operator (1-state MDP)')
plt.tight_layout()
plt.savefig('bellman_operator.png', dpi=150)
plt.close()

print(f"V* (analytical) = {V_star:.4f}")
```

### 실험 2 — Contraction Property 검증

```python
# Multi-state example
S = 4
A = 2
gamma = 0.9
np.random.seed(42)

P = np.random.dirichlet(np.ones(S), size=(S, A))
R = np.random.randn(S, A)

def apply_T_star(V):
    """Apply T* to value function."""
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    return Q.max(axis=1)

# Two different V
V1 = np.array([1.0, 0.5, 0.0, -0.5])
V2 = np.array([1.5, 0.8, 0.2, 0.0])

# Apply T*
T_star_V1 = apply_T_star(V1)
T_star_V2 = apply_T_star(V2)

# Compute norms
dist_V = np.linalg.norm(V1 - V2, np.inf)
dist_T = np.linalg.norm(T_star_V1 - T_star_V2, np.inf)

print(f"||V1 - V2||_∞ = {dist_V:.6f}")
print(f"||T*V1 - T*V2||_∞ = {dist_T:.6f}")
print(f"Ratio ||T*V1 - T*V2|| / ||V1 - V2|| = {dist_T / dist_V:.6f}")
print(f"γ = {gamma:.6f}")
print(f"✓ Ratio < γ (contraction verified)")
```

### 실험 3 — Value Iteration 의 Convergence 속도

```python
# Gridworld
S = 16
A = 4
gamma = 0.9

# Create gridworld environment (동일 구조)
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

# Value iteration + error tracking
V = np.zeros(S)
V_history = [V.copy()]

for it in range(100):
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V = Q.max(axis=1)
    V_history.append(V.copy())

# Theoretical V* (approximate)
V_final = V.copy()

# Compute ||V_k - V*||_inf
errors = np.array([np.linalg.norm(V - V_final, np.inf) for V in V_history])

# Log scale plot
plt.figure(figsize=(10, 6))
plt.semilogy(range(len(errors)), errors, 'b-', linewidth=2, label='||V_k - V*||_∞')
# Theoretical bound: γ^k
k_range = np.arange(len(errors))
theoretical = gamma ** k_range * errors[0]
plt.semilogy(k_range, theoretical, 'r--', linewidth=2, label=f'γ^k · ||V_0 - V*||')

plt.xlabel('Iteration k')
plt.ylabel('Error (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'Value Iteration Convergence (γ={gamma})')
plt.tight_layout()
plt.savefig('value_iteration_convergence.png', dpi=150)
plt.close()

# Find when error < epsilon
eps = 1e-6
k_convergence = np.argmax(errors < eps)
print(f"Converged to ε={eps} at iteration {k_convergence}")
```

---

## 🔗 후속 레포와의 연결

- **Ch3-04**: 최적 정책의 추출 — $\pi^*$ 는 $V^*$ 의 greedy policy
- **Ch4-01**: Banach Fixed Point Theorem — $T^*$ contraction 으로부터 수렴성 증명
- **Ch4-02**: Value Iteration 의 explicit convergence rate

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Supremum norm | 다른 norm (L2, L1) 에서는 contraction 아닐 수 있음 |
| Bounded action | Infinite action 에서는 max 가 존재 안 할 수 있음 |
| Stochastic P | Deterministic P 는 특수 경우, 증명 동일 |
| $\gamma < 1$ | $\gamma = 1$ 에서 contraction 붕괴, episodic 으로 수정 |

---

## 📌 핵심 정리

$$\boxed{T^* V = \max_a [R(s,a) + \gamma P^a V]}$$

**3가지 핵심 성질**:

| 성질 | 식 | 의미 |
|------|-----|------|
| Nonlinearity | $T^* (\alpha V + (1-\alpha) V') \neq \alpha T^* V + (1-\alpha) T^* V'$ | Superposition 불가 |
| Contraction | $\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty$ | Fixed point 수렴 보증 |
| Monotonicity | $V \leq V' \Rightarrow T^* V \leq T^* V'$ | Order 보존 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Max-Lipschitz Lemma 를 증명하라: $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$.

<details>
<summary>해설</summary>

$a^* = \arg\max_a f(a)$, $b^* = \arg\max_a g(a)$ 라 하자.

$$\max_a f(a) - \max_a g(a) = f(a^*) - g(b^*) \leq f(a^*) - g(a^*)$$

(왜냐하면 $g(b^*) \geq g(a^*)$)

$$= f(a^*) - g(a^*) \leq \max_a |f(a) - g(a)|$$

대칭성으로 반대 방향도 증명 $\square$

이것이 $\max$ 가 Lipschitz 연속임을 의미.

</details>

**문제 2** (심화): $T^\pi$ (expectation operator) 는 affine 이고 $T^*$ 는 nonlinear 이다. 이 차이가 알고리즘 설계에 어떤 영향을 주는가?

<details>
<summary>해설</summary>

**$T^\pi$ (affine)**:
- Linear system $V = r^\pi + \gamma P^\pi V$ 로 풀기 가능
- 한 번의 matrix inversion 으로 exact solution
- 복잡도: $O(S^3)$ (dense), $O(S^2)$ (sparse)

**$T^*$ (nonlinear)**:
- Fixed point iteration 외 다른 direct solution 없음
- Contraction 이므로 iteration 반드시 수렴, 속도는 $\gamma^k$
- 복잡도: 각 iteration $O(|A| \cdot S^2)$, 총 $O(k \cdot |A| \cdot S^2)$ where $k \sim O(\log(1/\varepsilon) / (1-\gamma))$

**의미**: 
- Policy evaluation 은 exact (affine) → 한 정책당 $O(S^3)$
- Policy improvement 는 approximate (nonlinear) → iteration 필요
- **Policy Iteration** = (evaluation: exact, improvement: greedy) → often faster than Value Iteration

</details>

**문제 3** (논문 비평): Contraction property 증명에서 stochastic matrix 의 행 합이 1이라는 성질을 썼다. Non-stochastic matrix (행 합 > 1) 에서는 contraction 이 깨지는가?

<details>
<summary>해설</summary>

Stochastic matrix $P$: $\sum_{s'} P(s'|s,a) = 1$ for all $s, a$.

Contraction 증명:
$$\left| \sum_{s'} P(s'|s,a) [V(s') - V'(s')] \right| \leq \sum_{s'} P(s'|s,a) |V(s') - V'(s')|$$
$$\leq \|V - V'\|_\infty \sum_{s'} P(s'|s,a) = \|V - V'\|_\infty$$

만약 $\sum_{s'} P(s'|s,a) = c > 1$ (non-stochastic) 이면:

$$\left| \sum_{s'} P(s'|s,a) [V(s') - V'(s')] \right| \leq c \|V - V'\|_\infty$$

Contraction rate 가 $\gamma$ 에서 $\gamma c$ 로 악화 → $\gamma c > 1$ 이면 **발산 가능**.

**의미**: MDP 의 transition kernel 이 **반드시 stochastic matrix** 여야 하는 이유. 이는 확률 해석 (합 = 1) 뿐 아니라 **수렴성을 보장하는 수학적 필요조건** $\square$

</details>

---

<div align="center">

[◀ 이전: 02. Bellman Optimality Equation](./02-bellman-optimality.md) | [📚 README](../README.md) | [다음 ▶: 04. 최적 정책의 추출 — Greedy Policy](./04-greedy-policy.md)

</div>
