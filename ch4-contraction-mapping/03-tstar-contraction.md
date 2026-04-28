# 03. $T^*$ 가 $\gamma$-Contraction (Nonlinear 의 경우) — Value Iteration 의 수렴 원리

## 🎯 핵심 질문

- **Optimal operator** $T^*$ 가 왜 $\gamma$-contraction 인가?
- $T^*$ 는 affine 이 아니라 **nonlinear** (max 연산) 인데, contraction 이 될 수 있는 이유는?
- **Max-Lipschitz 보조정리** 는 무엇이고, 어떻게 증명하는가?
- Sup-norm 에서의 contraction 이 왜 특별한가? (다른 norm 은 안 될까?)

---

## 🔍 왜 이 정리가 가장 강력한가

$T^\pi$ (정책 평가) 는 affine 이므로 contraction 이 자동입니다. 그러나 $T^*$ (최적 벨만) 는:

$$(T^* V)(s) = \max_a \left[r(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s')\right] = \max_a Q(s, a)$$

**Nonlinear** 입니다. Max 연산은 합이나 곱과 달라서 contraction 증명이 비자명합니다.

그러나 **Max-Lipschitz 보조정리** (이 문서의 핵심) 로부터:

$$\left\|\max_a f_a - \max_a g_a\right\|_\infty \leq \max_a \|f_a - g_a\|_\infty$$

이것이 $T^*$ 의 nonlinear contraction 을 보장합니다. 이것이 **Value Iteration 이 어떤 policy 의 evaluation 보다 더 강력한 이유** 입니다.

---

## 📐 수학적 선행 조건

### 필수
- Ch4-01: Banach Fixed Point Theorem
- Ch4-02: $T^\pi$ 의 affine contraction
- Lipschitz continuity 의 정의

### 선택
- Convex optimization: subdifferential, Danskin's theorem

---

## 📖 직각적 이해

### 왜 Max 는 Lipschitz 인가?

두 벡터 $a, b \in \mathbb{R}^n$ 에 대해:

$$|\max(a_i) - \max(b_i)| \leq \max_i |a_i - b_i|$$

**직관**: 최댓값 두 개의 차이는 대응 요소 차이의 최댓값으로 bound 됨.

**기하학적 그림**:

```
Vectorized functions:
    a = [2.0, 1.5, 3.0]  →  max(a) = 3.0
    b = [2.1, 1.3, 2.9]  →  max(b) = 2.9
    
    |max(a) - max(b)| = 0.1
    max_i |a_i - b_i| = max(0.1, 0.2, 0.1) = 0.2
    
    Inequality: 0.1 ≤ 0.2 ✓
```

### RL 의 맥락: Q-function의 변화

두 value functions $V, V'$ 에 대해 Q-function 을 계산:

$$Q(s, a) = r(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s')$$
$$Q'(s, a) = r(s, a) + \gamma \sum_{s'} P(s' | s, a) V'(s')$$

$$|Q(s, a) - Q'(s, a)| = \gamma \left|\sum_{s'} P(s'|s,a) (V(s') - V'(s'))\right| \leq \gamma \|V - V'\|_\infty$$

따라서:
$$\max_a Q(s, a) - \max_a Q'(s, a) \leq \max_a |Q(s, a) - Q'(s, a)| \leq \gamma \|V - V'\|_\infty$$

---

## ✏️ 엄밀한 정의

### 정의 4.3.1 — Optimal Bellman Operator

$$T^* V(s) := \max_a \left[r(s, a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right]$$

또는 Q-function 으로:
$$(T^* V)(s) = \max_a Q(s, a), \quad Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)}[V(s')]$$

### 정의 4.3.2 — Lipschitz Function / Operator

함수 $f: X \to \mathbb{R}$ 가 **$L$-Lipschitz continuous**:
$$|f(x) - f(y)| \leq L \cdot d(x, y) \quad \forall x, y \in X$$

연산자 $T: B(\mathcal{S}) \to B(\mathcal{S})$ 가 **$L$-Lipschitz** (norm 에서):
$$\|T(V) - T(V')\| \leq L \cdot \|V - V'\| \quad \forall V, V' \in B(\mathcal{S})$$

### 정의 4.3.3 — Pointwise 와 Sup-norm

Max 연산에는 두 가지 해석이 있습니다:
1. **Pointwise contraction**: 각 state $s$ 에서 $|(T^* V)(s) - (T^* V')(s)| \leq \gamma \|V - V'\|_\infty$
2. **Sup-norm contraction**: $\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty$

둘 다 성립합니다.

---

## 🔬 정리와 증명

### 보조정리 4.3.1 — Max-Lipschitz

벡터 $f_a, g_a \in \mathbb{R}^{|\mathcal{S}|}$ ($a \in \mathcal{A}$) 에 대해:

$$\left\|\max_a f_a - \max_a g_a\right\|_\infty \leq \max_a \|f_a - g_a\|_\infty$$

여기서 $\max_a$ 는 **element-wise maximum**:
$$(\max_a f_a)(s) := \max_a f_a(s)$$

**증명**:

임의 state $s$ 에 대해:
$$(\max_a f_a)(s) = \max_a f_a(s)$$
$$(\max_a g_a)(s) = \max_a g_a(s)$$

Let $a^* = \arg\max_a f_a(s)$. Then:

$$f_a^*(s) = \max_a f_a(s) \geq f_a(s) \quad \forall a$$
$$g_a^*(s) \geq g_a(s) \quad \forall a$$

따라서:
$$f_{a^*}(s) - g_{a^*}(s) \leq f_{a^*}(s) - \max_a g_a(s)$$

위 정렬:
$$f_{a^*}(s) - \max_a g_a(s) = (f_{a^*}(s) - g_{a^*}(s)) + (g_{a^*}(s) - \max_a g_a(s))$$
$$\leq |f_{a^*}(s) - g_{a^*}(s)| + 0 = |f_{a^*}(s) - g_{a^*}(s)| \leq \|f_{a^*} - g_{a^*}\|_\infty$$

Similarly, $\max_a g_a(s) - \max_a f_a(s) \leq \max_a \|f_a - g_a\|_\infty$.

따라서:
$$|(\max_a f_a)(s) - (\max_a g_a)(s)| \leq \max_a \|f_a - g_a\|_\infty$$

모든 state $s$ 에 대해 성립하므로:
$$\left\|\max_a f_a - \max_a g_a\right\|_\infty = \max_s |(\max_a f_a)(s) - (\max_a g_a)(s)| \leq \max_a \|f_a - g_a\|_\infty \quad \square$$

### 정리 4.3.1 — $T^*$ 가 $\gamma$-Contraction in Sup-Norm

$(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서:

$$\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty$$

**증명**:

Define $f_a(s) = r(s, a) + \gamma \sum_{s'} P(s'|s,a) V(s')$ and $g_a(s) = r(s, a) + \gamma \sum_{s'} P(s'|s,a) V'(s')$.

Then $T^* V(s) = \max_a f_a(s)$ and $T^* V'(s) = \max_a g_a(s)$.

By Max-Lipschitz lemma:
$$\|T^* V - T^* V'\|_\infty = \|\max_a f_a - \max_a g_a\|_\infty \leq \max_a \|f_a - g_a\|_\infty$$

For each action $a$:
$$\|f_a - g_a\|_\infty = \left\|\gamma \sum_{s'} P(s'|s,a) (V(s') - V'(s'))\right\|_\infty$$
$$= \gamma \left\|P_a (V - V')\right\|_\infty$$

여기서 $P_a$ 는 action $a$ 의 transition matrix (stochastic).

행 합이 1인 stochastic matrix 이므로:
$$\|P_a (V - V')\|_\infty \leq \|V - V'\|_\infty$$

따라서:
$$\|f_a - g_a\|_\infty \leq \gamma \|V - V'\|_\infty \quad \forall a$$

$$\max_a \|f_a - g_a\|_\infty \leq \gamma \|V - V'\|_\infty$$

결론:
$$\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty \quad \square$$

### 따름정리 4.3.1 — Value Iteration 수렴 (Banach 정리의 응용)

Iteration $V_{k+1} = T^* V_k$ (임의 초기값 $V_0$) 는 유일 고정점 $V^*$ 로 수렴:
$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$$

---

## 💻 NumPy 구현 검증

### 실험 1 — Max-Lipschitz 검증

```python
import numpy as np

# Action space에서의 벡터 함수들
n_states, n_actions = 5, 3
f_a = np.random.randn(n_actions, n_states)
g_a = f_a + 0.1 * np.random.randn(n_actions, n_states)

# Max-Lipschitz: ||max_a f_a - max_a g_a||_∞ ≤ max_a ||f_a - g_a||_∞
lhs = np.linalg.norm(f_a.max(axis=0) - g_a.max(axis=0), ord=np.inf)
rhs = np.max([np.linalg.norm(f_a[a] - g_a[a], ord=np.inf) for a in range(n_actions)])

print(f"LHS ||max_a f_a - max_a g_a||_∞ = {lhs:.6f}")
print(f"RHS max_a ||f_a - g_a||_∞       = {rhs:.6f}")
print(f"Max-Lipschitz holds: {lhs <= rhs + 1e-10}")
```

### 실험 2 — Value Iteration 수렴 (4-state)

```python
# 4-state MDP
S = 4
A = 2
gamma = 0.9
np.random.seed(42)

# Random MDP
R = np.random.randn(S, A)
P = np.random.dirichlet(np.ones(S), size=(S, A))

# Value Iteration
V = np.zeros(S)
errors = []

for k in range(150):
    # Bellman: V(s) = max_a [r(s,a) + γ Σ P(s'|s,a) V(s')]
    Q = R + gamma * (P @ V)
    V_new = Q.max(axis=1)
    error = np.linalg.norm(V_new - V, ord=np.inf)
    errors.append(error)
    V = V_new
    if k == 0:
        V_star = V_new.copy()

errors = np.array(errors)
a_priori = gamma ** np.arange(len(errors)) * np.abs(R).max() / (1 - gamma)

print(f"Final V: {V}")
print(f"Max error at k=50: {errors[49]:.2e}")
print(f"γ^50 bound: {a_priori[49]:.2e}")
```

### 실험 3 — 5×5 Gridworld Value Iteration Convergence

```python
# 5×5 gridworld
nx, ny = 5, 5
S = nx * ny
A = 4
gamma = 0.9
np.random.seed(42)

# Random transition with slip
rewards = np.random.randn(S, A)
P = np.zeros((S, A, S))
for s in range(S):
    for a in range(A):
        x, y = s // ny, s % ny
        if a == 0: nx_new, ny_new = max(0, x-1), y
        elif a == 1: nx_new, ny_new = min(nx-1, x+1), y
        elif a == 2: nx_new, ny_new = x, max(0, y-1)
        else: nx_new, ny_new = x, min(ny-1, y+1)
        s_prime = nx_new * ny + ny_new
        P[s, a, s_prime] += 0.95
        for s_rand in range(S):
            P[s, a, s_rand] += 0.05 / S

# Value Iteration
V = np.zeros(S)
errors_inf = []

for k in range(200):
    Q = rewards + gamma * np.einsum('sap,p->sa', P, V)
    V_new = Q.max(axis=1)
    error = np.linalg.norm(V_new - V, ord=np.inf)
    errors_inf.append(error)
    V = V_new

import matplotlib.pyplot as plt
errors_inf = np.array(errors_inf)

plt.figure(figsize=(10, 5))
plt.semilogy(errors_inf, 'b-', label='$\|V_k - V_{k-1}\|_\\infty$', linewidth=2)
plt.semilogy(gamma ** np.arange(len(errors_inf)), 'r--', label='$\gamma^k$', linewidth=2)
plt.xlabel('Iteration k')
plt.ylabel('Error (log scale)')
plt.legend()
plt.grid(True)
plt.title('5×5 Gridworld: Value Iteration Convergence')
plt.tight_layout()
plt.savefig('/tmp/vi_gridworld_5x5.png', dpi=120)
```

---

## 🔗 후속 레포와의 연결

- **Ch4-04**: Value Iteration 정지 기준
- **Ch4-05**: $\gamma \to 1$ 에서의 한계
- **Ch5**: Policy Iteration, Generalized Policy Iteration

---

## ⚖️ 가정과 한계

| 가정 | 대응 |
|------|------|
| Sup-norm 선택 | L2 norm 에서는 stochastic matrix 가 contraction 을 보장하지 않음 |
| Bounded reward | 필수 |
| Finite state (또는 compact) | Infinite state: function approximation |

---

## 📌 핵심 정리

$$\boxed{\text{Max-Lipschitz: } \|\max_a f_a - \max_a g_a\|_\infty \leq \max_a \|f_a - g_a\|_\infty}$$

$$\boxed{\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty}$$

---

## 🤔 생각해볼 문제

**문제 1**: 왜 sup-norm 을 선택하는가? L2 norm 에서는 왜 contraction 이 깨지는가?

<details>
<summary>해설</summary>

Stochastic matrix $P$ 에 대해 $\|Px\|_2 \leq \|x\|_2$ 가 항상 성립하지 않음. 예: $P$ 가 stochastic 이어도 spectral norm (largest singular value) 이 1 보다 클 수 있음. Sup-norm (row-maximum) 은 stochastic 의 구조를 활용. $\square$

</details>

**문제 2**: Max-Lipschitz 증명에서 왜 $a^* = \arg\max_a f_a(s)$ 를 "선택"하는가?

<details>
<summary>해설</summary>

최댓값의 정의: $f_{a^*} = \max_a f_a$. 이제 $f_{a^*}$ 와 $g_{a^*}$ 의 차이가 최댓값들의 차이를 bound. Optimality 에 의한 표준 논법. $\square$

</details>

**문제 3**: Nonlinear $T^*$ 와 affine $T^\pi$ 의 수렴 속도가 같은 이유는?

<details>
<summary>해설</summary>

둘 다 contraction constant $\gamma$ 를 가짐 → $k \geq \log(\epsilon(1-\gamma)) / \log(\gamma)$ iteration 필요. 그러나 VI 가 "모든 action 에 대해 동시에" 최적화하므로, PI 의 policy eval 단계 반복보다 상대적으로 빠를 수 있음. 실제로 PI 가 종종 더 빠름 (superpolynomial). $\square$

</details>

---

<div align="center">

[◀ 이전: 02. $T^\pi$ 가 $\gamma$-Contraction](./02-tpi-contraction.md) | [📚 README](../README.md) | [다음 ▶: 04. Value Iteration 수렴 보장](./04-value-iteration-convergence.md)

</div>
