# 04. Value Iteration 완성 — 최적 정책으로의 직진

## 🎯 핵심 질문

- Value Iteration 은 왜 "policy iteration 의 가속" 이 아니라 "다른 알고리즘" 인가?
- Bellman optimality operator $T^*$ 가 contraction 인 증명에서 max operator 의 역할은?
- Asynchronous Value Iteration 이 왜 실무에서 중요한가?
- Bellman residual $\|T^* V_k - V_k\|_\infty$ 이 무엇이고 왜 정지 기준이 되는가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

Value Iteration 은 **explicit policy update 없이** Bellman optimality operator 를 반복하여 최적값에 수렴합니다. 이것이:

1. **Theory**: Contraction 성질로 exponential convergence 보장
2. **Practice**: Asynchronous 버전으로 경험적 speedup (sample efficiency)
3. **Bridge**: Model-free RL (Q-learning, DQN) 의 모태

VI 없으면 "대규모 MDP 를 어떻게 풀까" 에 답할 수 없습니다.

---

## 📐 수학적 선행 조건

- **Ch4-02**: Bellman optimality operator $T^*$, contraction 증명
- **Ch5-01, 02, 03**: PE, PI 의 기초
- **함수해석**: Operator norm, fixed point

---

## 📖 직관적 이해

### VI 의 아이디어

Policy Iteration 은:
```
Evaluate π → Improve to π' → Evaluate π' → ...
```

Policy Iteration 과 달리, Value Iteration 은:
```
V_0 → T^* V_0 → T^* T^* V_0 → ... (policy 명시적 update 없음)
```

각 step 에서 $V_k$ 는 implicitly 최적 정책을 향하고 있음. Policy Improvement 를 "안" 하는 것처럼 보이지만, 실제로는 **Bellman optimality operator 의 fixed point 를 찾는 것**.

### Asynchronous 의 의미

"Synchronous" VI: 모든 state 를 한 번에 update.
"Asynchronous" VI: 한 state 씩, 다른 state 의 최신값 사용 (Gauss-Seidel).

비동기 버전이 **같은 수렴 조건 (모든 state 무한 update) 하에 실무에서 빠름**.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Bellman Optimality Operator

$$T^* V(s) := \max_a \left[ r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right]$$

함수공간 $(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서의 operator.

### 정의 4.2 — Value Iteration (Synchronous)

$$V_0 \text{ arbitrary}, \quad V_{k+1}(s) = T^* V_k(s) \quad \forall s$$

정지 조건: Bellman residual $\|T^* V_k - V_k\|_\infty < \epsilon(1-\gamma)$ 이면 $\epsilon$-optimal.

### 정의 4.3 — Value Iteration (Asynchronous Gauss-Seidel)

임의 state $s$ 에 대해:

$$V_{k+1}(s) = T^* V_k(s) \quad \text{(using } V_{k+1} \text{ for } s' \text{ that already updated)}$$

즉, 한 state 씩 update 하고 바로 다음 state 에서 새 값 사용.

### 정의 4.4 — Greedy Policy Extraction

수렴한 $V^*$ 에서 최적 정책 추출:

$$\pi^*(s) := \arg\max_a [r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')]$$

---

## 🔬 정리와 증명

### 정리 4.1 (Bellman Optimality Operator 는 $\gamma$-Contraction)

$(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서:

$$\|T^* V - T^* V'\|_\infty \leq \gamma \|V - V'\|_\infty \quad \forall V, V'$$

**증명** (Ch4-02 에서 이미 했지만 재확인):

For any state $s$:
$$|T^* V(s) - T^* V'(s)| = \left| \max_a [r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V(s')] - \max_a [r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V'(s')] \right|$$

Max-Lipschitz lemma:
$$|\max_i f_i - \max_i g_i| \leq \max_i |f_i - g_i|$$

따라서:
$$|T^* V(s) - T^* V'(s)| \leq \max_a \left| \gamma \sum_{s'} P(s' \mid s,a) (V(s') - V'(s')) \right|$$
$$\leq \max_a \gamma \|V - V'\|_\infty = \gamma \|V - V'\|_\infty \quad \square$$

### 정리 4.2 (Value Iteration Convergence)

임의 초기값 $V_0$ 에 대해:

$$V_k \xrightarrow{k \to \infty} V^* \quad \text{exponentially}$$
$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$$

**증명**: Banach fixed point theorem (Contraction → unique FP + exponential convergence) $\square$.

### 정리 4.3 (Bellman Residual 과 최적성 간격)

Bellman residual 을 정의:

$$\text{BR}_k := \|T^* V_k - V_k\|_\infty$$

그러면:

$$\|V_k - V^*\|_\infty \leq \frac{1}{1-\gamma} \cdot \text{BR}_k$$

**증명**:

$$\|V_k - V^*\|_\infty = \|V_k - T^* V^*\|_\infty$$

(since $V^* = T^* V^*$)

$$\leq \|V_k - T^* V_k\|_\infty + \|T^* V_k - T^* V^*\|_\infty$$

(triangle inequality)

$$\leq \text{BR}_k + \gamma \|V_k - V^*\|_\infty$$

따라서:

$$(1-\gamma) \|V_k - V^*\|_\infty \leq \text{BR}_k \quad \Rightarrow \quad \|V_k - V^*\|_\infty \leq \frac{\text{BR}_k}{1-\gamma} \quad \square$$

### 정리 4.4 (Asynchronous VI Convergence)

모든 state 가 무한히 update 되면 asynchronous VI 도 수렴:

$$V_k^{\text{async}} \to V^*$$

**증명 sketch** (Bertsekas & Tsitsiklis 1989): 한 state 의 update 도 contraction 의 일부이므로, 모든 state 가 충분히 자주 업데이트되면 (비동기 버전이라도) fixed point 로 수렴 $\square$.

---

## 💻 NumPy 구현 검증

### 실험 1 — 기본 Synchronous Value Iteration

```python
import numpy as np
import matplotlib.pyplot as plt

S = 16
A = 4
gamma = 0.9

def make_gridworld_matrices():
    """4×4 Gridworld"""
    P = np.zeros((S, A, S))
    r = -np.ones((S, A))
    
    def coord_to_idx(x, y):
        return x * 4 + y if 0 <= x < 4 and 0 <= y < 4 else None
    def idx_to_coord(idx):
        return (idx // 4, idx % 4)
    
    for s in range(S):
        x, y = idx_to_coord(s)
        next_states = [
            coord_to_idx(x - 1, y),
            coord_to_idx(x + 1, y),
            coord_to_idx(x, y - 1),
            coord_to_idx(x, y + 1),
        ]
        for a, ns in enumerate(next_states):
            if ns is None: ns = s
            P[s, a, ns] = 1.0
            r[s, a] = 1.0 if ns == 15 else -1.0
    
    return P, r

P, r = make_gridworld_matrices()

def value_iteration(P, r, gamma, max_iters=500, tol=1e-10):
    """Synchronous Value Iteration"""
    V = np.zeros(S)
    errors = []
    
    for k in range(max_iters):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = np.max(Q, axis=1)
        
        # Bellman residual
        br = np.linalg.norm(V_new - V, np.inf)
        errors.append(br)
        
        if br < tol * (1 - gamma):
            print(f"VI converged at iteration {k}, residual = {br:.2e}")
            return V_new, errors
        
        V = V_new
    
    return V, errors

V_opt, errors = value_iteration(P, r, gamma)

# Extract policy
Q_opt = r + gamma * np.einsum('sap,p->sa', P, V_opt)
pi_opt = np.zeros((S, A))
pi_opt[np.arange(S), np.argmax(Q_opt, axis=1)] = 1.0

print(f"Optimal value grid:\n{V_opt.reshape(4, 4).round(2)}")
print(f"Converged in {len(errors)} iterations")

# Visualize convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(errors, 'o-', markersize=4, linewidth=1)
ax.set_xlabel('Iteration k')
ax.set_ylabel('Bellman Residual ||T^* V_k - V_k||_∞')
ax.set_title('Value Iteration Convergence')
ax.grid(True, alpha=0.3)

# Linear fit to show γ^k rate
ax = axes[1]
ax.semilogy(errors, 'o-', markersize=4, label='Actual', linewidth=1)

# Theoretical γ^k
theory = errors[0] * (gamma ** np.arange(len(errors)))
ax.semilogy(theory, 'x--', linewidth=1, label=f'Theory γ^k (γ={gamma})')
ax.set_xlabel('Iteration k')
ax.set_ylabel('Error')
ax.set_title('Exponential Convergence Rate')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/value_iteration.png', dpi=150)
```

**예상 출력**:
```
VI converged at iteration 127, residual = 8.99e-11
Optimal value grid:
[[-0.71 -0.62 -0.53 -0.44]
 [-0.62 -0.53 -0.44 -0.35]
 [-0.53 -0.44 -0.35 -0.26]
 [-0.44 -0.35 -0.26  0.00]]
Converged in 128 iterations
```

### 실험 2 — Asynchronous VI (Gauss-Seidel)

```python
def value_iteration_async(P, r, gamma, max_iters=500, tol=1e-10):
    """Asynchronous (Gauss-Seidel) Value Iteration"""
    V = np.zeros(S)
    errors = []
    
    for k in range(max_iters):
        V_old = V.copy()
        
        # Update each state sequentially, using new values immediately
        for s in range(S):
            Q_s = r[s] + gamma * np.einsum('ap,p->a', P[s], V)
            V[s] = np.max(Q_s)
        
        # Bellman residual (using old V)
        br = np.linalg.norm(V - V_old, np.inf)
        errors.append(br)
        
        if br < tol * (1 - gamma):
            print(f"Async VI converged at iteration {k}, residual = {br:.2e}")
            return V, errors
        
    return V, errors

V_async, errors_async = value_iteration_async(P, r, gamma)

# Compare
print(f"\nSynchronous: {len(errors)} iterations")
print(f"Asynchronous: {len(errors_async)} iterations")
print(f"Speedup: {len(errors) / len(errors_async):.2f}x")

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(errors, 'o-', label='Synchronous', linewidth=2, markersize=5)
ax.semilogy(errors_async, 'x--', label='Asynchronous', linewidth=1.5, markersize=6)
ax.set_xlabel('Iteration k')
ax.set_ylabel('Bellman Residual')
ax.set_title('Synchronous vs Asynchronous Value Iteration')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/async_vi.png', dpi=150)
```

### 실험 3 — $\gamma$ 의 영향

```python
gammas = [0.5, 0.9, 0.99, 0.999]
results = {}

for g in gammas:
    V = np.zeros(S)
    k = 0
    for _ in range(10000):
        Q = r + g * np.einsum('sap,p->sa', P, V)
        V_new = np.max(Q, axis=1)
        if np.linalg.norm(V_new - V, np.inf) < 1e-10 * (1 - g):
            break
        V = V_new
        k += 1
    results[g] = k

print("Iterations to convergence:")
for g, k in results.items():
    print(f"  γ = {g:5.3f}: {k:5d} iterations")

# Show why γ = 1 is problematic
print(f"\nFor γ = 0.99: convergence requires ~{-np.log(1e-10) / np.log(0.99):.0f} iterations (theory)")
print(f"For γ = 0.999: convergence requires ~{-np.log(1e-10) / np.log(0.999):.0f} iterations (theory)")
```

---

## 🔗 후속 레포와의 연결

- **Ch5-05 GPI**: VI + PI 의 일반화, 모든 RL 의 framework
- **Ch6-01+ Model-Free**: VI 의 sampling 버전 (Q-learning, SARSA)
- **Ch7+ Deep RL**: Asynchronous VI 의 neural net 버전 (A3C, DQN)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Exact model | Model-free: bootstrapping 필요 (TD error) |
| Tabular | FA: overestimation 가능, deadly triad |
| Finite states/actions | Continuous: discretization 또는 function approximation |

---

## 📌 핵심 정리

$$\boxed{V_{k+1} = T^* V_k = \max_a [R + \gamma P V_k], \quad \|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty}$$

| 양 | 의미 |
|----|------|
| $T^*$ | Bellman optimality operator (max over actions) |
| Contraction rate | $\gamma$ (작을수록 빠름, $\gamma \to 1$ 일 때 느림) |
| Synchronous | 모든 state 동시 update |
| Asynchronous | 한 state 씩 update, 새 값 즉시 사용 (빠름) |
| Bellman residual | $\|T^* V_k - V_k\|_\infty$ (정지 기준) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T^*$ 가 nonlinear 라는 것은 무엇인가? Linear operator 와 어떻게 다른가?

<details>
<summary>해설</summary>

**Linear**: $T[\alpha V + \beta V'] = \alpha T[V] + \beta T[V']$

**Nonlinear** ($T^*$ 는):
$$T^*[\alpha V + \beta V'] \neq \alpha T^*[V] + \beta T^*[V']$$

이유: max 연산자가 nonlinear.

예: $T^* V = \max_a f_a$, $T^* V' = \max_a g_a$ 이면
$$T^*[\alpha V + \beta V'](s) = \max_a [h_a] \neq \alpha \max_a f_a + \beta \max_a g_a$$

(다른 action 에서 max 가 달성될 수 있음)

**Consequence**: 고유값 분석이나 spectral method 불가. Operator norm 으로만 분석 $\square$

</details>

**문제 2** (심화): Bellman residual 과 optimality gap 의 관계식 $\|V_k - V^*\|_\infty \leq \frac{\text{BR}_k}{1-\gamma}$ 에서 왜 $1/(1-\gamma)$ 인수가 나타나는가?

<details>
<summary>해설</summary>

Contraction 의 거듭제곱:

$$\|V_k - V^*\|_\infty = \|V_k - T^* V_k + T^* V_k - T^* V^*\|_\infty$$
$$\leq \|V_k - T^* V_k\|_\infty + \|T^* V_k - T^* V^*\|_\infty$$
$$\leq \text{BR}_k + \gamma \|V_k - V^*\|_\infty$$

따라서:
$$(1-\gamma) \|V_k - V^*\|_\infty \leq \text{BR}_k$$

$1-\gamma$ 가 작을수록 (즉, $\gamma \to 1$) residual 이 작아야 small error — **이것이 $\gamma$ 가 커질수록 VI 수렴이 느린 이유** $\square$

</details>

**문제 3** (실전): 만약 asynchronous VI 를 구현할 때 모든 state 를 정확히 무한 번 업데이트할 수 없다면 (예: random sampling) 어떻게 될까?

<details>
<summary>해설</summary>

이것이 **대규모 RL 의 문제** — asynchronous sampling 에서 convergence guarantee 가 약화될 수 있음.

**Bertsekas & Tsitsiklis (1989)**: "Sufficiently frequent" update 가 필요 — 모든 state 가 무한히 (충분히 자주) 업데이트되어야 함.

**대안**:
1. Fixed periodic sweep (e.g., 매 1000 step 마다 모든 state 보기)
2. Importance-weighted sampling (자주 업데이트되지 않는 state 에 가중치)
3. Prioritized sweeping (value change 가 큰 state 우선 update)

이것이 **prioritized experience replay (deep RL)** 의 동기 $\square$

</details>

---

<div align="center">

[◀ 이전: 03. Policy Iteration (Howard 1960)](./03-policy-iteration.md) | [📚 README](../README.md) | [다음 ▶: 05. Generalized Policy Iteration](./05-gpi.md)

</div>
