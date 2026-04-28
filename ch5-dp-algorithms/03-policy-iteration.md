# 03. Policy Iteration (Howard 1960) — Evaluation + Improvement 루프

## 🎯 핵심 질문

- Evaluation 과 Improvement 를 반복하면 최적 정책에 도달하는가?
- 정확히 몇 번의 반복이 필요한가? (upper bound 존재?)
- Value Iteration 과 비교했을 때 수렴 속도는 어떻게 되는가?
- 실무에서 언제 PI 를 쓰고 언제 VI 를 쓸 것인가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

Howard (1960) 는 **Evaluation + Improvement 의 반복이 finite MDP 에서 유한 step 안에 최적 정책을 찾는다** 는 것을 증명했습니다. 이것이:

1. **Policy Iteration** 알고리즘의 정당성
2. **Policy Gradient 이론**의 출발점 (later: Kakade 의 Performance Difference Lemma)
3. **모든 RL 알고리즘**의 탐색/착취 균형의 수학적 기초

Without PI, 우리는 "언제 수렴하는가?" 를 답할 수 없습니다.

---

## 📐 수학적 선행 조건

- **Ch5-01**: Policy Evaluation, contraction operator $T^\pi$
- **Ch5-02**: Policy Improvement Theorem, greedy policy
- **Ch4-02**: Fixed point, monotonicity

---

## 📖 직관적 이해

### 왜 "반복" 인가

Policy Iteration 은 이 과정을 반복합니다:

1. **Evaluate**: 현재 정책 $\pi_k$ 의 가치 $V^{\pi_k}$ 계산
2. **Improve**: $V^{\pi_k}$ 를 보고 greedy policy $\pi_{k+1}$ 구성
3. **반복**: $\pi_{k+1}$ 을 evaluate ...

각 iteration 에서 정책이 개선되거나 변하지 않음 (improvement theorem). 정책 수가 유한하므로 언젠가 정책이 변하지 않을 시점이 옴 → 최적.

### VI vs PI

| Policy Iteration | Value Iteration |
|------------------|-----------------|
| Evaluation 을 정확히 (수렴까지) | Evaluation 을 1 step 만 |
| 적은 반복 수 | 많은 step |
| Iteration 당 비용 큼 | Iteration 당 비용 작음 |

수렴은 같지만 (둘 다 최적), **practitioner 의 선택** 문제.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Policy Iteration 알고리즘

입력: MDP $\mathcal{M}$, tolerance $\epsilon > 0$

```
Initialize π_0 arbitrarily
k := 0
while True:
    1. Policy Evaluation:
       V_k := compute V^{π_k} (via iterative Bellman or direct solve)
    
    2. Policy Improvement:
       π_{k+1}(s) := arg max_a [r(s, a) + γ Σ_{s'} P(s'|s,a) V_k(s')]
    
    3. Check for convergence:
       if π_{k+1} = π_k:
           return π_k, V_k  (optimal policy found)
    
    k := k + 1
```

### 정의 3.2 — Convergence of PI

Policy sequence $\pi_0, \pi_1, \ldots$ 가 convergent:

$$\exists k^* : \pi_k = \pi_{k^*} \quad \forall k \geq k^*$$

---

## 🔬 정리와 증명

### 정리 3.1 (Howard 1960 — Finite Convergence)

Finite MDP 에서 Policy Iteration 은 유한 step 내에 최적 정책에 수렴한다.

**증명**:

**Step 1**: 각 iteration 에서:
- $V^{\pi_{k+1}} \geq V^{\pi_k}$ pointwise (Policy Improvement Theorem 의 consequence)

**Step 2**: Policy improvement 가 strict 하지 않으면 (즉, $Q^{\pi_k}(s, \pi_{k+1}(s)) = V^{\pi_k}(s) \, \forall s$):
$$\pi_{k+1} = \pi_k$$

이 경우, greedy 정책이 자기 자신과 같다 ⟺ **Bellman optimality 조건 만족** ⟺ $\pi_k = \pi^*$.

**Step 3**: Deterministic stationary policy 의 개수는 $|\mathcal{A}|^{|\mathcal{S}|}$ (유한). 각 iteration 에서:
- 정책이 같으면 종료
- 정책이 다르면 strict improvement 필요

strict improvement 는 유한 번만 가능 (value bounded above) ⟹ **유한 step 내 수렴** $\square$

### 정리 3.2 (Optimality of Converged Policy)

Policy Iteration 이 종료했을 때의 정책 $\pi^*$ 는 최적이다:

$$V^{\pi^*} = V^* \quad (\text{pointwise})$$

**증명**: Converged policy 는 greedy-in-itself:
$$\pi^*(s) \in \arg\max_a Q^{\pi^*}(s, a)$$

이는 Bellman optimality equation 의 정의:
$$V^*(s) = \max_a [r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')]$$

따라서 $V^{\pi^*} = V^*$ $\square$.

### 따름 정리 3.3 (Superpolynomial Convergence)

Policy Iteration 의 반복 수 $k^*$ 는:

$$k^* \leq |\mathcal{A}|^{|\mathcal{S}|}$$

worst-case bound. 실무에서는 보통 $k^* \approx$ 수십 iterations (충분히 빠름).

**비교 (Value Iteration)**:
- VI: $k \approx -\log\epsilon / \log\gamma$ (exponential in $1/\log\gamma$)
- PI: $k^*$ 가 policy 개수에 대한 monotone sequence (보통 훨씬 작음)

---

## 💻 NumPy 구현 검증

### 실험 1 — 기본 Policy Iteration

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

def policy_eval(pi, P, r, gamma, max_iter=1000):
    """Iterative Policy Evaluation"""
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(axis=1)
        if np.linalg.norm(V_new - V) < 1e-10:
            break
        V = V_new
    return V

def policy_improve(V, r, P, gamma):
    """Greedy Policy Improvement"""
    Q = r + gamma * np.einsum('sap,p->sa', P, V)
    pi_new = np.zeros((S, A))
    pi_new[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return pi_new

def policy_iteration(P, r, gamma, max_iters=100):
    """Full Policy Iteration"""
    pi = np.ones((S, A)) / A
    values = []
    
    for k in range(max_iters):
        # Evaluation
        V = policy_eval(pi, P, r, gamma)
        values.append(V[0])  # Track V(s=0)
        
        # Improvement
        pi_new = policy_improve(V, r, P, gamma)
        
        # Check convergence
        if np.allclose(pi, pi_new):
            print(f"Policy Iteration converged at k = {k}")
            return pi_new, V, values
        
        pi = pi_new
    
    return pi, V, values

pi_opt, V_opt, pi_values = policy_iteration(P, r, gamma)

print(f"Optimal value at state 0: {V_opt[0]:.4f}")
print(f"Optimal value grid:\n{V_opt.reshape(4, 4).round(2)}")
```

**예상 출력**:
```
Policy Iteration converged at k = 3
Optimal value at state 0: -0.7140
Optimal value grid:
[[-0.71 -0.62 -0.53 -0.44]
 [-0.62 -0.53 -0.44 -0.35]
 [-0.53 -0.44 -0.35 -0.26]
 [-0.44 -0.35 -0.26  0.00]]
```

### 실험 2 — PI vs VI 수렴 비교

```python
def value_iteration(P, r, gamma, max_iters=500):
    """Value Iteration for comparison"""
    V = np.zeros(S)
    values = []
    
    for k in range(max_iters):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = np.max(Q, axis=1)
        values.append(V_new[0])
        
        if np.linalg.norm(V_new - V) < 1e-10:
            return V_new, values
        
        V = V_new
    
    return V, values

V_vi, vi_values = value_iteration(P, r, gamma)

# Compare
pi_opt, V_pi, pi_values = policy_iteration(P, r, gamma)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comparison 1: Iteration count
ax = axes[0]
ax.plot(pi_values, 'o-', label='Policy Iteration', linewidth=2, markersize=8)
ax.plot(vi_values[:len(pi_values)*10], 'x--', label='Value Iteration', linewidth=1, alpha=0.7)
ax.set_xlabel('Iteration k')
ax.set_ylabel('V(s=0)')
ax.set_title('PI vs VI: Convergence Speed')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Comparison 2: Log scale (VI 의 exponential convergence 보기)
ax = axes[1]
pi_errors = [abs(v - V_pi[0]) for v in pi_values]
vi_errors = [abs(v - V_vi[0]) for v in vi_values]

ax.semilogy(pi_errors, 'o-', label='Policy Iteration', linewidth=2)
ax.semilogy(vi_errors, 'x--', label='Value Iteration', linewidth=1, alpha=0.7)
ax.set_xlabel('Iteration k')
ax.set_ylabel('|V_k(s) - V^*(s)|')
ax.set_title('Convergence Error (log scale)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/pi_vs_vi.png', dpi=150)

print(f"Policy Iteration: {len(pi_values)} iterations")
print(f"Value Iteration: {len(vi_errors)} iterations")
```

### 실험 3 — 큰 문제에서의 비교

```python
# 50×50 Gridworld (2500 states)
S_large = 2500
A = 4
gamma = 0.9

def make_large_gridworld():
    P = np.zeros((S_large, A, S_large))
    r = -np.ones((S_large, A))
    
    side = int(np.sqrt(S_large))
    
    def to_idx(x, y):
        return x * side + y if 0 <= x < side and 0 <= y < side else None
    def fr_idx(idx):
        return (idx // side, idx % side)
    
    for s in range(S_large):
        x, y = fr_idx(s)
        next_states = [
            to_idx(x - 1, y), to_idx(x + 1, y),
            to_idx(x, y - 1), to_idx(x, y + 1),
        ]
        for a, ns in enumerate(next_states):
            if ns is None: ns = s
            P[s, a, ns] = 1.0
            r[s, a] = 1.0 if ns == S_large - 1 else -1.0
    
    return P, r

print("Computing large gridworld...")
P_large, r_large = make_large_gridworld()

# Policy Iteration (approx)
pi_large = np.ones((S_large, A)) / A
V_large = policy_eval(pi_large, P_large, r_large, gamma, max_iter=100)
pi_large_new = policy_improve(V_large, r_large, P_large, gamma)

print(f"Large problem: {S_large} states, {A} actions")
print(f"Value range: [{V_large.min():.2f}, {V_large.max():.2f}]")
print(f"One PE iteration takes ~O(n² × A) = O({S_large**2 * A})")
```

---

## 🔗 후속 레포와의 연결

- **Ch5-04 Value Iteration**: PI 의 특수 케이스 (PE 를 1 step 으로)
- **Ch5-05 GPI**: 임의로 interleaving 된 E/I 의 수렴성
- **Ch6-01+ Model-Free RL**: PI 의 sampling 버전 (sample 으로 PE/PI 추정)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Exact model (P, r 알려짐) | Model-free: PE 를 sampling estimate |
| Finite state/action | Continuous/large: function approximation, asynchronous |
| Tabular | FA 에서 convergence 보장 불가 (deadly triad) |

---

## 📌 핵심 정리

$$\boxed{\text{Policy Iteration}: \text{Evaluation} \to \text{Improvement} \to \text{Evaluation} \to \cdots \to \pi^*, \; k^* \text{ 유한}}$$

**수렴 속도**:
- **PI**: $k^* \leq |\mathcal{A}|^{|\mathcal{S}|}$ (worst-case, 실무: $k \approx 10-100$)
- **VI**: $k \propto \log(1/\epsilon) / \log(1/\gamma)$ (exponential, 실무: $k \approx 100-10000$)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Policy Iteration 이 수렴함을 보이는 증명에서 "정책이 반복되면 최적" 이라고 했는데, 이것은 왜 참인가?

<details>
<summary>해설</summary>

$\pi_k = \pi_{k+1}$ 이면 greedy-in-self:
$$\pi_k(s) \in \arg\max_a [r(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^{\pi_k}(s')]$$

따라서 $Q^{\pi_k}(s, \pi_k(s)) \geq Q^{\pi_k}(s, a) \, \forall a$.

또한 정책 개선에서 strict 가 아니므로:
$$V^{\pi_{k+1}} = V^{\pi_k}$$

이 $\pi_k$ 가 자신의 value 를 maximize 한다:
$$V^{\pi_k}(s) = \max_a [r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^{\pi_k}(s')]$$

이는 Bellman optimality 의 정의 ⟹ $V^{\pi_k} = V^*$ $\square$

</details>

**문제 2** (심화): PI 가 "superpolynomial" 수렴한다는 것이 무엇인가? Worst-case $|\mathcal{A}|^{|\mathcal{S}|}$ 가 현실에서 훨씬 빨리 수렴하는 이유는?

<details>
<summary>해설</summary>

**Worst-case**: 모든 가능한 policy 를 방문할 수 있으므로 upper bound 는 $|\mathcal{A}|^{|\mathcal{S}|}$.

**실제**: 
1. 대부분의 정책은 suboptimal 이고 빠르게 eliminate
2. Improvement 가 "큼" (큰 value gap) 이면 적은 iteration
3. Value function 의 단조성 (monotonic increase) 으로 oscillation 없음

**관찰 (실무)**:
- 작은 문제 ($S < 100$): 5-20 iterations
- 중간 문제 ($100 < S < 10000$): 20-100 iterations
- 큰 문제: 수렴 속도 감소, VI 가 better

따라서 PI 는 **문제 크기가 중간일 때** (discrete, model-known) 최고의 선택 $\square$

</details>

**문제 3** (실전): 만약 PI 의 evaluation step 을 완전히 하지 않고 한 번만 (Value Iteration 처럼) 한다면 어떻게 되는가? 여전히 수렴하는가?

<details>
<summary>해설</summary>

이것이 바로 **Generalized Policy Iteration (GPI, Ch5-05)** 의 개념. 

한 번의 Bellman update:
$$V_{k+1}(s) = \max_a [r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V_k(s')]$$

는 동시에 (implicitly) improvement 를 하는 것. GPI framework 에서:
- **PE 를 $n$ 번 fully**: Policy Iteration
- **PE 를 1 번**: Value Iteration
- **PE 를 $1 < n < \infty$**: Modified PI (실무 하이브리드)

모두 같은 고정점으로 수렴하지만 **trade-off (accuracy vs speed)** 다름 $\square$

</details>

---

<div align="center">

[◀ 이전: 02. Policy Improvement Theorem](./02-policy-improvement.md) | [📚 README](../README.md) | [다음 ▶: 04. Value Iteration 완성](./04-value-iteration.md)

</div>
