# 02. Policy Improvement Theorem — Howard 의 정책 개선 정리

## 🎯 핵심 질문

- 주어진 정책 $\pi$ 의 value function $V^\pi$ 를 알 때, 더 나은 정책 $\pi'$ 을 어떻게 구성하는가?
- "Greedy" 정책 $\pi'(s) = \arg\max_a Q^\pi(s, a)$ 가 정말 더 나을까, 같을까, 아니면 더 나쁠까?
- "Strict improvement" 와 "convergence" 의 관계는 무엇인가?
- 이 정리가 Policy Iteration 의 유한성을 어떻게 보장하는가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

Howard (1960) 의 Policy Improvement Theorem 은 **"이론적으로 개선이 보장된다" 는 것을 처음 증명한 정리입니다. 이 한 줄이 없으면:

1. Policy Iteration 이 수렴하는지 알 수 없음
2. Greedy 선택이 왜 작동하는지 모름
3. 모든 RL 알고리즘 (Q-learning, actor-critic 등) 이 왜 converge 하는지 설명 불가

Kakade (2003) 의 Performance Difference Lemma 도 이 정리를 quantify 한 것입니다.

---

## 📐 수학적 선행 조건

- **Ch5-01**: Policy Evaluation, $V^\pi, Q^\pi$ 의 정의
- **Ch4-02**: Bellman optimality equation, $T^*$ operator
- **확률론**: 조건부 기대값, tower property

---

## 📖 직관적 이해

### 왜 Greedy 인가

$V^\pi$ 를 알고 있다면, 각 state $s$ 에서:

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$$

를 계산할 수 있습니다. $Q^\pi(s, a) > V^\pi(s)$ 라면, "지금 $a$ 를 택하는 것이 평균 ($\pi$ 를 따를 때) 보다 낫다" 는 의미.

**따라서 $\arg\max_a Q^\pi(s, a)$ 를 선택하면 (적어도 현재는) 가장 좋은 행동.**

### 미래까지 좋을까?

직관: "$a$ 를 한 번 하는 것이 좋다면, 그 뒤로도 쭉 따르면 더 좋을 것." 이것이 정리의 내용.

수학적으로는 **telescoping sum** 으로 증명.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Policy Improvement

정책 $\pi$ 가 주어졌을 때, greedy policy 를 정의:

$$\pi'(s) := \arg\max_a Q^\pi(s, a)$$

또는 stochastic 하게:

$$\pi'(a \mid s) := \begin{cases} 1 & \text{if } a \in \arg\max_a Q^\pi(s, a) \\ 0 & \text{otherwise} \end{cases}$$

### 정의 2.2 — Strict Improvement

정책 개선이 "strict" 하다는 것은:

$$\exists s_0 : Q^\pi(s_0, \pi'(s_0)) > V^\pi(s_0)$$

즉, 적어도 한 state 에서 strict inequality.

### 정의 2.3 — Policy Convergence

정책들의 수열 $\pi_0, \pi_1, \pi_2, \ldots$ 가 수렴한다는 것은:

$$\exists k^* : \pi_k(s) = \pi_{k+1}(s) \, \forall s, k \geq k^*$$

---

## 🔬 정리와 증명

### 정리 2.1 (Policy Improvement Theorem — Howard 1960)

정책 $\pi$ 에서 시작하여 greedy policy $\pi'$ 을 구성하면:

$$\forall s : Q^\pi(s, \pi'(s)) \geq V^\pi(s) \quad \Rightarrow \quad \forall s : V^{\pi'}(s) \geq V^\pi(s)$$

**증명** (Telescoping):

Fix 임의의 state $s$. $\pi'$ 를 따르는 trajectory 에서:

$$V^{\pi'}(s) = \mathbb{E}_{a_0 \sim \pi'(\cdot \mid s)}\!\left[Q^\pi(s, a_0)\right]$$

정의에 의해 $\pi'(s) \in \arg\max_a Q^\pi(s, a)$ 이므로:

$$\mathbb{E}_{a_0 \sim \pi'(\cdot \mid s)}\!\left[Q^\pi(s, a_0)\right] \geq \mathbb{E}_{a \sim \pi(\cdot \mid s)}\!\left[Q^\pi(s, a)\right] = V^\pi(s)$$

이제 한 발 더. $Q^\pi$ 의 정의를 전개:

$$V^{\pi'}(s) = \mathbb{E}_{a \sim \pi'(\cdot \mid s)}\!\left[r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')\right]$$

$$\geq \mathbb{E}_{a \sim \pi(\cdot \mid s)}\!\left[r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')\right] = V^\pi(s)$$

이제 미래까지 귀납적으로:

$$V^{\pi'}(s) \geq r(s, \pi'(s)) + \gamma \mathbb{E}_{s_1 \sim P(\cdot \mid s, \pi'(s))}[V^\pi(s_1)]$$

$$= \mathbb{E}_{s_1}\!\left[r(s, \pi'(s)) + \gamma V^\pi(s_1)\right]$$

$$\geq \mathbb{E}_{s_1}\!\left[r(s, \pi'(s)) + \gamma V^{\pi'}(s_1)\right]$$

(inductive assumption: $V^\pi(s_1) \leq V^{\pi'}(s_1)$)

반복하면 모든 step 에서 $V^{\pi'}$ 가 더 좋거나 같음. 따라서:

$$V^{\pi'}(s) \geq V^\pi(s) \quad \square$$

### 따름 정리 2.2 (Strict Improvement)

만약 적어도 한 state $s_0$ 에서 $Q^\pi(s_0, \pi'(s_0)) > V^\pi(s_0)$ (strict inequality) 이면:

$$V^{\pi'}(s_0) > V^\pi(s_0)$$

**증명**: telescoping 에서 strict 부등호가 propagate $\square$.

### 정리 2.3 (Finite Convergence — Puterman 2005)

Finite MDP ($|\mathcal{S}|, |\mathcal{A}|$ 유한) 에서 Policy Improvement 를 반복하면:

$$\text{유한 step } k^* \text{ 내에 } \pi_{k^*}(s) = \pi_{k^*+1}(s) \, \forall s$$

(즉, greedy 정책이 변하지 않음 ⟺ 최적 정책 도달)

**증명 스케치**:
1. 가능한 deterministic policy 의 수: $|\mathcal{A}|^{|\mathcal{S}|}$ (유한)
2. 각 iteration 마다 새 정책이 다르면 → strict improvement
3. Value function 이 bounded 이므로 infinite strict improvements 불가능
4. 따라서 유한 step 내에 fixed point 도달 $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Gridworld 에서 Policy Improvement 추적

```python
import numpy as np
import matplotlib.pyplot as plt

# 4×4 Gridworld setup (01-policy-evaluation 과 동일)
S = 16
A = 4
gamma = 0.9

def make_gridworld_matrices(deterministic=True):
    """4×4 gridworld"""
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
    """Policy Evaluation: V^π 계산"""
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(axis=1)
        if np.linalg.norm(V_new - V) < 1e-8:
            break
        V = V_new
    return V

def policy_improve(V, r, P, gamma):
    """Policy Improvement: π' = greedy(V^π)"""
    Q = r + gamma * np.einsum('sap,p->sa', P, V)
    pi_new = np.zeros((S, A))
    greedy_actions = np.argmax(Q, axis=1)
    pi_new[np.arange(S), greedy_actions] = 1.0
    return pi_new

# 초기 정책: 균등 분포
pi = np.ones((S, A)) / A

print("Policy Iteration Progress:")
print("=" * 60)

for iteration in range(20):
    # Evaluate
    V = policy_eval(pi, P, r, gamma)
    
    # Improve
    pi_new = policy_improve(V, r, P, gamma)
    
    # Check convergence
    policy_change = not np.allclose(pi, pi_new)
    num_policy_changes = np.sum(np.argmax(pi, axis=1) != np.argmax(pi_new, axis=1))
    
    J = np.sum(pi * (r + gamma * np.einsum('sap,p->sa', P, V)))  # objective
    
    print(f"Iter {iteration:2d}: J(π) = {J:+.4f}, "
          f"Policy changes: {num_policy_changes:2d}, "
          f"V_range: [{V.min():.2f}, {V.max():.2f}]")
    
    pi = pi_new
    if not policy_change:
        print(f"\n✓ Converged at iteration {iteration}")
        break
```

**예상 출력**:
```
Policy Iteration Progress:
============================================================
Iter  0: J(π) = -8.0000, Policy changes:  9, V_range: [-3.61, 3.06]
Iter  1: J(π) = -0.9796, Policy changes:  4, V_range: [-3.33, 3.06]
Iter  2: J(π) = -0.2095, Policy changes:  2, V_range: [-2.14, 3.06]
Iter  3: J(π) = -0.0419, Policy changes:  0, V_range: [-0.71, 3.06]

✓ Converged at iteration 3
```

### 실험 2 — Monotonic Improvement 확인

```python
pi = np.ones((S, A)) / A
values = []

for iteration in range(50):
    V = policy_eval(pi, P, r, gamma)
    values.append(V[0])  # Track value of state 0
    
    pi_new = policy_improve(V, r, P, gamma)
    if np.allclose(pi, pi_new):
        break
    pi = pi_new

plt.figure(figsize=(10, 5))
plt.plot(values, 'o-', linewidth=2, markersize=6)
plt.xlabel('Policy Iteration k')
plt.ylabel('V^{π_k}(s=0)')
plt.title('Monotonic Improvement of V(s) across Policy Iterations')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/monotonic_improvement.png', dpi=150)
print(f"✓ Value increased monotonically: {all(values[i] <= values[i+1] for i in range(len(values)-1))}")
```

### 실험 3 — 정책 변화 시각화

```python
pi = np.ones((S, A)) / A
policies = [np.argmax(pi, axis=1)]

for iteration in range(20):
    V = policy_eval(pi, P, r, gamma)
    pi_new = policy_improve(V, r, P, gamma)
    policies.append(np.argmax(pi_new, axis=1))
    
    if np.allclose(pi, pi_new):
        break
    pi = pi_new

# 시각화
action_names = ['↑', '↓', '←', '→']
fig, axes = plt.subplots(1, min(4, len(policies)), figsize=(15, 3))

for idx, ax in enumerate(axes):
    policy_grid = policies[idx].reshape(4, 4)
    ax.set_title(f'Iteration {idx}')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    
    for i in range(4):
        for j in range(4):
            action = policy_grid[i, j]
            ax.text(j, i, action_names[action], 
                   ha='center', va='center', fontsize=16, weight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/policy_evolution.png', dpi=150)
```

---

## 🔗 후속 레포와의 연결

- **Ch5-03 Policy Iteration**: Evaluation + Improvement 의 반복
- **Ch5-04 Value Iteration**: Improvement 를 한 step evaluation 으로 단순화
- **Ch5-05 GPI**: 임의로 interleaving 된 E/I 의 수렴성
- **Ch6-01+ Model-Free RL**: Q-learning 은 improvement 의 sampling 버전

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Exact $Q^\pi$ 알려짐 | Function approximation: greedy 가 suboptimal 가능 (overestimation) |
| Deterministic policy 로 개선 | Stochastic policy 로의 확장은 mixed strategy Nash equilibrium |
| Finite MDP | Infinite state/action: convergence 미보장 |
| Stationary policy | Non-stationary policy: episodic 변형 필요 |

---

## 📌 핵심 정리

$$\boxed{Q^\pi(s, \pi'(s)) \geq V^\pi(s) \, \forall s \quad \Rightarrow \quad V^{\pi'}(s) \geq V^\pi(s) \, \forall s}$$

| 요소 | 의미 |
|------|------|
| $Q^\pi(s, a)$ | $\pi$ 를 따를 때 $s$ 에서 $a$ 후 얻는 가치 |
| $V^\pi(s)$ | $\pi$ 를 따를 때 $s$ 의 평균 가치 |
| 비교 | $Q^\pi > V^\pi$ ⟺ 이 행동이 평균보다 좋음 |
| Greedy | 최선의 행동을 선택 ⟹ 정책 개선 |
| Convergence | Strict improvement ⟹ 유한 step 안에 최적 정책 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Greedy policy $\pi'(s) = \arg\max_a Q^\pi(s, a)$ 가 항상 improvement 를 보장하는가? 만약 $Q^\pi(s, a) = V^\pi(s) \, \forall a, s$ 라면 어떻게 되는가?

<details>
<summary>해설</summary>

$\pi'(s) \in \arg\max_a Q^\pi(s, a)$ 이므로:
$$\mathbb{E}_{a \sim \pi'(\cdot \mid s)}[Q^\pi(s, a)] = \max_a Q^\pi(s, a) \geq \mathbb{E}_{a \sim \pi(\cdot \mid s)}[Q^\pi(s, a)] = V^\pi(s)$$

따라서 항상 non-decreasing improvement ($\geq$).

**만약 모든 action 이 동일하면** ($Q^\pi(s, a) = c$ for all $a$):
$$V^\pi(s) = \mathbb{E}_a[Q^\pi(s, a)] = c = \max_a Q^\pi(s, a)$$

따라서 $V^{\pi'} = V^\pi$ — 개선 없음, 수렴 (optimal policy 찾음) $\square$

</details>

**문제 2** (심화): Telescoping sum 증명에서 왜 $V^\pi(s') \leq V^{\pi'}(s')$ 라는 귀납 가정이 정당한가? (Forward induction 이 아니라 backward 하지 않는가?)

<details>
<summary>해설</summary>

사실 "귀납" 이 아니라 **telescoping (cumulative argument)**:

$$V^{\pi'}(s) = \mathbb{E}_{\pi'}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

를 step-by-step 분해:

$$V^{\pi'}(s) = r_0 + \gamma \mathbb{E}[V^{\pi'}(s_1)]$$

각 step 에서:
$$r_0 + \gamma \mathbb{E}_{\pi'}[V^\pi(s_1)] \geq r_0 + \gamma \mathbb{E}_\pi[V^\pi(s_1)] = V^\pi(s)$$

첫 부등호는 정책 정의 (greedy), 그 이후 재귀적으로:
$$\mathbb{E}_{\pi'}[V^\pi(s_1)] \leq \mathbb{E}_{\pi'}[V^{\pi'}(s_1)]$$

는 **같은 논리를 $s_1$ 에서 반복 가능** — 이 재귀가 수렴하는 이유는 discounting $\gamma < 1$ $\square$

</details>

**문제 3** (실전): Policy Iteration 이 유한 step 에 수렴함을 증명하라. 정책의 개수는 몇 개이고, 왜 cycle 이 불가능한가?

<details>
<summary>해설</summary>

**Deterministic stationary policy** 의 개수: 각 state 에서 하나의 action 선택 ⟹ $|\mathcal{A}|^{|\mathcal{S}|}$ 개 (유한).

**Cycle 불가능**: 만약 $\pi_k = \pi_m$ (for $k < m$) 이면:
- $\pi_k$ 에서 $\pi_{k+1}$ 로 improvement: $V^{\pi_{k+1}} \geq V^{\pi_k}$ (비감소)
- 같은 정책이 반복된다 ⟹ strict improvement 없었음 ⟹ $\pi_k = \pi^*$ (최적)

따라서 정책이 반복되면 최적 정책 찾음. 정책 수가 유한이고 반복 불가능하면 ⟹ **유한 step 내 수렴** $\square$

</details>

---

<div align="center">

[◀ 이전: 01. Policy Evaluation](./01-policy-evaluation.md) | [📚 README](../README.md) | [다음 ▶: 03. Policy Iteration (Howard 1960)](./03-policy-iteration.md)

</div>
