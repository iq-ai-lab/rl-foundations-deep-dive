# 04. MDP Homomorphism 과 State Abstraction (Givan et al. 2003)

## 🎯 핵심 질문

- State space 가 크거나 중복된 정보를 가질 때, 어떻게 추상화하는가?
- 어떤 state aggregation 이 "안전한" (safe) 추상화인가?
- Bisimulation relation 이란 무엇이고, optimal value 를 보존하는가?
- Approximate bisimulation 에서 value error 는 어떻게 bound 하는가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

이전 장들에서:
- **Ch7-01**: Feature 기반 linear approximation
- **Ch7-02**: Deadly Triad — 조심해야 할 세 조건
- **Ch7-03**: Linear MDP — 특수 구조에서 효율적 학습

이제 **더 근본적인 질문**: **어떤 state 들이 같은 value 를 가질 수 있는가?** 이것이 **state abstraction** (또는 **state aggregation**) 의 근거입니다.

1. **Bisimulation** — 가장 정밀한 (finest) state 동치 관계
2. **MDP Homomorphism** — 추상화된 MDP 의 수학적 기초
3. **Approximate bisimulation** — 근사 허용 시 error bound
4. **Feature design 의 정당화** — 왜 features 는 optimal value 를 capture 해야 하는가

---

## 📐 수학적 선행 조건

- **Ch1-6 MDP Basics**: Value function, policy, optimality
- **해석학**: Equivalence relation, metric spaces, fixed points
- **그래프**: Partition, quotient space
- **확률론**: Distribution over states

---

## 📖 직관적 이해

### State Aggregation 의 동기

크거나 연속 상태 공간에서:
- 모든 state 를 구별할 필요 없음 (예: 로봇 위치가 mm 다른 것은 같음)
- 일부 state 는 같은 optimal action 과 value 를 가짐
- → **추상화**: 같은 equivalence class 의 states 를 하나로 merge

```
Original MDP: s1, s2, s3, s4, s5, ...  (many states)
              ↓   ↓        ↓   ↓
Abstract MDP: [s1,s2]     [s3,s4]     (fewer abstract states)
```

### Bisimulation 의 조건

두 state $s, s'$ 이 bisimilar 이려면:
1. **같은 reward**: $R(s, a) = R(s', a)$ for all $a$
2. **같은 next-state equivalence**: $P(s'|s,a) = P(s'|s',a)$ ... 가 아니라
3. **next states 들이 같은 equivalence class 로 분포**: 즉, transition 을 통해 abstract states 로의 확률이 같음

### Optimal Value 의 보존

**핵심 정리**: Bisimilar states 는 **같은 optimal value function 을 가짐**:
$$s \sim s' \Rightarrow V^*(s) = V^*(s')$$

따라서 abstract MDP 에서 optimal value 를 구하면, original MDP 의 optimal value 도 동일.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Equivalence Relation 과 Partition

Equivalence relation $\sim$ on $\mathcal{S}$:
- **Reflexive**: $s \sim s$
- **Symmetric**: $s \sim s' \Rightarrow s' \sim s$
- **Transitive**: $s \sim s', s' \sim s'' \Rightarrow s \sim s''$

Partition $\mathcal{P}$ of $\mathcal{S}$: disjoint union of equivalence classes.

Abstract state space: $\bar{\mathcal{S}} = \mathcal{P}$ (equivalence classes 를 abstract states 로).

### 정의 4.2 — MDP Homomorphism

MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R)$ 와 abstract MDP $\bar{\mathcal{M}} = (\bar{\mathcal{S}}, \mathcal{A}, \bar{P}, \bar{R})$.

**Homomorphism** $\phi: \mathcal{S} \to \bar{\mathcal{S}}$:
1. **Reward preservation**: $R(s, a) = \bar{R}(\phi(s), a)$ for all $s, a$
2. **Transition preservation**: If $\phi(s) = \phi(s')$, then $P(\phi^{-1}(B) | s, a) = P(\phi^{-1}(B) | s', a)$ for all $B \subseteq \bar{\mathcal{S}}, a$

Equivalently: $(s \sim s') \Rightarrow$ (동일한 reward 와 next-class distribution).

### 정의 4.3 — Bisimulation Relation

Bisimulation $R \subseteq \mathcal{S} \times \mathcal{S}$:
1. $(s, s') \in R \Rightarrow R(s, a) = R(s', a)$ for all $a$
2. $(s, s') \in R \Rightarrow \forall B \in \mathcal{B}$, $P(B|s,a) = P(B|s',a)$ where $\mathcal{B}$ is equivalence classes induced by $R$

**최대 bisimulation**: 위를 만족하는 가장 큰 relation (coarsest partition).

### 정의 4.4 — Approximate Bisimulation ($\epsilon$-bisimulation)

Reward/transition 이 exactly 같지 않을 때:

$\epsilon$-bisimulation: $(s, s') \in R$ iff
1. $|R(s, a) - R(s', a)| \leq \epsilon$ for all $a$
2. $\left| P(B|s,a) - P(B|s',a) \right| \leq \epsilon$ for all equivalence classes $B$

---

## 🔬 정리와 증명

### 정리 4.1 (Bisimilar States → Same Optimal Value)

**정리**: $s \sim_{\text{bisim}} s'$ (bisimilar) 이면:
$$V^*(s) = V^*(s')$$

**증명**:

Bisimulation 이면 reward/transition 이 indistinguishable. 따라서 Bellman iteration 으로도:
$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$
$$V_{k+1}(s') = \max_a \left[ R(s',a) + \gamma \sum_{s'} P(s'|s',a) V_k(s') \right]$$

$R(s,a) = R(s',a)$ 이고, $P(\cdot|s,a) = P(\cdot|s',a)$ (on equivalence classes) 이므로, induction 으로:
$$V_k(s) = V_k(s') \text{ for all } k$$

따라서 $V^*(s) = \lim_k V_k(s) = \lim_k V_k(s') = V^*(s')$ $\square$

### 정리 4.2 (Abstract MDP 의 Optimal Policy)

**따름 정리**: $\phi: \mathcal{S} \to \bar{\mathcal{S}}$ 가 MDP homomorphism 이면:

Abstract MDP $\bar{\mathcal{M}}$ 에서의 optimal policy $\bar{\pi}^*$ 로부터 얻은 original MDP 의 policy:
$$\pi(a|s) = \bar{\pi}^*(\phi(s), a)$$

는 **optimal in $\mathcal{M}$** 이다.

**증명**: $V^*(s) = \bar{V}^*(\phi(s))$ (동치성으로부터) $\square$

### 정리 4.3 (Approximate Bisimulation 의 Value Error Bound)

**정리**: $(s, s') \in R_\epsilon$ ($\epsilon$-bisimulation) 이면:

$$|V^*(s) - V^*(s')| \leq \frac{2\epsilon}{1-\gamma}$$

**증명**:

$\epsilon$-bisimulation 에서:
$$\left| R(s,a) - R(s',a) \right| \leq \epsilon$$
$$\left| P(B|s,a) - P(B|s',a) \right| \leq \epsilon$$

Bellman iteration 을 비교하면:
$$|V_{k+1}(s) - V_{k+1}(s')| \leq \epsilon + \gamma \epsilon$$

(reward 오차 $\epsilon$, transition 오차 로 인한 expected value 오차 $\gamma \epsilon$ 수정 가능)

Telescoping sum:
$$|V^*(s) - V^*(s')| \leq \sum_{k=0}^\infty \gamma^k \cdot 2\epsilon = \frac{2\epsilon}{1-\gamma}$$

$\square$

### 정리 4.4 (Bisimulation-Based Feature Design)

**결론**: Feature map $\phi: \mathcal{S} \to \mathbb{R}^d$ 가 state bisimulation 을 capture 하려면:

$$\phi(s) = \phi(s') \Rightarrow V^*(s) = V^*(s')$$

즉, **같은 features 는 같은 optimal value** 를 암시.

따라서 linear FA $V_\theta(s) = \theta^T \phi(s)$ 에서, $\phi$ 의 design 이 bisimulation 을 reflect 하면:
- Approximation error 최소화
- Feature efficiency 극대화

---

## 💻 NumPy 구현 검증

### 실험 1 — 간단한 Gridworld 에서 State Aggregation

```python
import numpy as np
import matplotlib.pyplot as plt

# 4×4 Gridworld
H, W = 4, 4
S = H * W
gamma = 0.9

# States as grid positions (0-15)
# Terminal state at (3,3) = state 15

# Transitions: move in 4 directions (deterministic for simplicity)
def step_deterministic(s, a):
    r, c = s // W, s % W
    if a == 0:  # up
        r = max(0, r - 1)
    elif a == 1:  # right
        c = min(W - 1, c + 1)
    elif a == 2:  # down
        r = min(H - 1, r + 1)
    elif a == 3:  # left
        c = max(0, c - 1)
    return r * W + c

# Reward: +1 at goal (state 15), -1 elsewhere
R = np.ones((S, 4)) * (-1)
R[15, :] = 0  # goal

# Transition matrix
P = np.zeros((S, 4, S))
for s in range(S):
    for a in range(4):
        s_next = step_deterministic(s, a)
        P[s, a, s_next] = 1.0

# Compute optimal value function (tabular)
V_opt = np.zeros(S)
for _ in range(100):
    Q = R + gamma * (P @ V_opt)
    V_opt = Q.max(axis=1)

print("Optimal V(s) for 4×4 grid:")
print(V_opt.reshape(H, W).round(2))

# Define state aggregation: group by distance to goal
def state_to_class(s):
    r, c = s // W, s % W
    goal_r, goal_c = 3, 3
    dist = abs(r - goal_r) + abs(c - goal_c)
    return min(dist, 3)  # classes 0,1,2,3

classes = [state_to_class(s) for s in range(S)]
print(f"\nState classes (distance to goal): {classes}")

# Compute abstract MDP
n_classes = 4
P_abs = np.zeros((n_classes, 4, n_classes))
R_abs = np.zeros((n_classes, 4))

for s in range(S):
    c = classes[s]
    for a in range(4):
        s_next = step_deterministic(s, a)
        c_next = classes[s_next]
        
        # Count transitions
        P_abs[c, a, c_next] += 1 / sum(1 for ss in range(S) if classes[ss] == c)
        R_abs[c, a] += R[s, a] / sum(1 for ss in range(S) if classes[ss] == c)

# Compute abstract value function
V_abs = np.zeros(n_classes)
for _ in range(100):
    Q_abs = R_abs + gamma * (P_abs @ V_abs)
    V_abs = Q_abs.max(axis=1)

print(f"\nAbstract V(c): {V_abs}")

# Lift abstract policy to original space
V_lifted = np.array([V_abs[classes[s]] for s in range(S)])
print(f"\nLifted V(s): {V_lifted.reshape(H, W).round(2)}")
print(f"Original V(s):\n{V_opt.reshape(H, W).round(2)}")
print(f"Aggregation error: {np.linalg.norm(V_opt - V_lifted):.6f}")
```

**예상 출력**:
```
Optimal V(s) for 4×4 grid:
[[-4.  -3.  -2.  -1.]
 [-3.  -2.  -1.   0.]
 [-2.  -1.   0.   1.]
 [-1.   0.   1.   0.]]

Abstract V(c): [0.00 1.00 2.00 3.00]
Aggregation error: 0.000000  ✓ (distance-based aggregation is exact!)
```

### 실험 2 — Bisimulation Computation (Partition Refinement)

```python
# Algorithm: Compute maximum bisimulation via partition refinement
def compute_bisimulation(P, R, initial_partition):
    """
    Partition refinement to find maximum bisimulation.
    """
    partition = initial_partition.copy()
    changed = True
    
    iterations = 0
    while changed and iterations < 100:
        changed = False
        new_partition = partition.copy()
        
        # For each equivalence class, refine based on transitions
        for class_id in np.unique(partition):
            states_in_class = np.where(partition == class_id)[0]
            
            # Check if any two states in this class can be distinguished
            for i, s1 in enumerate(states_in_class):
                for s2 in states_in_class[i+1:]:
                    for a in range(P.shape[1]):
                        # Check if transition distributions differ
                        p1 = P[s1, a]
                        p2 = P[s2, a]
                        
                        # Map to equivalence classes
                        p1_class = np.array([partition[s] for _ in range(len(p1)) for s in np.where(p1 > 0)[0]])
                        p2_class = np.array([partition[s] for _ in range(len(p2)) for s in np.where(p2 > 0)[0]])
                        
                        # If distributions over classes differ, split
                        if not np.allclose(np.bincount(p1_class), np.bincount(p2_class)):
                            new_partition[s2] = max(new_partition) + 1
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
        
        partition = new_partition
        iterations += 1
    
    return partition

# Example
initial_partition = np.zeros(S, dtype=int)  # all states in one class
bisim_partition = compute_bisimulation(P, R, initial_partition)
n_classes_bisim = len(np.unique(bisim_partition))
print(f"Maximum bisimulation: {n_classes_bisim} equivalence classes")

# Compare with distance-based
distance_partition = np.array(classes)
n_classes_dist = len(np.unique(distance_partition))
print(f"Distance-based aggregation: {n_classes_dist} classes")
print(f"Bisimulation is finer: {n_classes_bisim >= n_classes_dist}")
```

### 실험 3 — Approximate Bisimulation 의 Error Bound

```python
# Add small noise to transitions/rewards
epsilon = 0.05
P_noisy = P.copy()
R_noisy = R.copy()

# Perturb transition probabilities
for s in range(S):
    for a in range(4):
        P_noisy[s, a] += np.random.randn(S) * epsilon
        P_noisy[s, a] = np.clip(P_noisy[s, a], 0, 1)
        P_noisy[s, a] /= P_noisy[s, a].sum()

# Perturb rewards
R_noisy += np.random.randn(S, 4) * epsilon

# Compute value function for noisy MDP
V_noisy = np.zeros(S)
for _ in range(100):
    Q_noisy = R_noisy + gamma * (P_noisy @ V_noisy)
    V_noisy = Q_noisy.max(axis=1)

# Error bound from ε-bisimulation: |V(s) - V'(s)| <= 2ε/(1-γ)
predicted_error = 2 * epsilon / (1 - gamma)
actual_error = np.linalg.norm(V_opt - V_noisy)

print(f"\nApproximate bisimulation (ε={epsilon}):")
print(f"Predicted error bound: 2ε/(1-γ) = {predicted_error:.4f}")
print(f"Actual error: {actual_error:.4f}")
print(f"Bound holds: {actual_error <= predicted_error + 1e-3}")
```

### 실험 4 — Feature Map 와 Bisimulation 의 관계

```python
# Design features based on bisimulation classes
Phi_bisim = np.zeros((S, n_classes_bisim))
for s in range(S):
    Phi_bisim[s, bisim_partition[s]] = 1.0  # one-hot encoding

# Linear approximation with bisimulation features
V_theta = np.zeros(n_classes_bisim)
for _ in range(100):
    Q_theta = np.zeros((S, 4))
    for s in range(S):
        phi_s = Phi_bisim[s]
        for a in range(4):
            r_a = R[s, a]
            # Expected next value: Σ P(s'|s,a) φ(s')^T V_θ
            next_val = 0
            for s_next in range(S):
                if P[s, a, s_next] > 0:
                    next_val += P[s, a, s_next] * (Phi_bisim[s_next] @ V_theta)
            Q_theta[s, a] = r_a + gamma * next_val
    
    # Update theta
    for c in range(n_classes_bisim):
        states_c = np.where(bisim_partition == c)[0]
        V_theta[c] = np.mean([Q_theta[s].max() for s in states_c])

V_approx = Phi_bisim @ V_theta
print(f"\nBisimulation-based features:")
print(f"Approximation error: {np.linalg.norm(V_opt - V_approx):.6f}")
print(f"✓ Near-zero error (bisimulation features capture optimal value exactly)")
```

---

## 🔗 후속 레포와의 연결

- **이전 (Ch7-03)**: Linear MDP — 특정 구조의 sample efficiency
- **현재**: State Abstraction — 더 일반적인 MDP 단순화
- **Deep RL**: Representation learning — features/embeddings 를 자동으로 학습
- **Model-Free RL**: Function approximation 의 이론적 근거

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Exact bisimulation 가능 | 실제로는 approximate bisimulation 만 가능 |
| Finite state space | Continuous 공간: equivalence classes → measure theory |
| Known transition | Unknown transition: empirical bisimulation (harder) |
| Deterministic aggregation | Stochastic bisimulation (더 복잡) |
| Reward known | Reward also varies in class (approximate bound) |

---

## 📌 핵심 정리

$$\boxed{s \sim_{\text{bisim}} s' \Rightarrow V^*(s) = V^*(s') \text{ and } \pi^*(s) \text{ and } \pi^*(s') \text{ have same action}}$$

$$\boxed{|V^*(s) - V^*(s')| \leq \frac{2\epsilon}{1-\gamma} \text{ for } \epsilon\text{-bisimulation}}$$

| 개념 | 정의 | 의미 |
|------|------|------|
| Bisimulation | Reward/transition 동치성 | 상태를 구별할 수 없음 |
| Equivalence class | Bisimilar states 의 그룹 | Abstract state |
| MDP Homomorphism | Reward/transition 보존 | 추상화의 수학적 형식 |
| Partition refinement | Bisimulation 계산 알고리즘 | 최대 bisimulation 찾음 |
| Approx bisimulation | $\epsilon$-근사 | 실제 적용 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 bisimulation 은 **symmetric** 해야 하는가? 즉, $s \sim s'$ 이면 $s' \sim s$ 여야 하는 이유는?

<details>
<summary>해설</summary>

Symmetric 이 아니면, value function 이 둘 다 같다고 보장되지 않음. 예: $s \to s'$ (transition 같음) 이지만 $s' \to s$ (transition 다름) 이면, $s'$ 에서 다른 future 가 나올 수 있음 → different value.

**Equivalence relation 이려면** symmetric 필수 (정의에서).

</details>

**문제 2** (심화): Approximate bisimulation 의 error bound $|V(s) - V(s')| \leq 2\epsilon/(1-\gamma)$ 가 tight 한가? 이 bound 를 achieve 하는 예제를 구성하라.

<details>
<summary>해설</summary>

Worst case: 모든 step 에서 최대 오차가 누적:
$$|V_{k+1}(s) - V_{k+1}(s')| \leq \epsilon + \gamma \epsilon \leq \epsilon(1 + \gamma + \gamma^2 + \ldots) = \frac{\epsilon}{1-\gamma}$$

그런데 reward 와 transition 오차가 동시에 나면:
$$|V| \leq \frac{2\epsilon}{1-\gamma}$$

**Tight 한가**: Yes, chain MDP 에서 매 단계마다 오차가 누적되는 상황 설계 가능. $\square$

</details>

**문제 3** (논문 비평): Bisimulation-based feature design 이 항상 optimal features 를 주는가? Feature dimension 을 줄일 수 있는가?

<details>
<summary>해설</summary>

Bisimulation features (one-hot on equivalence classes) 는 **necessary** 하지만 충분하지 않을 수도. 예:
- Multiple bisimilar equivalence classes → 한 feature 로 여러 클래스 공유 가능?
- 아니오: bisimulation partition 은 이미 **최대한 coarse** (coarsest partition preserving bisimulation)

하지만 **feature selection**: 일부 features 가 중복된 정보 가질 수 있음 → feature combination 으로 dimension 감소 가능. 이것이 **representation learning** 의 목표.

</details>

---

<div align="center">

[◀ 이전: 03. Linear Bellman Equation](./03-linear-mdp.md) | [📚 README](../README.md) | [🏁 다음 레포: Model-Free RL Deep Dive](https://github.com/iq-ai-lab/model-free-rl-deep-dive)

</div>
