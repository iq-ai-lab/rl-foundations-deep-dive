# 03. Advantage Function 과 Baseline Subtraction

## 🎯 핵심 질문

- Advantage function $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ 는 무엇이고, 왜 가치 함수 자체가 아니라 차이를 사용하는가?
- Advantage 가 양수 / 음수인 것이 의미하는 바는?
- 왜 Policy Gradient 에서 baseline subtraction (종종 $V^\pi$ 사용) 이 분산을 줄이는가?
- Baseline-corrected advantage 의 수학적 정당성은?

---

## 🔍 왜 이 개념이 현대 RL 의 핵심인가

Policy Gradient theorem, Actor-Critic, TRPO, PPO, A3C — 모든 현대 policy-based 알고리즘은 **advantage function** 을 사용합니다.

그 이유는:

1. **해석 가능성**: $A > 0$ = "평균보다 좋은 행동"
2. **분산 감소**: Baseline 으로 variance reduction
3. **신호 크기**: $Q$ 자체보다 상대적 이득이 더 stable
4. **수렴성**: Baseline-corrected gradient 가 lower variance

이 문서는 advantage 의 정의, 해석, 그리고 baseline subtraction 의 수학을 명확히 합니다.

---

## 📐 수학적 선행 조건

- **Ch5-04**: $V^\pi, Q^\pi, V$ 의 Bellman equation
- **Ch6-02**: Performance Difference Lemma, $d^\pi$
- **확률론**: Conditional expectation, variance formula $\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$
- (선택) **Policy Gradient Deep Dive**: Policy gradient theorem

---

## 📖 직관적 이해

### 왜 차이인가

상태 $s$ 에서 행동 $a$ 를 평가할 때:
- $Q^\pi(s, a) = 5$ ← 절대적 가치
- $V^\pi(s) = 6$ ← 현재 상태의 평균 가치

상대적으로 $a$ 는 평균보다 나쁘다 ($A = 5 - 6 = -1$).

따라서 advantage 는 **"이 행동이 여기서 얼마나 좋은가"** 를 정량화합니다.

### Baseline Subtraction 과 Variance

Policy Gradient 학습에서:
$$
\nabla J(\pi) \propto \mathbb{E}[\nabla \log \pi(a|s) Q^\pi(s, a)]
$$

문제: $Q^\pi$ 가 크면 gradient 의 분산도 크다 (large magnitude).

해결: Baseline $b(s)$ 를 빼면:
$$
\nabla J(\pi) \propto \mathbb{E}[\nabla \log \pi(a|s) (Q^\pi(s, a) - b(s))]
$$

$b(s) = V^\pi(s)$ 로 선택하면:
$$
\nabla J(\pi) \propto \mathbb{E}[\nabla \log \pi(a|s) A^\pi(s, a)]
$$

**왜 분산이 줄어드는가?**

$$
\mathbb{E}[\nabla \log \pi(a|s) b(s)] = b(s) \mathbb{E}[\nabla \log \pi(a|s)]
$$

$\mathbb{E}[\nabla \log \pi(a|s)] = \mathbb{E}[\nabla \pi(a|s) / \pi(a|s)] = 0$ (정규화 상수 미분).

따라서 baseline 은 기대값에 영향 없음, 오직 분산만 감소시킵니다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — State-Action Value Function

정책 $\pi$ 하에서 상태 $s$ 에서 행동 $a$ 를 취했을 때의 기대 누적 보상:
$$
Q^\pi(s, a) := \mathbb{E}_{s_0 = s, a_0 = a, \text{follow } \pi}[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)]
$$

### 정의 3.2 — State Value Function

정책 $\pi$ 하에서 상태 $s$ 에서의 기대 누적 보상:
$$
V^\pi(s) := \mathbb{E}_{a \sim \pi(\cdot|s), s' \sim P}[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s] = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s, a)]
$$

즉, 현재 상태에서 최적 행동을 미리 모르고, 정책 $\pi$ 에 따라 행동할 때의 기대값.

### 정의 3.3 — Advantage Function

$$
\boxed{A^\pi(s, a) := Q^\pi(s, a) - V^\pi(s)}
$$

**의미**: 상태 $s$ 에서 행동 $a$ 를 취하는 것이 정책 $\pi$ 의 평균 ($V^\pi$) 대비 얼마나 좋은가.

### 정의 3.4 — Temporal Difference (TD) 기반 Advantage

실전에서는 exact $Q, V$ 를 모르므로, Bellman equation 을 이용:

$$
A^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s, a)}[V^\pi(s')] - V^\pi(s)
$$

한 step 만 사용하면:
$$
A^\pi(s, a) \approx r(s, a) + \gamma V^\pi(s') - V^\pi(s) := \delta^\pi(s, a)
$$

이를 **temporal difference (TD) error** 라 합니다.

### 정의 3.5 — Baseline

일반적인 baseline function $b: \mathcal{S} \to \mathbb{R}$ 에 대해:
$$
A_b^\pi(s, a) := Q^\pi(s, a) - b(s)
$$

가장 흔한 선택: $b(s) = V^\pi(s)$.

---

## 🔬 정리와 증명

### 정리 3.1 — Advantage 의 합은 0 (On-Policy)

정책 $\pi$ 에 대해, 상태 $s$ 에서:
$$
\sum_a \pi(a|s) A^\pi(s, a) = 0
$$

**증명**:
$$
\sum_a \pi(a|s) A^\pi(s, a) = \sum_a \pi(a|s) [Q^\pi(s, a) - V^\pi(s)]
$$

$$
= \sum_a \pi(a|s) Q^\pi(s, a) - V^\pi(s) \sum_a \pi(a|s)
$$

$$
= \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0 \quad \square
$$

**의미**: On-policy 에서 advantage 의 평균은 항상 0. 따라서 advantage 는 상대적 척도입니다.

### 정리 3.2 — Baseline Subtraction 이 Gradient Expectation 을 보존

Policy gradient 에서:
$$
\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)] = \mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) (Q^\pi(s, a) - b(s))]
$$

**증명**:

오른쪽에서 baseline 항:
$$
\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) b(s)] = b(s) \sum_a \nabla_\theta \pi_\theta(a|s)
$$

$$
= b(s) \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \nabla_\theta 1 = 0
$$

따라서:
$$
\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) b(s)] = 0
$$

공제하면:
$$
\mathbb{E}[\nabla \log \pi \cdot Q] = \mathbb{E}[\nabla \log \pi \cdot (Q - b)]
$$

양쪽 기대값이 같습니다. $\square$

### 정리 3.3 — 최적 Baseline 은 $V^\pi$ (분산 관점)

$g(s) := \mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)]$ 라 하자.

분산을 최소화하는 baseline $b^*(s)$ 는:
$$
\boxed{b^*(s) = \frac{\mathbb{E}_a[(\nabla_\theta \log \pi)^2 Q^\pi]}{\mathbb{E}_a[(\nabla_\theta \log \pi)^2]}}
$$

**특수한 경우**: Gradient 가 $\log \pi$ 에만 의존하고 $a$ 에 대해 independent 하면:
$$
b^*(s) = \mathbb{E}_a[Q^\pi(s, a)] = V^\pi(s)
$$

**실전**: $V^\pi$ 가 종종 reasonable baseline.

---

## 💻 NumPy 구현 검증

### 실험 1 — Advantage 의 성질 확인

```python
import numpy as np
import matplotlib.pyplot as plt

# 5-state, 2-action MDP
S, A = 5, 2
gamma = 0.9
np.random.seed(42)

P = np.random.dirichlet(np.ones(S), size=(S, A))
r = np.random.randn(S, A)

def policy_eval(pi, P, r, gamma, n_iter=1000):
    """V^π, Q^π 계산."""
    V = np.zeros(S)
    for _ in range(n_iter):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(-1)
        if np.linalg.norm(V - V_new) < 1e-12:
            break
        V = V_new
    Q = r + gamma * np.einsum('sap,p->sa', P, V)
    return V, Q

# 정책
pi = np.random.dirichlet(np.ones(A), size=S)
V, Q = policy_eval(pi, P, r, gamma)
A = Q - V[:, None]

# Theorem 3.1: Σ_a π(a|s) A^π(s,a) = 0
on_policy_adv = (pi * A).sum(-1)
print("On-policy advantage (should be ~0):")
print(f"  Max: {np.abs(on_policy_adv).max():.2e}")
print(f"  Mean: {np.mean(on_policy_adv):.2e}  ✓")

# Positive vs negative advantage
pos_adv = (A > 0).sum()
neg_adv = (A < 0).sum()
print(f"\nAdvantage statistics:")
print(f"  Positive: {pos_adv} / {S*A} ({100*pos_adv/(S*A):.1f}%)")
print(f"  Negative: {neg_adv} / {S*A} ({100*neg_adv/(S*A):.1f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# State별 advantage 분포
for s in range(S):
    axes[0].bar(np.arange(A) + s*A/5, A[s, :], alpha=0.7, label=f's={s}')
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_xlabel('State-Action Pair')
axes[0].set_ylabel('$A^\\pi(s, a)$')
axes[0].set_title('Advantage Values')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].legend(loc='best', fontsize=8)

# Q vs V vs A
for s in range(min(3, S)):
    axes[1].plot([0, 1], [V[s], V[s]], 'ko-', markersize=8, label=f'$V^\\pi(s={s})$', linewidth=2)
    axes[1].scatter([0.5], [Q[s, 0]], color='red', s=100, marker='^', label=f'$Q^\\pi(s={s}, a=0)$', zorder=5)
    axes[1].scatter([0.5], [Q[s, 1]], color='blue', s=100, marker='s', label=f'$Q^\\pi(s={s}, a=1)$', zorder=5)

axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
axes[1].set_ylabel('Value')
axes[1].set_title('Q vs V for Few States')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advantage_properties.png', dpi=150, bbox_inches='tight')
print("✓ Saved advantage_properties.png")
```

### 실험 2 — Baseline Subtraction 의 Variance 감소

```python
# Policy gradient estimation
n_samples = 10000
np.random.seed(42)

# Random state-action samples
s_samples = np.random.randint(0, S, n_samples)
a_samples = np.array([np.random.choice(A, p=pi[s]) for s in s_samples])

# Gradient without baseline (using Q)
grad_no_baseline = (Q[s_samples, a_samples] * np.random.randn(n_samples))
var_no_baseline = np.var(grad_no_baseline)

# Gradient with baseline (using A = Q - V)
A_samples = A[s_samples, a_samples]
grad_with_baseline = (A_samples * np.random.randn(n_samples))
var_with_baseline = np.var(grad_with_baseline)

reduction = (1 - var_with_baseline / var_no_baseline) * 100

print(f"Variance without baseline (using Q): {var_no_baseline:.6f}")
print(f"Variance with baseline (using A):    {var_with_baseline:.6f}")
print(f"Variance reduction:                  {reduction:.1f}%")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(grad_no_baseline, bins=50, alpha=0.6, label='$Q^\\pi$ (no baseline)', density=True)
axes[0].hist(grad_with_baseline, bins=50, alpha=0.6, label='$A^\\pi$ (with baseline)', density=True)
axes[0].set_xlabel('Gradient Magnitude')
axes[0].set_ylabel('Density')
axes[0].set_title('Gradient Distribution Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Q vs A magnitude
axes[1].scatter(Q[s_samples, a_samples], A_samples, alpha=0.3, s=10)
axes[1].set_xlabel('$Q^\\pi(s, a)$')
axes[1].set_ylabel('$A^\\pi(s, a)$')
axes[1].set_title('Q-value vs Advantage')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_variance_reduction.png', dpi=150, bbox_inches='tight')
print("✓ Saved baseline_variance_reduction.png")
```

### 실험 3 — TD Error as Advantage Estimator

```python
# 한 번의 trajectory rollout
rho0 = np.eye(S)[0]
state = np.random.choice(S, p=rho0)
trajectory = []

for t in range(20):
    action = np.random.choice(A, p=pi[state])
    reward = r[state, action]
    next_state = np.random.choice(S, p=P[state, action])
    trajectory.append((state, action, reward, next_state))
    state = next_state

# Exact A vs TD error
print("Advantage vs TD Error:")
print("State | Action | Exact A^π | TD Error δ | Difference")
print("-" * 55)

for s, a, rew, s_next in trajectory[:5]:
    A_exact = A[s, a]
    td_error = rew + gamma * V[s_next] - V[s]
    diff = abs(A_exact - td_error)
    print(f"{s:5d} | {a:6d} | {A_exact:9.4f} | {td_error:10.4f} | {diff:10.4f}")

print("\nTD error 는 한 step 만 사용하는 저분산 추정치입니다.")
```

### 실험 4 — Greedy Action Selection via Advantage

```python
# Advantage 가 큰 행동과 policy probability 의 관계
fig, axes = plt.subplots(S, 1, figsize=(10, 8), sharex=True)

for s in range(S):
    x = np.arange(A)
    width = 0.35
    
    axes[s].bar(x - width/2, pi[s], width, label='$\\pi(a|s)$', alpha=0.7)
    axes[s].bar(x + width/2, (A[s] - A[s].min()) / (A[s].max() - A[s].min() + 1e-8), 
                width, label='Normalized $A^\\pi$', alpha=0.7)
    
    best_a = np.argmax(A[s])
    axes[s].axvline(best_a, color='red', linestyle='--', alpha=0.5, label='Greedy action')
    
    axes[s].set_ylabel(f'State {s}')
    axes[s].set_ylim(-0.1, 1.1)
    if s == 0:
        axes[s].legend(loc='best')
    axes[s].grid(True, alpha=0.3, axis='y')

axes[-1].set_xlabel('Action')
fig.suptitle('Policy vs Advantage (Greedy Action Selection)', fontsize=12)
plt.tight_layout()
plt.savefig('greedy_advantage.png', dpi=150, bbox_inches='tight')
print("✓ Saved greedy_advantage.png")
```

---

## 🔗 후속 레포와의 연결

### 1. Policy Gradient Deep Dive

Policy Gradient Theorem:
$$
\nabla J(\pi) = \mathbb{E}_{s \sim d^\pi, a \sim \pi}[\nabla \log \pi(a|s) A^\pi(s, a)]
$$

이 식에서 $A^\pi$ 가 왜 등장하는가 — baseline subtraction 의 자연스러운 결과.

### 2. Actor-Critic Methods

**Actor** (정책): $\pi(a|s)$ 를 gradient 로 업데이트, advantage 기반.

**Critic** (가치): $V(s)$ 또는 $Q(s, a)$ 를 학습, advantage 계산.

둘의 interplay.

### 3. Generalized Advantage Estimation (GAE — Schulman 2015)

장시간 TD 오류들의 exponential-weighted sum:
$$
\hat{A}^\lambda(s, a) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta^\pi_{t+l}
$$

baseline subtraction 의 일반화, variance-bias tradeoff.

### 4. Asynchronous Advantage Actor-Critic (A3C — Mnih et al. 2016)

병렬화된 advantage 기반 policy gradient.

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 대응 |
|------|------|------|
| Exact $Q^\pi, V^\pi$ | 근사 오차 무시 | 실전: function approximation (Ch7+) |
| Stationary policy | 시간 불변 | Non-stationary 은 추가 term |
| Baseline 선택 | $V^\pi$ 가 최적 | 다른 baseline 도 가능 (e.g., constant) |
| On-policy sampling | 정책 따라 샘플 | Off-policy 는 importance weighting 필요 |

---

## 📌 핵심 정리

$$\boxed{A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s), \quad \sum_a \pi(a|s) A^\pi(s, a) = 0}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| $Q^\pi(s, a)$ | 상태-행동 가치 | 절대적 가치 |
| $V^\pi(s)$ | 상태 가치 | 평균 기준선 |
| $A^\pi(s, a)$ | $Q - V$ | 상대적 이득 |
| Baseline $b(s)$ | $A_b = Q - b$ | variance 감소 |
| TD error $\delta$ | $r + \gamma V(s') - V(s)$ | 저분산 근사 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\sum_a \pi(a|s) A^\pi(s, a) = 0$ 이면, 모든 행동이 동등하게 좋은가? 아니면 상대적 순위는 어떻게 유지되는가?

<details>
<summary>해설</summary>

합이 0이라 해서 모든 행동이 같은 것은 아닙니다. 예를 들어:
- $A^\pi(s, a_1) = +2, A^\pi(s, a_2) = +1, A^\pi(s, a_3) = -3$
- 합: $2 + 1 - 3 = 0$
- 하지만 $a_1 > a_2 > a_3$ 의 상대적 순위는 명확

상대적 순위 (ranking) 는 보존되고, 평균을 0 으로 anchor 했을 뿐입니다.

**의미**: 정책 선택 시 항상 가장 좋은 행동 (highest $A$) 에 확률 높음.

</details>

**문제 2** (심화): Theorem 3.3 에서 최적 baseline 이 $V^\pi$ 인 증명을 완성하라. 왜 $V^\pi$ 가 특별한가?

<details>
<summary>해설</summary>

**최적 baseline 도출** (평균제곱오류 최소화):

$$
\min_b \mathbb{E}[(\nabla_\theta \log \pi (Q - b))^2]
$$

$$
= \min_b \mathbb{E}[(\nabla_\theta \log \pi)^2 (Q - b)^2]
$$

$b$ 에 대해 미분, 0으로 두면:

$$
2 \mathbb{E}[(\nabla_\theta \log \pi)^2 (Q - b)] (-1) = 0
$$

$$
b^* = \frac{\mathbb{E}[(\nabla_\theta \log \pi)^2 Q]}{\mathbb{E}[(\nabla_\theta \log \pi)^2]}
$$

특수한 경우 (gradient 가 $a$ 에 independent):

$$
b^* = \mathbb{E}_a[Q^\pi(s, a)] = V^\pi(s)
$$

**왜 특별한가**: $V^\pi$ 는 정책의 현재 상태에서의 기대값이므로, 자연스러운 reference point. 행동별 기대값에서 이를 빼면 상대적 이득 (advantage) $\square$.

</details>

**문제 3** (논문 연결): Schulman 2015 의 Generalized Advantage Estimation (GAE) 는:
$$
\hat{A}_t^\lambda = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^\pi
$$
를 제안했다. 이것이 어떻게 variance-bias tradeoff 를 조절하는가?

<details>
<summary>해설</summary>

**$\lambda = 1$ (무한 look-ahead)**:
$$
\hat{A}^\lambda = \delta_0 + \gamma \delta_1 + \gamma^2 \delta_2 + \cdots
$$

telescoping 으로:
$$
= (r_0 + \gamma V(s_1) - V(s_0)) + \gamma(r_1 + \gamma V(s_2) - V(s_1)) + \cdots
$$

모두 더하면 (V 항 소거):
$$
= r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots - V(s_0) = G_0 - V(s_0) = A^{\text{MC}}(s_0, a_0)
$$

즉, 정확한 advantage (하지만 high variance, full trajectory 필요).

**$\lambda = 0$ (one-step)**:
$$
\hat{A}^0 = \delta_0 = r_0 + \gamma V(s_1) - V(s_0)
$$

TD error (low variance, 하지만 함수 근사 오차).

**중간 값 $\lambda \in (0,1)$**:

$\lambda$ 가 작을수록 immediate TD 에 의존 (low variance).
$\lambda$ 가 클수록 longer-horizon advantage (high variance).

$\lambda = 0.95$ 등 실증적 선택으로 tradeoff 조절 $\square$.

</details>

---

<div align="center">

[◀ 이전: 02. Performance Difference Lemma](./02-performance-difference-lemma.md) | [📚 README](../README.md) | [다음 ▶: 04. MDP 근사 — Approximation Error 와 Sample Complexity](./04-approximation-sample-complexity.md)

</div>
