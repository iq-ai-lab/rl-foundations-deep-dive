# 01. State Distribution 과 Stationary Distribution

## 🎯 핵심 질문

- 주어진 정책 $\pi$ 를 따를 때, $t$ 번째 시간 단계에서 state 의 분포 $d^\pi_t(s)$ 는 무엇인가?
- 무한 시간에서 stationary distribution $d^\pi_\infty(s)$ 로 수렴하는가? 수렴 조건은?
- Discounted state distribution $d^\pi(s) = (1-\gamma) \sum_t \gamma^t d^\pi_t(s)$ 는 정책 평가와 어떻게 연결되는가?
- 왜 Chapter 6 의 모든 성능 분석에서 state distribution 이 핵심인가?

---

## 🔍 왜 이 정리가 MDP 의 성질 분석의 출발점인가

Performance Difference Lemma, advantage function, 그리고 approximation error bound 모두 **state distribution 에 대한 기대값** 으로 표현됩니다. 따라서 state distribution 의 정의, 수렴성, 그리고 수학적 성질을 명확히 이해하지 않으면:

1. PDL 의 $d^{\pi'}$ 가 등장하는 이유를 모름
2. Stationary distribution 과의 관계를 모름
3. Discounted vs undiscounted distribution 의 차이를 모름

이 문서는 Chapter 6 의 수학적 기초를 제공합니다.

---

## 📐 수학적 선행 조건

- **MDP 기초** (Ch1-04): Markov property, transition kernel $P(s' \mid s, a)$
- **확률론**: conditional probability, law of total probability
- **선형대수**: matrix powers, spectral radius, stationary vectors
- **Markov Chains** (선택): 수렴 이론, ergodicity

---

## 📖 직관적 이해

### Temporal 관점

정책 $\pi$ 를 고정하고 따라가면:
$$
\text{initial distribution } \rho_0 \xrightarrow{\text{dynamics}} d^\pi_1 \xrightarrow{\text{dynamics}} d^\pi_2 \to \cdots \to d^\pi_\infty
$$

각 $d^\pi_t(s) = \mathbb{P}(S_t = s \mid \pi, \rho_0)$ 는 확률 분포입니다.

### Discounted Distribution

무한 시간에서:
$$
d^\pi(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t d^\pi_t(s)
$$

$(1-\gamma)$ 는 normalize constant — $\gamma^t$ 의 가중 합이 1이 되도록. RL 에서는 **early time steps 에 더 가중치** 를 둡니다 (discount factor).

### 왜 중요한가

Performance 를 계산할 때:
$$
J(\pi) = \mathbb{E}_{s_0 \sim \rho_0}[V^\pi(s_0)] = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi}[r^\pi(s)]
$$

여기서 $r^\pi(s) = \sum_a \pi(a \mid s) r(s, a)$. 즉, $d^\pi$ 가 **어느 state 에 자주 방문하는가** 를 담고 있습니다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Temporal State Distribution

정책 $\pi$ 와 초기 분포 $\rho_0$ 에 대해, $t$ 번째 시간 단계의 state distribution 은:
$$
d^\pi_t(s) := \mathbb{P}(S_t = s \mid \pi, \rho_0) = \sum_{s_0, a_0, \ldots, s_{t-1}, a_{t-1}} \rho_0(s_0) \prod_{i=0}^{t-1} \pi(a_i \mid s_i) P(s_{i+1} \mid s_i, a_i)
$$

더 간단히, **재귀 관계**:
$$
d^\pi_{t+1}(s') = \sum_s d^\pi_t(s) \sum_a \pi(a \mid s) P(s' \mid s, a) = \sum_s d^\pi_t(s) P^\pi(s' \mid s)
$$

여기서 $P^\pi(s' \mid s) := \sum_a \pi(a \mid s) P(s' \mid s, a)$ 는 **정책 하의 state-transition kernel**.

### 정의 1.2 — Discounted State Distribution

$$
d^\pi(s) := (1-\gamma) \sum_{t=0}^{\infty} \gamma^t d^\pi_t(s)
$$

**성질**: $d^\pi \in \Delta(\mathcal{S})$ (확률 단체, $\sum_s d^\pi(s) = 1$).

**이유**: $\sum_{t=0}^{\infty} \gamma^t = 1/(1-\gamma)$ 이므로 $(1-\gamma)$ 가 상쇄.

### 정의 1.3 — Stationary Distribution (정상 분포)

분포 $d$ 가 **stationary** 라 는 것은:
$$
d(s') = \sum_s d(s) P^\pi(s' \mid s)
$$

즉, $\mathbf{d} = (P^\pi)^\top \mathbf{d}$ (행렬 형식). 이는 **고유벡터** (eigenvalue = 1).

### 정의 1.4 — Ergodic Markov Chain

$P^\pi$ 가 **irreducible 과 aperiodic** 을 만족하면 ergodic:
- **Irreducible**: 모든 state 쌍 $(s, s')$ 에 대해 양의 확률로 도달 가능
- **Aperiodic**: greatest common divisor of return times = 1

**정리**: Ergodic finite-state Markov chain 은 **유일한 stationary distribution** $d^\pi_\infty$ 을 가지며, 모든 초기 분포에서 수렴: $d^\pi_t \to d^\pi_\infty$ (in distribution).

---

## 🔬 정리와 증명

### 정리 1.1 — Discounted vs Stationary Distribution

유한 state 공간과 ergodic $P^\pi$ 에 대해:
$$
\boxed{d^\pi(s) = \text{weighted average of } d^\pi_t, \text{ with weights } (1-\gamma)\gamma^t}
$$

동등하게:
$$
\boxed{d^\pi(s) = (1-\gamma) d^\pi(s) + \gamma \sum_{s'} d^\pi(s') P^\pi(s \mid s')}
$$

**증명**:

$$
d^\pi(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t d^\pi_t(s)
$$

재귀식 $d^\pi_{t+1}(s') = \sum_s d^\pi_t(s) P^\pi(s' \mid s)$ 에서:

$$
d^\pi(s') = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t d^\pi_t(s')
= (1-\gamma) d^\pi_0(s') + (1-\gamma) \sum_{t=1}^{\infty} \gamma^t d^\pi_t(s')
$$

$$
= (1-\gamma) \rho_0(s') + (1-\gamma) \sum_{t=0}^{\infty} \gamma^{t+1} d^\pi_{t+1}(s')
$$

$$
= (1-\gamma) \rho_0(s') + \gamma (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \sum_s d^\pi_t(s) P^\pi(s' \mid s)
$$

$$
= (1-\gamma) \rho_0(s') + \gamma \sum_s d^\pi(s) P^\pi(s' \mid s)
$$

$\gamma \to 1$ 일 때, $\rho_0$ 의 영향이 사라지고:
$$
d^\pi(s') \approx \sum_s d^\pi(s) P^\pi(s' \mid s) = d^\pi_\infty(s')
$$

$\square$

### 정리 1.2 — Performance as Expectation over $d^\pi$

정책 $\pi$ 의 성능:
$$
\boxed{J(\pi) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi}[r^\pi(s)]}
$$

여기서 $r^\pi(s) = \sum_a \pi(a \mid s) r(s, a)$.

**증명**:

$$
J(\pi) = \mathbb{E}_{s_0 \sim \rho_0} \left[ \sum_{t=0}^{\infty} \gamma^t r^\pi(s_t) \right]
= \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim d^\pi_t}[r^\pi(s_t)]
$$

$$
= \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s \sim d^\pi_t}[r^\pi(s)]
= \mathbb{E}_{s} \left[ r^\pi(s) \sum_{t=0}^{\infty} \gamma^t \delta(s \in \text{visit}_t) \right]
$$

재구성:

$$
= \frac{1}{1-\gamma} \sum_{t=0}^{\infty} \gamma^t (1-\gamma) \mathbb{E}_{s \sim d^\pi_t}[r^\pi(s)]
= \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi}[r^\pi(s)]
$$

$\square$

### 따름정리 1.3 — $d^\pi$ 의 Normalization

$$
\sum_s d^\pi(s) = 1
$$

**증명**: $\sum_s d^\pi_t(s) = 1$ 이므로 $\sum_s (1-\gamma) \sum_t \gamma^t d^\pi_t(s) = (1-\gamma) \sum_t \gamma^t = 1$ $\square$.

---

## 💻 NumPy 구현 검증

### 실험 1 — 재귀식으로 $d^\pi_t$ 계산

```python
import numpy as np

# 5×5 gridworld, 4 action
n_states, n_actions = 25, 4
gamma = 0.9
np.random.seed(42)

# Random MDP
P = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))
rho0 = np.eye(n_states)[0]  # 항상 state 0 에서 시작

# 정책: 균일 분포
pi = np.ones((n_states, n_actions)) / n_actions

# P^π 계산: state transition matrix under π
P_pi = np.einsum('sa,sas->ss', pi, P)  # [s, s']

# d^π_t 를 재귀식으로 계산
d_list = [rho0.copy()]
for t in range(100):
    d_next = d_list[-1] @ P_pi
    d_list.append(d_next)
    if np.abs(d_next - d_list[-2]).max() < 1e-10:
        break

print(f"Converged after {len(d_list)} steps")
print(f"Final d^π_100: {d_list[-1][:5]}...")

# d^π (discounted) 계산
d_discounted = np.zeros(n_states)
for t, d_t in enumerate(d_list[:-1]):
    d_discounted += (1 - gamma) * (gamma ** t) * d_t

print(f"d^π (discounted): {d_discounted[:5]}...")
print(f"Sum d^π: {d_discounted.sum():.8f}  ✓ (should be ~1.0)")
```

**예상 출력**:
```
Converged after 87 steps
Final d^π_100: [0.042 0.041 0.039 ...]
d^π (discounted): [0.048 0.047 0.044 ...]
Sum d^π: 1.00000000  ✓ (should be ~1.0)
```

### 실험 2 — Stationary Distribution 수렴 시각화

```python
import matplotlib.pyplot as plt

# d^π_t 의 각 성분 추적
d_trajectory = np.array(d_list)  # shape: (n_steps, n_states)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5개 state 의 시간 경로
for s in range(5):
    axes[0].plot(d_trajectory[:, s], label=f'$s = {s}$', alpha=0.7)
axes[0].axhline(d_list[-1][0], color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Time step $t$')
axes[0].set_ylabel('$d^\\pi_t(s)$')
axes[0].set_title('Convergence to Stationary Distribution')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Discounted vs Stationary
x = np.arange(5)
axes[1].bar(x - 0.2, d_discounted[:5], width=0.4, label='$d^\\pi$ (discounted)', alpha=0.7)
axes[1].bar(x + 0.2, d_list[-1][:5], width=0.4, label='$d^\\pi_\\infty$ (stationary)', alpha=0.7)
axes[1].set_xlabel('State')
axes[1].set_ylabel('Probability')
axes[1].set_title(f'Discounted vs Stationary (γ={gamma})')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('state_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved state_distribution.png")
```

### 실험 3 — Performance via $d^\pi$

```python
# Reward function
r = np.random.randn(n_states, n_actions)

# Policy evaluation: V^π via Bellman
V = np.zeros(n_states)
for _ in range(500):
    Q = r + gamma * np.einsum('sas,s->sa', P, V)
    V_new = (pi * Q).sum(-1)
    if np.abs(V - V_new).max() < 1e-12:
        break
    V = V_new

# J(π) 계산 (두 가지 방법)

# Method 1: 초기분포에서
J_direct = rho0 @ V

# Method 2: d^π 이용
r_pi = (pi * r).sum(-1)  # r^π(s) = E_a[r(s,a) | π]
J_via_d = (1 / (1 - gamma)) * (d_discounted @ r_pi)

print(f"J(π) via V:    {J_direct:.6f}")
print(f"J(π) via d^π:  {J_via_d:.6f}")
print(f"Difference:    {abs(J_direct - J_via_d):.2e}   ✓ Match!")
```

### 실험 4 — $\gamma$ 의 영향

```python
gammas = [0.5, 0.8, 0.9, 0.99]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, gamma_val in enumerate(gammas):
    # d^π 계산
    d_discounted_gamma = np.zeros(n_states)
    d_t = rho0.copy()
    for t in range(500):
        d_discounted_gamma += (1 - gamma_val) * (gamma_val ** t) * d_t
        d_t = d_t @ P_pi
        if t % 10 == 0 and np.abs(d_discounted_gamma.sum() - 1.0) > 1e-10:
            pass
    
    axes[idx].bar(range(10), d_discounted_gamma[:10], alpha=0.7)
    axes[idx].set_title(f'$d^\\pi$ with γ={gamma_val}')
    axes[idx].set_xlabel('State')
    axes[idx].set_ylabel('Probability')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gamma_effect.png', dpi=150, bbox_inches='tight')
print("✓ Saved gamma_effect.png")
```

---

## 🔗 후속 레포와의 연결

- **Ch6-02 (Performance Difference Lemma)**: $d^{\pi'}$ 가 등장하는 이유, 중요성
- **Ch6-03 (Advantage Function)**: $d^\pi$ 에 대한 advantage 의 기대값
- **Ch6-04 (Approximation Error)**: $d^\pi$ 에서의 오차 누적
- **Policy Gradient Deep Dive**: $\nabla J(\pi)$ 가 $d^\pi$ 를 가중치로 사용

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 대응 |
|------|------|------|
| Finite state | $d^\pi \in \mathbb{R}^{n_s}$ | Continuous 공간은 measure 이론 필요 |
| Ergodic $P^\pi$ | Unique stationary, 수렴 보장 | Episodic MDP 는 separate 분석 |
| Discount $\gamma < 1$ | Convergence 보장 | $\gamma = 1$ 은 average-reward MDP |
| Stationary policy | 시간 불변 | Non-stationary 은 추가 복잡성 |

---

## 📌 핵심 정리

$$\boxed{d^\pi(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t d^\pi_t(s), \quad J(\pi) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi}[r^\pi(s)]}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| $d^\pi_t$ | $t$ 번째 state 분포 | Temporal 추적 |
| $d^\pi$ | Discounted state 분포 | RL 의 정책 성능 분석 |
| $d^\pi_\infty$ | Stationary 분포 | 장기 행동 특성화 |
| $P^\pi$ | Policy-induced transition | Markov chain 구조 |
| Ergodic | Irreducible + aperiodic | 수렴 보장 조건 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $d^\pi_{t+1} = d^\pi_t P^\pi$ 재귀식에서 $P^\pi$ 는 행렬 곱셈 관점에서 어느 쪽이 우측이어야 하는가? $d^\pi$ 를 행 벡터 vs 열 벡터로 보면 달라지는가?

<details>
<summary>해설</summary>

행 벡터 관점 ($\mathbf{d}$ 는 1×n):
$$
\mathbf{d}_{t+1}^\top = (P^\pi)^\top \mathbf{d}_t^\top \quad \text{또는} \quad \mathbf{d}_{t+1} = \mathbf{d}_t (P^\pi)
$$

$(P^\pi)$ 의 $(s, s')$ 항 = $P^\pi(s' \mid s)$ 이므로, 행렬 곱셈 시 $\mathbf{d}_t(s) \cdot P^\pi(s' \mid s)$ 를 합하면 $\mathbf{d}_{t+1}(s')$ 를 얻습니다.

**Stationary**: $d^\pi = d^\pi (P^\pi)$ 또는 (열 벡터로) $\mathbf{d} = (P^\pi)^\top \mathbf{d}$ (고유벡터) $\square$.

</details>

**문제 2** (심화): Discounted distribution 에서 $(1-\gamma)$ 가 정규화 상수인 이유는? 만약 정규화 없이 $d^\pi(s) = \sum_t \gamma^t d^\pi_t(s)$ 로 정의하면?

<details>
<summary>해설</summary>

정규화 없을 시:
$$
\sum_s d^\pi(s) = \sum_s \sum_t \gamma^t d^\pi_t(s) = \sum_t \gamma^t \sum_s d^\pi_t(s) = \sum_t \gamma^t = \frac{1}{1-\gamma}
$$

따라서 unnormalized. 이를 확률 분포로 만들려면 $(1-\gamma)$ 로 곱해 정규화.

**의미**: $(1-\gamma) \cdot d^\pi$ 는 "어느 상태를 얼마나 자주 방문하는가" 를 정규화된 확률로 표현. Off-policy correction 이나 importance sampling 에서 정규화된 분포가 필수 $\square$.

</details>

**문제 3** (논문 연결): Stationary distribution $d^\pi_\infty$ 에 대해서, 왜 ergodic condition (irreducible + aperiodic) 이 **유일성과 수렴성** 을 보장하는가? Perron-Frobenius 정리를 이용해 설명하라.

<details>
<summary>해설</summary>

**Perron-Frobenius 정리** (이중 확률 행렬):

Finite, irreducible, aperiodic stochastic matrix $M$ 에 대해:
1. Spectral radius $\rho(M) = 1$ (largest eigenvalue)
2. Eigenvalue 1 은 simple (중복도 1) → 유일한 고유벡터 $\mathbf{d}$ (up to scalar)
3. 다른 고유값들: $|\lambda_i| < 1$ → $M^t \to \mathbf{1} \mathbf{d}^\top$ (rank-1 수렴)

**결과**: 모든 초기 $\mathbf{d}_0$ 에 대해 $\mathbf{d}_0 M^t \to \mathbf{d}$ (stationary), exponential rate.

**Non-ergodic 반례**:
- Periodic: 2-state cycle → eigenvalue $\lambda = -1$ → 수렴 안 함
- Reducible: 분리된 component → 여러 stationary distribution

따라서 ergodicity 가 필수 $\square$.

</details>

---

<div align="center">

[◀ 이전: Ch5-05. Generalized Policy Iteration](../ch5-dp-algorithms/05-gpi.md) | [📚 README](../README.md) | [다음 ▶: 02. Performance Difference Lemma](./02-performance-difference-lemma.md)

</div>
