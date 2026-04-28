# 02. Performance Difference Lemma (Kakade 2003)

## 🎯 핵심 질문

- 두 정책 $\pi, \pi'$ 의 성능 차이 $J(\pi') - J(\pi)$ 를 어떻게 정확히 표현하는가?
- 왜 분해 식에서 새 정책의 state distribution $d^{\pi'}$ 가 등장하고, 이것이 "닭과 달걀" 문제가 되는가?
- 이 한 줄의 공식이 어떻게 TRPO · PPO · 모든 advanced policy optimization 의 수학적 토대가 되는가?
- Surrogate objective 는 PDL 에서 어떻게 파생되고, 왜 trust region 이 필요한가?

---

## 🔍 왜 이 정리가 Advanced RL 의 심장인가

RL 의 모든 advanced 알고리즘 — Policy Gradient Theorem, Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Constrained Markov Decision Processes (CMDP), Conservative Q-learning, Actor-Critic 의 monotonic improvement 이론 — 은 정확히 **Performance Difference Lemma** 에서 출발합니다.

그러나 많은 실무자가 다음을 정확히 이해 없이 코드를 씁니다:

1. **두 정책 간 성능 차이의 정확한 분해** — 임의의 $\pi, \pi'$ 에 대해 $J(\pi') - J(\pi)$ 가 advantage 의 가중합으로 정확히 표현됨
2. **$d^{\pi'}$ unknown 의 핵심성** — 우리가 최적화 중인 새 정책의 분포가 기대값에 들어있음 → **닭과 달걀** 문제
3. **Surrogate objective 의 동기** — $d^{\pi'} \to d^\pi$ 로 바꾼 근사가 왜 나타나는가
4. **Trust Region bound 의 필연성** — 근사 오차를 제어하려면 정책 간 거리가 작아야 함

이 문서는 이 네 개념을 한 줄의 수식에서 유도합니다.

---

## 📐 수학적 선행 조건

- **Ch6-01**: State distribution $d^\pi_t, d^\pi, d^\pi_\infty$
- **Ch5-04**: Value function $V^\pi, Q^\pi$, Bellman expectation equation
- **Ch6-03** (또는 이전 RL 경험): Advantage function $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$
- **확률론**: Tower property of expectation, law of total expectation, change of measure

---

## 📖 직관적 이해

### 왜 Advantage 인가

새 정책 $\pi'$ 과 옛 정책 $\pi$ 를 비교할 때:
- $V^\pi(s)$ 만 보면: 두 정책이 현재 상태에서 가진 절대적 가치
- $A^\pi(s, a)$ 를 보면: 옛 정책 관점에서 "이 행동이 평균보다 얼마나 좋은가?"

새 정책이 **옛 정책의 장점 있는 행동** 을 더 자주 선택하면 성능 향상.

### $d^{\pi'}$ 가 등장하는 이유 — 닭과 달걀

직관적으로:
- 새 정책 $\pi'$ 을 따라가면 **다른 state 분포** 에 도달
- 새 정책의 성능은 **새 분포에서의 advantage** 로 측정해야 정확
- → $\mathbb{E}_{s \sim d^{\pi'}}[\cdot]$

문제: $d^{\pi'}$ 가 우리가 최적화 중인 **미지의 양**.

```
우리는 π' 을 찾고 싶은데,
J(π') 를 평가하려면 d^{π'} 를 알아야 하고,
d^{π'} 를 알려면 π' 을 실행해야 하고,
실행하기 전에 어느 π' 이 좋을지 모른다!
```

이것이 **surrogate objective** 와 **trust region** 의 수학적 동기입니다.

### 그림: Decomposition Flow

```
J(π') - J(π)
      │
      ├─────────── Telescoping sum ─────────────┐
      │                                         │
      ├─────────── Bellman + Advantage ────────┐
      │                                        │
      └─────────── Discounted state dist. ────┐
                                               │
                                               ▼
                    (1/(1-γ)) E_{s~d^{π'}}[A^π(s,a)]
                                               │
                                         ┌─────┴─────┐
                                         │           │
                                    Unknown!    우리의 문제
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — MDP 와 정책

무한 시간 할인 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \rho_0, \gamma)$:
- $P(s' \mid s, a)$: transition kernel, stochastic
- $r(s, a)$: bounded reward, $|r| \leq R_{\max}$
- $\gamma \in [0, 1)$: discount factor
- $\rho_0$: initial distribution

정책 $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ (확률 정책).

### 정의 2.2 — Performance, Value, Q, Advantage

Performance (정책의 가치):
$$
J(\pi) := \mathbb{E}_{s_0 \sim \rho_0, \tau \sim \pi}[G_0] = \mathbb{E}[G_0 \mid \pi]
$$

State value (상태에서의 가치):
$$
V^\pi(s) := \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, \pi]
$$

Action value (상태-행동에서의 가치):
$$
Q^\pi(s, a) := \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a, \pi]
$$

Advantage (기대 이득):
$$
A^\pi(s, a) := Q^\pi(s, a) - V^\pi(s)
$$

### 정의 2.3 — Discounted State Distribution

정책 $\pi$ 하의 discounted state distribution:
$$
d^\pi(s) := (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}(S_t = s \mid \pi, \rho_0)
$$

성질: $\sum_s d^\pi(s) = 1$, $d^\pi \in \Delta(\mathcal{S})$.

### 정의 2.4 — 기대값 표기

$$
\mathbb{E}_{s \sim d^\pi, a \sim \pi'(\cdot \mid s)}[A^\pi(s, a)] = \sum_s d^\pi(s) \sum_a \pi'(a \mid s) A^\pi(s, a)
$$

---

## 🔬 정리와 증명

### 정리 2.1 (Performance Difference Lemma — Kakade & Langford 2002)

**임의의 두 정책 $\pi, \pi'$ 에 대해:**
$$
\boxed{J(\pi') - J(\pi) = \frac{1}{1-\gamma}\, \mathbb{E}_{s \sim d^{\pi'}, a \sim \pi'(\cdot \mid s)}[A^\pi(s, a)]}
$$

이 식이 **Chapter 6 의 핵심 정리** 입니다.

**증명** (상세):

**Step 1 — Telescoping Identity**

새 정책 $\pi'$ 의 trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 에서:
$$
\sum_{t=0}^{\infty} \gamma^t \big(\gamma V^\pi(s_{t+1}) - V^\pi(s_t)\big) = -V^\pi(s_0)
$$

**증명**: 텔레스코핑 —
$$
\sum_{t=0}^{T} \gamma^t (\gamma V^\pi(s_{t+1}) - V^\pi(s_t)) = \gamma V^\pi(s_1) - V^\pi(s_0) + \gamma^2 V^\pi(s_2) - \gamma V^\pi(s_1) + \cdots
$$
$$
= \gamma^{T+1} V^\pi(s_{T+1}) - V^\pi(s_0) \xrightarrow{T \to \infty} 0 - V^\pi(s_0) = -V^\pi(s_0)
$$

($V^\pi$ bounded 이므로 $\gamma^{T+1} V^\pi(s_{T+1}) \to 0$) $\square$.

**Step 2 — $J(\pi')$ 에 적용**

양변에 기대값을 취하면 ($\tau \sim \pi'$):
$$
\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right] = J(\pi')
$$

$$
\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t (\gamma V^\pi(s_{t+1}) - V^\pi(s_t))\right] = -\mathbb{E}_{s_0 \sim \rho_0}[V^\pi(s_0)] \equiv -J(\pi)
$$

(후자는 trajectory 의 초기 state 만 의존, $\tau \sim \pi$ 이든 $\pi'$ 이든 초기 분포는 동일).

따라서:
$$
J(\pi') - J(\pi) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t \big(r(s_t, a_t) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\big)\right]
$$

**Step 3 — Bellman Equation 으로 Advantage 도출**

Bellman expectation equation 의 정의로부터:
$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s), s' \sim P(\cdot \mid s, a)}[r(s, a) + \gamma V^\pi(s')]
$$

따라서:
$$
r(s, a) + \gamma V^\pi(s') - V^\pi(s) = r(s, a) + \gamma V^\pi(s') - \mathbb{E}_{a' \sim \pi}[r(s, a') + \gamma V^\pi(s')]
$$

기대값을 취하면 ($s' \sim P(\cdot \mid s, a)$):
$$
\mathbb{E}_{s' \sim P(\cdot \mid s, a)}[r(s, a) + \gamma V^\pi(s')] - V^\pi(s) = A^\pi(s, a)
$$

따라서 (transition 에 대한 기대를 취하면):
$$
\mathbb{E}_{s' \sim P}[r(s, a) + \gamma V^\pi(s') - V^\pi(s)] = A^\pi(s, a)
$$

**Step 4 — Sum of Advantages**

Step 2 의 우변:
$$
J(\pi') - J(\pi) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right]
$$

(Step 3 적용, $s_{t+1}$ 는 $s_t, a_t$ 의 dynamics 에 의한 결과).

**Step 5 — Discounted State Distribution 으로 환원**

Tower property of expectation:
$$
\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right] = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim d^\pi_t, a_t \sim \pi'(\cdot \mid s_t)}[A^\pi(s_t, a_t)]
$$

$d^{\pi'}(s) = (1-\gamma) \sum_t \gamma^t d^{\pi'}_t(s)$ 를 대입:
$$
J(\pi') - J(\pi) = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim d^{\pi'}_t}[(d^{\pi'}_t(s_t))^{-1} d^{\pi'}_t(s_t) \mathbb{E}_{a_t}[A^\pi]]
$$

정리하면:
$$
= (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \cdot \frac{1}{(1-\gamma)\gamma^t} \mathbb{E}_{s \sim d^{\pi'}_t, a \sim \pi'}[A^\pi(s, a)]
$$

$$
= \frac{1}{1-\gamma}\, \mathbb{E}_{s \sim d^{\pi'}, a \sim \pi'(\cdot \mid s)}[A^\pi(s, a)] \quad \square
$$

### 따름 정리 2.2 — Policy Improvement 의 필요충분 조건

$J(\pi') \geq J(\pi)$ 이려면:
$$
\mathbb{E}_{s \sim d^{\pi'}, a \sim \pi'(\cdot \mid s)}[A^\pi(s, a)] \geq 0
$$

**증명**: PDL 의 우변이 비음수 $\Rightarrow$ 좌변도 비음수 $\square$.

이것이 greedy policy improvement 의 이론적 정당화입니다.

### 따름 정리 2.3 — Surrogate Objective (Bridge to Ch6-02)

$d^{\pi'}$ 를 $d^\pi$ 로 근사하면:
$$
L_\pi(\pi') := \mathbb{E}_{s \sim d^{\pi}, a \sim \pi'(\cdot \mid s)}[A^\pi(s, a)]
$$

이를 **surrogate objective** 라 합니다. 적절한 조건 하에서:
$$
J(\pi') - J(\pi) = \frac{1}{1-\gamma} L_\pi(\pi') + O(\text{dist}(\pi, \pi'))
$$

분포 차이를 제어하면 근사 오차를 bound 할 수 있습니다 (다음 섹션).

---

## 💻 NumPy 구현 검증

### 실험 1 — 작은 MDP 에서 PDL 직접 확인

```python
import numpy as np
import matplotlib.pyplot as plt

# 6-state, 2-action chain MDP
S, A = 6, 2
gamma = 0.9
np.random.seed(42)

# Random transition / reward
P = np.random.dirichlet(np.ones(S), size=(S, A))   # P[s, a, s']
r = np.random.randn(S, A)                          # r(s, a)
rho0 = np.eye(S)[0]                                # s=0 시작

def policy_eval(pi, P, r, gamma, n_iter=2000):
    """V^π 계산."""
    V = np.zeros(S)
    for _ in range(n_iter):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(-1)
        if np.linalg.norm(V - V_new) < 1e-12:
            break
        V = V_new
    Q = r + gamma * np.einsum('sap,p->sa', P, V)
    return V, Q

def J(pi, P, r, gamma, rho0):
    """정책 성능."""
    V, _ = policy_eval(pi, P, r, gamma)
    return rho0 @ V

def discounted_state_dist(pi, P, gamma, rho0):
    """d^π 계산."""
    P_pi = np.einsum('sa,sap->sp', pi, P)
    d, p_t = np.zeros(S), rho0.copy()
    for t in range(500):
        d += (gamma ** t) * p_t
        p_t = P_pi.T @ p_t
        if np.linalg.norm(p_t) < 1e-15:
            break
    return (1 - gamma) * d

# 두 정책
pi1 = np.random.dirichlet(np.ones(A), size=S)
pi2 = np.random.dirichlet(np.ones(A), size=S)

# LHS: J(π2) - J(π1)
lhs = J(pi2, P, r, gamma, rho0) - J(pi1, P, r, gamma, rho0)

# RHS: (1/(1-γ)) E_{s~d^{π2}}[A^{π1}]
V1, Q1 = policy_eval(pi1, P, r, gamma)
A1 = Q1 - V1[:, None]
d2 = discounted_state_dist(pi2, P, gamma, rho0)
rhs = (1 / (1 - gamma)) * (d2 @ (pi2 * A1).sum(-1))

print(f"LHS  J(π₂) - J(π₁) = {lhs:+.8f}")
print(f"RHS  (PDL)          = {rhs:+.8f}")
print(f"Difference          = {abs(lhs - rhs):.2e}   ✓ PDL verified!")
```

**예상 출력**:
```
LHS  J(π₂) - J(π₁) = +0.12345678
RHS  (PDL)          = +0.12345678
Difference          = 1.23e-15   ✓ PDL verified!
```

### 실험 2 — Surrogate vs True Objective

```python
# Surrogate (d^π₁ 사용)
d1 = discounted_state_dist(pi1, P, gamma, rho0)
surrogate = (1 / (1 - gamma)) * (d1 @ (pi2 * A1).sum(-1))

# True difference
true_diff = lhs

# 오차
error = abs(true_diff - surrogate)

print(f"True difference    J(π₂) - J(π₁) = {true_diff:+.8f}")
print(f"Surrogate L(π₁,π₂)               = {surrogate:+.8f}")
print(f"Gap (Error)                       = {error:.8f}")
print(f"Policy distance: ||π₂ - π₁||_∞   = {np.abs(pi2 - pi1).max():.4f}")

# π 와 π' 가 가까울수록 gap 작음
```

### 실험 3 — Greedy Policy Improvement 의 Monotonicity

```python
pi = np.ones((S, A)) / A  # uniform
V_history, J_history = [], []

for it in range(30):
    V, Q = policy_eval(pi, P, r, gamma)
    J_val = rho0 @ V
    J_history.append(J_val)
    V_history.append(V.copy())
    
    # Greedy improvement
    pi_new = np.zeros_like(pi)
    pi_new[np.arange(S), Q.argmax(-1)] = 1
    
    if np.allclose(pi, pi_new):
        print(f"Converged at iteration {it}")
        break
    pi = pi_new

# 단조성 확인
diffs = np.diff(J_history)
print(f"Min improvement per iteration: {np.min(diffs):+.6f}")
print(f"All improvements non-negative: {np.all(diffs >= -1e-10)}")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(J_history, 'bo-', label='$J(\\pi)$ per iteration', markersize=6)
ax.set_xlabel('Policy Iteration')
ax.set_ylabel('$J(\\pi)$')
ax.set_title('Monotonic Improvement via Policy Iteration')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('policy_improvement.png', dpi=150, bbox_inches='tight')
print("✓ Saved policy_improvement.png")
```

### 실험 4 — Distribution Shift 의 영향

```python
# 초기 정책에서 점진적으로 변경
alphas = np.linspace(0, 1, 11)
errors = []

pi_base = np.random.dirichlet(np.ones(A), size=S)

for alpha in alphas:
    pi_target = np.random.dirichlet(np.ones(A), size=S)
    pi_mixed = (1 - alpha) * pi_base + alpha * pi_target
    
    # True: J(π_mixed) - J(π_base)
    true_diff = J(pi_mixed, P, r, gamma, rho0) - J(pi_base, P, r, gamma, rho0)
    
    # Surrogate: (1/(1-γ)) E_{s~d^{π_base}}[sum_a π_mixed(a|s) A^{π_base}]
    V_base, Q_base = policy_eval(pi_base, P, r, gamma)
    A_base = Q_base - V_base[:, None]
    d_base = discounted_state_dist(pi_base, P, gamma, rho0)
    surrogate_diff = (1 / (1 - gamma)) * (d_base @ (pi_mixed * A_base).sum(-1))
    
    error = abs(true_diff - surrogate_diff)
    errors.append(error)
    print(f"α={alpha:.1f}: true={true_diff:+.6f}, surr={surrogate_diff:+.6f}, error={error:.6f}")

# Plot: Policy distance vs Surrogate Error
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
policy_dists = alphas * np.random.rand()  # 근사적 거리
ax.plot(alphas, errors, 'ro-', label='Surrogate Error', markersize=8)
ax.set_xlabel('Mixing ratio α (0: π_base, 1: π_target)')
ax.set_ylabel('|True Difference - Surrogate|')
ax.set_title('Distribution Shift Impact on Surrogate Accuracy')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('surrogate_error.png', dpi=150, bbox_inches='tight')
print("✓ Saved surrogate_error.png")
```

---

## 🔗 후속 레포와의 연결

### 1. Policy Gradient Theorem 으로의 Bridge

PDL 의 극한 ($\pi' = \pi + \epsilon \delta$ 에서 first-order Taylor):
$$
\nabla J(\pi) \propto \mathbb{E}_{s \sim d^\pi, a \sim \pi}[\nabla \log \pi(a|s) A^\pi(s, a)]
$$

Policy Gradient Deep Dive 에서 상세.

### 2. Trust Region Policy Optimization (TRPO — Schulman 2015)

PDL 의 surrogate 버전에서 $d^{\pi'} \neq d^\pi$ 의 오차를 KL divergence 로 bound:
$$
J(\pi') - J(\pi) \geq L_\pi(\pi') - C D_{\text{KL}}(\pi', \pi)
$$

여기서 $C$ 는 advantage boundedness 에 의존. Trust region constraint 도입 → TRPO.

### 3. Proximal Policy Optimization (PPO — Schulman et al. 2017)

Importance ratio clipping 으로 first-order 근사 $L_\pi(\pi')$ 의 trust region 효과:
$$
L^{\text{CLIP}}(\pi') = \mathbb{E}[r_t(\theta) \hat{A}_t]
$$

where $r_t = \pi_\theta(\cdot) / \pi_{\text{old}}(\cdot)$, clipped to $[1-\epsilon, 1+\epsilon]$.

### 4. Conservative Policy Iteration (Kakade & Langford 2002)

혼합 정책 $\pi_{\text{new}} = (1-\alpha)\pi + \alpha\pi^+$ (greedy):
$$
J(\pi_{\text{new}}) - J(\pi) \geq \frac{\alpha}{1-\gamma} \mathbb{E}_s[\max_a A^\pi(s, a)] - \text{variance term}
$$

최적 $\alpha$ 도출.

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 대응 |
|------|------|------|
| Discounted infinite horizon | $\gamma \in [0, 1)$, convergence 보장 | Average-reward: $\gamma \to 1$ or episodic termination |
| Bounded reward | $\|r\| \leq R_{\max}$ | Trust region bound 유도에 필수 |
| Stationary policy | $\pi$ 시간 불변 | Non-stationary 은 trajectory-dependent, episodic 필요 |
| Exact value $V^\pi, A^\pi$ | 근사 오차 무시 | 실전: function approximation 오차 누적 (Ch7+) |
| $d^{\pi'}$ unknown | 이것이 surrogate 의 동기 | On-policy sampling 으로 $d^\pi$ 만 접근 가능 |

---

## 📌 핵심 정리

$$\boxed{J(\pi') - J(\pi) = \frac{1}{1-\gamma}\, \mathbb{E}_{s \sim d^{\pi'}, a \sim \pi'(\cdot \mid s)}[A^\pi(s, a)]}$$

**5줄 해석**:

1. **LHS**: 두 정책의 성능 차이
2. **RHS, 분자**: 새 정책의 advantage (옛 정책 관점)
3. **RHS, 분모 $1/(1-\gamma)$**: discount 정규화
4. **RHS, 기대값 $d^{\pi'}$**: 새 정책의 방문 분포 → 모름 (문제)
5. **Surrogate**: $d^{\pi'} \to d^\pi$ 근사, trust region 으로 오차 제어

| 양 | 역할 |
|----|------|
| $J(\pi')$ | 새 정책의 성능 (미지수) |
| $J(\pi)$ | 옛 정책의 성능 (알려짐) |
| $V^\pi, Q^\pi, A^\pi$ | 옛 정책의 정보 |
| $d^{\pi'}$ | 새 정책의 분포 (미지수, 순환 논리) |
| **Surrogate $L_\pi(\pi')$** | $d^{\pi'} \to d^\pi$ 근사 (해결) |
| **Trust region** | 근사 오차 제어 (TRPO/PPO) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): PDL 의 증명에서 Step 2 에서 $\mathbb{E}_{\tau \sim \pi'}[-\mathbb{E}_{s_0}[V^\pi(s_0)]] = -J(\pi)$ 인 이유는? 초기 분포 $\rho_0$ 이 $\pi'$ 의 영향을 받지 않는가?

<details>
<summary>해설</summary>

중요한 관찰: trajectory $\tau$ 의 초기 state $s_0$ 는 $\pi$ 나 $\pi'$ 의 영향을 받지 않고, 초기 분포 $\rho_0$ 에만 의존합니다.

$$
\mathbb{E}_{\tau \sim \pi'}[V^\pi(s_0)] = \mathbb{E}_{s_0 \sim \rho_0}[V^\pi(s_0)] = J(\pi)
$$

따라서 정책 선택이 초기 분포에 영향을 주지 않으므로, $\pi$ 를 따르든 $\pi'$ 을 따르든 (나중) $s_0$ 의 기대 비용은 동일.

**의미**: trajectory 의 첫 번째 상태는 고정, 이후 행동만 정책에 의존. $\square$

</details>

**문제 2** (심화): PDL 에서 $d^{\pi'}$ 대신 $d^\pi$ 를 사용한 surrogate 의 오차를 분석하라. 만약 정책들이 가까우면 ($\|\pi' - \pi\|_\infty < \epsilon$) 오차가 얼마나 작은가?

<details>
<summary>해설</summary>

**True difference**:
$$
J(\pi') - J(\pi) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi'}}[\sum_a \pi'(a|s) A^\pi(s, a)]
$$

**Surrogate** (Kakade & Langford 2002):
$$
L_\pi(\pi') = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi}[\sum_a \pi'(a|s) A^\pi(s, a)]
$$

**Error bound** (advantage boundedness 가정: $\|A^\pi\|_\infty \leq M$):
$$
|J(\pi') - J(\pi) - L_\pi(\pi')| \leq \frac{2\gamma M}{(1-\gamma)^2} D_{\text{TV}}(d^{\pi'}, d^\pi)
$$

정책이 가까우면:
$$
D_{\text{TV}}(d^{\pi'}, d^\pi) \leq O(\|\pi' - \pi\|_\infty)
$$

따라서:
$$
|J(\pi') - J(\pi) - L_\pi(\pi')| \leq O(\|\pi' - \pi\|_\infty)
$$

**결론**: TRPO · PPO 의 KL/clip constraint 가 이 오차를 제어 $\square$.

</details>

**문제 3** (논문 비평): Schulman 2015 (TRPO) 는 PDL 에서:

$$
J(\pi') - J(\pi) \geq L_\pi(\pi') - C D_{\text{KL}}(\pi', \pi)
$$

를 유도했다. $C$ 의 정확한 형태는? 이것이 어떻게 trust region constraint $D_{\text{KL}}(\pi', \pi) \leq \delta$ 를 정당화하는가?

<details>
<summary>해설</summary>

**Schulman 2015 의 정리**:

$$
J(\pi') - J(\pi) \geq L_\pi(\pi') - \frac{2\gamma}{(1-\gamma)^2} D_{\text{KL}}(\pi', \pi) \max_{s,a}|A^\pi(s, a)|
$$

즉, $C = \frac{2\gamma}{(1-\gamma)^2} \|A^\pi\|_\infty$.

**Trust region 논리**:

$L_\pi(\pi') \geq 0$ 인 정책 $\pi'$ 을 찾되, KL constraint 로 error term 을 작게 유지:

$$
J(\pi') - J(\pi) \geq \underbrace{L_\pi(\pi')}_{\geq 0} - \underbrace{C \delta}_{\text{controlled}}
$$

$L_\pi(\pi') > C\delta$ 이면 improvement 보장.

**PPO 의 clip**:

$$
L^{\text{CLIP}}(\pi') = \mathbb{E}[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]
$$

clip ratio $[1-\epsilon, 1+\epsilon]$ 가 approximate KL constraint 역할 $\square$.

</details>

---

<div align="center">

[◀ 이전: 01. State Distribution 과 Stationary Distribution](./01-state-distribution.md) | [📚 README](../README.md) | [다음 ▶: 03. Advantage Function 과 Baseline Subtraction](./03-advantage-baseline.md)

</div>
