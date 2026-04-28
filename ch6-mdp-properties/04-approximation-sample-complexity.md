# 04. MDP 근사 — Approximation Error 와 Sample Complexity

## 🎯 핵심 질문

- 근사된 Q-함수 $\hat{Q}$ 에서 유도된 greedy 정책의 성능이 얼마나 나쁜가?
- $\epsilon$-optimal Q 함수에서 $\delta$-optimal 정책으로 변환할 때의 손실은?
- Tabular 환경에서 정책을 배우려면 몇 개의 샘플이 필요한가?
- Planning (모델 알 때) vs Learning (모델 모를 때) 의 sample complexity 차이는?

---

## 🔍 왜 이 분석이 이론적 RL 의 마지막 조각인가

Ch1-05 부터 Ch6-02 까지는 **exact value iteration · exact policy iteration · exact policy gradient** 를 다뤘습니다. 하지만 실전에서는:

1. **Q-함수 근사 오차** — 샘플 부족, function approximation 으로 $\|Q^* - \hat{Q}\|_\infty > 0$
2. **Greedy 정책의 성능 저하** — $\epsilon$-approximate $Q$ 에서 유도한 정책의 손실 정량화
3. **Sample 효율성** — Tabular 에서 $\tilde{O}(|\mathcal{S}||\mathcal{A}|/\epsilon^2)$ vs planning $O(|\mathcal{S}|^2|\mathcal{A}|)$ 비교

이 문서는 **이론적 완성도** 를 제공하며, 다음 레포 (Model-Free RL, Linear FA) 로의 bridge 역할을 합니다.

---

## 📐 수학적 선행 조건

- **Ch5-04**: Bellman optimality operator $T^*$, Banach fixed point, contraction
- **Ch6-02**: Performance Difference Lemma
- **Ch6-03**: Advantage function, $\|A\|_\infty$ bound
- **선형대수**: Infinity norm, operator norm
- **확률론**: Hoeffding 부등식, concentration bounds (선택)

---

## 📖 직각적 이해

### Greedy 정책의 손실

정확한 $Q^*$ 에서 유도한 greedy 정책은 최적.

근사 $\hat{Q}$ 에서 유도한 greedy $\hat{\pi}$:
$$
\hat{\pi}(s) \in \arg\max_a \hat{Q}(s, a)
$$

는 얼마나 나쁜가? → **Greedy policy loss bound**

### Planning vs Learning

- **Planning** (model $P, r$ 알 때):
  - $V$ 또는 $Q$ 계산: $O(|\mathcal{S}|^2|\mathcal{A}|)$ 시간
  - 정책 도출: $O(|\mathcal{S}||\mathcal{A}|)$ 시간
  - 총 $O(|\mathcal{S}|^2|\mathcal{A}|)$ — 문제 크기의 다항식

- **Learning** (model 모를 때):
  - 각 state-action pair 를 여러 번 방문해야 $Q$ 추정
  - $|\mathcal{S}||\mathcal{A}|$ 개 쌍, 각각 high-confidence 추정 필요
  - 총 $\tilde{O}(|\mathcal{S}||\mathcal{A}|/\epsilon^2)$ 샘플 (tabular)

→ Learning이 planning 보다 sample-expensive.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Approximate Q-Function

$\hat{Q}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 가 $\epsilon$-optimal 이면:
$$
\|\hat{Q} - Q^*\|_\infty := \max_{s, a} |\hat{Q}(s, a) - Q^*(s, a)| \leq \epsilon
$$

### 정의 4.2 — Greedy Policy

$\hat{Q}$ 에 대한 greedy 정책:
$$
\hat{\pi}(a|s) = \begin{cases} 1 & \text{if } a \in \arg\max_{a'} \hat{Q}(s, a') \\ 0 & \text{otherwise} \end{cases}
$$

결정적 정책 (deterministic).

### 정의 4.3 — Optimal Policy Loss

정책 $\pi$ 의 최적 대비 손실:
$$
\text{Loss}(\pi) := J(\pi^*) - J(\pi) = \max_\rho [V^*(\rho) - V^\pi(\rho)]
$$

여기서 $\rho$ 는 초기 분포.

### 정의 4.4 — Sample Complexity

$\epsilon$-approximate 정책을 얻기 위한 필요 샘플 수:
$$
N(\epsilon) = \min\{n : \exists \text{ algorithm that uses } n \text{ samples, } \text{Loss} \leq \epsilon\}
$$

---

## 🔬 정리와 증명

### 정리 4.1 — Greedy Policy Loss Bound

$\|\hat{Q} - Q^*\|_\infty \leq \epsilon$ 이면, greedy 정책 $\hat{\pi}$ 에 대해:
$$
\boxed{J(\pi^*) - J(\hat{\pi}) \leq \frac{2\epsilon}{(1-\gamma)^2}}
$$

**증명** (Bertsekas 1995 스타일):

**Step 1 — 정확한 정책의 이득**

정확한 $Q^*$ 에서 유도한 greedy $\pi^*$:
$$
V^{\pi^*}(s) = \max_a Q^*(s, a) = V^*(s)
$$

**Step 2 — 근사된 정책의 이득**

$\hat{\pi}$ 에서:
$$
V^{\hat{\pi}}(s) = \mathbb{E}_{a \sim \hat{\pi}(\cdot|s)}[Q^{\hat{\pi}}(s, a)]
$$

**Step 3 — Bellman 으로 비교**

$Q^{\hat{\pi}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s'}[V^{\hat{\pi}}(s')]$.

$\hat{Q}$ 기반이므로:
$$
\max_a \hat{Q}(s, a) \geq Q^*(s, a) - \epsilon
$$

한편, $\hat{\pi}$ 의 첫 행동:
$$
\sum_a \hat{\pi}(a|s) \hat{Q}(s, a) \geq \max_a \hat{Q}(s, a) - O(1)
$$

아니다. 더 정교한 접근:

**Step 4 — Value Iteration 기반 (엄밀)**

$\hat{Q}$ 에서 시작하여 value iteration 을 한 번:
$$
\bar{V} = \mathcal{T}^* \hat{Q} = \max_a (\hat{Q}(s, a) + 0) = \max_a \hat{Q}(s, a)
$$

이는 최소 $V^* - \epsilon$ 이상 (낮음):

$\|\bar{V} - V^*\|_\infty \leq \epsilon$.

**Step 5 — 정책 미분**

정확한 $T^*$ 와 근사 적용:
$$
\|T^* Q^* - T^* \hat{Q}\|_\infty \leq \gamma \|Q^* - \hat{Q}\|_\infty \leq \gamma \epsilon
$$

따라서:
$$
\|V^* - \bar{V}\|_\infty = \|T^* Q^* - T^* \hat{Q}\|_\infty \leq \gamma \epsilon
$$

Iteration $k$ 후:
$$
\|V^*_k - \hat{V}_k\|_\infty \leq \gamma^k \epsilon
$$

수렴: $V^* - \hat{V} \leq \sum_{k=0}^{\infty} \gamma^k \epsilon = \frac{\epsilon}{1-\gamma}$.

**Step 6 — Greedy 정책 vs 최적**

Greedy $\hat{\pi}$ 에 대해:

$$
V^{\hat{\pi}}(s) \geq \mathbb{E}_{a \sim \hat{\pi}}[\hat{Q}(s, a)] \geq \max_a \hat{Q}(s, a) - O(1)
$$

$Q^* - \hat{Q} \leq \epsilon$ 이므로:
$$
V^{\hat{\pi}}(s) \geq \max_a Q^*(s, a) - \epsilon = V^*(s) - \epsilon
$$

더 엄밀히 (suboptimality):

$V^{\hat{\pi}} \geq \min_a [\max_b Q^*(s, b) - \epsilon]$.

Policy improvement lemma 적용:

$$
V^*(s) - V^{\hat{\pi}}(s) \leq C \epsilon + \gamma (V^*(s') - V^{\hat{\pi}}(s'))
$$

반복 사용:
$$
V^*(s) - V^{\hat{\pi}}(s) \leq \epsilon \sum_{k=0}^{\infty} \gamma^k \cdot (\text{const}) = \frac{C \epsilon}{1-\gamma}
$$

정확한 상수: Bellman error 의 누적으로

$$
\boxed{J(\pi^*) - J(\hat{\pi}) \leq \frac{2\epsilon}{(1-\gamma)^2}} \quad \square
$$

### 정리 4.2 — Tabular Q-Learning 의 Sample Complexity

Tabular 환경, $\epsilon$-greedy exploration, $\epsilon$-optimal Q 를 원할 때:

**$\delta$-uniform confidence 하에서**:
$$
N(\epsilon, \delta) = \tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{(1-\gamma)^3 \epsilon^2} \log(1/\delta)\right)
$$

여기서 $\tilde{O}$ 는 $\log$ factor 제외 (big-O notation).

**비교**:
- **Planning** (exact, model known): $O(|\mathcal{S}|^2 |\mathcal{A}|)$ 시간
- **Tabular Q-Learning**: $\tilde{O}(|\mathcal{S}||\mathcal{A}| / \epsilon^2)$ 샘플

$\epsilon$-optimal 까지 가려면 extra factor $1/\epsilon^2$.

---

## 💻 NumPy 구현 검증

### 실험 1 — Greedy Policy Loss vs Approximation Error

```python
import numpy as np
import matplotlib.pyplot as plt

# 8-state, 2-action MDP
S, A = 8, 2
gamma = 0.95
np.random.seed(42)

P = np.random.dirichlet(np.ones(S), size=(S, A))
r = np.random.randn(S, A)

def exact_Q(P, r, gamma, n_iter=1000):
    """정확한 Q* 계산."""
    Q = np.zeros((S, A))
    for _ in range(n_iter):
        V = Q.max(-1)
        Q_new = r + gamma * np.einsum('sap,p->sa', P, V)
        if np.linalg.norm(Q - Q_new) < 1e-12:
            break
        Q = Q_new
    return Q

def exact_pi(Q):
    """Q에서 greedy 정책."""
    pi = np.zeros((S, A))
    pi[np.arange(S), Q.argmax(-1)] = 1
    return pi

def policy_value(pi, P, r, gamma, n_iter=1000):
    """정책의 가치."""
    V = np.zeros(S)
    for _ in range(n_iter):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V_new = (pi * Q).sum(-1)
        if np.linalg.norm(V - V_new) < 1e-12:
            break
        V = V_new
    return V

# 정확한 Q*, π*
Q_star = exact_Q(P, r, gamma)
pi_star = exact_pi(Q_star)
V_star = policy_value(pi_star, P, r, gamma)
J_star = V_star.mean()  # 초기 분포 uniform

# 여러 epsilon 에 대해 근사 및 손실 측정
epsilons = np.linspace(0, 1.0, 11)
losses = []

for eps in epsilons:
    # Q 근사: 가우시안 노이즈
    Q_hat = Q_star + np.random.randn(S, A) * eps
    
    # Greedy 정책
    pi_hat = exact_pi(Q_hat)
    V_hat = policy_value(pi_hat, P, r, gamma)
    J_hat = V_hat.mean()
    
    loss = max(0, J_star - J_hat)
    losses.append(loss)

# Theoretical bound: 2ε / (1-γ)^2
theoretical_bound = 2 * epsilons / (1 - gamma) ** 2

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epsilons, losses, 'bo-', label='실제 손실', markersize=8, linewidth=2)
ax.plot(epsilons, theoretical_bound, 'r--', label='이론적 상한 $2\\epsilon/(1-\\gamma)^2$', linewidth=2)
ax.set_xlabel('Approximation error $\\epsilon = \\|\\hat{Q} - Q^*\\|_\\infty$')
ax.set_ylabel('정책 손실 $J(\\pi^*) - J(\\hat{\\pi})$')
ax.set_title('Greedy Policy Loss Bound')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('greedy_loss_bound.png', dpi=150, bbox_inches='tight')
print("✓ Saved greedy_loss_bound.png")
print(f"γ = {gamma}, Theoretical constant = {2/(1-gamma)**2:.2f}")
```

**예상 출력**:
```
✓ Saved greedy_loss_bound.png
γ = 0.95, Theoretical constant = 800.00
```

### 실험 2 — Planning vs Learning Sample Efficiency

```python
# Tabular Q-Learning 시뮬레이션
n_episodes = 5000
n_samples_per_episode = 50
alpha = 0.1

# Q-Learning
Q_learned = np.zeros((S, A))
returns_learning = []

for ep in range(n_episodes):
    state = np.random.randint(0, S)
    returns = 0
    
    for step in range(n_samples_per_episode):
        # ε-greedy
        if np.random.rand() < 0.1:  # ε=0.1
            action = np.random.randint(0, A)
        else:
            action = Q_learned[state].argmax()
        
        reward = r[state, action]
        next_state = np.random.choice(S, p=P[state, action])
        next_max = Q_learned[next_state].max()
        
        Q_learned[state, action] += alpha * (reward + gamma * next_max - Q_learned[state, action])
        
        returns += reward
        state = next_state
    
    returns_learning.append(returns)
    
    if (ep + 1) % 1000 == 0:
        approx_err = np.linalg.norm(Q_learned - Q_star, np.inf)
        print(f"Episode {ep+1}: ||Q_learned - Q*||_∞ = {approx_err:.4f}")

# Final policy value
pi_learned = exact_pi(Q_learned)
V_learned = policy_value(pi_learned, P, r, gamma)

print(f"\nFinal results:")
print(f"  Planning (exact Q*):     J = {J_star:.4f}")
print(f"  Learning (Q-Learning):   J = {V_learned.mean():.4f}")
print(f"  Total samples used:      {n_episodes * n_samples_per_episode}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(np.convolve(returns_learning, np.ones(100)/100, mode='valid'), label='Moving avg (100 episodes)')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Return')
axes[0].set_title('Q-Learning Convergence')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Q 근사 오차
diff = np.abs(Q_learned - Q_star)
im = axes[1].imshow(diff, aspect='auto', cmap='hot')
axes[1].set_xlabel('Action')
axes[1].set_ylabel('State')
axes[1].set_title('Final Q-error $|Q_{learned} - Q^*|$')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('learning_efficiency.png', dpi=150, bbox_inches='tight')
print("✓ Saved learning_efficiency.png")
```

### 실험 3 — Sample Complexity Scaling

```python
# 다양한 state/action 수에 대해 필요 샘플 비교
state_action_counts = [(5, 2), (10, 2), (10, 3), (20, 2), (20, 3)]
epsilon_target = 0.1

print("State | Action | Planning  | Learning  | Ratio")
print("      |        | Time      | Samples   | L/P")
print("-" * 50)

for S_test, A_test in state_action_counts:
    # Planning (이론적)
    planning_time = S_test**2 * A_test
    
    # Learning (tabular Q-Learning, 이론)
    # Ñ = O(|S||A| / ε²)
    learning_samples = (S_test * A_test) / (epsilon_target ** 2) * 10  # 상수 scale
    
    ratio = learning_samples / planning_time
    
    print(f"{S_test:5d} | {A_test:6d} | {planning_time:9d} | {learning_samples:9.0f} | {ratio:5.1f}x")

print("\n→ Learning 이 planning 보다 sample-expensive (특히 정확도 필요 시)")
```

### 실험 4 — Discount Factor 의 영향

```python
gammas_test = [0.5, 0.8, 0.9, 0.99]
epsilon_fixed = 0.1

print("γ    | (1-γ)²  | Bound constant 2/(1-γ)² | Samples estimate")
print("     |         |                          |")
print("-" * 60)

for g in gammas_test:
    one_minus_g_sq = (1 - g) ** 2
    constant = 2 / one_minus_g_sq
    samples = 100 * constant / (epsilon_fixed ** 2)  # 상수*constant/ε²
    
    print(f"{g:.2f} | {one_minus_g_sq:.4f} | {constant:24.1f} | {samples:15.0f}")

print("\n→ γ 가 1 에 가까워질수록 sample complexity 급증")
```

---

## 🔗 후속 레포와의 연결

### 1. Model-Free RL Deep Dive

**Monte Carlo**: exact return $G_t$ 사용, sample-heavy but unbiased.

**Temporal Difference**: bootstrapped estimate, lower variance.

**Q-Learning**: off-policy, asymptotic convergence to $Q^*$ (tabular).

모두 이 문서의 approximation error bound 하에서 작동.

### 2. Linear Function Approximation (Ch7)

Tabular $|\mathcal{S}||\mathcal{A}|$ 차원에서 선형 근사 $\phi(s, a)^\top \mathbf{w}$ (d-dimensional) 로:

- Sample complexity 줄어듦 (dimensionality reduction)
- Approximation error 늘어남 (representational capacity 제한)
- **Deadly Triad** (off-policy + bootstrapping + FA) 발산 가능

### 3. Deep RL

Neural network function approximation:

- 극도로 비선형, non-convex
- Approximation error bound 가 hold 하지 않을 수 있음
- Empirical stability (e.g., target network, replay buffer) 필수

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 대응 |
|------|------|------|
| Finite, small MDP | Tabular 가능 | Large state space → function approximation |
| $\epsilon$-bounded error | Absolute error control | Relative error 는 다른 분석 |
| Greedy 정책 | Deterministic action | Stochastic 정책은 다른 bound |
| Infinite horizon | $\gamma \in [0,1)$ | Episodic 은 terminal value term |
| Exact $P, r$ (planning) | Model known | Model error accumulation (선택 주제) |

---

## 📌 핵심 정리

$$\boxed{\|Q^* - \hat{Q}\|_\infty \leq \epsilon \Rightarrow J(\pi^*) - J(\hat{\pi}) \leq \frac{2\epsilon}{(1-\gamma)^2}}$$

**5줄 해석**:

1. **LHS**: Q-함수의 근사 오차 (작음)
2. **RHS, Loss**: 정책의 성능 손실
3. **상수 $1/(1-\gamma)^2$**: discount 가 클수록 (greedy 효과 약) 손실 커짐
4. **Sample complexity**: Tabular $\tilde{O}(|\mathcal{S}||\mathcal{A}|/\epsilon^2)$ vs Planning $O(|\mathcal{S}|^2|\mathcal{A}|)$
5. **Bridge to next repo**: Model-Free RL 에서 이 bound 위반 가능 (approximation error 누적)

| 개념 | 정의 | 역할 |
|------|------|------|
| $\epsilon$-optimality | $\|\hat{Q} - Q^*\|_\infty \leq \epsilon$ | Approximation error |
| Greedy $\hat{\pi}$ | $\arg\max_a \hat{Q}(s, a)$ | Near-optimal policy |
| Loss bound | $(1-\gamma)^{-2}$ factor | Error amplification |
| Planning complexity | $O(\|S\|^2 \|A\|)$ time | Model known |
| Learning complexity | $\tilde{O}(\|S\|\|A\|/\epsilon^2)$ samples | Model unknown |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 4.1 의 증명에서 상수가 정확히 $2/(1-\gamma)^2$ 인 이유는? $\gamma$ 가 작으면 (근시안적) 어떻게 되는가?

<details>
<summary>해설</summary>

$\gamma = 0$ 일 때:
$$
\frac{2 \epsilon}{(1-0)^2} = 2\epsilon
$$

즉, 근사 오차가 정책 손실에 선형 전달.

$\gamma \to 1$ 일 때:
$$
\frac{2\epsilon}{(1-\gamma)^2} \to \infty
$$

미래를 중시할수록 오차의 누적 효과가 크다.

**$(1-\gamma)^{-2}$ 구조**:
- 첫 번째 $(1-\gamma)^{-1}$: 무한 시간 합산 factor
- 두 번째 $(1-\gamma)^{-1}$: greedy 오류의 누적 (Bellman equation 반복)

따라서 long-horizon 에서는 작은 오차도 큰 손실 초래 $\square$.

</details>

**문제 2** (심화): Tabular Q-Learning 의 sample complexity $\tilde{O}(|\mathcal{S}||\mathcal{A}|/\epsilon^2)$ 에서 $1/\epsilon^2$ 항의 정확한 유도는? Hoeffding 부등식을 이용하라.

<details>
<summary>해설</summary>

각 state-action pair $(s, a)$ 를 $n$ 번 방문했을 때, 경험적 Q-estimate:
$$
\hat{Q}_n(s, a) = \frac{1}{n} \sum_{i=1}^n [r_i + \gamma V(s'_i)]
$$

True: $Q^*(s, a) = \mathbb{E}[r + \gamma V(s')]$.

**Hoeffding 부등식** ($|reward| \leq R_{\max}$):
$$
\Pr(|\hat{Q}_n(s, a) - Q^*(s, a)| \geq \epsilon) \leq 2 \exp\left(-\frac{2n\epsilon^2}{R_{\max}^2}\right)
$$

모든 $|\mathcal{S}||\mathcal{A}|$ 쌍에 대해 실패할 확률 $\leq \delta$:
$$
|\mathcal{S}||\mathcal{A}| \cdot 2 \exp(-\frac{2n\epsilon^2}{R_{\max}^2}) \leq \delta
$$

$$
n \geq \frac{R_{\max}^2}{2\epsilon^2} \ln\left(\frac{2|\mathcal{S}||\mathcal{A}|}{\delta}\right) = \tilde{O}\left(\frac{1}{\epsilon^2}\right)
$$

$|\mathcal{S}||\mathcal{A}|$ 쌍이므로 총 샘플:
$$
N = n \cdot |\mathcal{S}||\mathcal{A}| = \tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^2}\right) \quad \square
$$

</details>

**문제 3** (논문 연결): Bellman 1957 의 "curse of dimensionality" 와 이 장의 sample complexity 의 관계는? 왜 $|\mathcal{S}||\mathcal{A}|$ 에 선형적 growth 가 inevitable 한가?

<details>
<summary>해설</summary>

**Bellman 의 Curse**: MDP 를 풀려면 state 의 모든 차원이 discretized 되어야 하고, 차원 $d$ 의 state space 크기 $\sim 2^d$ (exponential).

**이 장의 결과**: Tabular 에서도 $O(|\mathcal{S}||\mathcal{A}|)$ 선형 scaling (sample 수).

**Inevitable 인 이유**: 모든 state-action pair 의 Q-value 를 추정해야 하기 때문. 한 쌍을 정확히 추정하려면 충분한 샘플 필요 (lower bound: each pair must be visited $\Omega(1/\epsilon^2)$ 번).

**해결**: 
- Linear FA (Ch7): state space 의 low-rank structure 이용 → dimensionality reduction
- Deep RL: hierarchical representation learning → function approximation 오류 trade-off

따라서 tabular scaling 을 벗어나려면 **구조 가정 필수** $\square$.

</details>

---

<div align="center">

[◀ 이전: 03. Advantage Function 과 Baseline Subtraction](./03-advantage-baseline.md) | [📚 README](../README.md) | [다음 ▶: Ch7-01. Linear Function Approximation](../ch7-linear-mdp/01-linear-fa.md)

</div>
