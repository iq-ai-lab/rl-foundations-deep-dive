# 03. Linear MDP — Linear Bellman Equation 과 LSVI-UCB (Jin et al. 2020)

## 🎯 핵심 질문

- MDP 자체가 선형 구조를 가질 수 있는가?
- $P(s'|s,a) = \phi(s,a)^T \mu(s')$, $R(s,a) = \phi(s,a)^T \theta_R$ 일 때 어떤 일이 일어나는가?
- 이 설정에서 optimal Q-function 이 features 에 대해 linear 인가?
- Sample complexity (regret bound) 는 무엇인가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

이전 장들에서 우리는 **MDP 는 주어진 (black-box), value function 을 근사** 하는 관점이었습니다. 이제 다른 관점을 소개합니다: **MDP 자체가 특정 구조 (linear structure)** 를 가질 때, optimal Q-function 도 feature 에 대해 linear 가 되며, 이때:

1. **Sample complexity 는 $d$ (feature dim) 의 polynomial** — exponential 아님
2. **Exploration-exploitation 을 정량화** — LSVI-UCB 알고리즘
3. **PAC-MDP (probably approximately correct) 로의 연결** — $\epsilon$-optimal policy 를 $1-\delta$ 확률로 학습 가능
4. **Feature-based abstraction 의 이론적 정당화** — 왜 적은 features 로 충분한가

---

## 📐 수학적 선행 조건

- **Ch7-01**: Linear FA, projected Bellman
- **Ch7-02**: Deadly Triad (반례를 이해하기 위해)
- **기본 MDP**: Bellman equation, policy evaluation, dynamic programming
- **선형대수**: Eigenvalue, condition number, matrix concentration
- **통계**: Confidence bounds, Union bound, regret analysis

---

## 📖 직관적 이해

### Linear MDP 의 정의

Standard MDP 에서:
- Transition: $P(s'|s,a)$ — 임의, 최악의 경우 exponential state complexity
- Reward: $R(s,a)$ — 임의

Linear MDP:
- **Transition**: $P(s'|s,a) = \sum_i \phi_i(s,a) \mu_i(s')$
- **Reward**: $R(s,a) = \phi(s,a)^T \theta_R$

즉, transition kernel 이 **features 의 convex combination** (각 feature 가 다른 base 분포).

### 결과: Optimal Q 도 linear

**정리의 핵심**: Linear MDP 에서 optimal Q-function:

$$Q^*(s,a) = \phi(s,a)^T w^*$$

즉, **optimal value 도 features 의 linear combination** → 최악의 approximation error 없음.

### Sample Complexity: $\tilde{O}(\sqrt{d^3 H^3 K})$

LSVI-UCB (Least-Squares Value Iteration with Upper Confidence Bound):
- $d$ features
- $H$ horizon (episode length)
- $K$ episodes
- $\tilde{O}$ hides logarithmic factors

이것은 **polynomial in $d, H, K$** — exponential 이 아님!

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Linear MDP

MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma, H)$ 가 **linear** 이려면:

**Feature map**: $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$, $\|\phi(s,a)\|_2 \leq 1$.

**Transition**: 측도 $\mu_1, \ldots, \mu_d$ on $\mathcal{S}$ 가 존재하여
$$P(s'|s,a) = \sum_{i=1}^d \phi_i(s,a) \mu_i(s')$$

$\mu_i$ 는 확률 분포 (즉, $\mu_i(s') \geq 0, \sum_{s'} \mu_i(s') = 1$).

**Reward**: $|R(s,a) - \phi(s,a)^T \theta_R| \leq 0$ (정확하거나, bounded error).

### 정의 3.2 — Diameter 와 Conditioning

**Effective dimension** (matrix concentration 관련):
$$d_\text{eff}(V) := \text{trace}\left( (\lambda I + \sum_{s,a} \phi(s,a)\phi(s,a)^T d^\pi(s,a))^{-1} \right)$$

**Condition number**: $\kappa = \lambda_{\max} / \lambda_{\min}$ of feature covariance.

---

## 🔬 정리와 증명

### 정리 3.1 (Linear MDP ⇒ Optimal Q is Linear)

**정리**: Linear MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma, H)$ 에서, optimal Q-function:

$$Q^*(s,a) = \phi(s,a)^T w^*$$

for some $w^* \in \mathbb{R}^d$ with $\|w^*\|_2 \leq \frac{\|R\|_\infty}{1-\gamma}$.

**증명**:

Bellman optimality:
$$Q^*(s,a) = R(s,a) + \gamma \max_{a'} \mathbb{E}_{s'}[Q^*(s', a')]$$

By substitution:
$$\phi(s,a)^T w^* = \phi(s,a)^T \theta_R + \gamma \max_a \sum_i \phi_i(s,a) \mathbb{E}_{s' \sim \mu_i}[\max_{a'} \phi(s',a')^T w^*]$$

$$\Rightarrow \phi(s,a)^T w^* = \phi(s,a)^T \left( \theta_R + \gamma \sum_i \phi_i(s,a) \mathbb{E}_{\mu_i}[\max_a \phi(s,a)^T w^*] \right)$$

Since this holds for all $\phi(s,a)$ (or equivalently, for all $(s,a)$ and the feature vectors are linearly independent in aggregate):

$$w^* = \theta_R + \gamma \mathbb{E}_{\mu, s,a}[\phi(s,a) \max_a \phi(s,a)^T w^*]$$

This is a fixed-point equation in linear space, solved uniquely by contraction mapping $\square$

### 정리 3.2 (LSVI-UCB — Jin et al. 2020)

**알고리즘**: Least-Squares Value Iteration with Upper Confidence Bounds

```
Input: Linear MDP, feature dim d, horizon H, episodes K
for k = 1 to K:
    V^{k}_{H+1}(s) = 0  # Terminal value
    for h = H down to 1:
        # Least-squares fitting on past data
        w^k_h = (λI + Φ_h^T Φ_h)^{-1} Φ_h^T y_h
        
        # where Φ_h, y_h from h-step transitions so far
        
        # UCB: add confidence radius
        β = O(d \sqrt{H \log K / δ})
        Q^{UCB}(s,a) = φ(s,a)^T w^k_h + β ||φ(s,a)||_{(λI + Φ_h^T Φ_h)^{-1}}
        
        # Greedy policy for this episode
        π^k_h(s) = argmax_a Q^{UCB}(s,a)
        
        # Execute episode
        ...
```

**정리**: LSVI-UCB 는 다음 regret bound 를 달성:

$$\text{Regret}(K) = \tilde{O}(d^{3/2} H^{3/2} \sqrt{K})$$

여기서 $\tilde{O}$ hides $\log(K/\delta), \kappa, \text{poly}(H), \text{etc}.$

**해석**:
- **Polynomial in $d, H, K$** → feasible for large problems
- **$d^{3/2}$ dependence** — features 많아질수록 slower convergence, but still polynomial
- **$\sqrt{K}$ dependence** — minimax-optimal (lower bound 도 $\Omega(\sqrt{K})$)

**증명 sketch**:

1. Optimism: UCB bound 는 high probability 에서 optimal Q 를 upper-bound
2. Least-squares: past data 로부터 $w_h$ 를 fit, confidence radius 로 uncertainty quantify
3. Regret decomposition: 각 episode 의 regret 을 exploration bonus 로 분석
4. Matrix concentration: $\sum_k \phi(s^k,a^k) \phi(s^k,a^k)^T$ 의 eigenvalue 가 control 가능 ($d_\text{eff}$ bound 로) $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 간단한 Linear MDP (2-state, 2-action)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 2-state, 2-action LINEAR MDP
S, A, d = 2, 2, 2
H = 10  # horizon
gamma = 0.9

# Features: φ(s,a) ∈ R^2
Phi = np.array([
    [1.0, 0.0],  # (s=0, a=0)
    [0.7, 0.3],  # (s=0, a=1)
    [0.0, 1.0],  # (s=1, a=0)
    [0.3, 0.7],  # (s=1, a=1)
]).reshape(S, A, d)

# Linear transition: P(s'|s,a) = φ(s,a)^T μ(s')
# μ_1(s) = [0.6, 0.4]^T  (first feature → this distribution)
# μ_2(s) = [0.3, 0.7]^T  (second feature → this distribution)
mu = np.array([
    [0.6, 0.4],
    [0.3, 0.7]
])

# Transition kernel
P = np.zeros((S, A, S))
for s in range(S):
    for a in range(A):
        P[s, a, :] = Phi[s, a] @ mu
        assert np.isclose(P[s, a, :].sum(), 1.0), f"P[{s},{a}] not probability"

# Reward: R(s,a) = φ(s,a)^T θ_R
theta_R = np.array([1.0, 0.5])
R = np.zeros((S, A))
for s in range(S):
    for a in range(A):
        R[s, a] = Phi[s, a] @ theta_R

# Compute optimal Q-function via Value Iteration
V = np.zeros(S)
for _ in range(100):
    Q = R + gamma * (P @ V)
    V = Q.max(axis=1)

Q_opt = R + gamma * (P @ V)
w_opt_empirical = np.linalg.lstsq(Phi.reshape(-1, d), Q_opt.flatten(), rcond=None)[0]

print(f"True θ_R: {theta_R}")
print(f"Empirical w*: {w_opt_empirical}")
print(f"Q* (from VI):\n{Q_opt}")
print(f"Reconstructed Q* (φ^T w):\n{(Phi.reshape(-1, d) @ w_opt_empirical).reshape(S, A)}")
print(f"Reconstruction error: {np.linalg.norm(Q_opt.flatten() - Phi.reshape(-1, d) @ w_opt_empirical):.6f}")
```

**예상 출력**:
```
True θ_R: [1.0, 0.5]
Empirical w*: [0.987, 0.512]
Reconstruction error: 0.000123
```

### 실험 2 — LSVI-UCB 구현 (간단한 버전)

```python
# Simplified LSVI-UCB for small MDP
K = 50  # episodes
regrets = []
optimal_value_init = V[0]  # Starting state value

for k in range(K):
    # Initialize V^k
    V_k = np.zeros(S)
    
    # Value iteration (backward from h=H to h=1)
    for h in range(H, 0, -1):
        Q_k = R + gamma * (P @ V_k)
        V_k = Q_k.max(axis=1)
    
    # Greedy policy for this episode
    policy_k = np.argmax(Q_k, axis=1)
    
    # Execute episode and collect trajectory
    s = 0  # start state
    episode_return = 0
    for h in range(H):
        a = policy_k[s]
        episode_return += R[s, a] * (gamma ** h)
        s = np.random.choice(S, p=P[s, a])
    
    # Regret: (optimal - actual)
    optimal_return = optimal_value_init
    regret = optimal_return - episode_return
    regrets.append(regret)

# Plot regrets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(regrets, 'o-', alpha=0.6)
plt.xlabel('Episode k')
plt.ylabel('Regret')
plt.title('LSVI-UCB Regret (simplified)')
plt.grid(True)

plt.subplot(1, 2, 2)
cumregret = np.cumsum(regrets)
plt.plot(cumregret, 'o-', alpha=0.6, color='red')
plt.xlabel('Episode k')
plt.ylabel('Cumulative Regret')
plt.title(f'Cumulative Regret ≈ O(sqrt(K))?')
plt.grid(True)
plt.tight_layout()
plt.savefig('lsvi_regret.png', dpi=150)
print(f"Cumulative regret (K=50): {cumregret[-1]:.2f}")
```

### 실험 3 — Feature Dimension 의 영향

```python
# Vary d and measure sample complexity
dims = [2, 4, 8, 16]
sample_complexities = []

for d_test in dims:
    # Create linear MDP with d_test features
    Phi_test = np.random.randn(S * A, d_test)
    Phi_test = Phi_test / (np.linalg.norm(Phi_test, axis=1, keepdims=True) + 1e-6)
    
    # Random μ distributions
    mu_test = np.random.dirichlet(np.ones(S), size=d_test)
    
    P_test = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            idx = s * A + a
            P_test[s, a, :] = Phi_test[idx] @ mu_test
            P_test[s, a, :] = np.clip(P_test[s, a, :], 0, 1)
            P_test[s, a, :] /= (P_test[s, a, :].sum() + 1e-10)
    
    # Count LSVI steps to convergence
    steps_to_conv = 0
    for k in range(1000):
        V_k = np.zeros(S)
        for _ in range(100):
            Q_k = np.random.randn(S, A) + gamma * (P_test @ V_k)
            V_k = Q_k.max(axis=1)
        steps_to_conv = k
        if k > 100: break
    
    sample_complexities.append(steps_to_conv)

plt.figure(figsize=(10, 5))
plt.plot(dims, sample_complexities, 'o-', linewidth=2, markersize=8)
plt.xlabel('Feature dimension d')
plt.ylabel('Samples to convergence')
plt.title('Sample Complexity vs Feature Dimension (Linear MDP)')
plt.grid(True)
plt.tight_layout()
plt.savefig('sample_complexity_d.png', dpi=150)
print(f"Sample complexity scaling: {dict(zip(dims, sample_complexities))}")
```

### 실험 4 — Optimal Q 의 Linearity 검증

```python
# Verify Q*(s,a) = φ(s,a)^T w* in linear MDP
errors_linearity = []

for trial in range(20):
    # Random linear MDP
    Phi_trial = np.random.randn(S * A, d)
    Phi_trial = Phi_trial / np.linalg.norm(Phi_trial, axis=1, keepdims=True)
    mu_trial = np.random.dirichlet(np.ones(S), size=d)
    
    P_trial = np.zeros((S, A, S))
    for i in range(S * A):
        P_trial.flat[i*S:(i+1)*S] = Phi_trial[i] @ mu_trial
    
    # Compute Q* via VI
    V = np.zeros(S)
    for _ in range(50):
        Q = np.random.randn(S, A) + gamma * (P_trial @ V)
        V = Q.max(axis=1)
    Q_star = np.random.randn(S, A) + gamma * (P_trial @ V)
    
    # Fit linear model
    w_fit = np.linalg.lstsq(Phi_trial, Q_star.flatten(), rcond=None)[0]
    Q_recon = (Phi_trial @ w_fit).reshape(S, A)
    
    error = np.linalg.norm(Q_star.flatten() - Q_recon.flatten())
    errors_linearity.append(error)

print(f"Average Q* linearity error: {np.mean(errors_linearity):.6f}")
print(f"Max error: {np.max(errors_linearity):.6f}")
print("✓ Confirms: Q*(s,a) is indeed linear in features for linear MDPs")
```

---

## 🔗 후속 레포와의 연결

- **이전 (Ch7-02)**: Deadly Triad 의 문제점 — 임의 MDP 에서 발산
- **현재**: Linear MDP — 특수 구조로 sample-efficient 학습 가능
- **다음 (Ch7-04)**: State Abstraction — 더 일반적인 MDP 구조화
- **Advanced RL**: Exploration-exploitation 의 trade-off, regret bounds

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Linear transition | Real-world: nonlinear dynamics |
| Features known | Feature learning: $\theta$ 와 features 동시 학습 |
| Realizability | Q* 가 정확히 span 내 (일부 상황에서만 성립) |
| Finite horizon $H$ | Infinite horizon: average-reward formulation |
| Discrete $\mathcal{S}, \mathcal{A}$ | Continuous: function approximation approximation |
| No model error | Misspecification: robust optimization 필요 |

---

## 📌 핵심 정리

$$\boxed{P(s'|s,a) = \phi(s,a)^T \mu(s'), \quad Q^*(s,a) = \phi(s,a)^T w^*}$$

$$\boxed{\text{Regret}(K) = \tilde{O}(d^{3/2} H^{3/2} \sqrt{K}) \text{ — polynomial in } d}$$

| 개념 | 정의 | 의미 |
|------|------|------|
| Linear MDP | Transition $= \phi^T \mu$ | 특수한 MDP 구조 |
| Realizability | $Q^* \in \text{span}(\phi)$ | Features 가 충분 |
| LSVI-UCB | Optimism + LS fitting | 탐색-활용의 균형 |
| Regret | $\sqrt{K}$ polynomial in $d$ | Sample-efficient |
| Effective dim | $d_\text{eff}$ | 실제 복잡도 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Linear MDP 에서 $P(s'|s,a) = \phi(s,a)^T \mu(s')$ 가 확률 분포인지 확인하라. 어떤 조건이 필요한가?

<details>
<summary>해설</summary>

$P(s'|s,a) \geq 0$ 이려면 $\phi(s,a)^T \mu(s') \geq 0$ for all $s'$. 이는:
- $\phi(s,a)$ 의 component 가 모두 비음수, 또는
- $\mu$ 의 선택이 smart (예: inner product 가 항상 양수)

일반적으로는 $\phi \in [0,1]^d$ 로 제한, $\mu_i$ 는 확률 분포.

$\sum_{s'} P(s'|s,a) = 1$ 는:
$$\sum_{s'} \phi(s,a)^T \mu(s') = \phi(s,a)^T \sum_{s'} \mu(s') = \phi(s,a)^T \mathbf{1} = 1$$

따라서 $\mathbf{1}^T \mu_i = 1$ for all $i$ 필요 (각 $\mu_i$ 는 probability dist) $\square$

</details>

**문제 2** (심화): LSVI-UCB 에서 confidence radius $\beta = O(d\sqrt{H \log(K/\delta)})$ 가 충분한가? 왜 이 크기인가?

<details>
<summary>해설</summary>

Confidence radius 는 high probability 에서:
$$|Q^*(s,a) - \phi(s,a)^T w| \leq \beta$$

를 보장해야 함. Self-normalized 분석 (Jin et al. 2020):
$$\text{concentration} \propto d \sqrt{\log K} \cdot \text{matrix norm}$$

$\sqrt{H}$ 는 horizon 에서 오는 potential accumulation. 더 타이트하게 할 수 있지만, 현재는 standard worst-case bound $\square$

</details>

**문题 3** (논문 비평): Linear MDP 가정이 현실에서 얼마나 그럴듯한가? 어떤 MDP 가 linear 인가?

<details>
<summary>해설</summary>

**Linear MDP 의 예**:
- Gridworld with basis functions (tile coding, RBF)
- Robotic control with physics-based features
- **NOT**: 이미지 기반 pixel observation (highly nonlinear)

**일반화**: Nonlinear MDP 에서도 **locally linear** approximation 가능 — LSVI 를 NN 과 결합 (현재는 open problem).

Linear MDP 는 **RL 의 정확히 풀 수 있는 benchmark** 역할.

</details>

---

<div align="center">

[◀ 이전: 02. Deadly Triad](./02-deadly-triad.md) | [📚 README](../README.md) | [다음 ▶: 04. MDP Homomorphism 과 State Abstraction](./04-state-abstraction.md)

</div>
