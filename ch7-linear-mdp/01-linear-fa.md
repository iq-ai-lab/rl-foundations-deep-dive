# 01. Linear Function Approximation (Tsitsiklis & Van Roy 1997)

## 🎯 핵심 질문

- 무한 상태 공간에서 value function 을 어떻게 근사하는가?
- Linear function approximation 에서 $V_\theta(s) = \theta^T \phi(s)$ 모델이 왜 fundamental 인가?
- TD(0) learning 이 on-policy 설정에서 어디로 수렴하는가?
- Projected Bellman operator $\Pi T^\pi$ 의 contraction 성질이 무엇인가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

MDP 의 이론적 기초 (Ch1-6) 는 **finite state space 가정**에 기반합니다. 그러나 현실의 대부분 문제는 무한하거나 매우 높은 차원의 상태 공간을 가집니다. 이때 exact value function $V(s) \in \mathbb{R}$ 을 각 state 마다 저장할 수 없으므로:

1. **Function approximation 필수** — shared representation $\phi(s) \in \mathbb{R}^d$ 로 $d \ll |\mathcal{S}|$
2. **선형 모델의 수학적 우수성** — $V_\theta(s) = \theta^T \phi(s)$ 는 수렴 이론이 완전함 (Tsitsiklis & Van Roy 1997)
3. **Deep RL 의 출발점** — neural network 는 non-linear FA, linear FA 의 일반화
4. **딜레마 도입** — function approximation 이 off-policy 설정에서 **치명적 불안정성** 초래 (Ch7-02 Deadly Triad)

---

## 📐 수학적 선행 조건

- **Ch1-6 MDP Basics**: Bellman equation, value function, policy evaluation
- **선형대수**: inner product, projection operator, spectral radius, convergence
- **확률론**: Martingale convergence theorem (Robbins-Monro)
- **수학해석**: Norm, contraction, fixed point

---

## 📖 직관적 이해

### Function Approximation 의 필요성

State 가 continuous (e.g., 로봇의 관절 각도) 이거나 high-dimensional (e.g., image) 이면 tabular value table 불가능. 대신 **매개변수 $\theta$ 로 value function 전체를 표현**:

$$V_\theta(s) = \theta^T \phi(s)$$

여기서 $\phi: \mathcal{S} \to \mathbb{R}^d$ 는 **feature map** (사람이 설계하거나 NN 으로 학습).

### Linear FA 의 장점

$$V_\theta(s) = \sum_{i=1}^d \theta_i \phi_i(s)$$

- **볼록성 (convexity)** — loss landscape 가 convex → 수렴 보장 가능
- **닫힌 형태 해 (closed form)** — 일부 경우 $\theta^*$ 를 직접 계산 가능
- **해석 가능성** — 각 feature weight 의 의미 명확

### Projection 의 직관

Bellman operator $T^\pi$ 를 무한 차원 공간에 적용하면:

$$T^\pi V = r^\pi + \gamma P^\pi V$$

근데 $T^\pi V$ 가 반드시 span($\phi$) 내에 있지 않음. 따라서 **nearest projection**:

$$\Pi T^\pi V_\theta$$

이 operator 가 contraction 이면 → fixed point $V^*_\theta$ 로 수렴.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Linear Function Approximation

$\phi: \mathcal{S} \to \mathbb{R}^d$ 를 feature map. Value function 근사:

$$V_\theta(s) := \theta^T \phi(s), \quad \theta \in \mathbb{R}^d$$

On-policy policy $\pi$ 에 대해, Q-function 도:

$$Q_\theta(s, a) := \theta^T \phi(s, a), \quad \phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$$

### 정의 1.2 — Projected Bellman Operator

$\Pi$ 를 span($\Phi$) 으로의 orthogonal projection ($\Phi = [\phi(s_1), \ldots, \phi(s_n)] \in \mathbb{R}^{d \times n}$):

$$\Pi = \Phi (\Phi^T \Phi)^{-1} \Phi^T$$

Projected Bellman operator:

$$(\Pi T^\pi)(V) := \Pi(r^\pi + \gamma P^\pi V)$$

### 정의 1.3 — $L_2$ Norm with Stationary Distribution

상태 분포 $d^\pi$ (discounted stationary) 에 가중. Inner product 및 norm:

$$\langle u, v \rangle_{d^\pi} := \sum_s d^\pi(s) u(s) v(s), \quad \|u\|_{d^\pi} := \sqrt{\langle u, u \rangle_{d^\pi}}$$

---

## 🔬 정리와 증명

### 정리 1.1 (Tsitsiklis & Van Roy 1997 — On-Policy Convergence)

**가정**:
- Finite state/action space
- Policy $\pi$ given (fixed)
- Feature map $\phi(s) \in \mathbb{R}^d$, $\phi(s) \neq 0$
- Learning rate $\alpha_t$ 는 Robbins-Monro (decay, but $\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$)

**정리**: TD(0) iteration

$$\theta_{t+1} = \theta_t + \alpha_t (\delta_t) \phi(s_t)$$

여기서 $\delta_t = r_t + \gamma V_{\theta_t}(s_{t+1}) - V_{\theta_t}(s_t)$ (TD error),

**$\theta_t \to \theta^* $ a.s. (almost surely)**,

여기서 $\theta^*$ 는 **projected Bellman equation 의 유일 고정점**:

$$\theta^* = \arg\min_\theta \| \Pi T^\pi V_\theta - V_\theta \|_{d^\pi}^2$$

**증명 개요**:

**Step 1** — $\Pi T^\pi$ 가 $\|\cdot\|_{d^\pi}$ 에서 $\gamma$-contraction:

$$\|\Pi T^\pi V - \Pi T^\pi V'\|_{d^\pi} \leq \gamma \|V - V'\|_{d^\pi}$$

(Proof: $\Pi$ 는 orthogonal projection, $P^\pi$ 는 stochastic matrix → $\|\gamma P^\pi\|_{d^\pi} \leq \gamma$)

**Step 2** — Banach fixed point theorem → unique $V^*_\theta = (\Pi T^\pi) V^*_\theta$

**Step 3** — TD(0) 를 stochastic approximation (Robbins-Monro) 으로 분석. $\theta_t$ 가 ODE 를 follow:

$$\dot{\theta} = A \theta - b$$

여기서 $A = \mathbb{E}[\phi(s) (\phi(s) - \gamma \phi(s'))^T]$, $b = \mathbb{E}[\phi(s) r(s, \pi(s))]$.

The stable point 는 $\theta^* = A^{-1} b$, 이것이 projected Bellman 의 고정점과 일치 $\square$

### 정리 1.2 — Approximation Error 와 MSBE

On-policy 에서의 MSBE (Mean Square Bellman Error):

$$\text{MSBE}(\theta) := \| V_\theta - \Pi T^\pi V_\theta \|_{d^\pi}^2$$

TD(0) converge 점 $\theta^*$ 는 이 error 를 최소화. 근데 generalization error (optimal value 와의 거리):

$$\| V^*_\theta - V^\pi \|_{d^\pi} \leq \frac{1}{1-\gamma} \| V^\pi - \Pi V^\pi \|_{d^\pi}$$

즉, **projection 의 근본적 한계** — feature map 이 $V^\pi$ 를 정확히 span 하지 않으면 inevitable error.

---

## 💻 NumPy 구현 검증

### 실험 1 — 간단한 4-state MDP 에서 Linear TD(0)

```python
import numpy as np
import matplotlib.pyplot as plt

S, A, d = 4, 2, 2
gamma = 0.9
np.random.seed(42)

P = np.random.dirichlet(np.ones(S), size=(S, A))
R = np.random.randn(S, A)
pi = np.ones((S, A)) / A

Phi = np.eye(S)[:, :d]

V_pi = np.linalg.solve(np.eye(S) - gamma * (P * pi[:, None]).sum(1), 
                       (R * pi).sum(1))

def td_update(theta, s, r, s_next, alpha=0.01):
    delta = r + gamma * theta.T @ Phi[s_next] - theta.T @ Phi[s]
    theta = theta + alpha * delta * Phi[s]
    return theta, delta

theta = np.random.randn(d) / 10
history = [theta.copy()]

for ep in range(50):
    for step in range(100):
        s = np.random.randint(S)
        a = np.random.randint(A)
        s_next = np.random.choice(S, p=P[s, a])
        r = R[s, a]
        theta, delta = td_update(theta, s, r, s_next, alpha=0.01)
    history.append(theta.copy())
    
history = np.array(history)
V_theta = Phi @ history[-1]

print(f"True V^π:      {V_pi}")
print(f"TD(0) approx:  {V_theta}")
print(f"Error:         {np.abs(V_theta - V_pi).max():.4f}")
```

### 실험 2 — Feature Dimension 이 부족할 때

```python
Phi_poor = np.ones((S, 1))
theta_poor = np.random.randn(1) / 10

for _ in range(1000):
    s = np.random.randint(S)
    a = np.random.randint(A)
    s_next = np.random.choice(S, p=P[s, a])
    r = R[s, a]
    theta_poor, _ = td_update(theta_poor, s, r, s_next, alpha=0.01)

V_theta_poor = Phi_poor @ theta_poor
print(f"Constant feature approx: {V_theta_poor}")
print(f"Error: {np.abs(V_theta_poor.mean() - V_pi).max():.4f}")
```

### 실험 3 — Learning Rate Decay

```python
alphas = [0.01, 0.1, 0.001]
errors = []

for alpha_init in alphas:
    theta = np.random.randn(d) / 10
    for t in range(500):
        s = np.random.randint(S)
        a = np.random.randint(A)
        s_next = np.random.choice(S, p=P[s, a])
        r = R[s, a]
        alpha_t = alpha_init / (1 + t * 0.001)
        theta, _ = td_update(theta, s, r, s_next, alpha=alpha_t)
    
    V_theta_final = Phi @ theta
    errors.append(np.abs(V_theta_final - V_pi).max())

print(f"Errors vs alpha: {list(zip(alphas, errors))}")
```

### 실험 4 — Projection 의 시각화

```python
V_init = np.random.randn(S)

def bellman_pi(V):
    P_pi = (P * pi[:, None]).sum(0)
    return (R * pi).sum(1) + gamma * P_pi @ V

V_exact, V_proj = [V_init.copy()], [Phi @ (Phi.T @ V_init)]

for _ in range(30):
    V_exact.append(bellman_pi(V_exact[-1]))
    V_next = bellman_pi(V_proj[-1])
    V_proj_coef = np.linalg.lstsq(Phi, V_next, rcond=None)[0]
    V_proj.append(Phi @ V_proj_coef)

V_exact = np.array(V_exact)
V_proj = np.array(V_proj)

plt.figure(figsize=(10, 5))
for s in range(S):
    plt.plot(V_exact[:, s], 'o-', label=f'Exact s={s}', alpha=0.5)
    plt.plot(V_proj[:, s], 's--', label=f'Projected s={s}', alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend(fontsize=8)
plt.title('Bellman vs Projected Bellman Convergence')
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig('bellman_projection.png', dpi=150)
```

---

## 🔗 후속 레포와의 연결

- **이전 (Ch6)**: MDP approximation 의 이론적 근거
- **현재**: Value function 의 parametric approximation
- **다음 (Ch7-02)**: Deadly Triad — off-policy 에서 발산
- **Deep RL**: Non-linear approximation 및 techniques

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Finite state/action | Continuous/high-dim → feature discretization 또는 NN |
| Fixed $\pi$ (on-policy) | Off-policy 에서 발산 가능 |
| Linearity | Non-linear (NN) 에서는 수렴 보장 없음 |
| Feature map 주어짐 | Feature learning 에서는 complexity 증가 |
| $\|\cdot\|_{d^\pi}$ norm | Different norm → 다른 고정점 |
| Bounded reward | Unbounded → 발산 가능 |

---

## 📌 핵심 정리

$$\boxed{V_\theta(s) = \theta^T \phi(s), \quad \theta_{t+1} = \theta_t + \alpha_t \delta_t \phi(s_t) \to \theta^*}$$

$$\boxed{\|\Pi T^\pi V - \Pi T^\pi V'\|_{d^\pi} \leq \gamma\|V - V'\|_{d^\pi}}$$

| 개념 | 정의 | 의미 |
|------|------|------|
| Feature map | $\phi(s) \in \mathbb{R}^d$ | 상태의 저차원 표현 |
| Linear FA | $V_\theta(s) = \theta^T \phi(s)$ | 매개변수 기반 근사 |
| Projected Bellman | $\Pi T^\pi V$ | span 내에서의 반복 |
| TD(0) 수렴점 | $\theta^*$ | Projected Bellman 고정점 |
| MSBE | $\|V_\theta - \Pi T^\pi V_\theta\|_{d^\pi}$ | Bellman 오차 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): TD(0) update 에서 feature $\phi(s_t)$ 가 등장하는 이유는?

<details>
<summary>해설</summary>

TD error $\delta_t$ 를 줄이기 위해 **오차에 정비례하여 feature direction 으로 조정**. Feature learning 의 기초 개념.

</details>

**문제 2** (심화): Projected Bellman $\Pi T^\pi$ 가 $\|\cdot\|_{d^\pi}$ 에서 contraction 임을 증명하라.

<details>
<summary>해설</summary>

Orthogonal projection 은 norm 축소 (non-expansion). 따라서:
$$\|\Pi T^\pi V - \Pi T^\pi V'\|_{d^\pi} \leq \|\gamma P^\pi(V - V')\|_{d^\pi} \leq \gamma\|V - V'\|_{d^\pi}$$

</details>

**문제 3** (논문 비평): 왜 off-policy 에서는 이 수렴을 보장하지 못하는가?

<details>
<summary>해설</summary>

Off-policy 에서 behavior policy 와 target policy 의 분포 mismatch 로 인해 ODE 분석이 깨짐. TD gradient 가 biased 되고, 최악의 경우 $A$ 가 singular → 발산 가능.

</details>

---

<div align="center">

[◀ 이전: Ch6-04. MDP 근사 — Approximation Error 와 Sample Complexity](../ch6-mdp-properties/04-approximation-sample-complexity.md) | [📚 README](../README.md) | [다음 ▶: 02. Deadly Triad](./02-deadly-triad.md)

</div>
