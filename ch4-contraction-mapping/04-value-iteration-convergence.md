# 04. Value Iteration 수렴 보장 — 정지 기준과 선형 수렴의 실제 응용

## 🎯 핵심 질문

- **Value Iteration 의 수렴을 어떻게 보장하고 정량화하는가?**
- **정지 기준 (Stopping Criterion)** $\|V_{k+1} - V_k\|_\infty < \epsilon (1-\gamma)/\gamma$ 는 어디서 나오는가?
- **A Priori bound** vs **A Posteriori bound** 의 차이는?
- 실제로 $\gamma = 0.9$, $\epsilon = 10^{-6}$ 일 때 몇 iteration 이 필요한가?

---

## 🔍 왜 이 정리가 실무적 중요성을 갖는가

Ch4-01 에서 Banach Fixed Point Theorem 을 증명했고, Ch4-03 에서 $T^*$ 가 $\gamma$-contraction 임을 보였습니다. 이제 그것을 **실제 알고리즘에 적용** 합니다:

$$V_{k+1} = T^* V_k$$

의 수렴 속도와 **정지 기준을 유도** 하는 것이 목표입니다. 이것이:
- **이론과 실제의 다리**
- 알고리즘 설계의 핵심 (언제 멈출 것인가?)
- 함수근사 설정에서 발생하는 문제 (contraction 상실) 의 이해로 이어집니다.

---

## 📐 수학적 선행 조건

### 필수
- Ch4-01: Banach 정리, A Priori/Posteriori bound
- Ch4-03: $T^*$ 의 contraction 증명
- 삼각부등식, Telescoping sum

### 선택: Advanced
- Error analysis in numerical methods

---

## 📖 직관적 이해

### 수렴 속도: Exponential Decay

Contraction constant $\gamma < 1$ 이면:

$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$$

이는 **선형 수렴 (Linear Convergence)** 의 정의: 오차가 매 iteration 마다 **고정 배수 $\gamma$ 로 감소**.

예: $\gamma = 0.9$, $\|V_0 - V^*\|_\infty = 10$ 이면:
- $k = 10$: 오차 $\approx 10 \times 0.9^{10} \approx 3.5$
- $k = 100$: 오차 $\approx 10 \times 0.9^{100} \approx 2.7 \times 10^{-5}$

### Stopping Criterion: A Posteriori Bound

A Priori bound 는 $V^*$ 를 알아야 하는데, 우리는 $V^*$ 를 모릅니다. 대신:

$$\|V_k - V^*\|_\infty \leq \frac{\gamma}{1-\gamma} \|V_k - V_{k-1}\|_\infty$$

이것은 **계산 중 관찰 가능한 양** ($\|V_k - V_{k-1}\|_\infty$) 만으로 오차를 bound 합니다.

### 구체적 예시

$\gamma = 0.9$, $\epsilon = 10^{-6}$ 으로 수렴시키려면:

$$\|V_k - V^*\|_\infty < \epsilon$$

정지 기준:
$$\|V_k - V_{k-1}\|_\infty < \epsilon \frac{1-\gamma}{\gamma} = \epsilon \frac{0.1}{0.9} \approx 1.11 \times 10^{-7}$$

이를 만족하는 최소 $k$ 는 약 $130$ iteration.

---

## ✏️ 엄밀한 정의

### 정의 4.4.1 — Value Iteration Sequence

초기값 $V_0 \in B(\mathcal{S})$ 에서 시작하여:

$$V_{k+1} = T^* V_k = \max_a \left[r(s, a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right]$$

로 정의된 수열 $(V_k)_{k \geq 0}$.

### 정의 4.4.2 — Convergence and Errors

- **True error (A Priori)**: $e_k^{\text{true}} := \|V_k - V^*\|_\infty$
- **Observable error (A Posteriori)**: $\delta_k := \|V_k - V_{k-1}\|_\infty$
- **Stopping threshold**: $\epsilon > 0$ (desired accuracy)

---

## 🔬 정리와 증명

### 정리 4.4.1 — Value Iteration 수렴 (A Priori Bound)

$T^*$ 가 $\gamma$-contraction 이므로 (Theorem 4.3.1), 임의 초기값 $V_0$ 에서:

$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$$

더 구체적으로, $V^*$ 에 대한 상한을 알면:
$$\|V^*\|_\infty \leq \frac{R_{\max}}{1-\gamma}$$

따라서:
$$\|V_k - V^*\|_\infty \leq \gamma^k \left(\|V_0\|_\infty + \frac{R_{\max}}{1-\gamma}\right)$$

**증명**: Banach Fixed Point Theorem (Ch4-01, Theorem 4.1) 의 직접 적용. $\square$

### 정리 4.4.2 — Stopping Criterion (A Posteriori Bound)

$V_k$ 에서 다음 step 으로 진행할지 멈출지 결정하기 위해, Banach 정리의 A Posteriori bound:

$$\|V_k - V^*\|_\infty \leq \frac{\gamma}{1-\gamma} \|V_k - V_{k-1}\|_\infty$$

**증명**:

Telescoping sum (Ch4-01 참고):
$$\|V_k - V^*\|_\infty = \left\|\sum_{j=k}^{\infty} (V_j - V_{j+1})\right\|_\infty$$
$$\leq \sum_{j=k}^{\infty} \|V_j - V_{j+1}\|_\infty$$

$T^*$ 의 contraction 으로부터:
$$\|V_{j+1} - V_j\| = \|T^*(V_j) - T^*(V_{j-1})\| \leq \gamma \|V_j - V_{j-1}\|$$

따라서:
$$\|V_j - V_{j+1}\| \leq \gamma^{j-k+1} \|V_k - V_{k-1}\|$$

합산:
$$\|V_k - V^*\|_\infty \leq \sum_{j=k}^{\infty} \gamma^{j-k+1} \|V_k - V_{k-1}\|_\infty$$
$$= \|V_k - V_{k-1}\|_\infty \sum_{i=1}^{\infty} \gamma^i$$
$$= \|V_k - V_{k-1}\|_\infty \cdot \frac{\gamma}{1-\gamma} \quad \square$$

### 정리 4.4.3 — Practical Stopping Rule

원하는 정확도 $\epsilon$ 에 대해:

$$\|V_k - V_{k-1}\|_\infty < \epsilon \frac{1-\gamma}{\gamma}$$

이면 $\|V_k - V^*\|_\infty < \epsilon$ 를 **보장**.

**적용**: 
```python
while ||V_new - V|| > epsilon * (1 - gamma) / gamma:
    V = V_new
    V_new = T*(V)
```

### 정리 4.4.4 — 필요 Iteration 수 (Complexity)

목표 오차 $\epsilon$ 에 도달하려면:

$$k \geq \frac{\log(\epsilon \cdot (1-\gamma) / \|V_0 - V^*\|_\infty)}{\log \gamma}$$

또는 대략:
$$k \approx \frac{\log(1/\epsilon)}{\log(1/\gamma)} = \frac{\log(1/\epsilon)}{-\log \gamma}$$

$\gamma = 1 - 1/n$ (soft regime) 일 때:
$$k \approx n \log(1/\epsilon)$$

**예시**:
- $\gamma = 0.9$, $\epsilon = 10^{-6}$: $k \approx 130$
- $\gamma = 0.99$, $\epsilon = 10^{-6}$: $k \approx 1380$
- $\gamma = 0.999$, $\epsilon = 10^{-6}$: $k \approx 13800$

---

## 💻 NumPy 구현 검증

### 실험 1 — 5×5 Gridworld VI with Stopping Criterion

```python
import numpy as np
import matplotlib.pyplot as plt

# 5×5 gridworld
nx, ny = 5, 5
S = nx * ny
A = 4
gamma = 0.9
epsilon = 1e-6

np.random.seed(42)
rewards = np.random.randn(S, A)
P = np.zeros((S, A, S))
for s in range(S):
    for a in range(A):
        x, y = s // ny, s % ny
        if a == 0: nx_new, ny_new = max(0, x-1), y
        elif a == 1: nx_new, ny_new = min(nx-1, x+1), y
        elif a == 2: nx_new, ny_new = x, max(0, y-1)
        else: nx_new, ny_new = x, min(ny-1, y+1)
        s_prime = nx_new * ny + ny_new
        P[s, a, s_prime] += 0.95
        for s_rand in range(S):
            P[s, a, s_rand] += 0.05 / S

# Value Iteration with A Posteriori stopping
V = np.zeros(S)
V_old = V.copy()
errors_obs = []  # Observable: ||V_k - V_{k-1}||
errors_true = []  # True: ||V_k - V*||
k = 0
stop_threshold = epsilon * (1 - gamma) / gamma

while True:
    Q = rewards + gamma * np.einsum('sap,p->sa', P, V)
    V = Q.max(axis=1)
    delta = np.linalg.norm(V - V_old, ord=np.inf)
    errors_obs.append(delta)
    
    # True error (approximate V* after many iterations)
    if k == 0:
        V_star_approx = V.copy()
    elif k > 500:
        break
    
    if delta < stop_threshold:
        print(f"Stopped at iteration {len(errors_obs)}")
        break
    
    V_old = V.copy()
    k += 1

errors_obs = np.array(errors_obs)

# Theory: A Posteriori Bound
aposteriori_bound = gamma / (1 - gamma) * errors_obs

print(f"γ = {gamma}, ε = {epsilon:.0e}")
print(f"Stopping threshold: ε·(1-γ)/γ = {stop_threshold:.2e}")
print(f"Final ||V_k - V_{k-1}||_∞ = {errors_obs[-1]:.2e}")
print(f"Predicted true error bound: {aposteriori_bound[-1]:.2e}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(errors_obs, 'b-', label='$\|V_k - V_{k-1}\|_\\infty$', linewidth=2)
plt.axhline(stop_threshold, color='r', linestyle='--', label=f'Threshold $\\epsilon \\frac{{1-\\gamma}}{{\\gamma}}$')
plt.xlabel('Iteration k')
plt.ylabel('Observable Error (log)')
plt.legend()
plt.grid(True)
plt.title('Value Iteration: Observable Error')

plt.subplot(1, 2, 2)
plt.semilogy(aposteriori_bound, 'g-', label='$\\frac{\\gamma}{1-\\gamma} \|V_k - V_{k-1}\|_\\infty$', linewidth=2)
plt.axhline(epsilon, color='r', linestyle='--', label=f'Tolerance ε')
plt.xlabel('Iteration k')
plt.ylabel('True Error Bound (log)')
plt.legend()
plt.grid(True)
plt.title('Value Iteration: A Posteriori Bound')

plt.tight_layout()
plt.savefig('/tmp/vi_stopping_criterion.png', dpi=120)
```

### 실험 2 — 감마 값의 영향

```python
# Compare different gamma values
gammas = [0.5, 0.9, 0.99]
epsilons = 1e-6
colors = ['r', 'b', 'g']

plt.figure(figsize=(10, 5))

for gamma, color in zip(gammas, colors):
    # Theory: k ≥ log(ε(1-γ)) / log(γ)
    k_theory = np.ceil(np.log(epsilon * (1 - gamma)) / np.log(gamma))
    
    # Simulate a few iterations
    V = np.zeros(S)
    V_old = V.copy()
    errors = []
    
    for iter in range(int(k_theory) + 20):
        Q = rewards + gamma * np.einsum('sap,p->sa', P, V)
        V = Q.max(axis=1)
        delta = np.linalg.norm(V - V_old, ord=np.inf)
        errors.append(delta)
        V_old = V.copy()
    
    threshold = epsilon * (1 - gamma) / gamma
    plt.semilogy(errors, color=color, label=f'γ={gamma}, k_theory≈{k_theory:.0f}', linewidth=2)
    plt.axhline(threshold, color=color, linestyle='--', alpha=0.5)

plt.xlabel('Iteration k')
plt.ylabel('$\|V_k - V_{k-1}\|_\\infty$ (log)')
plt.legend()
plt.grid(True)
plt.title(f'Effect of γ on Convergence (ε={epsilon:.0e})')
plt.tight_layout()
plt.savefig('/tmp/gamma_effect.png', dpi=120)
```

---

## 🔗 후속 레포와의 연결

- **Ch5-01**: Policy Evaluation 의 정지 기준 (유사)
- **Ch5-02**: Policy Iteration 전체
- **Advanced RL**: Function approximation 에서 contraction 상실

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Finite state/action | Infinite: function approximation |
| Exact Bellman backup | Approximate (function approx): 오차 누적 |
| $\gamma < 1$ strictly | Average-reward: 다른 수렴 기준 필요 |
| Deterministic policy 최적 | Stochastic optimal policy: 약간 수정 |

---

## 📌 핵심 정리

$$\boxed{\text{A Priori: } \|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty}$$

$$\boxed{\text{A Posteriori: } \|V_k - V^*\|_\infty \leq \frac{\gamma}{1-\gamma} \|V_k - V_{k-1}\|_\infty}$$

$$\boxed{\text{Stopping: } \|V_k - V_{k-1}\|_\infty < \epsilon \frac{1-\gamma}{\gamma} \Rightarrow \|V_k - V^*\|_\infty < \epsilon}$$

---

## 🤔 생각해볼 문제

**문제 1**: A Posteriori bound 에서 계수 $\gamma/(1-\gamma)$ 는 어디서 나오는가?

<details>
<summary>해설</summary>

Telescoping: $V_k - V^* = \sum_{j=k}^∞ (V_j - V_{j+1})$ 이고, 각 차이가 $\gamma$ 배 감소하므로, 무한 합 $\sum γ^i = γ/(1-γ)$. $\square$

</details>

**문제 2**: 왜 $\epsilon(1-\gamma)/\gamma$ 이지 다른 상수가 아닌가?

<details>
<summary>해설</summary>

역으로 유도: $\|V_k - V^*\|_∞ < ε$ 를 원하므로, $\frac{γ}{1-γ} \|V_k - V_{k-1}\|_∞ < ε$ ⟹ $\|V_k - V_{k-1}\|_∞ < ε(1-γ)/γ$. $\square$

</details>

**문제 3**: $\gamma \to 1$ 일 때 필요 iteration 수는?

<details>
<summary>해설</summary>

$k \approx \log(1/ε) / (-\log γ) = \log(1/ε) / (1-γ + O((1-γ)^2)) \approx \log(1/ε) / (1-γ)$ 로 **발산**. 이것이 $γ=1$ (average-reward) 로 전환하는 이유. $\square$

</details>

---

<div align="center">

[◀ 이전: 03. $T^*$ 가 $\gamma$-Contraction](./03-tstar-contraction.md) | [📚 README](../README.md) | [다음 ▶: 05. $\gamma \to 1$ 에서의 한계](./05-gamma-limit.md)

</div>
