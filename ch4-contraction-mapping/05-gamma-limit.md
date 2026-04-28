# 05. $\gamma \to 1$ 에서의 한계 — Discounted 에서 Average-Reward 로의 전환

## 🎯 핵심 질문

- **$\gamma$ 가 1에 가까워질수록 무엇이 약해지는가?**
- Contraction rate $\gamma^k$ 는 $\gamma \to 1$ 일 때 왜 느려지는가?
- **$\gamma = 1$ 에서 contraction 이 깨지는 이유는?**
- Average-reward MDP, Cesàro 합, Blackwell optimality 는 무엇이고, 왜 필요한가?
- **$\gamma$ 선택의 trade-off** — 이론 vs 실무 사이의 긴장은?

---

## 🔍 왜 이 절이 중요한가

지금까지 **$\gamma < 1$** 을 당연한 가정으로 사용했습니다. Ch4-01 에서 Banach 정리도 $L < 1$ 을 요구합니다. 하지만 **왜 $\gamma = 1$ 이면 안 되는가?**

$$\text{If } \gamma = 1: \quad \|T^* V - T^* V'\|_\infty \leq 1 \cdot \|V - V'\|_\infty$$

이것은 nonexpansive 일뿐, contraction 이 아닙니다. 수렴이 보장되지 않습니다.

더 깊게는, **RL 의 문제 설정 자체가 변합니다**:
- $\gamma < 1$: **Discounted infinite-horizon** — 미래를 점점 할인
- $\gamma = 1$, episodic: **Finite-horizon** — episode 끝에서 멈춤
- $\gamma = 1$, continuing: **Average-reward** — 장기 평균 보상

이 절은 **Puterman 2005, Bertsekas 2019, Blackwell 1965** 의 결과를 소개하며, RL 의 **세 가지 설정**을 통합된 관점에서 이해합니다.

---

## 📐 수학적 선행 조건

### 필수
- Ch4-01~04: Contraction, Value Iteration
- 급수와 수렴 (analysis)

### 강화: Advanced Topics
- Cesàro average, Stolz-Cesàro theorem
- Spectral theory, ergodicity
- Blackwell optimal policies (Howard-Veinott canonical form)

---

## 📖 직관적 이해

### Convergence Rate Degradation

$$k \geq \frac{\log(1/\epsilon)}{\log(1/\gamma)} = \frac{\log(1/\epsilon)}{-\log \gamma}$$

**구체적 예**:
- $\gamma = 0.5$: $k \approx 20$ for $\epsilon = 10^{-6}$
- $\gamma = 0.9$: $k \approx 130$
- $\gamma = 0.99$: $k \approx 1380$
- $\gamma = 0.999$: $k \approx 13800$
- $\gamma \to 1$: $k \to \infty$

**기울기**: $1 - \gamma$ 가 작아질수록, 필요 iteration 은 **역수 속도**로 증가:

$$k \approx \frac{\log(1/\epsilon)}{1-\gamma}$$

### Three Problem Settings: Unified View

```
                 γ < 1, continuing
                (Discounted Infinite)
                     ↓
    ┌──────────────────────────────┐
    │ CONTRACTION MAPPING REGIME    │  ← Ch4: Banach theorem
    │ Linear Convergence: γ^k       │
    │ Value function: V(s)∈ℝ        │
    └──────────────────────────────┘
                     ↓
             γ → 1^- (soft regime)
            (contraction weakens)
                     ↙ ↘
        ┌────────────┘   └────────────┐
        │                             │
    Episodic              Average-Reward
   (γ=1, finite T)        (γ=1, ∞ horizon)
  "Terminal value"      "Cesàro average"
   V(terminal)=0       J = lim_T→∞ (1/T) Σ_t r_t
                      Blackwell optimal
```

---

## ✏️ 엄밀한 정의

### 정의 4.5.1 — Three Problem Classes

**클래스 1: Discounted Infinite-Horizon ($\gamma < 1$)**
$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right], \quad \text{converges absolutely since } \gamma < 1$$

**클래스 2: Episodic (Finite-Horizon, $\gamma = 1$)**

Trajectory length $T < \infty$ (e.g., maze reach terminal):
$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{T-1} r_t\right], \quad \text{finite sum}$$

Terminal state $s_T$ 에서 $V(s_T) = 0$ (관례).

**클래스 3: Average-Reward Continuing ($\gamma = 1$)**

No terminal state, but define:
$$J(\pi) := \limsup_{T \to \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=0}^{T-1} r_t\right]$$

또는 **Cesàro sense**:
$$\rho^\pi = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=0}^{T-1} r_t\right]$$

### 정의 4.5.2 — Cesàro Convergence

$(x_n)$ 이 **Cesàro 수렴**:
$$\lim_{N \to \infty} \frac{1}{N} \sum_{n=1}^{N} x_n = L$$

이는 $x_n \to L$ 보다 약한 조건. 진동하는 수열도 Cesàro 수렴 가능.

### 정의 4.5.3 — Blackwell Optimal Policy

정책 $\pi^*$ 이 **Blackwell optimal** ⟺
$$\exists \gamma_0 < 1: \forall \gamma \in (\gamma_0, 1), V_\gamma^{\pi^*} \geq V_\gamma^\pi \quad \forall \pi$$

(모든 충분히 큰 $\gamma$ 에 대해 최적)

또한 average-reward optimal: $\rho^{\pi^*} \geq \rho^\pi$ for all $\pi$.

---

## 🔬 정리와 증명

### 정리 4.5.1 — Convergence Rate as $\gamma \to 1$

Value Iteration 에 필요한 iteration 수:

$$k(\epsilon, \gamma) = \left\lceil \frac{\log(1/\epsilon)}{-\log \gamma} \right\rceil \approx \frac{\log(1/\epsilon)}{1-\gamma}$$

따라서:
$$\lim_{\gamma \to 1^-} k(\epsilon, \gamma) = \infty$$

**증명**:

Theorem 4.4.1 에서: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty \leq \gamma^k \frac{R_{\max}}{1-\gamma}$.

목표 오차 $\epsilon$ 달성:
$$\gamma^k \frac{R_{\max}}{1-\gamma} < \epsilon$$
$$\gamma^k < \epsilon (1-\gamma) / R_{\max}$$
$$k \log \gamma < \log(\epsilon(1-\gamma)/R_{\max})$$
$$k > \frac{\log(\epsilon(1-\gamma)/R_{\max})}{log \gamma} = \frac{\log(1/\epsilon) + \log((1-\gamma)/R_{\max})}{\log \gamma}$$

$\gamma \to 1^-$ 일 때, $\log \gamma \to 0^-$ 이므로 $k \to \infty$. 정확히:

$$k \sim \frac{\log(1/\epsilon)}{-(1-\gamma + O((1-\gamma)^2))} \sim \frac{\log(1/\epsilon)}{1-\gamma} \quad \square$$

### 정리 4.5.2 — Average-Reward Optimality

임의 MDP 에 대해:

1. **Average-reward optimal policy** $\pi^*$ 존재 (stationary)
2. Optimal average reward:
$$\rho^* = \max_\pi \limsup_{T \to \infty} \frac{1}{T} \mathbb{E}_\pi\left[\sum_{t=0}^{T-1} r_t\right]$$

3. Differential value function $W^\pi(s)$ 가 존재 (relative value):
$$W^\pi(s) = \lim_{T \to \infty} \left(V_T^\pi(s) - T \rho^\pi\right)$$

여기서 $V_T^\pi(s) = \mathbb{E}[\sum_{t=0}^{T-1} r_t | s_0 = s]$.

4. Bellman optimality equation:
$$\rho^* + W^*(s) = \max_a \left[r(s, a) + \sum_{s'} P(s'|s,a) W^*(s')\right]$$

(부차선택은 $\gamma \to 1$ 극한과 일치)

**증명 스케치** (Blackwell 1965):

- $\gamma < 1$ 에서 optimal policy $\pi_\gamma^*$ 존재
- $\gamma \to 1$ 극한에서 subsequence 수렴 → average-reward optimal policy
- Differential value equation 은 discount equation 의 극한

자세한 증명은 Puterman (2005, Ch 8) 참고.

### 정리 4.5.3 — Relation to Discounted and Episodic

**Episodic** (terminal state):
- Natural: $\gamma = 1$ (미래 할인 없음)
- Terminal $s_T$: $V(s_T) = 0$ (또는 terminal reward)
- Episode 가 끝날 때까지 누적 보상: $\sum_{t=0}^{T-1} r_t$

**Discounted** ($\gamma < 1$):
- Unbounded horizon 을 bounded 로 만듦
- 원래 episodic 문제를 $\gamma = 1$ 에서 "soft" truncation

**Average-Reward** ($\gamma = 1$, continuing):
- Episode 없음, never terminate
- Long-run 평균만 중요
- "균형 상태" (stationary distribution) 에서의 성능

---

## 💻 NumPy 구현 검증

### 실험 1 — Iteration Count vs $\gamma$

```python
import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-6
gamma_range = np.linspace(0.5, 0.999, 50)
k_needed = []

for gamma in gamma_range:
    # k_theory = ceil(log(epsilon*(1-gamma)) / log(gamma))
    k = np.ceil(np.log(epsilon * (1 - gamma)) / np.log(gamma))
    k_needed.append(k)

k_needed = np.array(k_needed)

# Compare with asymptotic: k ~ log(1/eps) / (1-gamma)
asymptotic = np.log(1/epsilon) / (1 - gamma_range)

plt.figure(figsize=(10, 6))
plt.semilogy(gamma_range, k_needed, 'b-', linewidth=2, label='Exact: $\log(\epsilon(1-\gamma))/\log(\gamma)$')
plt.semilogy(gamma_range, asymptotic, 'r--', linewidth=2, label='Asymptotic: $\log(1/\epsilon)/(1-\gamma)$')
plt.xlabel('$\gamma$')
plt.ylabel('Number of iterations k (log scale)')
plt.legend()
plt.grid(True)
plt.title(f'Value Iteration Complexity: $\\epsilon={epsilon:.0e}$')
plt.tight_layout()
plt.savefig('/tmp/gamma_iteration_count.png', dpi=120)

# Print some values
print("γ       | k (exact) | Asymptotic")
for i in [0, len(gamma_range)//4, len(gamma_range)//2, 3*len(gamma_range)//4, -1]:
    print(f"{gamma_range[i]:.3f}   | {k_needed[i]:6.0f}   | {asymptotic[i]:6.0f}")
```

### 실험 2 — Episodic vs Continuing Reward

```python
# Small 3-state MDP
S = 3
gamma_values = [0.5, 0.9, 0.99, 1.0]

# Fixed small reward
r = np.array([1.0, 0.5, -0.5])
P = np.array([
    [0.1, 0.6, 0.3],
    [0.2, 0.5, 0.3],
    [0.0, 0.0, 1.0]  # absorbing state
])

results = {}

for gamma in gamma_values[:-1]:  # Skip gamma=1 for now (discounted)
    V = np.zeros(S)
    V_prev = V.copy()
    
    for _ in range(300):
        V = r + gamma * (P @ V)
    
    results[gamma] = V.copy()
    print(f"γ={gamma}: V = {V}")

# Episodic: finite horizon T
print("\nEpisodic (finite horizon T):")
T = 10
V_epi = np.zeros(S)
for t in range(T):
    V_epi = r + (P @ V_epi)  # (γ=1 implicitly)
    if np.allclose(P[-1], [0, 0, 1]):  # absorbing, likely reached by T
        V_epi[-1] = 0

print(f"T={T}: V = {V_epi}")
```

### 실험 3 — Cesàro Convergence (Average-Reward)

```python
# Verify Cesàro convergence in average-reward setting
# Simple 2-state MDP with oscillating reward

S = 2
P = np.array([
    [0.5, 0.5],
    [0.5, 0.5]
])

# Rewards that cause oscillation
r = np.array([1.0, -1.0])

# Simulate trajectory (many steps)
s = 0
n_steps = 10000
rewards_traj = []

for t in range(n_steps):
    reward = r[s]
    rewards_traj.append(reward)
    s = np.random.choice(2, p=P[s])

rewards_traj = np.array(rewards_traj)

# Compute cumulative and average
cumsum = np.cumsum(rewards_traj)
time_steps = np.arange(1, n_steps+1)
cesaro_avg = cumsum / time_steps

plt.figure(figsize=(10, 6))
plt.plot(cesaro_avg[-1000:], 'b-', alpha=0.7, linewidth=1)
plt.axhline(np.mean(r), color='r', linestyle='--', label=f'Limiting average: {np.mean(r):.3f}')
plt.xlabel('Time t')
plt.ylabel('Cesàro Average: (1/t) $\sum_{{k=0}}^{{t-1}} r_k$')
plt.legend()
plt.grid(True)
plt.title('Average-Reward Convergence (Cesàro)')
plt.tight_layout()
plt.savefig('/tmp/cesaro_average.png', dpi=120)

print(f"Final Cesàro average: {cesaro_avg[-1]:.6f}")
print(f"Theoretical limit: {np.mean(r):.6f}")
```

---

## 🔗 후속 레포와의 연결

- **Ch5**: Policy Iteration, GPI (discounted 중심)
- **Model-Free RL Deep Dive**: Q-learning, actor-critic (discounted, $\gamma < 0.99$ 보통)
- **Advanced**: Average-reward algorithms, Blackwell optimal control

---

## ⚖️ 가정과 한계

| 설정 | $\gamma$ | 특징 | 수렴 | 사용처 |
|------|---------|------|------|--------|
| Discounted | $\gamma < 1$ | Bounded return | Linear ($\gamma^k$) | Robotics, standard RL |
| Episodic | $\gamma = 1$ | Finite horizon $T$ | Finite steps | Games, navigation |
| Average-Reward | $\gamma = 1$ | Continuing, Cesàro | Blackwell optimal | Manufacturing, operations |

---

## 📌 핵심 정리

$$\boxed{k(\epsilon, \gamma) \sim \frac{\log(1/\epsilon)}{1-\gamma} \to \infty \text{ as } \gamma \to 1^-}$$

**Three Problem Classes**:
1. **Discounted** ($\gamma < 1$): Contraction, linear convergence
2. **Episodic** ($\gamma = 1$, finite $T$): Natural, finite sum
3. **Average-Reward** ($\gamma = 1$, Cesàro): Blackwell optimal, more complex

---

## 🤔 생각해볼 문제

**문제 1**: 왜 average-reward 에서 "differential value function" $W(s)$ 가 필요한가?

<details>
<summary>해설</summary>

$\rho^*$ (단일 scalar) 로는 상태 간 정책 비교 불가. 각 상태의 "상대적 가치" $W(s) = V_∞(s) - \rho^* \cdot \infty$ 를 정의. Bellman: $\rho^* + W^*(s) = \max_a[\cdots]$. $\square$

</details>

**문제 2**: Blackwell optimal policy 는 모든 MDP 에 존재하는가? 증명 sketch?

<details>
<summary>해설</summary>

Puterman Thm: stationary Blackwell optimal policy 존재. 증명은 subsequence compactness: $\pi_\gamma$ (각 $\gamma$ 에서 optimal) 은 유한 정책 중에서, subsequence 수렴 → average-reward optimal. $\square$

</details>

**문제 3**: 실무에서 $\gamma$ 를 어떻게 선택하는가? $\gamma = 0.99$ vs $0.999$ vs $0.9999$?

<details>
<summary>해설</summary>

Trade-off:
- **더 큰 $\gamma$**: 더 "먼 미래" 고려, 수학적으로 더 어려움 ($k \to \infty$)
- **작은 $\gamma$**: 빠른 수렴, 단기 보상 편향
- **관례**: Robotics $\gamma=0.99$, Atari $\gamma=0.99$, continuous control $\gamma=0.95$
- **Guideline**: 문제의 time scale 에 따라. Episode 길이가 $T$ 면, $\gamma \approx (1 - 1/T)$. $\square$

</details>

---

<div align="center">

[◀ 이전: 04. Value Iteration 수렴 보장](./04-value-iteration-convergence.md) | [📚 README](../README.md) | [다음 ▶: Ch5-01. Policy Evaluation](../ch5-dp-algorithms/01-policy-evaluation.md)

</div>
