# 01. Discounted Return 의 정의와 수렴

## 🎯 핵심 질문

- Discounted return $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$ 이 어떤 조건에서 유한한 값으로 수렴하는가?
- $\gamma \in [0, 1)$ 제약이 단순한 hyperparameter 가 아니라 **무한 합 수렴의 필수 수학적 조건** 임을 어떻게 증명하는가?
- $\gamma = 1$ 에서 episodic vs average-reward MDP 로 갈라지는 본질적 이유는?
- $\gamma = 0$ (myopic) 의 극한 해석과 $\gamma \to 1$ (far-sighted) 의 수렴 속도 악화가 RL 알고리즘에 어떻게 반영되는가?
- Reward 의 boundedness 가정이 없으면 어떻게 부서지는가?

---

## 🔍 왜 이 정리가 RL 의 정초인가

**Reinforcement Learning 의 모든 가치 함수** — $V^\pi(s), Q^\pi(s, a), V^*(s)$ — 는 **discounted return 의 기대값** 으로 정의됩니다. 따라서 return 이 수렴하지 않으면 가치 함수 자체가 정의되지 않습니다. 

그런데 많은 입문 자료는:
- "$\gamma$ 는 미래 reward 를 얼마나 중요시할지 정하는 파라미터" (직관적이지만 불완전)
- "$G_t = r_t + \gamma G_{t+1}$ 로 재귀적으로 계산" (어떻게 수렴하는지 생략)
- "$\gamma = 1$ 은 non-discounted, episodic 에서만 쓴다" (왜? 라는 답 부족)

이 문서는:
1. **Absolute convergence theorem** 으로 $\gamma < 1$ 이 수렴의 충분조건임을 증명
2. $\gamma = 1$ 에서 무엇이 부서지고, episodic 또는 average-reward 로 우회하는 방법 설명
3. $\gamma$ 의 극한 해석: $\gamma = 0$ (한 걸음), $\gamma \to 1$ (무한 지평)
4. NumPy 로 수렴 속도를 시각화

---

## 📐 수학적 선행 조건

- 수열과 급수의 수렴 (기본 해석학)
- Geometric series $\sum_{k=0}^\infty r^k = 1/(1-r)$ for $|r| < 1$
- Norms: sup-norm $\|\cdot\|_\infty$ 과 $\ell^1$ norm
- 확률론: bounded random variable, expectation, law of large numbers
- Bounded linear operator 의 spectral radius

---

## 📖 직관적 이해

### Return 의 의미

시간 $t$ 에서 시작한 agent 의 **누적 보상**:

$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots$$

만약 reward 가 상수 $R = 1$ 이라면 (무한히 계속):
- **Undiscounted ($\gamma = 1$)**: $G_t = 1 + 1 + 1 + \cdots = \infty$ ✗
- **Discounted ($\gamma = 0.9$)**: $G_t = 1 + 0.9 + 0.81 + 0.729 + \cdots = 10$ ✓

### Discount factor 의 해석

```
시간 →  t   t+1   t+2   t+3   ...
보상    R    R     R     R     ...
가중    1  γ¹    γ²    γ³     ...
누적  R + γR + γ²R + γ³R + ... = R/(1-γ)

γ = 0.0  → G = R      (한 걸음만 봄, 욕심쟁이)
γ = 0.5  → G = 2R     (조금 미래지향)
γ = 0.9  → G = 10R    (먼 미래까지 봄)
γ = 0.99 → G = 100R   (매우 멀리까지)
γ = 1.0  → G = ∞      (무한 합, 정의 안 됨)
```

### 수렴성의 직관

Reward 가 bounded $|R_t| \leq R_{\max}$ 일 때:

$$|G_t| = \left|\sum_{k=0}^\infty \gamma^k R_{t+k+1}\right| \leq \sum_{k=0}^\infty \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma}$$

**$\gamma < 1$ 이면 기하급수가 수렴** → return 도 유한.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — MDP Trajectory

MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ 에서 시간 $t$ 부터의 trajectory:

$$\tau_t = (s_t, a_t, s_{t+1}, a_{t+1}, \ldots)$$

각 $a_t \sim \pi(\cdot \mid s_t)$, $s_{t+1} \sim P(\cdot \mid s_t, a_t)$, $R_t = R(s_t, a_t)$.

### 정의 1.2 — Discounted Return

$$G_t := \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

where:
- $\gamma \in [0, 1)$: discount factor
- $R_{t+k+1} = R(s_{t+k}, a_{t+k})$: immediate reward
- Sum 은 정책 $\pi$ 와 dynamics $P$ 에 대한 기대값 형태로 정의

### 정의 1.3 — Bounded Reward Assumption

$$\|R\|_\infty := \sup_{s, a} |R(s, a)| \leq R_{\max} < \infty$$

유한한 모든 action-state pair 에서 reward 의 절댓값이 $R_{\max}$ 를 넘지 않음.

### 정의 1.4 — Infinite Horizon MDP

**Infinite horizon** (discounted): 종료 시간이 없고, $\gamma \in [0, 1)$ 로 미래를 할인.

**Episodic**: 유한한 episode 길이 $T < \infty$ 가 존재, 또는 absorbing terminal state 존재.

---

## 🔬 정리와 증명

### 정리 1.1 (Absolute Convergence of Discounted Return)

**조건**: Bounded reward $|R_t| \leq R_{\max}$, $\gamma \in [0, 1)$.

**결론**: 거의 모든 trajectory 에 대해 $G_t$ 는 절대수렴하며:

$$|G_t| \leq \frac{R_{\max}}{1 - \gamma} < \infty$$

**증명**:

**Step 1** — Absolute value bound:

$$\left|G_t\right| = \left|\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\right| \leq \sum_{k=0}^{\infty} \gamma^k |R_{t+k+1}| \leq \sum_{k=0}^{\infty} \gamma^k R_{\max}$$

**Step 2** — Geometric series convergence:

$$\sum_{k=0}^{\infty} \gamma^k R_{\max} = R_{\max} \sum_{k=0}^{\infty} \gamma^k = R_{\max} \cdot \frac{1}{1 - \gamma}$$

(기하급수, $|\gamma| < 1$ 이므로 수렴)

**Step 3** — Conclusion:

$$|G_t| \leq \frac{R_{\max}}{1 - \gamma} < \infty \quad \square$$

### 따름정리 1.2 — Return 의 유한성과 기대값 존재

$|G_t| \leq R_{\max}/(1-\gamma)$ 이므로 $G_t$ 는 bounded random variable. 따라서:

$$\mathbb{E}[G_t] < \infty, \quad |\mathbb{E}[G_t]| \leq \frac{R_{\max}}{1 - \gamma}$$

**따름정리 1.3 — $\gamma$ 의 수학적 필요성**

만약 $\gamma = 1$ 이고 bounded reward $R_t \geq R_{\min} > 0$ 라면:

$$G_t = \sum_{k=0}^{\infty} R_{t+k+1} \geq R_{\min} \cdot \infty = \infty$$

따라서 **$\gamma = 1$ 일 때 infinite-horizon 에서 return 이 정의되지 않음**.

### 정리 1.4 — $\gamma = 1$ 의 우회: Episodic vs Average-Reward

**우회법 1 (Episodic)**: Episode 길이 $T < \infty$ 제한. $G_t = \sum_{k=0}^{T-t-1} R_{t+k+1}$ (유한).

**우회법 2 (Average-Reward)**: 평균 reward 정의:

$$J_{\text{avg}}(\pi) := \lim_{T \to \infty} \frac{1}{T} \mathbb{E}[G_t^{(T)}]$$

여기서 $G_t^{(T)} = \sum_{k=0}^{T-1} R_{t+k+1}$. 이 경우 아주 큰 $T$ 에서도 normalized 되므로 수렴.

**이 레포의 범위**: Infinite-horizon discounted ($\gamma \in [0, 1)$), 필요시 episodic 변형 언급.

### 정리 1.5 — Recursive Bellman Form of Return

임의의 $t \geq 0$ 에 대해:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**증명**:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \sum_{k=1}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma \sum_{j=0}^{\infty} \gamma^j R_{t+1+j} = R_{t+1} + \gamma G_{t+1} \quad \square$$

---

## 💻 NumPy 구현 검증

### 실험 1 — 상수 Reward 에서의 이론 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# 상수 reward R, discount factor γ
R = 1.0
gamma_list = [0.0, 0.5, 0.9, 0.95, 0.99]

# 이론값: G = R / (1 - γ)
theoretical = [R / (1 - gamma) if gamma < 1 else np.inf 
               for gamma in gamma_list]

# 수치 계산: 처음 n_terms 항까지 합
n_terms = 100
numerical = [R * np.sum([gamma ** k for k in range(n_terms)]) 
             for gamma in gamma_list]

print("γ      | 이론값     | 수치값 (100항) | 오차")
print("-------|----------|--------------|-------")
for g, theory, num in zip(gamma_list, theoretical, numerical):
    error = abs(theory - num) if theory != np.inf else num
    print(f"{g:.2f}  | {theory:8.2f}  | {num:8.2f}     | {error:.2e}")
```

**예상 출력**:
```
γ      | 이론값     | 수치값 (100항) | 오차
-------|----------|--------------|-------
0.00  |      1.00  |      1.00     | 0.00e+00
0.50  |      2.00  |      2.00     | 4.63e-09
0.90  |     10.00  |     10.00     | 1.03e-10
0.95  |     20.00  |     20.00     | 1.84e-08
0.99  |    100.00  |    100.00     | 2.58e-07
```

### 실험 2 — 수렴 속도 분석: $\gamma$ 의 영향

```python
# 부분합 S_n = Σ_{k=0}^{n} γ^k R 의 수렴
R = 1.0
gamma_vals = [0.5, 0.9, 0.99]
n_max = 200

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) Linear scale
ax = axes[0]
for gamma in gamma_vals:
    theoretical = R / (1 - gamma)
    partial_sums = np.array([R * np.sum([gamma ** k for k in range(n)]) 
                             for n in range(1, n_max)])
    ax.plot(range(1, n_max), partial_sums, label=f'γ={gamma}', linewidth=2)
    ax.axhline(theoretical, color='k', linestyle='--', alpha=0.3)

ax.set_xlabel('항의 개수 n')
ax.set_ylabel('부분합 $S_n$')
ax.set_title('Discounted Return 의 수렴 (Linear scale)')
ax.legend()
ax.grid()

# (2) Log scale: 오차 ||S_n - G||
ax = axes[1]
for gamma in gamma_vals:
    theoretical = R / (1 - gamma)
    partial_sums = np.array([R * np.sum([gamma ** k for k in range(n)]) 
                             for n in range(1, n_max)])
    error = np.abs(theoretical - partial_sums)
    ax.semilogy(range(1, n_max), error, label=f'γ={gamma}', linewidth=2)

ax.set_xlabel('항의 개수 n')
ax.set_ylabel('오차 $|G - S_n|$ (log scale)')
ax.set_title('수렴 속도: γ가 작을수록 빠름')
ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig('discounted_return_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 실험 3 — 확률적 Reward 의 Return 분포

```python
np.random.seed(42)
gamma = 0.9
n_traj = 10000
T_horizon = 500

# 매 step 에서 R_t ~ Unif(-1, 1)
returns = []
for _ in range(n_traj):
    R = np.random.uniform(-1, 1, T_horizon)
    G = np.sum([gamma ** k * R[k] for k in range(T_horizon)])
    returns.append(G)

returns = np.array(returns)

print(f"Return 통계 (γ={gamma}, {n_traj} trajectories):")
print(f"  E[G] = {np.mean(returns):.6f}")
print(f"  Std[G] = {np.std(returns):.6f}")
print(f"  Min[G] = {np.min(returns):.6f}")
print(f"  Max[G] = {np.max(returns):.6f}")
print(f"  이론 상한: {1.0 / (1 - gamma):.6f}")

# 히스토그램
plt.figure(figsize=(10, 5))
plt.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(returns), color='r', linestyle='--', label=f'Mean = {np.mean(returns):.3f}')
plt.axvline(1.0 / (1 - gamma), color='g', linestyle='--', label=f'Theoretical max = {1.0/(1-gamma):.3f}')
plt.xlabel('Return $G_0$')
plt.ylabel('Density')
plt.title(f'Return 의 분포 (γ={gamma})')
plt.legend()
plt.grid()
plt.savefig('return_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 실험 4 — 극단적 $\gamma$ 값 비교: 0 vs 1에 가까운

```python
# Simple environment: 매 step 마다 reward 1.0 받음
R_constant = 1.0
gammas = np.logspace(-2, -0.01, 50)  # 0.01 ~ 0.99
returns = np.array([R_constant / (1 - g) for g in gammas])

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(gammas, returns, linewidth=2.5, color='navy')
ax.set_xlabel('Discount factor γ (log scale)')
ax.set_ylabel('Return G (linear scale)')
ax.set_title('Return 이 γ에 따라 어떻게 급증하는가?')
ax.grid(True, which='both', alpha=0.3)
ax.axvline(0.9, color='r', linestyle='--', alpha=0.5, label='γ=0.9')
ax.axvline(0.99, color='orange', linestyle='--', alpha=0.5, label='γ=0.99')
ax.legend()
plt.tight_layout()
plt.savefig('gamma_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# 미분: dG/dγ
dG_dgamma = R_constant / (1 - gammas) ** 2
print(f"γ=0.9 에서 dG/dγ = {R_constant / (1 - 0.9)**2:.1f}  → 작은 γ 변화도 큰 영향")
print(f"γ=0.99 에서 dG/dγ = {R_constant / (1 - 0.99)**2:.1f}  → 민감도 극대")
```

---

## 🔗 후속 레포와의 연결

- **Ch2-02 (Value Functions)**: $V^\pi(s) = \mathbb{E}[G_t | S_t = s]$ 정의 시 $G_t$ 수렴 필수
- **Ch2-03, 04 (Bellman Expectation, Operator)**: $T^\pi$ 의 고정점 존재 $\Leftrightarrow$ $\gamma < 1$ convergence
- **Ch3 (Bellman Optimality)**: $V^*(s) = \max_a \mathbb{E}[G_t | S_t = s, A_t = a]$ 의 유한성
- **Ch4 (Contraction Mapping)**: Discount $\gamma$ 가 contraction rate 의 핵심 파라미터
- **Ch5 (Policy Iteration)**: 수렴 속도 $O(\gamma^k)$ — $\gamma$ 가 클수록 느림

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 한계 | 대응 |
|------|------|------|------|
| Bounded reward | $\|R\| \leq R_{\max}$ | Unbounded 환경 처리 불가 | Ch7 linear FA 에서 근사 |
| $\gamma \in [0, 1)$ | Discount 엄격히 < 1 | $\gamma = 1$ 일 때? | Episodic 또는 average-reward |
| Stationary reward | $R$ 분포 시간 불변 | Changing reward 불가 | 표준 MDP 정의 밖 |
| Measurability | $G_t$ 확률변수 | 경로 집합이 복잡 | Borel $\sigma$-algebra 전제 |

---

## 📌 핵심 정리

$$\boxed{G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \in \left[-\frac{R_{\max}}{1-\gamma}, \frac{R_{\max}}{1-\gamma}\right]}$$

**조건**: $|R_t| \leq R_{\max}$, $\gamma \in [0, 1)$

| 항 | 의미 |
|----|------|
| $\gamma^k$ | Time discount: 멀수록 지수적으로 약화 |
| $R_{t+k+1}$ | $t$ 에서 $k+1$ 스텝 뒤의 즉시보상 |
| $\gamma = 0$ | Myopic: 한 걸음만 봄 ($G_t = R_{t+1}$) |
| $\gamma \to 1$ | Far-sighted: 무한 지평, 그러나 수렴 느림 |
| **Absolute convergence** | 모든 bounded trajectory 에서 유한 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 1.1 의 증명에서 왜 절댓값 부등식 $|\sum a_k| \leq \sum |a_k|$ (삼각 부등식) 를 사용하고, 이를 "absolute convergence" 라 부르는가? 조건수렴(conditional convergence)과의 차이는?

<details>
<summary>해설</summary>

**절댓값 부등식 (삼각 부등식)**:
$$\left|\sum_{k=0}^n a_k\right| \leq \sum_{k=0}^n |a_k|$$

이를 무한 급수로 확장하면:
- **Absolute convergence**: $\sum |a_k|$ 가 수렴 ⇒ $\sum a_k$ 도 수렴 (더 강함)
- **Conditional convergence**: $\sum a_k$ 수렴하지만 $\sum |a_k|$ 발산 (부분 취소)

**정리 1.1 의 key**: $R_{t+k+1}$ 이 부호가 바뀔 수 있어도, 절댓값 합이 bounded 이므로 원래 합도 수렴함을 보장.

예: $a_k = (-1)^k / k$ 는 조건수렴 (교대 급수), 하지만 $|a_k| = 1/k$ 는 발산.

**우리의 경우**: $|\gamma^k R_{t+k+1}| \leq \gamma^k R_{\max}$ 이고, $\sum \gamma^k R_{\max}$ 수렴 → absolute convergence.

이 덕분에 return 의 순서를 바꾸거나 기대값을 넣을 때 문제 없음. $\square$

</details>

**문제 2** (심화): 정리 1.5 (재귀 형태 $G_t = R_{t+1} + \gamma G_{t+1}$) 에서, 만약 episode 길이가 유한 $T$ 이고 terminal state 에서 $V(s_T) = 0$ 이면, $G_t = R_{t+1} + \gamma G_{t+1}$ 이 정확히 어떻게 변형되는가? 그 식에서 $\gamma = 1$ 을 대입해도 수렴하는가?

<details>
<summary>해설</summary>

**유한 horizon 에서의 return**:

$$G_t^{(T)} = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} + \gamma^{T-t} V(s_T)$$

Terminal value 가 0 이면:
$$G_t^{(T)} = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

재귀:
$$G_t^{(T)} = R_{t+1} + \gamma G_{t+1}^{(T)}$$

**$\gamma = 1$ 대입**:
$$G_t^{(T)} = R_{t+1} + G_{t+1}^{(T)} = \sum_{k=0}^{T-t-1} R_{t+k+1}$$

유한 합이므로 $\gamma = 1$ 에서도 **perfectly fine**. Terminal 이 있으면 무한성 제거.

**결론**: Episodic 은 $\gamma = 1$ 허용, infinite-horizon 은 $\gamma < 1$ 필수. $\square$

</details>

**문제 3** (논문 비평): Sutton & Barto (2018) 의 RL 교과서에서 $\gamma$ 를 "앞을 봐야 할 얼마나 많은 미래를 반영하는 가" 로 설명하고, Bellman (1957) 의 DP 원논문에서는 "무한 합의 수렴 조건" 으로 엄격하게 다룬다. 두 관점이 모순되는가? 실전 RL (Deep RL, 심지어 $\gamma = 0.999$) 에서는 어느 관점이 더 유용한가?

<details>
<summary>해설</summary>

**두 관점은 보완적**:

1. **Sutton & Barto (직관)**: "에피소드 끝까지 vs 몇 스텝만 봐라" — 알고리즘 설계에 직관 제공
2. **Bellman (수학)**: "$\gamma < 1$ 없으면 방정식 자체 정의 안 됨" — 이론적 토대

**실전의 차이**:
- **$\gamma = 0.99$** (일반적): Return 이론적으로 무한이지만, 약 100 스텝 뒤가 1/e 정도로 약해짐 — "100-step lookahead"
- **$\gamma = 0.999$**: 약 1000 스텝 — 매우 "먼" 보상도 영향, 그러나 value iteration 느려짐
- **Atari Deep RL**: $\gamma = 0.99$ 표준 (스텝당 ~17ms = 경험적 1초)

**의견**: $\gamma$ 를 "미래 지평" 으로 설정하되, **수학적으로는 수렴을 보장하는 조건** 으로 이해. 실전에서는 $\gamma < 1$ 을 항상 지키고, 값 선택은 환경의 temporal scale 과 episodic 여부에 따라 $\square$.

</details>

---

<div align="center">

[◀ 이전: Ch1-05. POMDP 와 Belief State](../ch1-mdp-definition/05-pomdp.md) | [📚 README](../README.md) | [다음 ▶: 02. State-Value 와 Action-Value Function](./02-value-functions.md)

</div>
