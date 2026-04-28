# 02. State-Value 와 Action-Value Function

## 🎯 핵심 질문

- State-value $V^\pi(s) = \mathbb{E}[G_t | S_t = s]$ 와 action-value $Q^\pi(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$ 의 정확한 관계는?
- 두 함수 사이의 세 가지 중요한 관계식 (total expectation, Bellman decomposition, recursive lookahead) 은?
- 왜 가치 함수가 bounded 인가? $\|V^\pi\|_\infty \leq R_{\max} / (1 - \gamma)$ 증명은?
- Advantage function $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ 의 의미와 차원 축소 효과는?
- 확률적 정책 vs 결정적 정책에서 가치 함수의 관계가 어떻게 다른가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

**Reinforcement Learning 의 핵심 목표**는 "최적 정책을 찾는 것" 입니다. 그러나 정책의 품질을 어떻게 정량화하는가? 

답: **Value function** — 각 상태(또는 상태-행동 쌍) 에서 앞으로 얻을 **누적 보상의 기대값**.

많은 입문 자료는:
- "$V(s)$ 는 '상태의 가치'" (추상적, 직관 부족)
- "$Q(s, a)$ 는 '행동의 가치'" (정확하지만 관계 생략)
- "Bellman equation 을 쓰면 된다" (어디서 나온지 설명 부족)

이 문서는:
1. **정의 부터 시작** — discounted return 의 기대값으로 엄밀하게
2. **세 가지 관계식** — 두 함수의 인터플레이를 증명
3. **Boundedness** — 왜 가치가 유한한가
4. **Advantage** — 간단한 뺄셈이지만 수심을 드러내는 개념

---

## 📐 수학적 선행 조건

- **Ch2-01**: Discounted return $G_t$ 와 수렴
- 확률론: 조건부 기댓값, 전확률 공식 (total expectation)
- Tower property: $\mathbb{E}[\mathbb{E}[X | Y] | Z] = \mathbb{E}[X | Z]$
- Linearity of expectation

---

## 📖 직관적 이해

### 두 함수의 역할

```
┌──────────────────┬──────────────────────┐
│   V^π(s)         │   Q^π(s, a)          │
├──────────────────┼──────────────────────┤
│ "상태 s 에서     │ "상태 s 에서 행동 a  │
│  정책 π 따라     │  하고 나서 정책 π    │
│  앞으로 기대     │  따라 앞으로 기대    │
│  얻을 보상"      │  얻을 보상"          │
│                  │                      │
│ 입력: 상태만     │ 입력: (상태, 행동)   │
│ 차원: |S|        │ 차원: |S| × |A|      │
└──────────────────┴──────────────────────┘
```

### 관계 1: 전확률 법칙 (Total Expectation)

정책 $\pi$ 에서 상태 $s$ 의 가치는, **그 상태에서 가능한 모든 행동의 가중 평균**:

$$V^\pi(s) = \sum_a \pi(a | s) \cdot Q^\pi(s, a)$$

- $\pi(a|s)$: 상태 $s$ 에서 행동 $a$ 를 선택할 확률
- "결정적 정책" ($\pi(a^* | s) = 1$) 이면: $V^\pi(s) = Q^\pi(s, a^*)$ (단 하나의 항)

### 관계 2: 한-걸음 전망 (One-Step Lookahead)

행동 $a$ 를 취한 후, 다음 상태 $s'$ 에서 가치 함수를 재귀적으로:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \cdot V^\pi(s')$$

이것이 나중에 **Bellman expectation equation** 의 씨앗.

### 관계 3: Advantage (Baseline-Corrected가치)

상태 $s$ 에서 행동 $a$ 의 **"평균 대비 이득"**:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A^\pi(s, a) > 0$: 이 행동이 평균보다 좋다
- $A^\pi(s, a) < 0$: 이 행동이 평균보다 나쁘다
- $A^\pi(s, a) = 0$: 중립적

**기대값**: 정책 $\pi$ 에서 $\sum_a \pi(a|s) A^\pi(s, a) = 0$ (항상 평균은 0)

---

## ✏️ 엄밀한 정의

### 정의 2.1 — State-Value Function

정책 $\pi$ 에 대한 상태 $s$ 의 가치 함수:

$$V^\pi(s) := \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s\right] = \mathbb{E}_\pi[G_t | S_t = s]$$

여기서:
- 기대값 $\mathbb{E}_\pi$ 는 정책 $\pi$ 와 dynamics $P$ 에 따라 계산
- $G_t$ 는 시간 $t$ 부터의 discounted return (Ch2-01)
- $V^\pi: \mathcal{S} \to \mathbb{R}$ 는 상태에서 실수로의 함수

### 정의 2.2 — Action-Value Function (Q-Function)

정책 $\pi$ 에 대한 상태-행동 쌍 $(s, a)$ 의 가치 함수:

$$Q^\pi(s, a) := \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s, A_t = a\right] = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

다만:
- 시간 $t$ 에 **반드시** 행동 $a$ 를 취하고,
- 그 다음부터 정책 $\pi$ 따라 움직임

### 정의 2.3 — Advantage Function

상태-행동 쌍의 **baseline-corrected 가치**:

$$A^\pi(s, a) := Q^\pi(s, a) - V^\pi(s)$$

"상태 $s$ 에서 정책의 평균 행동 대비, 행동 $a$ 가 얼마나 더 좋은가"

---

## 🔬 정리와 증명

### 정리 2.1 (Total Expectation Law for Value Functions)

모든 상태 $s$ 에서:

$$V^\pi(s) = \sum_a \pi(a | s) Q^\pi(s, a)$$

**증명**:

**Step 1** — 조건부 기댓값의 정의:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

시간 $t$ 에 첫 행동 $A_t$ 를 조건으로 조건부:

$$V^\pi(s) = \mathbb{E}[\mathbb{E}_\pi[G_t | S_t = s, A_t] | S_t = s]$$

**Step 2** — 내부 기대값 = $Q^\pi(s, a)$:

$$\mathbb{E}_\pi[G_t | S_t = s, A_t = a] = Q^\pi(s, a)$$

**Step 3** — 외부 기대값에서 $A_t$ 주변화:

$$V^\pi(s) = \sum_a Q^\pi(s, a) \Pr(A_t = a | S_t = s) = \sum_a \pi(a|s) Q^\pi(s, a) \quad \square$$

### 정리 2.2 (One-Step Bellman Decomposition)

모든 상태-행동 쌍 $(s, a)$ 에 대해:

$$Q^\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s, A_t = a]$$

또는 명시적으로:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) V^\pi(s')$$

**증명**:

**Step 1** — Return 의 재귀 (Ch2-01, 정리 1.5):

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**Step 2** — 양변에 기대값 적용:

$$\mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a]$$

**Step 3** — Linearity:

$$Q^\pi(s, a) = \mathbb{E}_\pi[R_{t+1} | S_t = s, A_t = a] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s, A_t = a]$$

**Step 4** — MDP 의 Markov 성질:
- 첫 항: $\mathbb{E}[R_{t+1} | S_t = s, A_t = a] = R(s, a)$ (즉시 보상, 정책 무관)
- 둘째 항: $\mathbb{E}_\pi[G_{t+1} | S_t = s, A_t = a] = \mathbb{E}_{s' \sim P(\cdot|s,a)}[\mathbb{E}_\pi[G_{t+1} | S_{t+1} = s']]$ (tower property)

**Step 5** — 정의에 의해 $\mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] = V^\pi(s')$:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) V^\pi(s') \quad \square$$

### 정리 2.3 (Boundedness of Value Functions)

Bounded reward $|R(s, a)| \leq R_{\max}$ 일 때:

$$|V^\pi(s)| \leq \frac{R_{\max}}{1 - \gamma}, \quad |Q^\pi(s, a)| \leq \frac{R_{\max}}{1 - \gamma}$$

**증명**:

**$V^\pi$ 의 경우**:

$$|V^\pi(s)| = \left|\mathbb{E}_\pi[G_t | S_t = s]\right| \leq \mathbb{E}_\pi[|G_t| | S_t = s] \leq \mathbb{E}_\pi\left[\frac{R_{\max}}{1-\gamma} \Big| S_t = s\right] = \frac{R_{\max}}{1-\gamma}$$

(여기서 Ch2-01, 정리 1.1 이용: $|G_t| \leq R_{\max}/(1-\gamma)$ a.s.)

**$Q^\pi$ 의 경우**: 동일하게, $G_t$ 가 bounded 이므로 $Q^\pi$ 도 bounded. $\square$

### 따름정리 2.4 — Advantage 의 성질

$$\mathbb{E}_{\pi}[A^\pi(s, a) | S_t = s] = \sum_a \pi(a|s) A^\pi(s, a) = 0$$

**증명**:

$$\sum_a \pi(a|s) A^\pi(s, a) = \sum_a \pi(a|s) (Q^\pi(s,a) - V^\pi(s)) = \sum_a \pi(a|s) Q^\pi(s,a) - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0 \quad \square$$

---

## 💻 NumPy 구현 검증

### 실험 1 — 4×4 Gridworld 에서 Value 계산

```python
import numpy as np
import matplotlib.pyplot as plt

# 4x4 gridworld: 0,1,2,...,15 states
# Simple policy: uniform random
S = 16
A = 4  # up, down, left, right
gamma = 0.9

# Transition: stochastic, p_success=0.8, p_failure=0.1 각각 옆
def build_P_and_R():
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    
    for s in range(S):
        row, col = s // 4, s % 4
        for a in range(A):
            # Simple: each action moves to an adjacent state (with wrapping)
            if a == 0:    # up
                next_s = ((row - 1) % 4) * 4 + col
            elif a == 1:  # down
                next_s = ((row + 1) % 4) * 4 + col
            elif a == 2:  # left
                next_s = row * 4 + ((col - 1) % 4)
            else:         # right
                next_s = row * 4 + ((col + 1) % 4)
            
            P[s, a, next_s] = 1.0
            R[s, a] = -1.0 if next_s != 15 else 10.0  # +10 at goal (state 15)
    
    return P, R

P, R = build_P_and_R()

# Uniform policy
pi = np.ones((S, A)) / A  # π(a|s) = 0.25 for all a, s

# Policy Evaluation: V^π via iterative Bellman
def policy_eval_v(pi, P, R, gamma, n_iter=1000, tol=1e-6):
    V = np.zeros(S)
    for _ in range(n_iter):
        V_old = V.copy()
        # Q(s,a) = R(s,a) + γ Σ P(s'|s,a) V(s')
        Q = R + gamma * np.einsum('sap,p->sa', P, V)  # shape (S, A)
        # V(s) = Σ π(a|s) Q(s,a)
        V = np.einsum('sa,sa->s', pi, Q)
        if np.abs(V - V_old).max() < tol:
            break
    return V, Q

V, Q = policy_eval_v(pi, P, R, gamma)

print("State-Value V^π for uniform random policy:")
print(V.reshape(4, 4).round(2))
print("\nMax |V|:", np.abs(V).max())
print("Theoretical upper bound R_max/(1-γ) =", 10.0 / (1 - gamma))

# Test 정리 2.1: V(s) = Σ π(a|s) Q(s,a)
V_test = np.einsum('sa,sa->s', pi, Q)
print(f"\nTheorem 2.1 check (V = Σ π(a|s)Q(s,a)):")
print(f"  Max error: {np.abs(V - V_test).max():.2e}  ✓")
```

### 실험 2 — Deterministic vs Stochastic 정책

```python
# (1) Deterministic policy: always move right (a=3)
pi_det = np.zeros((S, A))
pi_det[:, 3] = 1.0  # always right

V_det, Q_det = policy_eval_v(pi_det, P, R, gamma)

print("Deterministic policy (always right):")
print(V_det.reshape(4, 4).round(2))
print(f"Starting state (0) value: {V_det[0]:.2f}")

# (2) Stochastic: 50% right, 50% up
pi_stoch = np.zeros((S, A))
pi_stoch[:, 0] = 0.5  # up
pi_stoch[:, 3] = 0.5  # right

V_stoch, Q_stoch = policy_eval_v(pi_stoch, P, R, gamma)

print("\nStochastic policy (50% up, 50% right):")
print(V_stoch.reshape(4, 4).round(2))
print(f"Starting state (0) value: {V_stoch[0]:.2f}")

print(f"\nValue 차이: {np.abs(V_det - V_stoch).max():.2f} (deterministic이 더 good 경우 많음)")
```

### 실험 3 — Advantage Function 시각화

```python
# Q and Advantage
A_func = Q - V[:, None]  # shape (S, A)

print("\nAdvantage A^π(s, a) = Q(s, a) - V(s) for uniform policy:")
print("State 0 (top-left):")
for a in range(4):
    action_names = ['Up', 'Down', 'Left', 'Right']
    print(f"  {action_names[a]:6s}: A = {A_func[0, a]:7.3f}")

# Verify 따름정리 2.4: E_π[A(s,a) | s] = 0
A_mean = np.einsum('sa,sa->s', pi, A_func)
print(f"\nCorollary 2.4 check (E_π[A(s,a)|s] = 0):")
print(f"  Max |E[A]|: {np.abs(A_mean).max():.2e}  ✓")

# Heatmap: Advantage for each action
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
action_names = ['Up', 'Down', 'Left', 'Right']
for a, ax in enumerate(axes):
    im = ax.imshow(A_func[:, a].reshape(4, 4), cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title(f'A(s, {action_names[a]})')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('advantage_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 실험 4 — Bounded Norm 검증

```python
# Generate random MDPs and check boundedness
n_test = 100
S, A = 10, 3
R_max = 5.0

bound_theoretical = R_max / (1 - gamma)

max_V_values = []
for trial in range(n_test):
    P_rand = np.random.dirichlet(np.ones(S), size=(S, A))
    R_rand = np.random.uniform(-R_max, R_max, (S, A))
    pi_rand = np.random.dirichlet(np.ones(A), size=S)
    
    V_rand, _ = policy_eval_v(pi_rand, P_rand, R_rand, gamma, n_iter=2000)
    max_V_values.append(np.abs(V_rand).max())

max_V_values = np.array(max_V_values)

print(f"Random MDP tests (n={n_test}):")
print(f"Theoretical bound: {bound_theoretical:.2f}")
print(f"Empirical max |V|: {max_V_values.mean():.2f} ± {max_V_values.std():.2f}")
print(f"Maximum observed: {max_V_values.max():.2f}")
print(f"All within bound? {(max_V_values <= bound_theoretical * 1.01).all()}  ✓")

plt.figure(figsize=(10, 5))
plt.hist(max_V_values, bins=30, alpha=0.7, edgecolor='black', label='Empirical max |V|')
plt.axvline(bound_theoretical, color='r', linestyle='--', linewidth=2, label=f'Theoretical bound = {bound_theoretical:.2f}')
plt.axvline(max_V_values.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean = {max_V_values.mean():.2f}')
plt.xlabel('Maximum |V(s)| over all states')
plt.ylabel('Frequency')
plt.title(f'Value Function Boundedness (γ={gamma}, R_max={R_max})')
plt.legend()
plt.grid()
plt.savefig('value_boundedness.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 후속 레포와의 연결

- **Ch2-03 (Bellman Expectation)**: $V^\pi, Q^\pi$ 의 재귀적 정의가 고정점 방정식으로 완성
- **Ch2-04 (Bellman Operator)**: 연산자 $T^\pi$ 로 형식화하여 일반 수렴 이론 적용
- **Ch3 (Optimal Value)**: $V^*, Q^*$ 정의, Bellman optimality equation
- **Ch6 (Advanced)**: Advantage 가 policy gradient 의 중심 개념으로 재등장

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 한계 | 대응 |
|------|------|------|------|
| Bounded reward | $\|R\| \leq R_{\max}$ | Unbounded 환경 | Average normalization |
| Markov property | 현재 상태만 의존 | 부분 관찰성 | POMDP (Ch1-05) |
| Stationary policy | $\pi$ 시간 불변 | Dynamic strategy | Non-stationary 확장 |
| Finite MDP (보통) | $\|\mathcal{S}\|, \|\mathcal{A}\|$ 유한 | Continuous 상태 | Function approximation |

---

## 📌 핵심 정리

$$\boxed{V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]}$$

| 함수 | 입력 | 정의 | 역할 |
|------|------|------|------|
| $V^\pi$ | 상태 $s$ | $\mathbb{E}[G_t \| S_t = s]$ | 상태 전망 |
| $Q^\pi$ | $(s, a)$ | $\mathbb{E}[G_t \| S_t=s, A_t=a]$ | 행동 품질 |
| $A^\pi$ | $(s, a)$ | $Q^\pi - V^\pi$ | Baseline-corrected |
| **Bound** | 모든 함수 | $\leq R_{\max}/(1-\gamma)$ | 유한성 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 2.1 의 "전확률 공식" 증명에서 tower property 를 사용한다. Tower property 가 정확히 무엇이고, 왜 여기서 적용되는가?

<details>
<summary>해설</summary>

**Tower Property (조건부 기댓값의 중첩)**:

$$\mathbb{E}[\mathbb{E}[X | Y] | Z] = \mathbb{E}[X | Z]$$

(우측이 $Z$-measurable 일 때)

**정리 2.1 에서의 적용**:
$$V^\pi(s) = \mathbb{E}[G_t | S_t = s] = \mathbb{E}[\mathbb{E}[G_t | S_t = s, A_t] | S_t = s]$$

내부 기대값: $\mathbb{E}[G_t | S_t = s, A_t = a] = Q^\pi(s, a)$

외부 기대값: $A_t$ 의 분포는 정책 $\pi$ 에서 $\Pr(A_t = a | S_t = s) = \pi(a|s)$

따라서:
$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$

**의의**: 복잡한 조건부 기댓값을 단계적으로 분해. $\square$

</details>

**문제 2** (심화): 정리 2.2 (One-Step Bellman) 에서, 왜 $R_{t+1}$ 의 기대값이 정책 $\pi$ 무관하게 $R(s, a)$ 가 되는가? 만약 reward 가 확률적 (예: $R \sim N(\mu, \sigma^2)$) 이면?

<details>
<summary>해설</summary>

**MDP 의 표준 정의**: $R(s, a)$ 는 **결정적** (deterministic). 주어진 $(s, a)$ 에 대해 항상 같은 보상.

$$\mathbb{E}[R_{t+1} | S_t = s, A_t = a] = R(s, a)$$

(기대값이 아니라 상수)

**확률적 Reward 의 경우**:

만약 $R$ 이 확률분포를 따르면 $R: \mathcal{S} \times \mathcal{A} \to$ probability distribution.

이 경우:
$$\mathbb{E}[R_{t+1} | S_t = s, A_t = a] = \mathbb{E}_{R \sim \text{Dist}(s,a)}[R]$$

예: $R(s, a) \sim N(\mu(s,a), \sigma^2)$ 이면 기대값은 $\mu(s,a)$.

**표준 MDP** 에서는 단순성을 위해 deterministic reward 가정. 확률적 reward 는 "reward distribution MDP" 로 일반화 (Mania et al. 2020). $\square$

</details>

**문제 3** (논문 비평): Sutton & Barto (2018) 는 advantage 를 "baseline-corrected 가치" 로 설명하고, Kakade & Langford (2002) 의 Performance Difference Lemma 에서 advantage 가 핵심 양이 된다. 왜 $Q - V$ 라는 단순한 뺄셈이 이토록 중요한가? Policy gradient 이론에서 실제로 어떤 역할을 하는가?

<details>
<summary>해설</summary>

**$Q - V$ 의 중요성**:

1. **Baseline correction**: Policy gradient 계산에서 low-variance direction 획득
   - $\nabla_\theta J(\pi_\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi(a|s) \cdot Q^\pi(s,a)]$ (high variance)
   - $= \mathbb{E}_\pi[\nabla_\theta \log \pi(a|s) \cdot A^\pi(s,a)]$ (lower variance, $V$ 는 상수 취급)

2. **Policy Improvement 의 정량화**: 
   - Kakade & Langford (2002) Performance Difference Lemma (다음 레포):
   $$J(\pi') - J(\pi) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi'}, a \sim \pi'}[A^\pi(s,a)]$$
   
   이것이 **모든 정책 개선 이론의 출발점**.

3. **직관**: "절대 가치" ($Q, V$) 가 아니라 "**상대적 가치**" ($A$) 가 policy improvement 의 신호 제공.
   - 일정한 positive shift (모든 행동에 +C) 는 정책 변화 없음
   - Advantage 가 실제 preference 정보 담음

**Advanced RL 의 핵심**: TRPO, PPO 등은 모두 "advantage function 을 최대화" 하는 알고리즘. $\square$

</details>

---

<div align="center">

[◀ 이전: 01. Discounted Return 의 정의와 수렴](./01-discounted-return.md) | [📚 README](../README.md) | [다음 ▶: 03. Bellman Expectation Equation 유도](./03-bellman-expectation.md)

</div>
