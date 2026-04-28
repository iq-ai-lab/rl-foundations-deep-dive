# 04. Operator 표기법: $T^\pi V = r^\pi + \gamma P^\pi V$

## 🎯 핵심 질문

- Bellman expectation equation 을 "연산자 (operator)" 로 형식화하면 무엇을 얻는가?
- $r^\pi, P^\pi$ 의 정의와 정책 유도 과정은?
- 선형 연산자 $T^\pi$ 의 성질은? Affine map 이란?
- 고정점 (fixed point) $V^\pi = T^\pi V^\pi$ 의 의미와 uniqueness 는?
- Operator norm 이란? $\|T^\pi V - T^\pi V'\|_\infty$ 가 왜 중요한가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

**Bellman expectation equation** (Ch2-03) 은 상태 $s$ 별로 쓴 것입니다:

$$V^\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

그러나 이것은 **함수방정식 (functional equation)** 입니다 — 우변이 $V^\pi$ 자체를 포함합니다.

**Operator 표기법** 은:
1. **함수들의 공간**을 명확히 정의 (bounded functions on $\mathcal{S}$)
2. $T^\pi$ 를 **함수→함수의 변환**으로 형식화
3. Bellman equation 을 **고정점 방정식** $V^\pi = T^\pi V^\pi$ 로 표현

**왜 중요한가?**
- **Fixed Point Theorem 적용**: Banach fixed point theorem (Ch4) 을 쓸 수 있게 됨
- **Convergence 증명**: $T^\pi$ 가 contraction 이면 value iteration 수렴 보장
- **일반 수학**: 함수해석학의 표준 도구로 RL 을 분석

---

## 📐 수학적 선행 조건

- **Ch2-03**: Bellman expectation equation (scalar form)
- 함수공간: $B(\mathcal{S})$ = bounded functions on $\mathcal{S}$
- Norm: sup-norm $\|f\|_\infty = \sup_{s \in \mathcal{S}} |f(s)|$
- 선형대수: 선형 연산자, affine map
- Matrix notation (stochastic matrix)

---

## 📖 직관적 이해

### Operator 의 의미

함수 $V$ 를 받아서 **그 함수를 한 스텝 Bellman backup 한 새 함수**를 만드는 기계:

```
입력 V (전 상태의 가치) → [T^π 연산자] → 출력 T^π V (새 가치)

예: V = [1.0, 2.5, 3.0, ...]
         ↓ (Bellman backup)
    T^π V = [1.5, 2.8, 3.2, ...]  (한 스텝 더 정확해짐)
```

### 정책-유도 (Induced) 객체들

원래 MDP 의 $P(s'|s, a), R(s, a)$ 를, 정책 $\pi$ 에 따라 "정책적 버전" 으로 평균:

```
r^π(s) := Σ_a π(a|s) · R(s,a)
        = "상태 s 에서 정책 π 따를 때 한 스텝의 기대 보상"

P^π(s'|s) := Σ_a π(a|s) · P(s'|s,a)
          = "상태 s 에서 정책 π 따를 때 다음 상태 분포"
```

이 둘을 Bellman 에 대입:

```
V(s) = r^π(s) + γ Σ_{s'} P^π(s'|s) V(s')
```

### Affine 성질

$T^\pi$ 는 **선형 + 상수**:

$$T^\pi V = r^\pi + \gamma P^\pi V$$

- "$r^\pi$": 상수항
- "$\gamma P^\pi V$": $V$ 에 대해 선형

이를 "affine" 이라 부름 (선형 + 평행이동).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 함수 공간

$$B(\mathcal{S}) := \{f: \mathcal{S} \to \mathbb{R} : \|f\|_\infty < \infty\}$$

$\mathcal{S}$ 에서 실수로의 bounded 함수들의 집합. 표준 norm:

$$\|f\|_\infty := \sup_{s \in \mathcal{S}} |f(s)|$$

**완비 거리공간 (complete metric space)**: $(B(\mathcal{S}), d_\infty)$ where $d_\infty(f, g) = \|f - g\|_\infty$.

### 정의 4.2 — Policy-Induced Reward

정책 $\pi$ 에 대한 기대 보상:

$$r^\pi(s) := \sum_a \pi(a|s) R(s, a)$$

**성질**: $\|r^\pi\|_\infty \leq R_{\max}$.

### 정의 4.3 — Policy-Induced Transition

정책 $\pi$ 에서의 상태 전이:

$$P^\pi(s'|s) := \sum_a \pi(a|s) P(s'|s,a)$$

각 상태 $s$ 에서:
- $\sum_{s'} P^\pi(s'|s) = 1$ (stochastic matrix)
- $P^\pi(s'|s) \geq 0$ (확률)

**Matrix form**: $P^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$, 각 열이 확률 분포.

### 정의 4.4 — Bellman Operator (정책 평가)

선형 연산자 $T^\pi: B(\mathcal{S}) \to B(\mathcal{S})$:

$$\boxed{(T^\pi V)(s) := r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) V(s')}$$

또는 벡터 표기:

$$T^\pi \mathbf{V} = \mathbf{r}^\pi + \gamma P^\pi \mathbf{V}$$

### 정의 4.5 — 고정점 (Fixed Point)

$V^* \in B(\mathcal{S})$ 가 $T^\pi$ 의 고정점:

$$V^* = T^\pi V^*$$

즉, 연산자를 적용해도 변하지 않는 함수.

---

## 🔬 정리와 증명

### 정리 4.1 (Affine Operator)

$T^\pi: B(\mathcal{S}) \to B(\mathcal{S})$ 는 affine 연산자이다:

$$T^\pi(\alpha V + (1-\alpha) V') = \alpha (T^\pi V) + (1-\alpha) (T^\pi V')$$

for all $V, V' \in B(\mathcal{S})$ and $\alpha \in [0, 1]$.

**증명**:

$$T^\pi(\alpha V + (1-\alpha)V')(s) = r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) [\alpha V(s') + (1-\alpha) V'(s')]$$

$$= r^\pi(s) + \alpha \gamma \sum_{s'} P^\pi(s'|s) V(s') + (1-\alpha) \gamma \sum_{s'} P^\pi(s'|s) V'(s')$$

$$= \alpha [r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) V(s')] + (1-\alpha) [r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) V'(s')]$$

$$= \alpha (T^\pi V)(s) + (1-\alpha) (T^\pi V')(s) \quad \square$$

### 정리 4.2 (Monotonicity)

$V \geq V'$ (pointwise) 이면 $T^\pi V \geq T^\pi V'$ (pointwise).

**증명**:

$V(s') \geq V'(s')$ for all $s'$ 이면:

$$(T^\pi V)(s) = r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) V(s') \geq r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) V'(s') = (T^\pi V')(s) \quad \square$$

### 정리 4.3 (Boundedness of $T^\pi$)

모든 $V \in B(\mathcal{S})$ 에 대해:

$$\|T^\pi V\|_\infty \leq R_{\max} + \gamma \|V\|_\infty$$

**증명**:

$$|(T^\pi V)(s)| = |r^\pi(s) + \gamma \sum_{s'} P^\pi(s'|s) V(s')|$$

삼각부등식:

$$\leq |r^\pi(s)| + \gamma \left|\sum_{s'} P^\pi(s'|s) V(s')\right|$$

$$\leq R_{\max} + \gamma \sum_{s'} P^\pi(s'|s) |V(s')|$$ 

(stochastic $P^\pi$ 이므로 $\sum_{s'} P^\pi(s'|s) = 1$)

$$\leq R_{\max} + \gamma \|V\|_\infty \quad \square$$

### 정리 4.4 (고정점의 존재과 유일성 — 예고)

**정리** (Banach Fixed Point Theorem — Ch4 에서 정식화):

$T^\pi$ 가 $\gamma$-contraction 이면, 유일한 고정점 $V^\pi$ 가 존재하고:

$$V_{k+1} = T^\pi V_k \to V^\pi \quad \text{as } k \to \infty$$

**스케치** (정규 증명은 Ch4):

$T^\pi$ 가 contraction 이려면:

$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty \quad \forall V, V' \in B(\mathcal{S})$$

Affine 이므로:

$$\|T^\pi V - T^\pi V'\|_\infty = \|\gamma P^\pi (V - V')\|_\infty \leq \gamma \|P^\pi (V-V')\|_\infty$$

$P^\pi$ 가 stochastic (행의 합 = 1) 이므로:

$$\|P^\pi (V-V')\|_\infty = \sup_s \left| \sum_{s'} P^\pi(s'|s) [V(s') - V'(s')] \right| \leq \sup_s \sum_{s'} P^\pi(s'|s) |V(s') - V'(s')| \leq \|V - V'\|_\infty$$

따라서:

$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$$

이것이 $\gamma < 1$ 을 필요로 하는 이유 (Ch2-05 에서 완성). $\square$

### 정리 4.5 (Vector-Matrix Form 의 고정점)

유한 MDP 에서, 고정점은 선형방정식:

$$(I - \gamma P^\pi) \mathbf{V}^\pi = \mathbf{r}^\pi$$

의 유일한 해이다.

**증명**:

$$\mathbf{V}^\pi = T^\pi \mathbf{V}^\pi = \mathbf{r}^\pi + \gamma P^\pi \mathbf{V}^\pi$$

$$(I - \gamma P^\pi) \mathbf{V}^\pi = \mathbf{r}^\pi$$

가역성은 정리 2-03 의 Ch2-05 note 참고 $\square$.

---

## 💻 NumPy 구현 검증

### 실험 1 — Operator 의 정의와 고정점

```python
import numpy as np
from scipy.linalg import solve

# 작은 예제: 3-state MDP
S = 3
A = 2
gamma = 0.9

# 간단한 transition 과 reward
P = np.array([  # P[s, a, s']
    [[0.5, 0.5, 0.0], [0.0, 0.3, 0.7]],  # state 0
    [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]],  # state 1
    [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],  # state 2
])

R = np.array([
    [1.0, 2.0],   # state 0
    [0.5, 1.5],   # state 1
    [-1.0, 0.0],  # state 2
])

# 정책: 각 상태에서 action 선택 확률
pi = np.array([
    [0.4, 0.6],  # state 0
    [0.5, 0.5],  # state 1
    [0.3, 0.7],  # state 2
])

# r^π 계산
r_pi = np.einsum('sa,sa->s', pi, R)
print("r^π:", r_pi)

# P^π 계산
P_pi = np.einsum('sa,sap->sp', pi, P)
print("\nP^π:\n", P_pi)
print("Row sums (should be all 1):", P_pi.sum(axis=1))

# Operator T^π 정의
def bellman_operator(V, r_pi, P_pi, gamma):
    """(T^π V)(s) = r^π(s) + γ Σ P^π(s'|s) V(s')"""
    return r_pi + gamma * (P_pi @ V)

# (1) 초기 V = 0 에서 시작
V = np.zeros(S)
print("\n--- Iteration ---")
for k in range(20):
    V_new = bellman_operator(V, r_pi, P_pi, gamma)
    error = np.abs(V_new - V).max()
    print(f"k={k:2d}: V = {V_new.round(3)}, max_change = {error:.4f}")
    if error < 1e-6:
        print(f"Converged at k={k}")
        break
    V = V_new

V_iter = V
print(f"\nConverged V from iteration: {V_iter}")

# (2) 선형방정식으로 직접 풀기
# (I - γ P^π) V = r^π
A_matrix = np.eye(S) - gamma * P_pi
V_linear = solve(A_matrix, r_pi)
print(f"Direct solve:               {V_linear}")
print(f"Difference: {np.abs(V_iter - V_linear).max():.2e}")
```

### 실험 2 — Contraction 성질 검증

```python
# T^π V - T^π V' 와 V - V' 의 거리 비교
V1 = np.array([1.0, 2.0, 0.5])
V2 = np.array([0.5, 1.5, 1.0])

TV1 = bellman_operator(V1, r_pi, P_pi, gamma)
TV2 = bellman_operator(V2, r_pi, P_pi, gamma)

dist_V = np.abs(V1 - V2).max()   # ||V1 - V2||_∞
dist_TV = np.abs(TV1 - TV2).max()  # ||T^π V1 - T^π V2||_∞

print("Contraction property check:")
print(f"  ||V1 - V2||_∞  = {dist_V:.6f}")
print(f"  ||T^π V1 - T^π V2||_∞ = {dist_TV:.6f}")
print(f"  Ratio = {dist_TV / dist_V:.6f} (should be ≤ γ = {gamma})")
print(f"  Is contraction? {dist_TV <= gamma * dist_V}")
```

### 실험 3 — 여러 초기값에서의 수렴

```python
import matplotlib.pyplot as plt

# 다양한 초기 V 에서 iteration
V_inits = [
    np.zeros(S),
    np.ones(S),
    np.array([10.0, -10.0, 5.0]),
    np.array([-100.0, 100.0, 50.0]),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for V_init in V_inits:
    V = V_init.copy()
    errors = []
    
    for k in range(100):
        V_new = bellman_operator(V, r_pi, P_pi, gamma)
        error = np.abs(V_new - V_linear).max()  # V_linear = ground truth
        errors.append(error)
        V = V_new
    
    axes[0].semilogy(range(len(errors)), errors, 
                     label=f'V_0 = {V_init.round(1)}', linewidth=2)

axes[0].set_xlabel('Iteration k')
axes[0].set_ylabel('||V_k - V*||_∞ (log scale)')
axes[0].set_title('Convergence from Different Initializations')
axes[0].legend()
axes[0].grid()

# 수렴율 확인: e_k ≈ γ^k e_0
k_range = np.arange(50)
initial_error = np.abs(V_inits[2] - V_linear).max()
theoretical = initial_error * (gamma ** k_range)

V = V_inits[2].copy()
actual = []
for k in range(50):
    V = bellman_operator(V, r_pi, P_pi, gamma)
    actual.append(np.abs(V - V_linear).max())

axes[1].semilogy(k_range, actual, 'o-', label='Actual error', linewidth=2)
axes[1].semilogy(k_range, theoretical, '--', label=f'γ^k e_0 (γ={gamma})', linewidth=2)
axes[1].set_xlabel('Iteration k')
axes[1].set_ylabel('Error (log scale)')
axes[1].set_title(f'Linear Convergence Rate γ^k')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('operator_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 실험 4 — Affine 성질 검증

```python
# T^π(α V + (1-α)V') = α T^π V + (1-α) T^π V'

V1 = np.array([1.0, 2.0, 0.5])
V2 = np.array([2.0, 1.0, 3.0])

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

print("Affine property check:")
print("α    | LHS (T^π mix) | RHS (mix T^π) | Error")
print("-" * 55)

for alpha in alphas:
    V_mix = alpha * V1 + (1 - alpha) * V2
    
    # LHS: T^π(α V1 + (1-α)V2)
    T_mix = bellman_operator(V_mix, r_pi, P_pi, gamma)
    
    # RHS: α T^π V1 + (1-α) T^π V2
    T_V1 = bellman_operator(V1, r_pi, P_pi, gamma)
    T_V2 = bellman_operator(V2, r_pi, P_pi, gamma)
    mix_T = alpha * T_V1 + (1 - alpha) * T_V2
    
    error = np.abs(T_mix - mix_T).max()
    print(f"{alpha:.2f} | {str(T_mix.round(2)):13s} | {str(mix_T.round(2)):13s} | {error:.2e}")
```

---

## 🔗 후속 레포와의 연결

- **Ch2-05 (고유성)**: $(I - \gamma P^\pi)$ 가역성 → closed-form $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$
- **Ch4 (Contraction)**: $T^\pi$ 가 $\gamma$-contraction 증명 → Banach fixed point
- **Ch5 (Value Iteration)**: $V_{k+1} = T^\pi V_k$ 의 수렴성
- **Ch5 (Policy Iteration)**: Policy evaluation 에서 $T^\pi$ 반복
- **Ch3 (Optimality)**: Bellman optimality operator $T^*$ (nonlinear greedy)

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 한계 | 대응 |
|------|------|------|------|
| Bounded functions | $B(\mathcal{S})$ 에서 작업 | Unbounded reward | Normalization |
| Sup-norm | $\|\cdot\|_\infty$ 사용 | 다른 norm 은? | $\ell^1, \ell^2$ 도 유사 |
| Finite MDP | 행렬 연산 | Continuous 상태 | Operator 추상성 보존 |
| Stationary $\pi$ | 정책 고정 | Non-stationary 정책 | 일반화 가능하지만 복잡 |

---

## 📌 핵심 정리

$$\boxed{T^\pi V = r^\pi + \gamma P^\pi V, \quad V^\pi = T^\pi V^\pi}$$

| 기호 | 정의 | 성질 |
|------|------|------|
| $T^\pi$ | Bellman operator | Affine, monotone |
| $r^\pi(s)$ | $\sum_a \pi(a\|s) R(s,a)$ | Policy-averaged reward |
| $P^\pi$ | $\sum_a \pi(a\|s) P(s'\|s,a)$ | Stochastic matrix |
| $V^\pi$ | $T^\pi V^\pi$ 의 고정점 | Unique for $\gamma < 1$ |
| **Contraction rate** | $\gamma \in [0,1)$ | Convergence speed |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 4.1 (Affine property) 의 증명을 따라가며, 어느 단계에서 $P^\pi$ 의 stochastic 성질 (행의 합 = 1) 이 사용되는가? 만약 행의 합이 1이 아니면?

<details>
<summary>해설</summary>

**정리 4.1 증명에서**:

$T^\pi$ 의 affine 성질은 우변이 $V$ 에 대해 선형이므로 자동 유도됨:

$$T^\pi(\alpha V + (1-\alpha)V') = r^\pi + \gamma P^\pi [\alpha V + (1-\alpha)V']$$
$$= r^\pi + \alpha \gamma P^\pi V + (1-\alpha) \gamma P^\pi V'$$

**$P^\pi$ stochastic 이 아닐 경우**:

행의 합이 $c(s) \neq 1$ 이면, 예를 들어 $c(s) = 1.1$ (확률 합이 초과):

$$T^\pi V = r + \gamma P V$$

여전히 affine 이지만, **contraction property 가 깨짐**:

$$\|T^\pi V - T^\pi V'\|_\infty = \gamma \|P(V-V')\|_\infty$$

여기서 $\|P(V-V')\|_\infty$ 가 $\|V-V'\|_\infty$ 보다 최대 $\max_s c(s)$ 배 커질 수 있음:

$$\|P(V-V')\|_\infty \leq \max_s c(s) \cdot \|V-V'\|_\infty$$

따라서 contraction rate 는 $\gamma \max_s c(s)$, 이것이 < 1이어야 수렴.

**결론**: Stochastic $P^\pi$ 가 아니면 contraction bound 가 약해짐. $\square$

</details>

**문제 2** (심화): 정리 4.3 에서 $\|T^\pi V\|_\infty \leq R_{\max} + \gamma \|V\|_\infty$ 이다. 이것과 "contraction" 의 관계는? 왜 contraction 은 이 부등식이 아니라 $\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$ 를 요구하는가?

<details>
<summary>해설</summary>

**두 부등식의 역할**:

1. **정리 4.3 ($\|T^\pi V\|_\infty$ bound)**: "연산자가 유한한 함수를 유한한 함수로 보낸다" — 함수공간 밖으로 나가지 않음
2. **Contraction ($\|T^\pi V - T^\pi V'\|_\infty$ bound)**: "거리를 줄인다" — 두 함수 사이 차이가 매번 $\gamma$ 배로 축소

**왜 다른가?**

- 정리 4.3: **Absolute boundedness** — 원점으로부터의 거리
- Contraction: **Relative boundedness** — 두 점 사이의 거리

**예**:
- $T^\pi V = 10$ (상수함수) 이면 $\|T^\pi V\|_\infty = 10$ (bounded)
- 하지만 $T^\pi V = T^\pi V'$ (identical) 이면 $\|T^\pi V - T^\pi V'\|_\infty = 0$ (극단적 수축)

**Contraction 의 필요성**:

Fixed point theorem 을 증명하려면:
$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty \to 0$$

이것은 정리 4.3 의 absolute bound 만으로는 불가능. **상대 거리** 감소가 핵심. $\square$

</details>

**문題 3** (논文 비評): Bellman (1957) 은 "dynamic programming" 의 원리로 "optimal substructure" 를 강조했고, 이것이 가치 함수의 재귀 구조로 이어졌다. 현대 functional analysis 관점에서는 operator 의 fixed point 로 본다. 역사적으로 어느 관점이 먼저였고, 수학적으로 어느 것이 더 fundamental 한가?

<details>
<summary>해설</summary>

**역사적 순서**:

1. **Bellman (1957)** — "Optimality Principle": 최적 정책의 부분구조가 최적 ⇒ 재귀 방정식
2. **Puterman (1990s–2000s)** — Markov decision processes 교과서에서 operator 형식화
3. **Modern functional analysis** — Complete metric space 에서 fixed point theorem 적용

**수학적 위상**:

| 관점 | Bellman | Operator |
|------|---------|----------|
| 발원 | 최적화 원리 | 함수 공간의 위상 |
| 직관 | "큰 문제 = 한 스텝 + 작은 문제" | "T 의 고정점" |
| 증명 | Contraction 성질로 유도 | Banach theorem 직접 적용 |
| 일반성 | RL에 특화 | 모든 fixed point 문제 |

**결론**:

- **Bellman**: 원래 동기, intuitive
- **Operator**: 수학적으로 more powerful, 일반 정리 활용 가능

이 레포는 **역사적 순서 (Bellman → 정의들 → operator → fixed point theorem)** 를 따름. $\square$

</details>

---

<div align="center">

[◀ 이전: 03. Bellman Expectation Equation 유도](./03-bellman-expectation.md) | [📚 README](../README.md) | [다음 ▶: 05. Value Function 의 고유성과 존재성](./05-value-uniqueness.md)

</div>
