# 05. Value Function 의 고유성과 존재성

## 🎯 핵심 질문

- Value function $V^\pi$ 의 **존재성과 유일성**을 어떻게 증명하는가?
- Closed-form 해 $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ 는 어떻게 유도되는가?
- $(I - \gamma P^\pi)$ 가 **가역인 조건**은? Spectral radius 와의 관계는?
- Neumann series $\sum_{k=0}^\infty (\gamma P^\pi)^k$ 가 왜 수렴하는가?
- $\gamma < 1$ 이 수학적으로 왜 **반드시 필요**한가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

**Ch2-04** 에서 우리는 Bellman expectation 을 고정점 방정식 $V^\pi = T^\pi V^\pi$ 로 표현했습니다. 하지만 이것이 **정말 해가 존재하는가? 유일한가?** 라는 근본 질문이 남습니다.

이 문서는:
1. **Fixed point theorem** (Banach) 을 적용하여 **존재성 + 유일성** 증명
2. **선형대수적 해** — 행렬 형태 $(I - \gamma P^\pi) V = r^\pi$ 의 직접 풀이
3. **Spectral analysis** — 왜 $\rho(\gamma P^\pi) < 1$ 이 핵심인가
4. **Neumann series** — 무한 급수로 $V^\pi$ 를 표현

이것을 모르면:
- Value Iteration 이 왜 수렴하는지 증명 불가
- Policy Iteration 의 "policy evaluation" 을 정당화 불가
- $\gamma = 1$ 환경이 왜 부서지는지 이해 불가

---

## 📐 수학적 선행 조건

- **Ch2-04**: Bellman operator $T^\pi$, affine 성질, fixed point 개념
- 선형대수: 행렬 가역성, eigenvalue, spectral radius
- 함수공간: $(B(\mathcal{S}), \|\cdot\|_\infty)$, complete metric space
- Banach fixed point theorem (직관만 여기서, 정규는 Ch4)
- 급수 수렴: Neumann series

---

## 📖 직관적 이해

### 가역성의 의미

행렬 $A = I - \gamma P^\pi$ 를 생각해봅시다:

```
연립방정식: A V = r
풀이:       V = A^{-1} r  (A 가역일 때만)

우리의 경우: (I - γP^π) V = r^π
            V = (I - γP^π)^{-1} r^π
```

**A 가 가역인가?** → 모든 eigenvalue 가 0이 아닌가?

$I - \gamma P^\pi$ 의 eigenvalue 들:
- $\lambda$ 가 $I$ 의 eigenvalue → 1
- $\mu$ 가 $P^\pi$ 의 eigenvalue → $\gamma \mu$ 가 $\gamma P^\pi$ 의 eigenvalue
- 따라서 $1 - \gamma \mu$ 가 $I - \gamma P^\pi$ 의 eigenvalue

**$\gamma < 1$ 이고 $|\mu| \leq 1$ (stochastic matrix) 이면**:

$$|1 - \gamma \mu| \geq 1 - \gamma |\mu| \geq 1 - \gamma > 0$$

모든 eigenvalue 가 0이 아님 → **가역!**

### Neumann Series 의 직관

$$V = (I - \gamma P^\pi)^{-1} r^\pi = \sum_{k=0}^\infty (\gamma P^\pi)^k r^\pi$$

이것을 풀어 쓰면:

$$V = r^\pi + \gamma P^\pi r^\pi + \gamma^2 (P^\pi)^2 r^\pi + \cdots$$

**의미**:
- $r^\pi$: 첫 스텝 보상
- $\gamma P^\pi r^\pi$: 한 스텝 뒤의 기대 보상 (폐기된 $\gamma$배)
- $\gamma^2 (P^\pi)^2 r^\pi$: 두 스텝 뒤의 기대 보상 ($\gamma^2$배)
- ...

**이것이 정확히 $V^\pi$ 의 정의** (무한 지평의 discounted return)!

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Spectral Radius

행렬 (또는 연산자) $A$ 의 spectral radius:

$$\rho(A) := \max\{|\lambda| : \lambda \text{ is eigenvalue of } A\}$$

### 정의 5.2 — Stochastic Matrix

행렬 $P \in \mathbb{R}^{n \times n}$ 이 stochastic:

$$P(i,j) \geq 0 \quad \forall i,j, \quad \sum_j P(i,j) = 1 \quad \forall i$$

(모든 행의 합이 1, 모든 원소 비음수)

**성질**: Stochastic matrix 의 spectral radius $\rho(P) \leq 1$.

### 정의 5.3 — Matrix Norm (Op Norm)

$$\|A\|_{\text{op}} := \max_{\|x\| \neq 0} \frac{\|Ax\|}{\|x\|}$$

**성질**: $\|P\|_\infty = 1$ (stochastic $P$ 에 대해, sup-norm 사용).

---

## 🔬 정리와 증명

### 정리 5.1 (Spectral Radius of Stochastic Matrix)

$P$ 가 stochastic matrix 이면:

$$\rho(P) \leq 1$$

**증명**:

**Step 1** — Eigenvalue-eigenvector 관계:

$\lambda$ 가 $P$ 의 eigenvalue, $x$ 가 associated eigenvector:

$$Px = \lambda x$$

**Step 2** — 크기 비교:

$$|Px| = |\lambda x| = |\lambda| |x|$$

$$\|P x\|_\infty \leq \|P\|_\infty \|x\|_\infty = 1 \cdot \|x\|_\infty$$ 

(stochastic matrix 이므로 $\|P\|_\infty = 1$)

**Step 3** — 결합:

$$|\lambda| \|x\|_\infty = \|Px\|_\infty \leq \|x\|_\infty$$

따라서 $|\lambda| \leq 1$ $\square$.

### 정리 5.2 (가역성: $(I - \gamma P^\pi)$ is Invertible)

$P^\pi$ 가 stochastic matrix 이고 $\gamma \in [0, 1)$ 이면, $(I - \gamma P^\pi)$ 는 가역이고:

$$(I - \gamma P^\pi)^{-1} = \sum_{k=0}^\infty (\gamma P^\pi)^k$$

(Neumann series, 절대수렴)

**증명**:

**Step 1** — Spectral radius 계산:

$\mu$ 가 $P^\pi$ 의 eigenvalue ⇒ $|\mu| \leq 1$ (정리 5.1).

$\gamma P^\pi$ 의 eigenvalue: $\gamma \mu$ ⇒ $|\gamma \mu| \leq \gamma < 1$.

따라서:

$$\rho(\gamma P^\pi) \leq \gamma < 1$$

**Step 2** — $(I - \gamma P^\pi)$ 의 가역성:

$\lambda$ 가 $I - \gamma P^\pi$ 의 eigenvalue ⇒ $\lambda = 1 - \gamma \mu$ (for some eigenvalue $\mu$ of $P^\pi$).

$|\gamma \mu| < 1$ 이므로 $|1 - \gamma \mu| \geq 1 - \gamma > 0$.

따라서 모든 eigenvalue 가 0이 아님 ⇒ $(I - \gamma P^\pi)$ 가역. $\square$

**Step 3** — Neumann series:

기하급수:

$$(I - X)^{-1} = \sum_{k=0}^\infty X^k \quad \text{if } \|X\| < 1$$

$X = \gamma P^\pi$ 에 적용:

$$\|(gamma P^\pi)\|_\infty \leq \gamma \|P^\pi\|_\infty = \gamma < 1$$

따라서:

$$(I - \gamma P^\pi)^{-1} = \sum_{k=0}^\infty (\gamma P^\pi)^k \quad \square$$

### 정리 5.3 (Closed-Form Solution for $V^\pi$)

유한 MDP 에서, 고정점 $V^\pi$ 는 유일하며:

$$\boxed{V^\pi = (I - \gamma P^\pi)^{-1} r^\pi = \sum_{k=0}^\infty (\gamma P^\pi)^k r^\pi}$$

**증명**:

**Step 1** — 선형방정식 형태:

Bellman equation $V^\pi = r^\pi + \gamma P^\pi V^\pi$ 를 정렬:

$$(I - \gamma P^\pi) V^\pi = r^\pi$$

**Step 2** — 정리 5.2 에서 $(I - \gamma P^\pi)$ 가역:

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

**Step 3** — Neumann series:

$$(I - \gamma P^\pi)^{-1} = \sum_{k=0}^\infty (\gamma P^\pi)^k$$

대입:

$$V^\pi = \sum_{k=0}^\infty (\gamma P^\pi)^k r^\pi \quad \square$$

### 정리 5.4 (Element-wise Interpretation of Neumann Series)

$$V^\pi(s) = \sum_{k=0}^\infty \sum_{s_1, \ldots, s_k} \gamma^k P^\pi(s_1|s) \cdots P^\pi(s_k|s_{k-1}) r^\pi(s_k)$$

**의미**: $s$ 에서 시작하여, $k$ 스텝 동안 $s \to s_1 \to \cdots \to s_k$ 경로로 이동할 확률 × $k$ 스텝 뒤의 discounted 보상, 모두 합산.

이것이 정확히 **discounted return 의 기대값**.

### 정리 5.5 (Banach Fixed Point — 직관)

$(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서 $T^\pi$ 가 contraction mapping:

$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$$

**그러면**:

1. 유일한 고정점 $V^\pi$ 존재
2. 임의 초기 $V_0$ 에서 $V_{k+1} = T^\pi V_k$ 수렴:
   $$\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty \to 0$$

(정규 증명은 Ch4)

---

## 💻 NumPy 구현 검증

### 실험 1 — Closed-form 해 계산 (행렬 역행)

```python
import numpy as np
from scipy.linalg import solve, inv
import matplotlib.pyplot as plt

# 4x4 Gridworld
S = 16
A = 4
gamma = 0.9

def build_gridworld():
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        row, col = s // 4, s % 4
        actions = [
            ((row - 1) % 4, col),
            ((row + 1) % 4, col),
            (row, (col - 1) % 4),
            (row, (col + 1) % 4)
        ]
        for a, (nr, nc) in enumerate(actions):
            next_s = nr * 4 + nc
            P[s, a, next_s] = 1.0
            R[s, a] = -1.0 if next_s != 15 else 10.0
    return P, R

P, R = build_gridworld()
pi = np.ones((S, A)) / A

# Policy-induced quantities
r_pi = np.einsum('sa,sa->s', pi, R)
P_pi = np.einsum('sa,sap->sp', pi, P)

# 방법 1: 선형방정식 풀기 (I - γP^π)V = r^π
print("Method 1: Solve (I - γP^π)V = r^π")
A_matrix = np.eye(S) - gamma * P_pi
V_solve = solve(A_matrix, r_pi)
print("V^π shape:", V_solve.shape)
print("V^π grid:\n", V_solve.reshape(4, 4).round(2))

# 방법 2: 행렬 역행 직접 계산
print("\nMethod 2: (I - γP^π)^{-1} r^π")
A_inv = inv(A_matrix)
V_inv = A_inv @ r_pi
print("Max diff from solve:", np.abs(V_solve - V_inv).max())

# 가역성 확인
print("\nInvertibility check:")
print("det(I - γP^π) =", np.linalg.det(A_matrix))
print("Eigenvalues of (I - γP^π):")
eigvals = np.linalg.eigvals(A_matrix)
print("  Min |eigenvalue| =", np.abs(eigvals).min())
print("  All non-zero?", np.all(np.abs(eigvals) > 1e-10))

print("\nSpectral radius of γP^π:")
eigvals_P = np.linalg.eigvals(gamma * P_pi)
rho = np.abs(eigvals_P).max()
print("  ρ(γP^π) =", rho)
print("  ρ < γ =", rho, "<", gamma, "?", rho < gamma)
```

### 실험 2 — Neumann Series 수렴

```python
# (I - γP^π)^{-1} = Σ (γP^π)^k

print("Neumann Series Convergence:")
print("k    | ||Σ_{j=0}^k (γP^π)^j|| | Error to truth")
print("-" * 55)

partial_sum = np.eye(S)
for k in range(30):
    if k > 0:
        partial_sum = partial_sum + (gamma * P_pi) ** k
    
    # Approximate (I - γP^π)^{-1}
    A_inv_approx = partial_sum @ A_matrix
    error_to_identity = np.abs(A_inv_approx - np.eye(S)).max()
    
    if k % 5 == 0 or k < 5:
        norm_partial = np.linalg.norm(partial_sum, ord=np.inf)
        print(f"{k:2d}   | {norm_partial:24.6f} | {error_to_identity:.2e}")
    
    if error_to_identity < 1e-8:
        print(f"Converged at k={k}")
        break

# V^π 직접 계산 from Neumann
print("\nV^π from Neumann series:")
partial_sums_V = np.zeros((S, 15))
partial = np.zeros(S)
for k in range(15):
    partial = partial + (gamma * P_pi) ** k @ r_pi if k == 0 else partial + (gamma * P_pi) ** k @ r_pi
    partial_sums_V[:, k] = partial

# 정확한 값과 비교
V_true = V_solve
errors = [np.abs(partial_sums_V[:, k] - V_true).max() for k in range(15)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(range(15), errors, 'o-', linewidth=2, markersize=6)
ax.axhline(1e-8, color='r', linestyle='--', alpha=0.5, label='Numerical precision')
ax.set_xlabel('Neumann series terms k')
ax.set_ylabel('||V_k - V*||_∞ (log scale)')
ax.set_title('Convergence of Neumann Series for V^π')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('neumann_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 실험 3 — $\gamma$ 에 따른 가역성 (spectral radius)

```python
gammas = np.linspace(0.0, 1.0, 50)
spectral_radii_P = []
is_invertible = []

for g in gammas:
    A_mat = np.eye(S) - g * P_pi
    eigvals = np.linalg.eigvals(A_mat)
    min_abs_eigval = np.abs(eigvals).min()
    is_invertible.append(min_abs_eigval > 1e-10)
    
    # ρ(γP^π)
    eigvals_gP = np.linalg.eigvals(g * P_pi)
    rho_gP = np.abs(eigvals_gP).max()
    spectral_radii_P.append(rho_gP)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) Spectral radius of γP^π
ax = axes[0]
ax.plot(gammas, spectral_radii_P, 'b-', linewidth=2, label='ρ(γP^π)')
ax.plot(gammas, gammas, 'r--', linewidth=2, label='γ')
ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
ax.fill_between(gammas, 0, 1, where=(np.array(spectral_radii_P) < np.array(gammas)), alpha=0.2, color='green', label='ρ < γ (invertible)')
ax.set_xlabel('Discount factor γ')
ax.set_ylabel('Spectral radius')
ax.set_title('ρ(γP^π) vs γ')
ax.legend()
ax.grid()

# (b) Invertibility status
ax = axes[1]
ax.scatter(gammas[is_invertible], [1]*sum(is_invertible), 
          color='green', s=30, label='Invertible (ρ < 1)')
ax.scatter(gammas[~np.array(is_invertible)], [0]*sum(~np.array(is_invertible)), 
          color='red', s=30, label='Not invertible (ρ ≥ 1)')
ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='γ = 1')
ax.set_xlabel('Discount factor γ')
ax.set_ylabel('Invertible?')
ax.set_ylim([-0.1, 1.1])
ax.set_title('Invertibility of (I - γP^π)')
ax.legend()
ax.grid(axis='x')

plt.tight_layout()
plt.savefig('gamma_invertibility.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 실험 4 — Value Iteration 수렴 속도 (선형수렴)

```python
# V_{k+1} = T^π V_k 의 오차: ||V_k - V*||_∞ ≤ γ^k ||V_0 - V*||_∞

V_true = V_solve
V_init = np.zeros(S)  # 또는 np.random.randn(S)

errors_actual = []
errors_theoretical = np.abs(V_init - V_true).max() * (gamma ** np.arange(100))

V = V_init.copy()
for k in range(100):
    errors_actual.append(np.abs(V - V_true).max())
    Q = R + gamma * np.einsum('sap,p->sa', P, V)
    V = np.einsum('sa,sa->s', pi, Q)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) Log scale: 선형수렴 시각화
ax = axes[0]
ax.semilogy(range(100), errors_actual, 'o-', label='Actual error', linewidth=2, markersize=4)
ax.semilogy(range(100), errors_theoretical, '--', label=f'γ^k ||V_0 - V*||_∞ (γ={gamma})', linewidth=2)
ax.set_xlabel('Iteration k')
ax.set_ylabel('Error ||V_k - V*||_∞ (log scale)')
ax.set_title('Linear Convergence of Value Iteration')
ax.legend()
ax.grid()

# (b) Ratio: 실제 오차 감소율
ax = axes[1]
ratios = [errors_actual[k+1] / errors_actual[k] for k in range(len(errors_actual)-1)]
ax.plot(range(len(ratios)), ratios, 'o-', linewidth=2, markersize=4)
ax.axhline(gamma, color='r', linestyle='--', linewidth=2, label=f'Theoretical rate γ = {gamma}')
ax.set_xlabel('Iteration k')
ax.set_ylabel('Error ratio ||V_{k+1} - V*|| / ||V_k - V*||')
ax.set_title('Contraction Rate (should converge to γ)')
ax.legend()
ax.grid()
ax.set_ylim([0.8, 1.0])

plt.tight_layout()
plt.savefig('value_iteration_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 후속 레포와의 연결

- **Ch4 (Contraction Mapping)**: Banach fixed point theorem 정규 증명
- **Ch5 (Value Iteration)**: $\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$ 의 정지 기준 유도
- **Ch5 (Policy Iteration)**: Policy evaluation 에서 $(I - \gamma P^\pi)^{-1} r^\pi$ 계산
- **Ch6 (Advanced Properties)**: Stationary distribution, discounted occupancy measure 와의 연결

---

## ⚖️ 가정과 한계

| 가정 | 의미 | 한계 | 대응 |
|------|------|------|------|
| Finite MDP | $\|\mathcal{S}\|, \|\mathcal{A}\|$ 유한 | Continuous 상태 | Operator 추상성 유지 |
| $\gamma < 1$ | Spectral radius 제약 | $\gamma = 1$ 무한합 | Episodic 또는 평균보상 |
| Bounded $r^\pi$ | $\|r^\pi\|_\infty \leq R_{\max}$ | Unbounded reward | 결과 재정의 |
| 정책 고정 | $\pi$ 유지 | 정책 변화 | Ch5 에서 iteration |

---

## 📌 핵심 정리

$$\boxed{V^\pi = (I - \gamma P^\pi)^{-1} r^\pi = \sum_{k=0}^\infty (\gamma P^\pi)^k r^\pi}$$

**조건**: $\gamma \in [0, 1)$, $P^\pi$ stochastic

| 항 | 역할 |
|----|------|
| $(I - \gamma P^\pi)$ | Bellman equation 의 선형 형태 |
| $\rho(\gamma P^\pi) < 1$ | 가역성 보장 (spectral radius) |
| Neumann series | 무한 급수로 해를 표현 |
| **선형 수렴** | $\gamma^k$ 속도 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 5.2 의 Neumann series $(I - \gamma P^\pi)^{-1} = \sum_{k=0}^\infty (\gamma P^\pi)^k$ 에서, 왜 "절대 수렴" 이 중요한가? 어떻게 보장되는가?

<details>
<summary>해설</summary>

**절대 수렴의 정의**:

급수 $\sum a_k$ 가 절대수렴 ⇔ $\sum |a_k|$ 수렴.

**우리의 경우**:

$\|\gamma P^\pi\|_\infty < 1$ 이므로, norm 으로의 급수:

$$\sum_{k=0}^\infty \|\gamma P^\pi\|_\infty^k = \sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma} < \infty$$

**절대 수렴의 필요성**:

수렴 급수 $\sum a_k$ 에 대해:
- $\sum |a_k|$ 수렴 → 항의 순서 바꿔도 같은 값 (unconditional convergence)
- $\sum |a_k|$ 발산 → 순서 바꾸면 다른 값 (conditional convergence 의 Riemann rearrangement)

**RL 에서**:

$(\gamma P^\pi)^k$ 의 순서를 바꾸거나 (예: matrix 곱셈 순서) 일부 항만 계산할 때도, 절대수렴 덕분에 결과가 일정 → **robust computability**. $\square$

</details>

**문題 2** (심化): 정리 5.1 에서 stochastic matrix 의 spectral radius $\rho(P) \leq 1$ 을 증명했다. 등호 $\rho(P) = 1$ 인 경우는 언제 가능한가? 예를 들어?

<details>
<summary>해설</summary>

**$\rho(P) = 1$ 이 되는 경우**:

$P$ 가 stochastic 이고 $\rho(P) = 1$ ⇔ **eigenvalue 1 이 존재** (doubly stochastic 또는 cyclic).

**예시 1 (Cyclic)**:

$$P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

(상태 0 ↔ 1 번갈아 이동)

Eigenvalue: $1, -1$ → $\rho(P) = 1$.

**예시 2 (Doubly stochastic / 정상 분포)**:

$$P = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$$

Eigenvalue: $1, 0$ → $\rho(P) = 1$.

**우리의 경우**:

$\gamma < 1$ 이므로 $\rho(\gamma P) = \gamma \cdot \rho(P) \leq \gamma < 1$ → always < 1.

**$\gamma = 1$ 문제**:

만약 $\gamma = 1$ 이면, $\rho(P) = 1$ 이 가능 → $(I - P)$ 가 non-invertible.

이것이 **$\gamma = 1$ 이 infinite-horizon discounted 에서 부서지는 이유**. $\square$

</details>

**問題 3** (論文 評論): Puterman (2005) 는 "MDP standard text" 에서 $(I - \gamma P^\pi)^{-1}$ 형식을 선호하고, Bertsekas & Tsitsiklis (1996) 는 operator/functional analysis 관점을 강조한다. 계산 관점에서 어느 관점이 더 실용적인가?

<details>
<summary>해설</summary>

**비교**:

| 관점 | Puterman (행렬) | Bertsekas (함수공간) |
|------|---|---|
| 구현 | `V = solve((I - γP)V = r)` | `V = V; for k: V = T^π V` |
| 안정성 | $(I - γP)$ 수치 조건 수 주의 | Iteration 안정 (contraction) |
| 속도 | $O(n^3)$ (행렬 역행) | $O(n^2) \times k$ iterations |
| 메모리 | Full matrix 저장 | Vector 만 필요 |
| 특성 | 유한 단계 정확해 | 무한 단계 수렴 |
| 함수근사 | 직접 어려움 | 자연스러운 일반화 |

**실전**:

- **Tabular RL (small $n$)**: 행렬 풀이 가능, 이론적으로도 clean
- **Large-scale / Deep RL**: Iteration 필수 (행렬 저장 불가)
- **이론 분석**: Function space 관점 (contraction, GPI 등)

**이 레포의 선택**: 둘 다 제시 — closed-form $(I - \gamma P^\pi)^{-1}$ 를 보인 후, Neumann series 로 iteration 과의 연결. $\square$

</details>

---

<div align="center">

[◀ 이전: 04. Operator 표기법](./04-bellman-operator.md) | [📚 README](../README.md) | [다음 ▶: Ch3-01. Optimal Value Function 의 정의](../ch3-bellman-optimality/01-optimal-value-function.md)

</div>
