# 02. $T^\pi$ 가 $\gamma$-Contraction in Sup-Norm — Policy Evaluation 의 수렴 보장

## 🎯 핵심 질문

- **Policy evaluation operator** $T^\pi$ 가 왜 contraction 인가?
- Affine operator 의 contraction 을 증명하는 핵심은 stochastic matrix $P^\pi$ 의 성질인가?
- $\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$ 의 증명이 왜 간단하면서도 강력한가?
- Policy Iteration 에서 Policy Evaluation step 의 수렴이 보장되는 원리는?

---

## 🔍 왜 이 정리가 Policy Iteration 의 기초인가

Value Iteration 은 $T^*$ (최적 operator) 를 반복하지만, Policy Iteration 은 고정 정책 $\pi$ 에 대한 $T^\pi$ 를 반복합니다. 이것이 "정책 평가 (Policy Evaluation)" 단계입니다:

$$V_{k+1}^\pi = T^\pi V_k^\pi = r^\pi + \gamma P^\pi V_k^\pi$$

이 iteration 이 수렴하려면 $T^\pi$ 가 contraction 이어야 합니다. 다행히 $T^\pi$ 는:
- **Affine** (선형 + 상수항): $T^\pi V = r^\pi + \gamma P^\pi V$
- **Stochastic matrix**: $P^\pi$ 의 각 행이 확률 분포 → $\|P^\pi\|_\infty = 1$

이 두 성질로부터 **$\gamma$-contraction** 이 자동으로 따릅니다.

---

## 📐 수학적 선행 조건

### 필수
- Banach Fixed Point Theorem (Ch4-01)
- Stochastic matrix 의 정의 — 각 행 합이 1
- Matrix norm: $\|P\|_\infty = \max_s \sum_{s'} |P(s, s')|$ (row sum norm)

### 선택
- Spectral radius $\rho(P) = \max |\lambda_i|$
- Neumann series $\sum_{k=0}^\infty \gamma^k P^\pi = (I - \gamma P^\pi)^{-1}$

---

## 📖 직관적 이해

### Affine Operator 의 축소

**Affine operator** $T: V \mapsto a + BV$ (상수항 $a$ + 행렬 $B$) 에서:

$$T(V) - T(V') = B(V - V')$$

따라서 거리 축소는 $B$ 의 norm 에만 달려 있습니다.

**$T^\pi$ 의 경우**:
$$T^\pi V - T^\pi V' = r^\pi + \gamma P^\pi V - (r^\pi + \gamma P^\pi V') = \gamma P^\pi(V - V')$$

거리:
$$\|T^\pi V - T^\pi V'\|_\infty = \|\gamma P^\pi (V - V')\|_\infty$$

**Stochastic matrix 의 성질**: $P^\pi$ 의 각 행이 확률분포이므로:
$$\sum_{s'} P^\pi(s, s') = 1 \quad \forall s$$

이것은:
$$\|(P^\pi V)(s)\| = \left|\sum_{s'} P^\pi(s, s') V(s')\right| \leq \sum_{s'} P^\pi(s, s') |V(s')| \leq \max_{s'} |V(s')| = \|V\|_\infty$$

따라서 $\|P^\pi V\|_\infty \leq \|V\|_\infty$.

**결론**:
$$\|T^\pi V - T^\pi V'\|_\infty = \gamma \|P^\pi(V - V')\|_\infty \leq \gamma \|V - V'\|_\infty$$

즉, $T^\pi$ 는 $\gamma$-contraction.

---

## ✏️ 엄밀한 정의

### 정의 4.2.1 — Policy Evaluation Operator

고정 정책 $\pi$ 에 대해:

$$T^\pi V(s) := r^\pi(s) + \gamma \sum_{s'} P^\pi(s, s') V(s')$$

또는 행렬 표기:
$$T^\pi V = r^\pi + \gamma P^\pi V$$

여기서:
- $r^\pi(s) = \sum_a \pi(a|s) r(s, a)$ — 평균 immediate reward
- $P^\pi(s, s') = \sum_a \pi(a|s) P(s'|s, a)$ — 정책에 따른 전이 kernel

### 정의 4.2.2 — Stochastic Matrix (Row-Stochastic)

행렬 $P \in \mathbb{R}^{n \times n}$ 가 row-stochastic:
$$P(s, s') \geq 0 \quad \forall s, s', \quad \sum_{s'} P(s, s') = 1 \quad \forall s$$

**성질**: $\|Px\|_\infty \leq \|x\|_\infty$ (sup-norm 에서)

### 정의 4.2.3 — Policy Evaluation: Fixed Point Equation

$V^\pi = T^\pi V^\pi$ 를 정책 $\pi$ 의 **value function** 또는 **value vector** 라 부르며:

$$V^\pi(s) = r^\pi(s) + \gamma \sum_{s'} P^\pi(s, s') V^\pi(s')$$

이는 **Bellman expectation equation** 이기도 합니다.

---

## 🔬 정리와 증명

### 정리 4.2.1 — $T^\pi$ 가 $\gamma$-Contraction

$(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서 정책 $\pi$ 에 대한 operator $T^\pi$ 는 $\gamma$-contraction:

$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$$

**증명**:

$$\|T^\pi V - T^\pi V'\|_\infty = \|r^\pi + \gamma P^\pi V - r^\pi - \gamma P^\pi V'\|_\infty$$
$$= \|\gamma P^\pi(V - V')\|_\infty$$
$$= \gamma \|P^\pi(V - V')\|_\infty$$

$P^\pi$ 가 row-stochastic 이므로, 임의 $s$ 에 대해:

$$|(P^\pi(V - V'))(s)| = \left|\sum_{s'} P^\pi(s, s') (V(s') - V'(s'))\right|$$
$$\leq \sum_{s'} P^\pi(s, s') |V(s') - V'(s')|$$
$$\leq \sum_{s'} P^\pi(s, s') \|V - V'\|_\infty$$
$$= \|V - V'\|_\infty \quad \text{(행 합이 1이므로)}$$

따라서:
$$\|P^\pi(V - V')\|_\infty = \max_s |(P^\pi(V - V'))(s)| \leq \|V - V'\|_\infty$$

결론:
$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty \quad \square$$

### 정리 4.2.2 — Policy Evaluation 의 수렴 (Banach 정리의 응용)

Policy Evaluation iteration $V_k^{\pi} := T^\pi V_{k-1}^\pi$ (임의 초기값 $V_0^\pi$) 는 유일 고정점 $V^\pi$ 로 수렴:

$$\|V_k^\pi - V^\pi\|_\infty \leq \gamma^k \|V_0^\pi - V^\pi\|_\infty$$

**증명**: Theorem 4.1 (Banach Fixed Point Theorem) 에서 $T = T^\pi$ 를 대입.

### 정리 4.2.3 — Closed Form Solution (Finite MDP)

정책 $\pi$ 가 고정되면, Bellman equation $V^\pi = r^\pi + \gamma P^\pi V^\pi$ 는 선형:

$$(I - \gamma P^\pi) V^\pi = r^\pi$$

$P^\pi$ 의 spectral radius $\rho(P^\pi) < 1$ (finite stochastic matrix) 이므로 $(I - \gamma P^\pi)$ 는 가역:

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

또한 Neumann series:
$$V^\pi = \sum_{k=0}^{\infty} (\gamma P^\pi)^k r^\pi$$

이는 $V^\pi(s) = r^\pi(s) + \gamma \sum_{s'} P^\pi(s, s') V^\pi(s')$ 의 다른 형태입니다.

---

## 💻 NumPy 구현 검증

### 실험 1 — Simple Policy의 Value Function 수렴

```python
import numpy as np
import matplotlib.pyplot as plt

# 4-state MDP
S = 4
gamma = 0.9
np.random.seed(0)

# Deterministic policy π(s) = fixed action
pi_action = np.array([0, 1, 0, 1])  # state -> action

# Random transition (state-action -> state')
P_full = np.random.dirichlet(np.ones(S), size=(S, 4))  # P[s, a, s']
R_full = np.random.randn(S, 4)

# 정책에 따른 P^π, r^π
P_pi = np.array([P_full[s, a] for s, a in enumerate(pi_action)])
r_pi = np.array([R_full[s, a] for s, a in enumerate(pi_action)])

# Iteration: V_k = r^π + γ P^π V_{k-1}
V = np.zeros(S)
errors = []

for k in range(100):
    V_new = r_pi + gamma * (P_pi @ V)
    error = np.linalg.norm(V_new - V, ord=np.inf)
    errors.append(error)
    V = V_new

# Closed form: V^π = (I - γP^π)^{-1} r^π
V_star = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)

errors = np.array(errors)
a_priori = gamma ** np.arange(len(errors)) * np.abs(r_pi).max() / (1 - gamma)

print(f"Final V (iteration): {V}")
print(f"True  V (closed form): {V_star}")
print(f"Max difference: {np.abs(V - V_star).max():.2e}")

plt.semilogy(errors, 'b-', label='Actual error', linewidth=2)
plt.semilogy(a_priori, 'r--', label='$\\gamma^k$ bound', linewidth=2)
plt.xlabel('Iteration k')
plt.ylabel('$\|V_k^\\pi - V^\\pi\|_\\infty$ (log)')
plt.legend()
plt.grid(True)
plt.title('Policy Evaluation Convergence')
plt.tight_layout()
plt.savefig('/tmp/policy_eval_convergence.png', dpi=120)
```

### 실험 2 — Stochastic Policy 의 contraction 검증

```python
# Stochastic policy (여러 action 에 확률 분배)
pi = np.random.dirichlet(np.ones(4), size=S)

# P^π = Σ_a π(a|s) P(s, a, s')
P_pi_stochastic = np.einsum('sa,sap->sp', pi, P_full)
r_pi_stochastic = np.einsum('sa,sa->s', pi, R_full)

# Two random value functions
V1 = np.random.randn(S)
V2 = np.random.randn(S)

# Apply T^π to both
TV1 = r_pi_stochastic + gamma * (P_pi_stochastic @ V1)
TV2 = r_pi_stochastic + gamma * (P_pi_stochastic @ V2)

# Check contraction
dist_before = np.linalg.norm(V1 - V2, ord=np.inf)
dist_after = np.linalg.norm(TV1 - TV2, ord=np.inf)
bound = gamma * dist_before

print(f"||T^π(V1) - T^π(V2)||_∞   = {dist_after:.6f}")
print(f"γ · ||V1 - V2||_∞          = {bound:.6f}")
print(f"Contraction verified: {dist_after <= bound + 1e-10}")
```

---

## 🔗 후속 레포와의 연결

- **Ch4-03**: $T^*$ 의 nonlinear contraction (max 연산)
- **Ch4-04**: Value Iteration + Policy Evaluation 의 GPI 통합
- **Ch5**: Policy Iteration, Q-learning, Actor-Critic 의 수렴

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Finite state/action | Infinite state: function approximation 필요 |
| Deterministic policy (또는 stochastic) | 주어진 고정 정책 |
| $\gamma < 1$ | 평균보상 MDP 로의 확장 필요 |
| Bounded reward | 무한 reward 의 수렴 불보장 |

---

## 📌 핵심 정리

$$\boxed{\|T^\pi V - T^\pi V'\|_\infty = \gamma \|P^\pi(V - V')\|_\infty \leq \gamma \|V - V'\|_\infty}$$

**Policy Evaluation** 은 Banach 정리에 의해 유일 $V^\pi$ 로 선형 수렴.

---

## 🤔 생각해볼 문제

**문제 1**: Stochastic matrix $P^\pi$ 에서 왜 $\|P^\pi V\|_\infty \leq \|V\|_\infty$ 인가? 만약 $P^\pi$ 의 행 합이 1이 아니라면?

<details>
<summary>해설</summary>

행 합이 1이므로 $|(P^\pi V)(s)| = |\sum_{s'} P^\pi(s, s') V(s')| \leq \max_s |V(s)|$. 행 합이 1이 아니면 벡터의 크기가 커질 수 있음. $\square$

</details>

**문제 2**: Closed form $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ 가 존재하려면 $(I - \gamma P^\pi)$ 가 가역이어야 하는가? 왜?

<details>
<summary>해설</summary>

Stochastic matrix $P^\pi$ 의 spectral radius $\rho(P^\pi) = 1$ (largest eigenvalue) 이므로, $\gamma < 1$ 이면 $\gamma P^\pi$ 의 spectral radius $\gamma < 1$. 따라서 $I - \gamma P^\pi$ 는 invertible. $\square$

</details>

**문제 3**: Policy Iteration 에서 Policy Evaluation 을 정확히 할지 (폐형식) vs 대략적으로 (몇 iteration) 할지의 trade-off?

<details>
<summary>해설</summary>

Bellman 이 "truncated policy evaluation" 을 제안: $n$ iteration 만 수행하면 $O(\gamma^n)$ 오차로 정책 개선 가능. 현대에는 GPI (임의 interleaving) 로 통합. $\square$

</details>

---

<div align="center">

[◀ 이전: 01. Banach Fixed Point Theorem](./01-banach-fixed-point.md) | [📚 README](../README.md) | [다음 ▶: 03. $T^*$ 가 $\gamma$-Contraction](./03-tstar-contraction.md)

</div>
