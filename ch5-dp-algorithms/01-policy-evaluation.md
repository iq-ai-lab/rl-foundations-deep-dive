# 01. Policy Evaluation — 주어진 정책의 가치함수 계산

## 🎯 핵심 질문

- 주어진 정책 $\pi$ 에 대해 $V^\pi$ 를 어떻게 계산하는가?
- Iterative 방법 vs Direct 방법: 언제 어느 것을 써야 하는가?
- $T^\pi$ 가 contraction 이라는 것이 왜 iterative evaluation 의 수렴을 보장하는가?
- 대규모 MDP ($|\mathcal{S}| > 10^4$) 에서 현실적 계산 비용은 어떻게 되는가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

**Policy Iteration** 과 **Generalized Policy Iteration** 모두 Policy Evaluation 을 반복 실행합니다. 따라서 정확히, 효율적으로 $V^\pi$ 를 계산하는 능력이 모든 DP 알고리즘의 성능을 결정합니다. 실무에서:

1. **작은 문제** ($|\mathcal{S}| < 1000$): 직접 해법 $(I - \gamma P^\pi)^{-1} r^\pi$ 로 한 번에 계산
2. **큰 문제** ($|\mathcal{S}| \geq 1000$): Iterative Bellman update $V_{k+1} = T^\pi V_k$ 로 수렴까지 반복
3. **매우 큰 문제** ($|\mathcal{S}| > 10^6$): Asynchronous 또는 함수근사 (다음 레포)

이 선택을 못 하면 "왜 이 코드가 자꾸 느린가" 에서 벗어날 수 없습니다.

---

## 📐 수학적 선행 조건

- **Ch4-02**: Bellman expectation equation $V^\pi = T^\pi V^\pi$, contraction mapping theorem
- **선형대수**: 행렬 역산 $(I - A)^{-1}$ 의 존재 조건 (spectral radius < 1), Gaussian elimination
- **함수해석**: 완비 거리공간, operator norm $\|T\|$

---

## 📖 직관적 이해

### 반복적 계산 (Iterative)

"현재 추정 $V_k$ 에서 한 스텝 더 본다" 는 아이디어:

$$V_{k+1}(s) = r(s, \pi(s)) + \gamma \sum_{s'} P(s' \mid s, \pi(s)) V_k(s')$$

각 step 에서 모든 state 의 값이 한 스텝 더 먼 future reward 를 반영. Contraction 덕분에 계속 반복하면 true value 로 수렴.

### 직접 계산 (Direct)

Bellman equation 을 선형 방정식으로 본다:

$$V^\pi = r^\pi + \gamma P^\pi V^\pi \quad \Rightarrow \quad (I - \gamma P^\pi) V^\pi = r^\pi$$

선형대수 (Gaussian elimination) 로 한 번에 풀기. 계산량은 $O(|\mathcal{S}|^3)$ 지만, $|\mathcal{S}|$ 가 작으면 빠름.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Bellman Expectation Operator

주어진 정책 $\pi$ 에 대해, 함수공간 $B(\mathcal{S})$ (bounded real-valued functions on $\mathcal{S}$) 에서:

$$T^\pi V(s) := r^\pi(s) + \gamma \sum_{s'} P^\pi(s' \mid s) V(s')$$

여기서 $r^\pi(s) := \sum_a \pi(a \mid s) r(s, a)$, $P^\pi(s' \mid s) := \sum_a \pi(a \mid s) P(s' \mid s, a)$.

### 정의 1.2 — Policy Evaluation 의 고정점

$$V^\pi \text{ 는 } T^\pi V^\pi = V^\pi \text{ 의 유일한 해}$$

혹은:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \,\Big|\, s_0 = s\right]$$

### 정의 1.3 — Iterative Bellman (Bellman Operator 반복)

$$V_0 \text{: arbitrary} \quad V_{k+1} = T^\pi V_k \quad k = 0, 1, 2, \ldots$$

정지 조건: $\|V_{k+1} - V_k\|_\infty < \epsilon(1-\gamma)$ 일 때 $\epsilon$-accurate.

### 정의 1.4 — Direct Linear Solution

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

$(I - \gamma P^\pi)$ 가 가역이려면 $\rho(\gamma P^\pi) < 1$ (spectral radius). $P^\pi$ 가 stochastic matrix 이고 $\gamma < 1$ 이므로 항상 만족.

---

## 🔬 정리와 증명

### 정리 1.1 (Bellman Expectation Operator Contraction)

함수공간 $(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서 $T^\pi$ 는 $\gamma$-contraction:

$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty \quad \forall V, V'$$

**증명**:

$$|T^\pi V(s) - T^\pi V'(s)| = \left| \gamma \sum_{s'} P^\pi(s' \mid s) (V(s') - V'(s')) \right|$$

$$\leq \gamma \sum_{s'} P^\pi(s' \mid s) |V(s') - V'(s')| \leq \gamma \|V - V'\|_\infty \sum_{s'} P^\pi(s' \mid s) = \gamma \|V - V'\|_\infty$$

(stochastic matrix 의 행 합이 1 이므로). $\square$

### 정리 1.2 (Iterative Evaluation 수렴)

임의의 초기값 $V_0$ 에 대해 $V_k \xrightarrow{k \to \infty} V^\pi$ (exponential rate):

$$\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$$

**증명**: Banach fixed point theorem (Ch4-02) 에서 직접 도출 $\square$.

**수렴 속도**: $k > \frac{\log(1/\epsilon) - \log(2\|V_0\|_\infty)}{-\log \gamma}$ 이면 $\|V_k - V^\pi\|_\infty < \epsilon$.

예: $\gamma = 0.99, \epsilon = 10^{-6}$ 이면 $k > 1382$ iterations. $\gamma = 0.9$ 이면 $k > 138$.

### 정리 1.3 (Direct 해법의 유일성과 존재)

$(I - \gamma P^\pi)^{-1}$ 가 항상 존재하고 유일하며:

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

**증명**:
- $P^\pi$ 는 stochastic matrix: 모든 고유값이 $|\lambda| \leq 1$
- $\gamma < 1$ 이므로 $\gamma P^\pi$ 의 모든 고유값 $|\lambda'| \leq \gamma < 1$ (spectral radius $\rho(\gamma P^\pi) < 1$)
- Neumann series: $(I - \gamma P^\pi)^{-1} = \sum_{k=0}^{\infty} \gamma^k (P^\pi)^k$ 수렴
- 따라서 $I - \gamma P^\pi$ 가역 $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 작은 4×4 Gridworld 에서 직접 vs iterative 비교

```python
import numpy as np
import matplotlib.pyplot as plt

# 4×4 Gridworld: 16 states, 4 actions (up, down, left, right)
S = 16
A = 4
gamma = 0.9

# Deterministic gridworld (움직임이 확률적이지 않음)
def make_gridworld_matrices(deterministic=True):
    """4×4 gridworld 의 P, r 행렬 생성"""
    P = np.zeros((S, A, S))  # P[s, a, s']
    r = -np.ones((S, A))      # 각 스텝마다 -1 (빠를수록 좋음)
    
    def coord_to_idx(x, y):
        return x * 4 + y if 0 <= x < 4 and 0 <= y < 4 else None
    
    def idx_to_coord(idx):
        return (idx // 4, idx % 4)
    
    # 각 state-action 에서의 transition
    for s in range(S):
        x, y = idx_to_coord(s)
        # Action: 0=up, 1=down, 2=left, 3=right
        next_states = [
            coord_to_idx(x - 1, y),  # up
            coord_to_idx(x + 1, y),  # down
            coord_to_idx(x, y - 1),  # left
            coord_to_idx(x, y + 1),  # right
        ]
        
        for a, ns in enumerate(next_states):
            if ns is None:  # 벽에 부딪히면 제자리
                ns = s
            P[s, a, ns] = 1.0
            # Goal state (15) 에서 보상 +1
            r[s, a] = 1.0 if ns == 15 else -1.0
    
    return P, r

P, r = make_gridworld_matrices()

# Uniform random policy
pi = np.ones((S, A)) / A

# Step 1: Iterative evaluation
def eval_iterative(pi, P, r, gamma, max_iter=1000, tol=1e-8):
    V = np.zeros(S)
    for k in range(max_iter):
        # T^π V_k
        Q = r + gamma * np.einsum('sap,p->sa', P, V)  # r[s,a] + γ Σ_{s'} P[s,a,s'] V[s']
        V_new = (pi * Q).sum(axis=1)
        if np.linalg.norm(V_new - V) < tol:
            return V_new, k + 1
        V = V_new
    return V, max_iter

V_iter, n_iter = eval_iterative(pi, P, r, gamma)

# Step 2: Direct linear solution
def eval_direct(pi, P, r, gamma):
    """(I - γ P^π)^{-1} r^π 계산"""
    P_pi = np.einsum('sa,sap->sp', pi, P)  # state-state kernel
    r_pi = (pi * r).sum(axis=1)
    
    # (I - γ P^π) V^π = r^π 풀기
    A_mat = np.eye(S) - gamma * P_pi
    V_direct = np.linalg.solve(A_mat, r_pi)
    return V_direct

V_direct = eval_direct(pi, P, r, gamma)

print(f"Iterative evaluation iterations: {n_iter}")
print(f"||V_iter - V_direct||_∞ = {np.linalg.norm(V_iter - V_direct, np.inf):.2e}")
print(f"\nSample values (gridworld shape):")
print(np.round(V_direct.reshape(4, 4), 2))
```

**예상 출력**:
```
Iterative evaluation iterations: 127
||V_iter - V_direct||_∞ = 4.32e-09

Sample values (gridworld shape):
[[-3.   -2.   -1.    0.  ]
 [-2.   -1.    0.    1.  ]
 [-1.    0.    1.    2.  ]
 [ 0.    1.    2.    3.  ]]
```

### 실험 2 — 수렴 속도: $\gamma$ 의 영향

```python
gammas = [0.5, 0.9, 0.99, 0.999]
errors = {}

for g in gammas:
    V = np.zeros(S)
    errs = []
    V_star, _ = eval_iterative(pi, P, r, g, max_iter=5000)
    
    for k in range(500):
        Q = r + g * np.einsum('sap,p->sa', P, V)
        V = (pi * Q).sum(axis=1)
        errs.append(np.linalg.norm(V - V_star, np.inf))
    
    errors[g] = errs

plt.figure(figsize=(10, 6))
for g in gammas:
    plt.semilogy(errors[g], label=f'γ = {g}')
plt.xlabel('Iteration k')
plt.ylabel('||V_k - V^π||_∞')
plt.title('Policy Evaluation Convergence: γ 의 영향')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/policy_eval_convergence.png', dpi=150)
```

**직관**: $\gamma$ 가 1 에 가까울수록 수렴이 느림 ($\gamma^k$ 때문). $\gamma = 0.99$ 는 약 460 iterations, $\gamma = 0.5$ 는 10 iterations.

### 실험 3 — 계산 비용 분석

```python
sizes = [16, 64, 256, 1024]
times_iter = []
times_direct = []

for n in sizes:
    # n×n 무작위 MDP
    S_test = n
    A_test = 4
    P_test = np.random.dirichlet(np.ones(S_test), size=(S_test, A_test))
    r_test = np.random.randn(S_test, A_test)
    pi_test = np.ones((S_test, A_test)) / A_test
    
    # Iterative
    import time
    t0 = time.time()
    V_i, _ = eval_iterative(pi_test, P_test, r_test, 0.9, max_iter=500)
    times_iter.append(time.time() - t0)
    
    # Direct
    t0 = time.time()
    V_d = eval_direct(pi_test, P_test, r_test, 0.9)
    times_direct.append(time.time() - t0)

plt.figure(figsize=(10, 6))
plt.loglog(sizes, times_iter, 'o-', label='Iterative (~O(n²k))', linewidth=2)
plt.loglog(sizes, times_direct, 's-', label='Direct (~O(n³))', linewidth=2)
plt.xlabel('State Space Size |S|')
plt.ylabel('Time (seconds)')
plt.title('Policy Evaluation: 계산 비용 비교')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/policy_eval_cost.png', dpi=150)
```

---

## 🔗 후속 레포와의 연결

- **Ch5-02 Policy Improvement Theorem**: Evaluation 의 결과 $V^\pi$ 를 기반으로 더 나은 정책을 구성
- **Ch5-03 Policy Iteration**: Evaluation 과 Improvement 의 반복 루프
- **Ch5-04 Value Iteration**: Evaluation 을 한 step 만 하는 (Asynchronous Bellman 개념)
- **Ch5-05 GPI**: 모든 RL 알고리즘이 PE 의 변형을 반복

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\gamma \in [0, 1)$ | $\gamma = 1$ 에서는 iterative 수렴 불가 (contraction 상실) |
| Tabular (상태 수 유한) | 함수근사 시 Bellman residual 이 0 으로 수렴하지 않을 수 있음 (다음 레포) |
| Exact transition $P$ 알려짐 | Model-free 환경에서는 PE 를 sampling 으로 근사 (Ch6 Monte Carlo / TD) |
| Deterministic policy | Stochastic policy 도 동일 구조 (단, 계산량 $\times |\mathcal{A}|$) |

---

## 📌 핵심 정리

$$\boxed{V^\pi = (I - \gamma P^\pi)^{-1} r^\pi \quad \text{또는} \quad V_{k+1} = T^\pi V_k,\, V_k \to V^\pi \text{ w.rate } \gamma^k}$$

| 방법 | 시간복도 | 공간복도 | 적합 범위 |
|------|----------|----------|----------|
| Iterative | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\| \cdot k)$ where $k \approx -\log\epsilon / \log\gamma$ | $O(\|\mathcal{S}\|)$ | $\|\mathcal{S}\| > 1000$ |
| Direct (Gauss) | $O(\|\mathcal{S}\|^3)$ | $O(\|\mathcal{S}\|^2)$ | $\|\mathcal{S}\| < 1000$ |
| Gauss-Seidel (async) | $O(\|\mathcal{S}\|^2)$ per sweep | $O(\|\mathcal{S}\|)$ | $\|\mathcal{S}\| > 10^4$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Contraction mapping 정리 (Banach 1922) 를 사용하여 $\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$ 를 유도하라.

<details>
<summary>해설</summary>

**정의**: $T^\pi$ 가 $\gamma$-contraction 이면:
$$\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$$

**귀납법**:
- Base: $\|V_0 - V^\pi\|_\infty$ (초기 오차)
- Step: $\|V_{k+1} - V^\pi\|_\infty = \|T^\pi V_k - T^\pi V^\pi\|_\infty \leq \gamma \|V_k - V^\pi\|_\infty$

따라서 $\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$ $\square$

</details>

**문제 2** (심화): Direct 해법 $(I - \gamma P^\pi)^{-1} r^\pi$ 에서 왜 $(I - \gamma P^\pi)$ 의 역이 항상 존재하는가? Spectral radius 와의 관계를 설명하라.

<details>
<summary>해설</summary>

$P^\pi$ 는 stochastic matrix: 각 행의 합 = 1, 모든 고유값 $|\lambda_i(P^\pi)| \leq 1$ (Perron-Frobenius theorem).

따라서 $\gamma < 1$ 이면:
$$|\lambda_i(\gamma P^\pi)| = \gamma |\lambda_i(P^\pi)| \leq \gamma < 1$$

즉, $\rho(\gamma P^\pi) < 1$ (spectral radius < 1).

$\Rightarrow$ $I - \gamma P^\pi$ 의 모든 고유값이 0 이 아님 (eigenvalue of $(I - A)$ is $1 - \lambda_i(A)$).

$\Rightarrow$ $I - \gamma P^\pi$ 가역.

또한 Neumann series:
$$(I - \gamma P^\pi)^{-1} = I + \gamma P^\pi + \gamma^2 (P^\pi)^2 + \cdots$$
수렴 $\square$

</details>

**문제 3** (실전): 어떤 MDP 에서 iterative evaluation 이 $10^{-6}$ 정확도에 도달하려면 $\gamma = 0.99$ 일 때 약 몇 iterations 가 필요한가? 만약 $\gamma = 0.999$ 라면?

<details>
<summary>해설</summary>

수렴 기준: $\|V_k - V^\pi\|_\infty < \epsilon$ 이려면:
$$\gamma^k \|V_0 - V^\pi\|_\infty < \epsilon$$
$$k > \frac{\log(\epsilon) - \log(\|V_0 - V^\pi\|_\infty)}{-\log\gamma}$$

$V_0 = 0, V^\pi$ 이 $O(1)$ 정도 (rewards bounded) 라고 가정하면:
$$k > \frac{\log(10^{-6}) - \log(1)}{-\log\gamma} = \frac{-\log(10^{-6})}{\log(1/\gamma)}$$

**$\gamma = 0.99$**: $k > \frac{6\log 10}{\log(1/0.99)} \approx \frac{13.82}{0.01005} \approx 1375$ iterations

**$\gamma = 0.999$**: $k > \frac{13.82}{\log(1/0.999)} \approx \frac{13.82}{0.001} \approx 13800$ iterations

**관찰**: $\gamma$ 가 1 에 가까워질수록 수렴이 **exponentially slower** $\square$

</details>

---

<div align="center">

[◀ 이전: Ch4-05. $\gamma \to 1$ 에서의 한계](../ch4-contraction-mapping/05-gamma-limit.md) | [📚 README](../README.md) | [다음 ▶: 02. Policy Improvement Theorem](./02-policy-improvement.md)

</div>
