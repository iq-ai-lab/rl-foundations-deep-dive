# 01. Banach Fixed Point Theorem — $\gamma$-Contraction 의 수렴 보장

## 🎯 핵심 질문

- **Banach Fixed Point Theorem** 이란 무엇이고, 왜 RL 의 수학적 기초인가?
- 완비 거리공간 $(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서 contraction mapping 이 유일한 고정점을 갖는 이유는?
- Value Iteration $V_{k+1} = T^* V_k$ 가 수렴하고, 그 속도가 $\gamma^k$ 로 bound 되는 이유는?
- Functional Analysis 의 추상 정리가 구체적 RL 알고리즘과 어떻게 연결되는가?

---

## 🔍 왜 이 정리가 RL 의 기초인가

Banach Fixed Point Theorem (1922) 은 **Pure Mathematics** 에서 출발한 정리입니다. 그러나 20세기 후반 RL 의 수학적 정초를 세운 Sutton, Puterman, Bertsekas 는 이 정리를 RL 에 적용하면서 다음을 보장합니다:

1. **Value Iteration 의 유일 해 존재** — Bellman equation $V^* = T^* V^*$ 는 단순 재귀 정의가 아니라, 유일한 해를 갖는 **고정점 방정식**
2. **선형 수렴 속도 (Linear Convergence)** — $V_k$ 가 $V^*$ 로 **지수적으로 빠르게** 수렴: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$
3. **정지 기준의 유도** — $\|V_{k+1} - V_k\|_\infty < \epsilon (1-\gamma)/\gamma$ 일 때 $\|V_k - V^*\|_\infty < \epsilon$ 를 **보장**
4. **Functional Analysis 와 RL 의 다리** — Sup-norm space, stochastic matrix, contraction 의 개념이 Policy Iteration, Q-learning, 현대 Deep RL 까지 모두에 관통

이것이 RL 이 **단순 알고리즘 모음** 이 아니라 **수학적 학문** 이 되는 지점입니다.

---

## 📐 수학적 선행 조건

### 필수: Metric Spaces and Completeness
- 거리공간 $(X, d)$ — open ball, closed ball, convergence
- Cauchy sequence — $\forall \epsilon > 0, \exists N: d(x_n, x_m) < \epsilon$ for $n, m > N$
- Complete metric space — 모든 Cauchy sequence 가 수렴
- **예시**: $(\mathbb{R}, |·|)$ 완비, $(\mathbb{Q}, |·|)$ 불완비

### 필수: Normed Vector Spaces
- Norm $\|\cdot\|: X \to \mathbb{R}_{\geq 0}$ — positive definiteness, homogeneity, triangle inequality
- Banach space — complete normed vector space
- **Sup-norm (Uniform norm)**: $\|f\|_\infty = \sup_{s \in \mathcal{S}} |f(s)|$ for bounded functions

### 선택: Functional Analysis (고급)
- Operator norm, spectral radius
- Contraction의 기하학적 의미

---

## 📖 직관적 이해

### Contraction 의 기하학

**Contraction mapping** $T: X \to X$ 는 모든 점 쌍에 대해 "거리를 줄이는" 함수입니다.

```
▼ 두 점 x, y 의 거리가 T 를 거친 후 γ 배로 축소

    x ─────→ T(x)
    │         │
    │ d(x,y)  │ γ·d(x,y)  (축소됨!)
    │         │
    y ─────→ T(y)

    d(T(x), T(y)) ≤ γ · d(x, y), γ < 1
```

**Iteration 의 효과**: 초기값 $x_0$ 에서 시작해 $x_{k+1} = T(x_k)$ 를 반복하면:
- 매 iteration 마다 거리가 $\gamma$ 배로 줄어듦
- 수렴 속도 $k \approx -\log(\epsilon) / \log(\gamma)$
- $\gamma = 0.9$ 이면 $k \approx 22$ 에서 $\epsilon = 10^{-2}$, $k \approx 130$ 에서 $\epsilon = 10^{-6}$

### RL 의 맥락

**Value Function** 는 $\mathcal{S} \to \mathbb{R}$ 인 bounded function 이고, sup-norm 은 "모든 state 에서 동시에" 거리를 측정:

$$\|V - V'\|_\infty = \max_s |V(s) - V'(s)|$$

Bellman operator $T^*$ 가 contraction 이면, Value Iteration 은 **guaranteed 수렴**.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Metric Space & Distance

**거리공간** $(X, d)$ 는 집합 $X$ 와 함수 $d: X \times X \to \mathbb{R}_{\geq 0}$ 의 쌍으로, 다음을 만족:
1. (양정성) $d(x, y) = 0 \Leftrightarrow x = y$
2. (대칭성) $d(x, y) = d(y, x)$
3. (삼각부등식) $d(x, z) \leq d(x, y) + d(y, z)$

**Convergence**: $(x_n)$ 이 $x$ 로 수렴 ($x_n \to x$) ⟺ $\forall \epsilon > 0, \exists N: d(x_n, x) < \epsilon$ for $n > N$.

### 정의 4.2 — Cauchy Sequence & Completeness

**Cauchy 수열**: $\forall \epsilon > 0, \exists N: d(x_n, x_m) < \epsilon$ for all $n, m > N$.

**완비 (Complete)**: 모든 Cauchy 수열이 수렴.

### 정의 4.3 — Contraction Mapping

함수 $T: X \to X$ 가 **$L$-Contraction** (L은 Lipschitz 상수):
$$d(T(x), T(y)) \leq L \cdot d(x, y) \quad \forall x, y \in X, \quad L < 1$$

$L = \gamma$ 로 표기하면 **$\gamma$-Contraction**.

### 정의 4.4 — Bounded Functions & Sup-Norm Space

**Bounded function space** $B(\mathcal{S})$:
$$B(\mathcal{S}) = \{f: \mathcal{S} \to \mathbb{R} \mid \|f\|_\infty < \infty\}$$

**Sup-norm**: $\|f\|_\infty = \sup_{s \in \mathcal{S}} |f(s)|$.

**거리**: $d(f, g) = \|f - g\|_\infty = \sup_s |f(s) - g(s)|$.

$(B(\mathcal{S}), \|\cdot\|_\infty)$ 는 **Banach space** (complete normed vector space).

---

## 🔬 정리와 증명

### 정리 4.1 (Banach Fixed Point Theorem — 1922)

완비 거리공간 $(X, d)$ 와 $\gamma$-contraction $T: X \to X$ ($\gamma < 1$) 에 대해:

1. **유일성**: 고정점 $x^* \in X$ 가 유일하게 존재: $T(x^*) = x^*$
2. **수렴**: 임의 $x_0 \in X$ 에서 시작한 iteration $x_{k+1} = T(x_k)$ 는 $x^*$ 로 수렴
3. **수렴 속도 (A Priori Bound)**:
$$d(x_k, x^*) \leq \gamma^k \cdot d(x_0, x^*)$$

4. **정지 기준 (A Posteriori Bound)**:
$$d(x_k, x^*) \leq \frac{\gamma}{1-\gamma} d(x_k, x_{k-1})$$

**증명**:

**Step 1 — Monotone decay of distances.**

$x_{k+1} = T(x_k)$ 에서:
$$d(x_{k+1}, x_k) = d(T(x_k), T(x_{k-1})) \leq \gamma \cdot d(x_k, x_{k-1})$$

따라서:
$$d(x_k, x_{k-1}) \leq \gamma^{k-1} d(x_1, x_0)$$

**Step 2 — Cauchy sequence.**

삼각부등식으로 $n > m$:
$$d(x_n, x_m) \leq \sum_{j=m}^{n-1} d(x_{j+1}, x_j) \leq d(x_1, x_0) \sum_{j=m}^{n-1} \gamma^j \leq \frac{\gamma^m}{1-\gamma} d(x_1, x_0)$$

$m \to \infty$ 일 때 $d(x_n, x_m) \to 0$ → Cauchy sequence.

**Step 3 — Convergence by completeness.**

$(X, d)$ 완비이므로 $(x_k)$ 는 어떤 $x^* \in X$ 로 수렴.

**Step 4 — Fixed point.**

Continuity of $T$:
$$x^* = \lim_k x_{k+1} = \lim_k T(x_k) = T(\lim_k x_k) = T(x^*)$$

따라서 $T(x^*) = x^*$.

**Step 5 — Uniqueness.**

$T(x^*) = x^*$ 와 $T(y^*) = y^*$ 면:
$$d(x^*, y^*) = d(T(x^*), T(y^*)) \leq \gamma \cdot d(x^*, y^*)$$

$(1 - \gamma) d(x^*, y^*) \leq 0$ 이고 $\gamma < 1$ 이므로 $d(x^*, y^*) = 0$ → $x^* = y^*$ $\square$.

**Step 6 — A Priori Bound (정확한 속도).**

$$d(x_k, x^*) \leq \gamma^k \cdot d(x_0, x^*)$$

이는 Step 2 의 부분합으로부터, 그리고 $T$ 의 repeated application 으로 도출.

---

## 💻 NumPy 구현 검증

### 실험 1 — 추상 Contraction 의 수렴

```python
import numpy as np
import matplotlib.pyplot as plt

# 간단한 1D contraction: T(x) = 0.7*x + 0.5
def T(x):
    return 0.7 * x + 0.5

# 고정점: x* = 0.7*x* + 0.5 => x* = 5/3
x_star_theory = 0.5 / (1 - 0.7)

gamma = 0.7
x0 = 0.0
iterations = 50
x_seq = [x0]

for k in range(iterations):
    x_seq.append(T(x_seq[-1]))

x_seq = np.array(x_seq)
errors = np.abs(x_seq - x_star_theory)

# A Priori Bound: ||x_k - x*|| <= gamma^k * ||x_0 - x*||
a_priori_bound = (gamma ** np.arange(iterations + 1)) * np.abs(x0 - x_star_theory)

print(f"Theory x* = {x_star_theory:.10f}")
print(f"Iteration 10: x = {x_seq[10]:.10f}, error = {errors[10]:.2e}")
print(f"A Priori Bound at k=10: {a_priori_bound[10]:.2e}")
print(f"Match: {np.allclose(errors[10], a_priori_bound[10], atol=1e-10)}")
```

---

## 🔗 후속 레포와의 연결

### 다음 문서들의 기초
1. **02. $T^\pi$ 가 $\gamma$-Contraction** — affine operator 로서 contraction
2. **03. $T^*$ 가 $\gamma$-Contraction** — nonlinear operator 로서 max-Lipschitz contraction
3. **04. Value Iteration 수렴 보장** — Banach 정리의 직접 응용
4. **Advanced RL** — Bellman backup 의 확장, 함수근사에서의 불안정성

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Complete metric space | Finite state RL 에서 자동 만족 |
| $\gamma < 1$ strictly | $\gamma = 1$ 이면 contraction 깨짐 |
| Bounded reward | Contraction 상수가 유지되려면 필수 |
| Stationary operator | Time-varying 은 별도 분석 필요 |

---

## 📌 핵심 정리

$$\boxed{(X, d) \text{ complete, } T \text{ } \gamma\text{-contraction} \Rightarrow \exists ! x^*: T(x^*) = x^*, \quad \|x_k - x^*\| \leq \gamma^k \|x_0 - x^*\|}$$

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 $\gamma < 1$ 이어야 contraction 이 되는가?

<details>
<summary>해설</summary>

$\gamma = 1$ 이면 nonexpansive 는 되지만 contraction 은 아님. 반복 후 거리가 줄어들지 않음. $\square$

</details>

**문제 2** (심화): Incomplete space 에서 contraction iteration 이 수렴하지 않는 구체적 예시?

<details>
<summary>해설</summary>

Rational numbers 에서 Newton's method: $x_{k+1} = (x_k + 2/x_k)/2$ 는 $\sqrt{2}$ 로 수렴하지만, $\sqrt{2} \notin \mathbb{Q}$. $\square$

</details>

**문제 3** (논문 비평): Bellman (1957) 은 Banach 정리를 명시적으로 사용하지 않았다. Puterman (2005) 에서 왜 Banach 정리를 도입했는가?

<details>
<summary>해설</summary>

Finite MDP 에서는 Bellman 의 주장이 충분했으나, continuous state 및 함수근사 설정에서는 Banach 정리의 완비성, 유일성, 수렴속도가 필수. Modern RL 의 엄밀함을 확보. $\square$

</details>

---

<div align="center">

[◀ 이전: Ch3-05. Deterministic 최적 정책의 존재](../ch3-bellman-optimality/05-deterministic-optimal.md) | [📚 README](../README.md) | [다음 ▶: 02. $T^\pi$ 가 $\gamma$-Contraction](./02-tpi-contraction.md)

</div>
