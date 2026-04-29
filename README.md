<div align="center">

# 🧭 RL Foundations Deep Dive

### MDP 의 **6-tuple 정의**

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$$

### 를 **외우는 것** 과,

### **Markov 성질**

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)$$

### 이 왜 dynamic programming 가능성의 **필수 조건** 인지, **stationary Markovian policy** 만으로 최적성이 충분 (Puterman 2005) 함을 증명할 수 있는 것은 **다르다.**

<br/>

> ***Bellman optimality equation***
>
> $$V^\star(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\star(s') \right]$$
>
> *을 **쓰는 것** 과, Bellman optimality operator $T^\star$ 의 **$\gamma$-contraction***
>
> $$\|T^\star V - T^\star V'\|_\infty \leq \gamma \|V - V'\|_\infty$$
>
> *을 증명하고, **Banach fixed point theorem** 으로 Value Iteration 의 유일한 고정점 존재와 linear convergence*
>
> $$\|V_k - V^\star\|_\infty \leq \gamma^k \|V_0 - V^\star\|_\infty$$
>
> *를 한 줄씩 유도할 수 있는 것은 다르다.*
>
> *Policy Iteration 을 **이름으로 아는 것** 과, Howard (1960) 의 정리 — **policy evaluation + policy improvement 의 반복이 finite MDP 에서 유한 step 내 최적 정책 도달** — 을 policy 의 유한성 + strict improvement 로 증명할 수 있는 것은 다르다.*
>
> ***Performance Difference Lemma** (Kakade 2003)*
>
> $$V^{\pi'}(\rho) - V^\pi(\rho) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi'}}\bigl[ \mathbb{E}_{a \sim \pi'}[A^\pi(s, a)] \bigr]$$
>
> *가 왜 Policy Gradient theorem · TRPO · PPO 의 **모든 monotonic improvement 이론의 출발점** 인지 알고 쓰는 것은 다르다.*
>
> *`γ ∈ [0, 1)` 의 제약이 단순한 hyperparameter 가 아니라 **bounded reward 에서의 무한 합 수렴 조건** 이고, `γ = 1` 에서 episodic vs average-reward MDP 로 갈라지는 본질적 분기점임을 알고 쓰는 것은 다르다.*

<br/>

**다루는 정리·알고리즘 (이론 계보순)**

Bellman 1957 *Dynamic Programming* · Howard 1960 *Policy Iteration* · Banach 1922 *Fixed Point Theorem* · Blackwell 1965 *Discounted DP* · Puterman 2005 *MDP 표준* · Bertsekas & Tsitsiklis 1996 *Neuro-DP* · Kakade 2003 *Performance Difference Lemma* · Tsitsiklis & Van Roy 1997 *TD with Linear FA* · Sutton & Barto 2018 *RL Bible* · Szepesvári 2010 *Algorithms for RL* · Jin et al. 2020 *LSVI-UCB*

<br/>

**핵심 질문**

> Markov Decision Process · Bellman equation · Dynamic Programming 은 왜 RL 의 **수학적 정초** 이며, **Markov 성질 · $\gamma$-contraction · Banach fixed point · Howard 의 monotonic improvement** 가 어떻게 **Value Iteration · Policy Iteration · GPI** 의 수렴을 보장하는가 — Bellman 1957 부터 현대 RL 의 Performance Difference Lemma 까지 한 줄씩 유도합니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557C?style=flat-square)](https://matplotlib.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-0081A5?style=flat-square)](https://gymnasium.farama.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-1.0-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-280개-success?style=flat-square)](./README.md)
[![Proofs](https://img.shields.io/badge/엄밀한_증명-130+개-9c27b0?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/표준_예제_재현-15개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

RL 입문 자료는 대부분 **"Bellman equation 은 재귀이고, 풀면 된다"** 또는 **"Value Iteration 을 돌리면 수렴한다"** 에서 멈춥니다. 하지만 Bellman equation 이 **재귀 정의일 뿐인지 수렴 보장이 있는 정리인지**, $T^*$ 가 왜 sup-norm 에서 $\gamma$-contraction 인지, 왜 $\gamma \in [0, 1)$ 이어야 하고 $\gamma = 1$ 에서 무엇이 부서지는지, Policy Iteration 이 왜 **유한 step 안에** 최적 정책을 보장하는지, GPI 사각 다이어그램이 왜 **모든 RL 알고리즘의 통합 프레임** 인지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "MDP 는 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 로 정의된다" | **Puterman 2005** — 6-tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ 의 각 요소의 measurability, transition kernel $P(\cdot \mid s, a)$ 가 stochastic kernel 이어야 하는 이유, **Markov 성질** $P(s_{t+1} \mid h_t) = P(s_{t+1} \mid s_t, a_t)$ 가 어떻게 history 의 무한 차원을 state 의 유한 차원으로 환원해 DP 를 가능케 하는지 $\square$ |
| "Stationary policy 만 보면 된다" | **Puterman 정리** — finite MDP, discounted infinite-horizon 에서 **deterministic stationary Markovian policy 중 최적이 존재**. History-dependent · stochastic policy 가 더 좋아질 수 없음을 contraction · monotonicity 로 증명 $\square$ |
| "Bellman equation 은 $V(s) = r + \gamma \sum P V(s')$ 이다" | **재귀의 수학적 위상** — Bellman expectation equation $V^\pi = T^\pi V^\pi$, optimality equation $V^* = T^* V^*$. $T^\pi V := r^\pi + \gamma P^\pi V$ 가 affine, $T^*$ 가 nonlinear. **고유성**: $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ 로 closed form (finite MDP), $(I - \gamma P^\pi)$ 가역성을 spectral radius $\rho(\gamma P^\pi) < 1$ 로 증명 $\square$ |
| "$\gamma$ 는 미래 reward 의 가중치다" | **수학적 필요성** — bounded reward $\|R\|_\infty \leq R_{\max}$ 에서 $G_t = \sum_k \gamma^k R_{t+k+1}$ 가 absolute convergent 이려면 $\gamma \in [0, 1)$. $\gamma = 1$ 에서는 episodic ($\exists T < \infty$ terminal) 또는 **average-reward MDP** $J = \lim_{T \to \infty} (1/T) \mathbb{E}[\sum_{t=0}^{T-1} R_t]$ 로 분기. $\gamma \to 1$ 에서 contraction rate 약화, value iteration 수렴 속도 $O(\gamma^k)$ 가 무너짐 |
| "Value Iteration 은 수렴한다" | **Banach Fixed Point Theorem** — 완비 거리공간 $(B(\mathcal{S}), \|\cdot\|_\infty)$ 에서 $T^*$ 가 $\gamma$-contraction $\Rightarrow$ 유일 고정점 $V^*$ 존재 + iteration $V_{k+1} = T^* V_k$ 가 $V^*$ 로 linear rate 수렴 $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$ $\square$. **정지 기준** $\|V_{k+1} - V_k\|_\infty < \epsilon (1-\gamma) / \gamma$ 의 도출 |
| "$T^\pi$ 와 $T^*$ 둘 다 contraction 이다" | **증명의 핵심**: $T^\pi$ 는 **affine**, contraction 은 $\|\gamma P^\pi (V - V')\|_\infty \leq \gamma \|V - V'\|_\infty$ 에서 ($P^\pi$ 가 stochastic matrix → 행 합 1). $T^*$ 는 **nonlinear** ($\max_a$), $\|\max_a f - \max_a g\|_\infty \leq \max_a \|f - g\|_\infty$ 의 max-Lipschitz 보조정리로 contraction $\square$. sup-norm 선택의 본질: 모든 state 에서 동시 보장 |
| "Policy Iteration 은 evaluation + improvement 이다" | **Howard 1960** — $Q^\pi(s, \pi'(s)) \geq V^\pi(s) \Rightarrow V^{\pi'} \geq V^\pi$ pointwise $\square$ (policy improvement theorem). Greedy improvement 가 strict 한 한 새 정책. **Finite MDP 에서 유한 step 수렴**: deterministic stationary policy 의 수가 $|\mathcal{A}|^{|\mathcal{S}|}$ 로 유한, 매 iteration 마다 strict improvement (혹은 종료). **superpolynomial** 수렴 — VI 보다 빠름 |
| "GPI 는 PI 의 일반화다" | **Sutton & Barto** — Policy evaluation 과 improvement 의 **임의 interleaving** 이 모두 같은 고정점으로 수렴. **모든 RL 알고리즘** (MC, TD, Q-learning, Actor-Critic, Deep RL) 이 GPI 의 instance — 각 알고리즘이 두 프로세스의 다른 trade-off 를 선택. 이것이 RL 통합 관점의 핵심 |
| "Performance Difference Lemma 는 알고 있다" | **Kakade 2003** — $V^{\pi'}(\rho) - V^\pi(\rho) = (1/(1-\gamma))\, \mathbb{E}_{s \sim d^{\pi'}}[\sum_a \pi'(a\|s)\, A^\pi(s, a)]$ $\square$. **유도**: telescoping sum + advantage 정의 + discounted state distribution $d^\pi(s) = (1-\gamma) \sum_t \gamma^t d^\pi_t(s)$. 이 한 줄이 **Policy Gradient theorem · TRPO · PPO 의 동기** — Advanced RL 레포로의 bridge |
| "Linear FA 는 Deep RL 의 단순 버전이다" | **Tsitsiklis & Van Roy 1997** — TD(0) with linear FA 가 **on-policy 에서 수렴** (projected Bellman equation 의 고정점). **Off-policy** 에서는 Baird's counterexample 처럼 **발산 가능**. **Deadly Triad**: ① Off-policy ② Bootstrapping ③ Function Approximation 이 동시에 있으면 발산 가능 — Deep RL 의 근본 불안정성 |
| 알고리즘의 나열 | NumPy + matplotlib 으로 **VI/PI 를 4×4 / 5×5 Gridworld 에서 직접 구현** · **수렴 곡선 $\|V_k - V^*\|_\infty$ 을 log-scale 로 plot** · **$\gamma$ 별 수렴 속도 비교** ($\gamma = 0.5, 0.9, 0.99$) · **Contraction 시각화** (두 $V$ 사이 거리가 $\gamma$ 배로 줄어드는 애니메이션) · **Howard 의 정책 변화 추적** (PI iteration 별 정책 grid) · **Sutton & Barto 표준 예제 재현** (Jack's Car Rental, Blackjack, Gridworld) 까지 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Probability Theory]            ─┐
[Stochastic Processes]          ─┤
[Convex Optimization]           ─┼─►  이 레포  ──► [Model-Free RL Deep Dive]
[Functional Analysis]           ─┤   "왜 MDP·Bellman·DP 가         MC · TD · Q-learning
[Mathematical Statistics]       ─┘    RL 의 정초인가"               (model-unknown 환경)
                                          │
                                          ├──► [Policy Gradient Deep Dive]
                                          │     PG theorem · Natural PG · Actor-Critic
                                          │     (PDL 이 출발점)
                                          │
                                          ├──► [Deep RL Deep Dive]
                                          │     DQN · DDPG · Double Q · Replay Buffer
                                          │     (Function Approximation 의 실전 문제)
                                          │
                                          ├──► [Advanced RL Deep Dive]
                                          │     TRPO · PPO · SAC · TD3
                                          │     (PDL → monotonic improvement bound)
                                          │
                                          └──► [RL Theory Deep Dive]
                                                Sample complexity · Regret · PAC-MDP

         │
         ├── [Probability Theory]    조건부 기댓값 · Markov chain → Ch1, Ch2
         ├── [Stochastic Processes]  Markov chain · stationary distribution → Ch1, Ch6
         ├── [Convex Optimization]   Banach fixed point · contraction → Ch4, Ch5
         ├── [Functional Analysis]   Complete metric space · operator → Ch4
         └── [Mathematical Stats]    수렴 이론 · LLN · 측도 → Ch1, Ch7
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability Theory Deep Dive** (조건부 기댓값, Markov chain), **Stochastic Processes Deep Dive** (Markov chain, stationary distribution), **Convex Optimization Deep Dive** (Banach fixed point) 를 선행 지식으로 전제합니다. **Functional Analysis Deep Dive** (complete metric space, contraction mapping operator) 와 **Mathematical Statistics Deep Dive** (수렴 이론) 는 Ch4 의 contraction 증명과 Ch7 의 linear FA 분석에서 권장됩니다.

> 💡 **이 레포의 핵심 기여**: Chapter 4 (Contraction Mapping) 와 Chapter 5 (Dynamic Programming) 는 RL 의 **수학적 기둥** 입니다. 전자는 "왜 Bellman iteration 이 수렴하는가" 의 functional-analytic 토대 (Banach fixed point + $\gamma$-contraction), 후자는 "왜 PI 가 유한 step 에 끝나는가" 의 algorithmic 정수 (Howard 의 monotonic improvement). 이 두 축을 완전히 이해한 후 Chapter 6 (Performance Difference Lemma) 을 읽으면 **Policy Gradient · TRPO · PPO 의 모든 이론이 자연스럽게** 따라옵니다.

> 🟢 **이 레포의 성격**: 여기서 다루는 주제 — **MDP 의 공리적 정의, Bellman equation, $\gamma$-contraction, Howard 의 PI, GPI, Performance Difference Lemma** — 는 **반세기 동안 정착된 고전 이론** 입니다. 레포는 "최신 SOTA" 가 아니라 **"모든 현대 RL 의 수학적 정초"** 를 제공합니다. Model-Free RL · Deep RL · Advanced RL 로 진행하기 전 반드시 한 번은 거쳐야 하는 통과 의례입니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-MDP_정의-1565C0?style=for-the-badge)](./ch1-mdp-definition/01-mdp-tuple.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Bellman_Expectation-1565C0?style=for-the-badge)](./ch2-bellman-expectation/01-discounted-return.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Bellman_Optimality-1565C0?style=for-the-badge)](./ch3-bellman-optimality/01-optimal-value-function.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Contraction_Mapping-1565C0?style=for-the-badge)](./ch4-contraction-mapping/01-banach-fixed-point.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-DP_Algorithms-1565C0?style=for-the-badge)](./ch5-dp-algorithms/01-policy-evaluation.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-MDP_Properties-1565C0?style=for-the-badge)](./ch6-mdp-properties/01-state-distribution.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Linear_MDP-1565C0?style=for-the-badge)](./ch7-linear-mdp/01-linear-fa.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Markov Decision Process 의 공리적 정의

> **핵심 질문:** MDP 의 6-tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ 의 각 요소는 왜 그 형태여야 하는가? Markov 성질 $P(s_{t+1} \mid h_t) = P(s_{t+1} \mid s_t, a_t)$ 이 왜 dynamic programming 가능성의 **필수 조건** 인가? Stationary Markovian policy 가 왜 history-dependent · stochastic policy 보다 더 좋아질 수 없는가 (Puterman 2005)? Finite-horizon, infinite-horizon, average-reward MDP 의 본질적 차이는? POMDP 가 어떻게 belief state $b(s)$ 로 full MDP 에 환원되며 그 비용은?

<details>
<summary><b>6-tuple 정의부터 POMDP 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. MDP 의 6-tuple 정의](./ch1-mdp-definition/01-mdp-tuple.md) | **정의**: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ — state space (measurable), action space, transition kernel $P(\cdot\|s,a)$ (stochastic kernel), reward $R(s, a)$ (bounded measurable), discount $\gamma \in [0,1)$, initial $\rho_0 \in \Delta(\mathcal{S})$. **Measurability** 의 필수성 — expectation 의 정의 가능성. **Finite vs Borel-measurable** state space 의 분기 |
| [02. Markov 성질과 그 결과](./ch1-mdp-definition/02-markov-property.md) | **정의**: $P(s_{t+1} \mid s_t, a_t, s_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)$ $\square$. **결과**: history $h_t$ 의 무한 차원을 state $s_t$ 의 유한 차원으로 환원 → DP 의 **subproblem 분해 가능성**. Markov 성질 없이는 Bellman equation 자체가 정의되지 않음 |
| [03. Policy 의 종류와 Stationary Policy 충분성](./ch1-mdp-definition/03-policy-types.md) | **분류**: deterministic vs stochastic, history-dependent vs Markovian vs stationary. **Puterman 정리**: finite MDP, discounted infinite-horizon 에서 **deterministic stationary Markovian** policy 중 최적이 존재 $\square$. 증명: Bellman optimality + greedy 가 deterministic 이고 state-only 의존 |
| [04. Finite-Horizon vs Infinite-Horizon vs Average Reward](./ch1-mdp-definition/04-horizon-types.md) | **Episodic** ($T < \infty$, terminal): $J = \mathbb{E}[\sum_{t=0}^{T} R_t]$, time-dependent value $V_t(s)$ 가능. **Discounted infinite-horizon** ($\gamma < 1$): $J = \mathbb{E}[\sum_t \gamma^t R_t]$ — 본 레포의 주 대상. **Average-reward** ($\gamma = 1$): $J = \lim_T (1/T) \mathbb{E}[\sum_{t<T} R_t]$ — 다른 수학 (Cesaro 합, Blackwell optimality) |
| [05. POMDP 와 Belief State](./ch1-mdp-definition/05-pomdp.md) | **POMDP**: $(\mathcal{S}, \mathcal{A}, \mathcal{O}, P, O, R, \gamma)$ — observation $o$ 만 주어짐. **Belief state** $b(s) = \mathbb{P}(s_t = s \mid h_t) \in \Delta(\mathcal{S})$. **정리**: belief MDP $(\mathcal{B}, \mathcal{A}, P_b, R_b, \gamma)$ 가 **full MDP** — belief space 가 continuous (계산 부담). Bayes update $b' \propto O(o\|s') \sum P(s'\|s,a) b(s)$ |

</details>

<br/>

### 🔹 Chapter 2: Return, Value Function 과 Bellman Expectation Equation

> **핵심 질문:** Discounted return $G_t = \sum_k \gamma^k R_{t+k+1}$ 의 수렴은 왜 $\gamma \in [0, 1)$ 에서 보장되는가 — bounded reward 와의 관계는? $V^\pi(s) = \mathbb{E}^\pi[G_t \mid S_t = s]$ 와 $Q^\pi(s, a) = \mathbb{E}^\pi[G_t \mid S_t = s, A_t = a]$ 의 관계 $V^\pi(s) = \sum_a \pi(a\|s) Q^\pi(s, a)$ 는 어떻게 유도되는가? Bellman expectation equation 이 어떻게 재귀가 되는가? $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ 가 왜 closed-form 해이며 $(I - \gamma P^\pi)$ 의 가역성은 어떻게 보장되는가?

<details>
<summary><b>Discounted Return 부터 Operator 표기법까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Discounted Return 의 정의와 수렴](./ch2-bellman-expectation/01-discounted-return.md) | **정의**: $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$. **정리**: $\|R\|_\infty \leq R_{\max}, \gamma \in [0, 1)$ 이면 $|G_t| \leq R_{\max} / (1-\gamma)$ 로 absolute convergent $\square$. $\gamma = 1$ 에서의 처리 — episodic ($T < \infty$) 만 의미, infinite-horizon 발산. $\gamma = 0$ 의 myopic 한계 |
| [02. State-Value 와 Action-Value Function](./ch2-bellman-expectation/02-value-functions.md) | **정의**: $V^\pi(s) = \mathbb{E}^\pi[G_t \mid S_t = s]$, $Q^\pi(s, a) = \mathbb{E}^\pi[G_t \mid S_t = s, A_t = a]$. **관계 1**: $V^\pi(s) = \sum_a \pi(a\|s)\, Q^\pi(s, a)$ — total expectation. **관계 2**: $Q^\pi(s, a) = R(s,a) + \gamma \sum_{s'} P(s'\|s, a)\, V^\pi(s')$ — one-step lookahead $\square$. Boundedness: $\|V^\pi\|_\infty \leq R_{\max}/(1-\gamma)$ |
| [03. Bellman Expectation Equation 유도](./ch2-bellman-expectation/03-bellman-expectation.md) | **정리**: $V^\pi(s) = \sum_a \pi(a\|s) [R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V^\pi(s')]$ $\square$. **유도**: tower property $\mathbb{E}[G_t \mid S_t] = \mathbb{E}[\mathbb{E}[G_t \mid S_t, A_t] \mid S_t]$ + return 의 재귀 $G_t = R_{t+1} + \gamma G_{t+1}$. $Q^\pi$ 의 재귀: $Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'\|s,a) \sum_{a'} \pi(a'\|s') Q^\pi(s', a')$ |
| [04. Operator 표기법: $T^\pi V = r^\pi + \gamma P^\pi V$](./ch2-bellman-expectation/04-bellman-operator.md) | **정의**: $r^\pi(s) := \sum_a \pi(a\|s) R(s,a)$ — policy-induced reward vector. $P^\pi(s' \| s) := \sum_a \pi(a\|s) P(s'\|s,a)$ — policy-induced stochastic matrix. **Operator**: $(T^\pi V)(s) = r^\pi(s) + \gamma \sum_{s'} P^\pi(s'\|s) V(s')$. **선형성**: $T^\pi(\alpha V + \beta V') = \alpha T^\pi V + \beta T^\pi V' + (1-\alpha-\beta) r^\pi$ — affine. $V^\pi = T^\pi V^\pi$ 가 고정점 |
| [05. Value Function 의 고유성과 존재성](./ch2-bellman-expectation/05-value-uniqueness.md) | **Closed-form**: $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ (finite MDP) $\square$. **가역성**: $P^\pi$ 가 stochastic matrix 이므로 spectral radius $\rho(P^\pi) \leq 1$ → $\rho(\gamma P^\pi) \leq \gamma < 1$ → $(I - \gamma P^\pi)$ 의 모든 eigenvalue 가 0 이 아님. **Neumann series**: $(I - \gamma P^\pi)^{-1} = \sum_k (\gamma P^\pi)^k$ 가 절대수렴 |

</details>

<br/>

### 🔹 Chapter 3: Bellman Optimality Equation 과 최적 정책

> **핵심 질문:** $V^*(s) = \sup_\pi V^\pi(s)$ 가 모든 state 에서 **동시에** 같은 정책으로 달성 가능한 이유는 무엇인가? Bellman optimality equation $V^*(s) = \max_a [R(s,a) + \gamma \sum P V^*(s')]$ 는 어떻게 expectation equation 의 max 로 도출되는가? Optimality operator $T^* V := \max_\pi T^\pi V$ 의 의미는? Greedy policy $\pi^*(s) = \arg\max_a Q^*(s, a)$ 가 왜 최적이며, 유일하지 않을 수 있지만 성능이 같은 이유는? Finite MDP 에서 deterministic stationary policy 중 최적이 항상 존재하는가?

<details>
<summary><b>Optimal Value 부터 Greedy Policy 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Optimal Value Function 의 정의](./ch3-bellman-optimality/01-optimal-value-function.md) | **정의**: $V^*(s) = \sup_\pi V^\pi(s)$, $Q^*(s, a) = \sup_\pi Q^\pi(s, a)$. **정리**: 모든 state 에서 **동시에** 최대값을 달성하는 정책 $\pi^*$ 존재 $\square$ — pointwise sup 이 동시에 attained (Puterman, contraction argument). Boundedness $\|V^*\|_\infty \leq R_{\max}/(1-\gamma)$ |
| [02. Bellman Optimality Equation](./ch3-bellman-optimality/02-bellman-optimality.md) | **정리**: $V^*(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V^*(s')]$, $Q^*(s, a) = R(s,a) + \gamma \sum_{s'} P(s'\|s, a) \max_{a'} Q^*(s', a')$ $\square$. **유도**: $V^*$ 가 고정점 + $\sup$ 이 $\max$ 로 attained (finite action). Bellman expectation equation 의 자연스러운 일반화 |
| [03. Bellman Optimality Operator $T^*$](./ch3-bellman-optimality/03-optimality-operator.md) | **정의**: $(T^* V)(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V(s')]$. **관계**: $T^* V = \max_\pi T^\pi V$ (pointwise) $\square$. **Nonlinear**: $T^*$ 는 max 로 인해 affine 이 아니지만 monotone ($V \leq V' \Rightarrow T^* V \leq T^* V'$). Ch4 에서 $\gamma$-contraction 증명 |
| [04. 최적 정책의 추출 — Greedy Policy](./ch3-bellman-optimality/04-greedy-policy.md) | **정의**: $\pi^*(s) = \arg\max_a Q^*(s, a)$. **정리**: $V^*$ 주어지면 greedy policy 가 최적 $\square$. **유일성 부재**: 동률이 있으면 여러 greedy 가 모두 최적, **성능은 같음**. Tie-breaking 의 임의성과 epsilon-greedy 의 수학적 위상 |
| [05. Deterministic 최적 정책의 존재](./ch3-bellman-optimality/05-deterministic-optimal.md) | **정리**: finite MDP, discounted infinite-horizon 에서 **deterministic stationary** policy 중 최적이 항상 존재 $\square$. **유도**: $\arg\max$ 가 deterministic, state-only 의존 → stationary Markovian. **Stochastic policy 의 불필요성**: linear in $\pi(\cdot \|s)$ 인 expected return 이 vertex 에서 최댓값 → deterministic |

</details>

<br/>

### 🔹 Chapter 4: Contraction Mapping 과 수렴 증명

> **핵심 질문:** Banach fixed point theorem 이 왜 RL 의 모든 수렴 이론의 기둥인가? $T^\pi$ 가 sup-norm 에서 $\gamma$-contraction 임을 어떻게 증명하는가 — stochastic matrix $P^\pi$ 의 $\|\cdot\|_\infty$ induced norm 이 1 이라는 사실이 어떻게 활용되는가? $T^*$ 는 nonlinear 인데도 여전히 $\gamma$-contraction 인 이유는 — max-Lipschitz 보조정리는? Value Iteration 의 linear convergence rate $\gamma^k$ 와 정지 기준 $\|V_{k+1} - V_k\|_\infty < \epsilon (1-\gamma) / \gamma$ 는 어떻게 도출되는가?

<details>
<summary><b>Banach Fixed Point 부터 VI 수렴까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Banach Fixed Point Theorem](./ch4-contraction-mapping/01-banach-fixed-point.md) | **정리**: 완비 거리공간 $(X, d)$, $T: X \to X$ 가 contraction with $L < 1$ ⇒ ① 유일 고정점 $x^*$ 존재, ② $x_{n+1} = T(x_n)$ 이 $x^*$ 로 수렴, ③ rate: $d(x_n, x^*) \leq L^n d(x_0, x^*) / (1-L)$ $\square$. **공간 선택**: $(B(\mathcal{S}), \|\cdot\|_\infty)$ 가 RL 의 표준 — bounded function space + sup-norm 으로 완비. Functional Analysis Deep Dive 와의 교차 |
| [02. $T^\pi$ 가 $\gamma$-Contraction in Sup-Norm](./ch4-contraction-mapping/02-tpi-contraction.md) | **정리**: $\|T^\pi V - T^\pi V'\|_\infty \leq \gamma\, \|V - V'\|_\infty$ $\square$. **증명**: $T^\pi V - T^\pi V' = \gamma P^\pi (V - V')$ → $\|\gamma P^\pi (V-V')\|_\infty \leq \gamma\, \|P^\pi\|_\infty \|V-V'\|_\infty = \gamma\, \|V-V'\|_\infty$ ($P^\pi$ stochastic ⇒ $\|P^\pi\|_\infty = 1$). **sup-norm 의 본질**: 모든 state 에서 동시 보장 |
| [03. $T^*$ 가 $\gamma$-Contraction (Nonlinear 의 경우)](./ch4-contraction-mapping/03-tstar-contraction.md) | **정리**: $T^*$ 는 nonlinear 이지만 여전히 $\gamma$-contraction $\square$. **보조정리** (max-Lipschitz): $\|\max_a f(a) - \max_a g(a)\|_\infty \leq \max_a \|f(a) - g(a)\|_\infty$. **증명**: $(T^* V)(s) - (T^* V')(s) = \max_a [\cdot] - \max_a [\cdot] \leq \max_a |\cdot| \leq \gamma \|V - V'\|_\infty$ |
| [04. Value Iteration 수렴 보장](./ch4-contraction-mapping/04-value-iteration-convergence.md) | **정리**: $V_{k+1} = T^* V_k$ ⇒ $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$ — linear rate $\square$. **정지 기준**: $\|V_{k+1} - V_k\|_\infty < \epsilon (1-\gamma)/\gamma$ ⇒ $\|V_k - V^*\|_\infty < \epsilon$ (triangle + telescoping). **Practical**: $V_0 = 0$ 에서 시작, $\epsilon = 10^{-6}$, $\gamma = 0.9$ 면 $k \approx 130$ |
| [05. $\gamma \to 1$ 에서의 한계](./ch4-contraction-mapping/05-gamma-limit.md) | **현상**: $\gamma \to 1^-$ 에서 contraction rate 약화, $\gamma^k$ 수렴이 $1$ 근처에서 매우 느림 — 필요한 iteration $k = O(1/(1-\gamma))$. **극한 $\gamma = 1$**: contraction 깨짐 → average-reward MDP 로 전환 (Cesaro 합, Blackwell optimality, Howard-Veinott canonical form). $\gamma$ 선택의 trade-off |

</details>

<br/>

### 🔹 Chapter 5: Dynamic Programming 알고리즘

> **핵심 질문:** Policy evaluation 의 iterative ($V \leftarrow T^\pi V$) vs direct solve ($(I - \gamma P^\pi)^{-1} r^\pi$) 의 복잡도 trade-off 는? Policy improvement theorem $Q^\pi(s, \pi'(s)) \geq V^\pi(s) \Rightarrow V^{\pi'} \geq V^\pi$ pointwise 의 telescope 증명은? Howard 의 Policy Iteration 이 왜 finite MDP 에서 **유한 step** 안에 최적 정책을 보장하는가? Asynchronous VI (Gauss-Seidel) 의 수렴 조건은? Generalized Policy Iteration (GPI) 가 왜 모든 RL 알고리즘의 통합 관점인가?

<details>
<summary><b>Policy Evaluation 부터 GPI 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Policy Evaluation](./ch5-dp-algorithms/01-policy-evaluation.md) | **목표**: 주어진 $\pi$ 에 대해 $V^\pi$ 계산. **Iterative**: $V_{k+1} = T^\pi V_k$ — $O(\|\mathcal{S}\|^2 \|\mathcal{A}\| / (1-\gamma))$ ($\epsilon$-tolerance). **Direct**: $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ — $O(\|\mathcal{S}\|^3)$ (Gaussian elimination). **선택**: 큰 $\|\mathcal{S}\|$ 에서 iterative, 작은 $\|\mathcal{S}\|$ 에서 direct (수치 안정성). $T^\pi$ contraction 이 iterative 보장 |
| [02. Policy Improvement Theorem](./ch5-dp-algorithms/02-policy-improvement.md) | **정리** (Howard 1960): $Q^\pi(s, \pi'(s)) \geq V^\pi(s)\, \forall s \Rightarrow V^{\pi'}(s) \geq V^\pi(s)\, \forall s$ $\square$. **증명**: telescope $V^{\pi'} - V^\pi = \mathbb{E}^{\pi'}[\sum_t \gamma^t (Q^\pi(S_t, \pi'(S_t)) - V^\pi(S_t))] \geq 0$. **Strict 의 의미**: greedy improvement 가 적어도 한 state 에서 strict 인 한 새 정책. Suboptimal 정책의 strict improvement 보장 |
| [03. Policy Iteration (Howard 1960)](./ch5-dp-algorithms/03-policy-iteration.md) | **알고리즘**: ① Evaluate $V^{\pi_k}$, ② Improve $\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s, a)$, ③ 반복. **정리**: finite MDP 에서 **유한 step** 내 최적 도달 $\square$ — deterministic stationary policy 의 수가 $\|\mathcal{A}\|^{\|\mathcal{S}\|}$ 로 유한, 매 iteration 마다 strict improvement (혹은 종료). **Superpolynomial** 수렴 — VI 보다 빠름 |
| [04. Value Iteration 완성](./ch5-dp-algorithms/04-value-iteration.md) | **알고리즘**: $V_{k+1}(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V_k(s')]$. **Bellman residual**: $\|T^* V_k - V_k\|_\infty$ 로 진척도 측정. **Asynchronous VI** (Gauss-Seidel): 한 번에 한 state 씩 update, **수렴 보장 유지** (모든 state 가 무한히 update 되면) $\square$. **Modified PI**: PE 를 1번만 → VI 와 PI 의 보간 |
| [05. Generalized Policy Iteration (GPI)](./ch5-dp-algorithms/05-gpi.md) | **개념**: PE 와 PI 의 **임의 interleaving**, 두 프로세스가 각자 진행하면서 같은 고정점으로 수렴. **사각 다이어그램**: $\pi \xrightarrow{\text{eval}} V^\pi \xrightarrow{\text{improve}} \pi'$ 의 사이클. **모든 RL 의 통합**: MC, TD, Q-learning, Actor-Critic, Deep RL 모두 GPI 의 instance — 두 프로세스의 다른 trade-off (sample efficiency vs stability) |

</details>

<br/>

### 🔹 Chapter 6: MDP 의 성질과 Performance Difference

> **핵심 질문:** State distribution $d^\pi_t(s)$ 와 discounted state distribution $d^\pi(s) = (1-\gamma) \sum_t \gamma^t d^\pi_t(s)$ 가 어떻게 Markov chain 의 stationary distribution 으로 환원되는가? Performance Difference Lemma (Kakade 2003) 의 한 줄이 어떻게 **Policy Gradient · TRPO · PPO 의 모든 monotonic improvement 이론** 의 출발점인가? Advantage $A^\pi = Q^\pi - V^\pi$ 의 baseline subtraction 이 왜 estimation variance 를 줄이는가? MDP 근사의 sample complexity 와 planning vs learning 의 분리는?

<details>
<summary><b>State Distribution 부터 Sample Complexity 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. State Distribution 과 Stationary Distribution](./ch6-mdp-properties/01-state-distribution.md) | **정의**: $d^\pi_t(s) = \mathbb{P}(S_t = s \mid \pi, \rho_0)$. **재귀**: $d^\pi_{t+1}(s') = \sum_s P^\pi(s'\|s)\, d^\pi_t(s)$. **Discounted state distribution**: $d^\pi(s) = (1-\gamma) \sum_t \gamma^t d^\pi_t(s)$ ($d^\pi \in \Delta(\mathcal{S})$). **Stochastic Processes 와의 연결**: $d^\pi_t \to d^\pi_\infty$ (stationary distribution) 단, ergodic Markov chain $P^\pi$ 가정. $V^\pi(\rho_0) = (1/(1-\gamma)) \mathbb{E}_{s \sim d^\pi}[r^\pi(s)]$ |
| [02. Performance Difference Lemma (Kakade 2003)](./ch6-mdp-properties/02-performance-difference-lemma.md) | **정리**: $V^{\pi'}(\rho) - V^\pi(\rho) = (1/(1-\gamma))\, \mathbb{E}_{s \sim d^{\pi'}}[\sum_a \pi'(a\|s)\, A^\pi(s, a)]$ $\square$. **유도**: telescoping sum + advantage 정의 + discounted state distribution. **본질적 문제**: $d^{\pi'}$ 는 새 정책의 분포 — **unknown** (rollout 전엔 모름). 이 한 줄이 TRPO/PPO 의 surrogate 동기 — Advanced RL 레포로의 bridge |
| [03. Advantage Function 과 Baseline Subtraction](./ch6-mdp-properties/03-advantage-baseline.md) | **정의**: $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$. **의미**: $A^\pi(s, a) > 0$ 이면 "$\pi$ 의 평균보다 $a$ 가 좋다", $< 0$ 이면 나쁘다. **Variance reduction**: estimation 시 $\hat{Q} - V^\pi$ 가 $\hat{Q}$ 보다 분산 작음 (zero-mean baseline). **GAE** 의 효시 — Policy Gradient 레포에서 깊이 다룸 |
| [04. MDP 근사 — Approximation Error 와 Sample Complexity](./ch6-mdp-properties/04-approximation-sample-complexity.md) | **정리**: $\|V^* - V^{\hat{\pi}}\|_\infty \leq 2\epsilon / (1-\gamma)^2$ — $\epsilon$-optimal Q 의 greedy 가 $2\epsilon/(1-\gamma)^2$-optimal $\square$. **Planning vs Learning**: model 알면 planning ($\|\mathcal{S}\|^2 \|\mathcal{A}\|$ 시간), 모르면 learning ($\tilde{O}(\|\mathcal{S}\|\|\mathcal{A}\|/(1-\gamma)^3 \epsilon^2)$ samples for tabular). **Bridge to Model-Free RL** |

</details>

<br/>

### 🔹 Chapter 7: Linear MDP 와 Function Approximation 기초

> **핵심 질문:** Linear function approximation $V_\theta(s) = \theta^T \phi(s)$ 에서 TD(0) 가 on-policy 에서 수렴 (Tsitsiklis & Van Roy 1997) 하지만 off-policy 에서 발산 가능한 이유는? **Deadly Triad** — Off-policy + Bootstrapping + FA — 가 동시에 있으면 발산할 수 있는 메커니즘은 (Baird's counterexample)? Linear MDP 가정 $P(s'\|s,a) = \phi(s, a)^T \mu(s')$ 가 어떻게 polynomial sample complexity 를 가능하게 하는가 (Jin et al. 2020)? MDP homomorphism 과 state abstraction 의 safe condition 은?

<details>
<summary><b>Linear FA 부터 State Abstraction 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Linear Function Approximation](./ch7-linear-mdp/01-linear-fa.md) | **모델**: $V_\theta(s) = \theta^T \phi(s)$, $\phi: \mathcal{S} \to \mathbb{R}^d$. **TD(0) update**: $\theta \leftarrow \theta + \alpha (r + \gamma \theta^T \phi(s') - \theta^T \phi(s))\, \phi(s)$. **정리** (Tsitsiklis & Van Roy 1997): on-policy 에서 projected Bellman equation 의 고정점으로 수렴 $\square$ — projection $\Pi$ onto span($\phi$), $\Pi T^\pi$ 가 contraction in $\|\cdot\|_{d^\pi}$ |
| [02. Deadly Triad — Off-policy + Bootstrapping + FA](./ch7-linear-mdp/02-deadly-triad.md) | **정리**: 세 요소가 **동시** 에 있으면 TD/Q-learning 이 발산 가능 $\square$. **Baird's counterexample**: 7-state MDP, linear FA, off-policy TD 가 $\theta \to \infty$. **메커니즘**: off-policy 의 weighting 이 contraction 깨뜨림. **회피책**: GTD (gradient TD), Emphatic TD, Retrace, V-trace. **Deep RL 의 근본 문제** — Deep RL 레포로의 bridge |
| [03. Linear Bellman Equation — MDP 의 특수 구조](./ch7-linear-mdp/03-linear-mdp.md) | **정의**: $P(s'\|s,a) = \phi(s, a)^T \mu(s')$, $R(s,a) = \phi(s,a)^T \theta_R$. **결과**: $Q^*(s, a) = \phi(s, a)^T w^*$ 로 linear (in features) $\square$. **Sample complexity** (Jin et al. 2020 LSVI-UCB): $\tilde{O}(\sqrt{d^3 H^3 K})$ regret — feature dim $d$ 에 polynomial. **PAC-MDP 로의 통로** |
| [04. MDP Homomorphism 과 State Abstraction](./ch7-linear-mdp/04-state-abstraction.md) | **정의**: $\phi: \mathcal{S} \to \bar{\mathcal{S}}$ 가 동일한 value function 을 induce. **Safe abstraction 조건**: $\phi(s_1) = \phi(s_2) \Rightarrow V^*(s_1) = V^*(s_2)$ + transition consistency. **Bi-simulation relation** (Givan 2003): 가장 정밀한 safe partition. **Approximate**: $\epsilon$-bisimulation 의 value error bound |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch1-03 의 Stationary Policy 충분성 증명에 contraction + monotonicity 의 두 단계 분리 보강, Ch4-02 의 $T^\pi$ contraction 증명에 stochastic matrix 의 $\|\cdot\|_\infty$ induced norm 이 1 이라는 사실을 명시적 보조정리로 추출, Ch5-03 의 Howard PI 유한 수렴 증명에 deterministic policy 수의 유한성 + strict improvement 의 조합을 step-by-step 으로 세분화, Ch6-02 의 Performance Difference Lemma 유도에 telescoping sum 의 각 단계를 expanded form 으로 추가, Ch7-02 의 Deadly Triad 에 Baird's counterexample 의 NumPy 재현 코드 추가했습니다. **11-섹션 문서 골격이 전체 33개 문서에서 일관**됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **표준 예제 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 수렴 곡선·plot 을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Markov Property** | $P(s_{t+1} \mid h_t) = P(s_{t+1} \mid s_t, a_t)$ — DP 가능성의 필수 조건 | [Ch1-02](./ch1-mdp-definition/02-markov-property.md) |
| **Stationary Policy 충분성 (Puterman 2005)** | Finite MDP, discounted infinite-horizon 에서 deterministic stationary 가 최적 | [Ch1-03](./ch1-mdp-definition/03-policy-types.md) |
| **Discounted Return 수렴** | $\|R\|_\infty \leq R_{\max}, \gamma < 1 \Rightarrow \|G_t\| \leq R_{\max}/(1-\gamma)$ | [Ch2-01](./ch2-bellman-expectation/01-discounted-return.md) |
| **Bellman Expectation Equation** | $V^\pi(s) = \sum_a \pi(a\|s)[R + \gamma \sum P V^\pi]$ | [Ch2-03](./ch2-bellman-expectation/03-bellman-expectation.md) |
| **Closed-Form $V^\pi$** | $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ + 가역성 (spectral radius < 1) | [Ch2-05](./ch2-bellman-expectation/05-value-uniqueness.md) |
| **Bellman Optimality Equation** | $V^*(s) = \max_a[R + \gamma \sum P V^*]$ | [Ch3-02](./ch3-bellman-optimality/02-bellman-optimality.md) |
| **Greedy Policy Optimality** | $V^*$ 주어지면 $\pi^*(s) = \arg\max_a Q^*$ 가 최적 | [Ch3-04](./ch3-bellman-optimality/04-greedy-policy.md) |
| **Banach Fixed Point Theorem** | 완비 거리공간의 contraction → 유일 고정점 + linear convergence | [Ch4-01](./ch4-contraction-mapping/01-banach-fixed-point.md) |
| **$T^\pi$ Contraction** | $\|T^\pi V - T^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$ — affine + sup-norm | [Ch4-02](./ch4-contraction-mapping/02-tpi-contraction.md) |
| **$T^*$ Contraction (Nonlinear)** | Max-Lipschitz 보조정리로 $\gamma$-contraction 증명 | [Ch4-03](./ch4-contraction-mapping/03-tstar-contraction.md) |
| **Value Iteration Linear Convergence** | $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$ + 정지 기준 | [Ch4-04](./ch4-contraction-mapping/04-value-iteration-convergence.md) |
| **Policy Improvement Theorem (Howard)** | $Q^\pi(s, \pi'(s)) \geq V^\pi(s) \Rightarrow V^{\pi'} \geq V^\pi$ pointwise | [Ch5-02](./ch5-dp-algorithms/02-policy-improvement.md) |
| **Policy Iteration 유한 수렴** | Finite MDP 에서 유한 step 내 최적 — superpolynomial | [Ch5-03](./ch5-dp-algorithms/03-policy-iteration.md) |
| **Asynchronous VI 수렴** | Gauss-Seidel 한 state 씩 update, 모든 state 무한 update 시 수렴 | [Ch5-04](./ch5-dp-algorithms/04-value-iteration.md) |
| **Generalized Policy Iteration** | PE/PI 임의 interleaving — 모든 RL 알고리즘의 통합 관점 | [Ch5-05](./ch5-dp-algorithms/05-gpi.md) |
| **Performance Difference Lemma (Kakade 2003)** | $V^{\pi'} - V^\pi = (1/(1-\gamma))\, \mathbb{E}_{s \sim d^{\pi'}}[A^\pi]$ | [Ch6-02](./ch6-mdp-properties/02-performance-difference-lemma.md) |
| **Approximation-Optimality Bound** | $\epsilon$-optimal Q 의 greedy → $2\epsilon/(1-\gamma)^2$-optimal | [Ch6-04](./ch6-mdp-properties/04-approximation-sample-complexity.md) |
| **TD(0) Linear FA 수렴 (Tsitsiklis 1997)** | On-policy projected Bellman equation 의 고정점으로 수렴 | [Ch7-01](./ch7-linear-mdp/01-linear-fa.md) |
| **Deadly Triad Divergence** | Off-policy + Bootstrap + FA 동시 → 발산 가능 (Baird) | [Ch7-02](./ch7-linear-mdp/02-deadly-triad.md) |
| **Linear MDP Sample Complexity** | $P = \phi^T \mu \Rightarrow \tilde{O}(\sqrt{d^3 H^3 K})$ regret (LSVI-UCB) | [Ch7-03](./ch7-linear-mdp/03-linear-mdp.md) |

> 💡 **챕터별 문서·정리/정의 수** (실측):
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 MDP 정의 | 5 | 38 |
> | Ch2 Bellman Expectation | 5 | 42 |
> | Ch3 Bellman Optimality | 5 | 36 |
> | Ch4 Contraction Mapping | 5 | 45 |
> | Ch5 DP 알고리즘 | 5 | 47 |
> | Ch6 MDP Properties | 4 | 38 |
> | Ch7 Linear MDP | 4 | 34 |
> | **합계** | **33** | **280** |
>
> 추가로 **130+ 엄밀한 $\square$ 증명 + 99 연습문제 (모두 해설 포함) + 100+ NumPy 실험 코드 (`### 실험 N` 형식)**.
>
> Ch6, Ch7 은 **4 문서** 로 구성 — Performance Difference Lemma 와 Linear FA 는 핵심만 다룸 (후속 레포에서 확장). Ch1–5 의 5 문서와 의도적 차이.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다 — **순수 NumPy + Matplotlib** 으로 충분 (PyTorch · GPU 불필요).

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
seaborn==0.13.0
gymnasium==0.29.0       # OpenAI Gym 후속 (FrozenLake · CliffWalking 등)
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
sympy==1.12             # 기호 계산 (Bellman equation 풀이 검증)
networkx==3.2           # MDP 그래프 시각화
```

```bash
# 환경 설치 (CPU 기준, GPU 불필요)
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 seaborn==0.13.0 \
            gymnasium==0.29.0 tqdm==4.66.0 jupyter==1.0.0 \
            sympy==1.12 networkx==3.2

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — Gridworld 에서 VI vs PI 수렴 비교 (Ch4-04, Ch5-03, Ch5-04)
import numpy as np
import matplotlib.pyplot as plt


class GridworldMDP:
    """4x4 Gridworld, 상좌하우 4방향, terminal at (size-1, size-1)"""

    def __init__(self, size=4, gamma=0.9):
        self.size       = size
        self.gamma      = gamma
        self.n_states   = size * size
        self.n_actions  = 4  # up, left, down, right
        self.terminal   = (size - 1) * size + (size - 1)

    def step_prob(self, s, a):
        """P(s' | s, a): deterministic, terminal 은 absorbing"""
        if s == self.terminal:
            return {s: 1.0}, 0.0
        row, col = divmod(s, self.size)
        dr, dc   = [(-1, 0), (0, -1), (1, 0), (0, 1)][a]
        r_new    = max(0, min(self.size - 1, row + dr))
        c_new    = max(0, min(self.size - 1, col + dc))
        s_new    = r_new * self.size + c_new
        reward   = 1.0 if s_new == self.terminal else -0.04
        return {s_new: 1.0}, reward


def value_iteration(mdp, tol=1e-6, max_iter=1000):
    V       = np.zeros(mdp.n_states)
    history = [V.copy()]
    for k in range(max_iter):
        V_new = V.copy()
        for s in range(mdp.n_states):
            vals = []
            for a in range(mdp.n_actions):
                trans, r = mdp.step_prob(s, a)
                vals.append(r + mdp.gamma * sum(p * V[sp] for sp, p in trans.items()))
            V_new[s] = max(vals)
        diff = np.max(np.abs(V_new - V))
        V    = V_new
        history.append(V.copy())
        if diff < tol:
            break
    return V, history


def policy_iteration(mdp, tol=1e-6):
    pi     = np.zeros(mdp.n_states, dtype=int)
    n_iter = 0
    while True:
        n_iter += 1
        # ── Policy evaluation (iterative)
        V = np.zeros(mdp.n_states)
        for _ in range(1000):
            V_new = V.copy()
            for s in range(mdp.n_states):
                trans, r  = mdp.step_prob(s, pi[s])
                V_new[s]  = r + mdp.gamma * sum(p * V[sp] for sp, p in trans.items())
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        # ── Policy improvement (greedy)
        pi_new = np.zeros_like(pi)
        for s in range(mdp.n_states):
            vals = []
            for a in range(mdp.n_actions):
                trans, r = mdp.step_prob(s, a)
                vals.append(r + mdp.gamma * sum(p * V[sp] for sp, p in trans.items()))
            pi_new[s] = int(np.argmax(vals))
        if np.array_equal(pi, pi_new):
            break  # 수렴 (Howard 정리)
        pi = pi_new
    return pi, V, n_iter


mdp                 = GridworldMDP(size=5, gamma=0.9)
V_vi, history       = value_iteration(mdp)
pi_pi, V_pi, n_pi   = policy_iteration(mdp)

print(f"VI iterations: {len(history) - 1}")
print(f"PI iterations: {n_pi}")  # PI 는 보통 5~10 step 안에 끝남 (Howard)

# ── 수렴 속도 plot: ‖V_k - V*‖_∞ 가 γ^k 로 감소 (Ch4-04)
V_star = V_vi
errs   = [np.max(np.abs(V - V_star)) for V in history]
plt.figure(figsize=(8, 5))
plt.semilogy(errs, 'o-', label='actual')
k = np.arange(len(errs))
plt.semilogy(errs[0] * (mdp.gamma ** k), '--', label=r'theoretical $\gamma^k$')
plt.xlabel('VI iteration')
plt.ylabel(r'$\|V_k - V^*\|_\infty$ (log)')
plt.title(rf'Value Iteration linear convergence: $\gamma$={mdp.gamma}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ── V* 시각화 (Ch3-04): 5x5 Gridworld 의 optimal value
plt.figure(figsize=(6, 6))
plt.imshow(V_vi.reshape(mdp.size, mdp.size), cmap='viridis')
for s in range(mdp.n_states):
    r, c = divmod(s, mdp.size)
    plt.text(c, r, f'{V_vi[s]:.2f}', ha='center', va='center', color='white')
plt.title(r'$V^*(s)$ on 5x5 Gridworld (goal at bottom-right)')
plt.colorbar()
plt.show()

# 대표 실험 ② — γ별 수렴 속도 비교 (Ch4-05)
# γ = 0.5, 0.9, 0.99 에서 ‖V_k - V*‖_∞ 의 log-scale 비교
# → γ → 1 에서 contraction rate 약화를 시각적으로 확인

# 대표 실험 ③ — Howard 의 정책 변화 추적 (Ch5-03)
# PI iteration 별 policy grid (화살표) 가 어떻게 진화하는지 시각화
# → "유한 step 내 수렴" 의 직접 관찰

# 대표 실험 ④ — Performance Difference Lemma 검증 (Ch6-02)
# 두 policy π, π' 에 대해 V^{π'} - V^π 를 직접 계산 vs PDL 공식으로 계산
# → 두 값이 정확히 일치하는지 numerically 확인
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 정리가 RL 의 기초인가** | 해당 이론·알고리즘이 RL 의 어떤 핵심 문제를 푸는지 |
| 3 | 📐 **수학적 선행 조건** | Prob · Stoch · Convex · FA 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Markov 성질 · Bellman 재귀 · contraction · GPI 사각 다이어그램의 기하학적 직관 |
| 5 | ✏️ **엄밀한 정의** | MDP 6-tuple · Value function · Bellman operator · contraction · advantage 등 |
| 6 | 🔬 **정리와 증명** | Banach · Bellman · Howard · Performance Difference 의 step-by-step 유도 |
| 7 | 💻 **NumPy 구현 검증** | 4 가지 실험 (`### 실험 1` ~ `### 실험 4`) — Gridworld · CliffWalk · FrozenLake · 시각화 |
| 8 | 🔗 **후속 레포와의 연결** | Model-Free RL · Policy Gradient · Deep RL · Advanced RL 로의 확장 경로 |
| 9 | ⚖️ **가정과 한계** | Markov · finite · stationary · known dynamics — 깨지면 무엇이 부서지는가 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ 핵심 수식 + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 기초 / 심화 / 논문 비평 의 3 문제, `<details>` 펼침 해설 |

> 📚 **연습문제 총 99개** (33 문서 × 3 문제): **기초 / 심화 / 논문 비평** 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. Markov 성질의 측도론적 정의부터 Bellman expectation equation 손 유도, $T^\pi$ contraction 증명, Howard 의 PI 수렴 step-counting, Performance Difference Lemma 의 telescoping, Baird's counterexample 까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 450~550줄 (정의·증명·코드·연습문제 포함) 기준 **약 50분~1시간 20분**. 전체 33문서는 약 **30~40시간** 상당 (증명 재구성·NumPy 실험 재현 포함 시 50시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "RL 을 배우지만 Bellman equation 이 왜 수렴하는지 모른다" — 입문 투어 (1주, 약 10~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  MDP 의 6-tuple 정의
       Ch1-02  Markov 성질과 그 결과
Day 2  Ch2-01  Discounted Return
       Ch2-03  Bellman Expectation Equation
Day 3  Ch3-01  Optimal Value Function
       Ch3-02  Bellman Optimality Equation
Day 4  Ch4-01  Banach Fixed Point Theorem
       Ch4-02  T^π Contraction
Day 5  Ch4-04  Value Iteration 수렴
       Ch5-01  Policy Evaluation
Day 6  Ch5-03  Policy Iteration (Howard)
       Ch5-05  Generalized Policy Iteration
Day 7  Ch6-02  Performance Difference Lemma (Advanced RL 으로의 bridge)
```

</details>

<details>
<summary><b>🟡 "Bellman·DP 의 모든 증명을 정복한다" — 이론 집중 (2주, 약 20~24시간)</b></summary>

<br/>

```
1주차 — MDP · Bellman · Contraction
  Day 1    Ch1-01~03   MDP 6-tuple + Markov + Stationary policy 충분성
  Day 2    Ch1-04~05   Horizon types + POMDP
  Day 3    Ch2-01~03   Return + Value functions + Bellman expectation
  Day 4    Ch2-04~05   Operator 표기 + Closed-form V^π
  Day 5    Ch3-01~03   V* + Bellman optimality + T*
  Day 6    Ch3-04~05   Greedy policy + Deterministic 최적 존재
  Day 7    Ch4-01~02   Banach + T^π contraction

2주차 — DP · Performance Difference · Linear FA
  Day 1    Ch4-03~05   T* contraction + VI 수렴 + γ → 1 한계
  Day 2    Ch5-01~02   Policy Evaluation + Improvement Theorem
  Day 3    Ch5-03~04   Policy Iteration (Howard) + Async VI
  Day 4    Ch5-05      Generalized Policy Iteration
  Day 5    Ch6-01~02   State distribution + Performance Difference Lemma
  Day 6    Ch6-03~04   Advantage + Approximation/Sample complexity
  Day 7    Ch7-01~04   Linear FA + Deadly Triad + Linear MDP + State abstraction
```

</details>

<details>
<summary><b>🔴 "RL 의 수학적 기초를 완전 정복한다" — 전체 정복 (6주, 약 30~40시간 + 표준 예제 재현 10~15시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — MDP 의 공리적 정의
         → MDP 6-tuple 의 measurability 손 검증
         → Markov 성질의 측도론적 정식화
         → Puterman 의 stationary policy 충분성 증명 재구성
         → POMDP belief MDP 환원 직접 확인

2주차   Chapter 2 전체 — Value Function 과 Bellman Expectation
         → Discounted return 의 absolute convergence 증명
         → V^π, Q^π 의 관계 손 유도
         → Bellman expectation equation 의 tower property 기반 유도
         → Closed-form V^π = (I - γP^π)^{-1} r^π 의 가역성 (Neumann series)

3주차   Chapter 3 전체 — Bellman Optimality
         → V*(s) = sup_π V^π(s) 의 동시 attain 증명
         → Bellman optimality equation 의 fixed point 위상
         → Greedy policy 가 V* 에서 최적임을 증명
         → Finite MDP 에서 deterministic stationary 최적 존재

4주차   Chapter 4 전체 — Contraction Mapping
         → Banach fixed point 의 완비성 가정 분리
         → T^π contraction 을 stochastic matrix property 로 유도
         → T* contraction 의 max-Lipschitz 보조정리 증명
         → Value Iteration 의 γ^k convergence + 정지 기준 도출
         → γ → 1 에서 Cesaro 합으로의 전환 (average-reward MDP 맛보기)

5주차   Chapter 5 전체 — Dynamic Programming
         → Policy evaluation 의 iterative vs direct 복잡도 비교
         → Policy improvement theorem 의 telescoping 증명
         → Howard 1960 의 finite step 수렴 (deterministic policy 수 + strict improvement)
         → Asynchronous VI (Gauss-Seidel) 의 수렴 보장
         → GPI 사각 다이어그램으로 모든 RL 알고리즘 분류

6주차   Chapter 6 + Chapter 7 + 종합
         → Discounted state distribution 의 Markov chain 환원
         → Performance Difference Lemma 손 유도 (telescoping + advantage)
         → Approximation-optimality bound: ε-Q → 2ε/(1-γ)^2-optimal
         → Tsitsiklis 1997 TD with linear FA 수렴 증명
         → Baird's counterexample NumPy 재현 — Deadly Triad 직접 확인
         → "Model-Free RL · Policy Gradient · Deep RL · Advanced RL" 로의 다음 단계 지도
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 조건부 기댓값 · Markov chain · 측도론 | **Ch1, Ch2** (Markov 성질, expectation 정의) |
| [stochastic-processes-deep-dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) | Markov chain · stationary distribution · ergodicity | **Ch1, Ch6** (state distribution) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | Banach fixed point · contraction · KKT | **Ch4 전체** (수렴 증명) |
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | Complete metric space · operator theory | **Ch4** (Banach + sup-norm 의 정당화) |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | LLN · CLT · 수렴 이론 | **Ch7** (sample complexity) |
| [model-free-rl-deep-dive](https://github.com/iq-ai-lab/model-free-rl-deep-dive) *(다음)* | MC · TD · Q-learning · SARSA | **이 레포 전체** 의 자연스러운 후속 |
| [policy-gradient-deep-dive](https://github.com/iq-ai-lab/policy-gradient-deep-dive) | PG theorem · Natural PG · Actor-Critic | **Ch6-02** (PDL 이 출발점) |
| [deep-rl-deep-dive](https://github.com/iq-ai-lab/deep-rl-deep-dive) | DQN · DDPG · Replay Buffer | **Ch7** (FA 의 실전 문제) |
| [advanced-rl-deep-dive](https://github.com/iq-ai-lab/advanced-rl-deep-dive) | TRPO · PPO · SAC · TD3 | **Ch6-02** (PDL → monotonic improvement) |

> 💡 이 레포는 **"MDP·Bellman·DP 가 왜 모든 현대 RL 의 수학적 정초인가, $\gamma$-contraction 과 Banach fixed point 가 왜 수렴 보장의 근원인가"** 에 집중합니다. Probability 에서 Markov chain 과 조건부 기댓값을, Stochastic Processes 에서 stationary distribution 을, Convex Optimization 에서 Banach fixed point 를, Functional Analysis 에서 sup-norm 과 operator 를 익힌 후 오면 Chapter 4 (contraction) 와 Chapter 5 (Howard 의 PI) 의 증명이 훨씬 자연스럽습니다. **Performance Difference Lemma (Ch6-02)** 가 Policy Gradient · Advanced RL 레포의 출발점입니다.

---

## 📖 Reference

### 🏛️ MDP · Dynamic Programming 표준 교과서
- **Reinforcement Learning: An Introduction** (Sutton & Barto, 2nd ed., 2018) — **RL 바이블**, Bellman·VI·PI·GPI 의 표준 자료
- **Markov Decision Processes: Discrete Stochastic Dynamic Programming** (Puterman, 2005) — **MDP 표준**, stationary policy 충분성 증명 원전
- **Dynamic Programming and Optimal Control, Vol. 2** (Bertsekas, 4th ed., 2012) — Discounted MDP · contraction 의 정수
- **Neuro-Dynamic Programming** (Bertsekas & Tsitsiklis, 1996) — 근사 DP 의 표준
- **Algorithms for Reinforcement Learning** (Szepesvári, 2010) — 수학적으로 가장 엄밀한 짧은 책
- **Stochastic Optimal Control: The Discrete-Time Case** (Bertsekas & Shreve, 1996) — Borel-measurable MDP

### 🌱 고전 원전
- **Dynamic Programming** (Bellman, 1957) — **Bellman equation 효시**
- **Dynamic Programming and Markov Processes** (Howard, 1960) — **Policy Iteration 효시**
- **Discounted Dynamic Programming** (Blackwell, 1965) — Discounted MDP 의 수학적 정초
- **Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales** (Banach, 1922) — **Fixed Point Theorem 원전**
- **The Theory of Dynamic Programming** (Bellman, 1954) — Bellman equation 의 첫 발표

### 🔬 현대 RL 이론
- **A Theoretical Analysis of Approximate Policy Iteration** (Munos, 2003)
- **Approximately Optimal Approximate Reinforcement Learning** (Kakade & Langford, 2002) — **Performance Difference Lemma**
- **On the Sample Complexity of Reinforcement Learning** (Kakade, 2003 PhD thesis) — PDL 통합 정식화
- **A Theory of Regularized MDPs with Entropy Regularization** (Geist et al., 2019)
- **Performance Bounds for $\lambda$-Policy Iteration and Application to the Game of Tetris** (Scherrer, 2013)

### 📐 Function Approximation 의 수학
- **An Analysis of Temporal-Difference Learning with Function Approximation** (Tsitsiklis & Van Roy, 1997) — **TD with linear FA 수렴 정식화**
- **Off-Policy Temporal-Difference Learning with Function Approximation** (Precup et al., 2001)
- **Provably Efficient Reinforcement Learning with Linear Function Approximation** (Jin et al., 2020) — **LSVI-UCB**
- **Bellman-Consistent Pessimism for Offline Reinforcement Learning** (Xie et al., 2021)

### 🌐 Average-Reward · Risk-Sensitive MDP
- **Average Reward Reinforcement Learning** (Mahadevan, 1996)
- **R-Learning: A Reinforcement Learning Method to Maximize the Average Reward** (Schwartz, 1993)
- **Risk-Sensitive and Robust Decision-Making: A CVaR Optimization Approach** (Chow et al., 2015)

### 🛠️ 구현 · Library
- **OpenAI Spinning Up in Deep RL** (Achiam, 2018) — Tabular RL 코드 예시
- **Gymnasium** (Towers et al., 2023) — FrozenLake, CliffWalking 등 표준 toy 환경
- **CleanRL** (Huang et al., 2022) — 단일 파일 구현 참조 (후속 레포)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"Bellman equation 을 호출하는 것과 — Bellman 1957 부터 Banach 1922 의 fixed point theorem 으로 $T^*$ 가 sup-norm 에서 $\gamma$-contraction 임을 증명하여 Value Iteration 의 linear convergence 를 한 줄씩 유도 · Howard 1960 으로 Policy Iteration 이 finite MDP 에서 유한 step 내 수렴함을 deterministic policy 수의 유한성 + strict improvement 로 증명 · Puterman 2005 로 stationary Markovian policy 가 history-dependent · stochastic 보다 더 좋아질 수 없음을 contraction 과 monotonicity 로 유도 · Kakade 2003 으로 Performance Difference Lemma 가 어떻게 Policy Gradient · TRPO · PPO 의 모든 monotonic improvement 이론의 출발점인지 telescoping sum 으로 도출 · Tsitsiklis 1997 로 TD with linear FA 가 on-policy 에서 수렴하지만 off-policy 에서 Deadly Triad 로 발산 가능함을 Baird's counterexample 로 직접 확인 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
