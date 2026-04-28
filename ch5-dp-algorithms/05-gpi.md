# 05. Generalized Policy Iteration (GPI) — 모든 RL 알고리즘의 통합 틀

## 🎯 핵심 질문

- Policy Evaluation (PE) 과 Policy Improvement (PI) 를 "섞어" 하면 어떻게 되는가?
- 왜 MC, TD, Q-learning, Actor-Critic, Deep RL 이 모두 GPI 의 instance 인가?
- GPI 의 고정점이 최적 정책인가? 어떻게 증명하는가?
- 이 한 개의 프레임이 어떻게 수십 년의 RL 발전을 통합하는가?

---

## 🔍 왜 이 정리가 RL 의 필수 이해인가

Sutton & Barto (2018) 가 제시한 **Generalized Policy Iteration (GPI)** 은 RL 의 모든 알고리즘을 하나의 틀로 봅니다:

- **Policy Iteration**: 양쪽 fully (강함, 느림)
- **Value Iteration**: Improvement 를 1-step PE (빠름, 덜 정확)
- **Monte Carlo**: PE 를 rollout (sample 기반)
- **Temporal Difference**: PE 를 bootstrapping (1-step 학습)
- **Q-learning**: Off-policy PE + PI
- **Actor-Critic**: PE 를 actor, PI 를 critic
- **Deep RL**: 모든 위의 것을 NN 으로

"왜 이 알고리즘이 수렴할까" 에 대한 답은 항상 "GPI 프레임이므로" 입니다.

---

## 📐 수학적 선행 조건

- **Ch5-01~04**: PE, PI, VI 의 모든 정의와 정리
- **확률론**: Concentration inequality (선택)
- **Functional analysis**: Operator 이론의 기본

---

## 📖 직관적 이해

### GPI 의 직관

Policy Iteration 과 Value Iteration 은 **극단의 두 case**:

- **PI**: E 를 완전히 (수렴까지), 그 후 I
- **VI**: E 를 1 번만, 그 후 I

GPI 는: E 와 I 를 **임의로 interleave** 해도 같은 고정점으로 수렴한다고 말합니다.

```
        V                              π
        ↑                              ↑
        │   E: evaluate π              │   I: improve on V
        │   ←─────────────────         │   ←──────────────
    π───┤                          V───┤
        │   I: improve to π'       E: evaluate π'
        │   ──────────────────→        ──────────────→
        
    사각 다이어그램: 두 경로 (π→V→π', π→V→π' via different routes)
    가 같은 고정점으로 수렴
```

### 왜 임의의 interleaving 인가

직관적으로, policy 는 value function 을 통해서만 개선되고, value function 은 policy 에 따라서만 정의되므로:
- E 를 많이 하면: 정확하지만 느림
- E 를 적게 하면: 빠르지만 부정확
- 둘의 balance (e.g., TD) 는: 중간 속도, 중간 정확도

하지만 **모두 같은 고정점으로 수렴** — E 와 I 사이의 다른 trade-off 일 뿐.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Generalized Policy Iteration

정책 (또는 value function) 을 업데이트하되, 두 프로세스가 모두 "충분히" 일어나면 GPI:

**Process A (Evaluation)**: 임의의 방식으로 $V$ 가 $V^\pi$ 방향으로 이동
**Process B (Improvement)**: Policy 가 $V$ 에 대한 greedy 로 개선

조건:
1. E 프로세스: 어떤 정책 $\pi$ 에 대해 $V \to V^\pi$ (같은 속도 무관)
2. I 프로세스: $\pi$ 가 $\arg\max_a [r + \gamma P V]$ 방향으로 개선
3. Interleaving: E 와 I 를 임의로 섞음 (각각 무한 번 일어남)

**결과**: E 와 I 의 고정점 $V^*, \pi^*$ 로 수렴.

### 정의 5.2 — Trade-off 의 양적화

각 알고리즘의 E/I balance:

$$\text{Sample Efficiency} \approx f(\text{how much E}), \quad \text{Convergence Speed} \approx g(\text{how much I})$$

- **PI**: $f$ 높음, $g$ 낮음 (E fully, 후 I)
- **VI**: $f$ 낮음, $g$ 높음 (E minimal, I immediately)
- **TD**: $f, g$ 중간 (E: 1 step, I: frequently)

---

## 🔬 정리와 증명

### 정리 5.1 (GPI Convergence — Sutton & Barto 2018)

충분히 자주 E 와 I 가 일어나는 GPI 는 $V^*, \pi^*$ 로 수렴한다.

**증명 개요** (informal):

**Step 1**: E 가 충분히 일어나면 $V_k \to V^{\pi_k}$ (어떤 속도든).

**Step 2**: $V^{\pi_k}$ 가 고정되면 I 는 monotonic improvement:
$$V^{\pi_{k+1}} \geq V^{\pi_k}$$

**Step 3**: Policy space 가 유한 (tabular) 이므로, monotonic improvement 는 finite steps 내 convergence (policy iteration 정리와 동일).

**Step 4**: Converged 시 정책은 greedy-in-self:
$$\pi^* \in \arg\max_a [r + \gamma P V^{\pi^*}]$$

이는 Bellman optimality 의 정의 ⟹ $V^{\pi^*} = V^*$ $\square$

### 정리 5.2 (Model-Free GPI)

Model (P, r) 을 몰라도 GPI 는 작동한다. E 프로세스를:

**Sampling 으로 approximation**:
- MC: Full rollout (return estimate)
- TD: 1-step bootstrap (TD target)
- Q-learning: Off-policy bootstrap

하면 같은 고정점으로 수렴 (조건: exploration + learning rate decay).

**증명 sketch**: Stochastic Approximation Theory (Robbins-Monro) — 노이즈가 있는 update 도 평균적으로 같은 fixed point 로 수렴 $\square$.

### 정리 5.3 (Function Approximation 과의 관계)

Tabular 에서 function approximation 으로 확장할 때, GPI 틀은 유지되지만:

1. **Deadly Triad** (Sutton 2015): Off-policy + Bootstrapping + FA = 발산 가능
2. **해결책**: (a) Tabular 로 돌아가기, (b) FA 제한 (linear), (c) Gradient TD, (d) Experience replay + frozen target (DQN)

GPI 틀은 항상 유효하지만, 실무 안정성은 별개 $\square$.

---

## 💻 NumPy 구현 검증

### 실험 1 — Policy Iteration vs Value Iteration as GPI instances

```python
import numpy as np
import matplotlib.pyplot as plt

S = 16
A = 4
gamma = 0.9

def make_gridworld_matrices():
    """4×4 Gridworld"""
    P = np.zeros((S, A, S))
    r = -np.ones((S, A))
    
    def coord_to_idx(x, y):
        return x * 4 + y if 0 <= x < 4 and 0 <= y < 4 else None
    def idx_to_coord(idx):
        return (idx // 4, idx % 4)
    
    for s in range(S):
        x, y = idx_to_coord(s)
        next_states = [
            coord_to_idx(x - 1, y),
            coord_to_idx(x + 1, y),
            coord_to_idx(x, y - 1),
            coord_to_idx(x, y + 1),
        ]
        for a, ns in enumerate(next_states):
            if ns is None: ns = s
            P[s, a, ns] = 1.0
            r[s, a] = 1.0 if ns == 15 else -1.0
    
    return P, r

P, r = make_gridworld_matrices()

def policy_eval_partial(pi, P, r, gamma, n_iters=1):
    """Partial PE: exactly n_iters of Bellman update"""
    V = np.zeros(S)
    for _ in range(n_iters):
        Q = r + gamma * np.einsum('sap,p->sa', P, V)
        V = (pi * Q).sum(axis=1)
    return V

def policy_improve(V, r, P, gamma):
    """Policy Improvement: greedy policy"""
    Q = r + gamma * np.einsum('sap,p->sa', P, V)
    pi_new = np.zeros((S, A))
    pi_new[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return pi_new

# GPI with different PE depths
def gpi_with_eval_depth(P, r, gamma, eval_depth, max_iters=50):
    """
    eval_depth: how many PE iterations per cycle
    - eval_depth = inf: Policy Iteration
    - eval_depth = 1: Value Iteration (approx)
    """
    pi = np.ones((S, A)) / A
    values = []
    
    for k in range(max_iters):
        V = policy_eval_partial(pi, P, r, gamma, n_iters=eval_depth)
        values.append(V[0])
        
        pi_new = policy_improve(V, r, P, gamma)
        if np.allclose(pi, pi_new):
            break
        pi = pi_new
    
    return values

# Compare different eval depths
depths = [1, 5, 10, 100]
results = {}

for d in depths:
    vals = gpi_with_eval_depth(P, r, gamma, d)
    results[d] = vals
    print(f"Eval depth {d:3d}: converged in {len(vals):3d} policy iterations")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for d in depths:
    ax.plot(results[d], 'o-', label=f'Eval depth = {d}', markersize=6, linewidth=2)
ax.set_xlabel('Policy Iteration Step')
ax.set_ylabel('V(s=0)')
ax.set_title('GPI with Different E/I Balance')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Error vs iterations
ax = axes[1]
V_optimal = results[100][-1] if len(results[100]) > 0 else 0
for d in depths:
    errors = [abs(v - V_optimal) for v in results[d]]
    ax.semilogy(errors, 'o-', label=f'Eval depth = {d}', markersize=6, linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('|V(s) - V^*(s)|')
ax.set_title('Convergence Error')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/gpi_eval_depth.png', dpi=150)
```

**예상 출력**:
```
Eval depth   1: converged in  50 policy iterations
Eval depth   5: converged in   8 policy iterations
Eval depth  10: converged in   4 policy iterations
Eval depth 100: converged in   3 policy iterations
```

### 실험 2 — Q-learning as GPI (off-policy)

```python
def q_learning_gpi(P, r, gamma, n_steps=5000, alpha=0.1, epsilon=0.1):
    """
    Q-learning: implicit GPI
    - E: 1-step TD bootstrap (r + γ max Q(s', a'))
    - I: implicit in max operator (greedy action)
    """
    Q = np.zeros((S, A))
    values = []
    
    s = 0  # Start state
    for step in range(n_steps):
        # Behavior: epsilon-greedy (exploration)
        if np.random.rand() < epsilon:
            a = np.random.randint(A)
        else:
            a = np.argmax(Q[s])
        
        # Environment
        next_s = np.argmax(P[s, a])  # Deterministic env
        reward = r[s, a]
        
        # TD update (implicit evaluation)
        target = reward + gamma * np.max(Q[next_s])
        Q[s, a] += alpha * (target - Q[s, a])
        
        # Track V(s) estimate
        values.append(np.max(Q[s]))
        
        s = next_s
    
    return Q, values

Q_ql, ql_values = q_learning_gpi(P, r, gamma, n_steps=2000)

# Extract learned value function
V_ql = np.max(Q_ql, axis=1)

print(f"Q-learning learned value grid:\n{V_ql.reshape(4, 4).round(2)}")

# Plot Q-learning convergence
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ql_values, alpha=0.7, linewidth=1)
ax.set_xlabel('Step')
ax.set_ylabel('max_a Q(s, a)')
ax.set_title('Q-learning Convergence (GPI with off-policy E)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/qlearning_gpi.png', dpi=150)
```

### 실험 3 — GPI 사각 다이어그램

```python
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 8))

# States: π, V
pi_box = patches.FancyBboxPatch((0.1, 0.7), 0.2, 0.15, boxstyle="round,pad=0.01",
                                 edgecolor='blue', facecolor='lightblue', linewidth=2)
v_box = patches.FancyBboxPatch((0.7, 0.7), 0.2, 0.15, boxstyle="round,pad=0.01",
                                edgecolor='red', facecolor='lightcoral', linewidth=2)
pi_opt_box = patches.FancyBboxPatch((0.1, 0.2), 0.2, 0.15, boxstyle="round,pad=0.01",
                                     edgecolor='blue', facecolor='lightblue', linewidth=2)
v_opt_box = patches.FancyBboxPatch((0.7, 0.2), 0.2, 0.15, boxstyle="round,pad=0.01",
                                    edgecolor='red', facecolor='lightcoral', linewidth=2)

ax.add_patch(pi_box)
ax.add_patch(v_box)
ax.add_patch(pi_opt_box)
ax.add_patch(v_opt_box)

# Labels
ax.text(0.2, 0.775, 'π', fontsize=20, weight='bold', ha='center')
ax.text(0.8, 0.775, 'V^π', fontsize=20, weight='bold', ha='center')
ax.text(0.2, 0.275, 'π*', fontsize=20, weight='bold', ha='center')
ax.text(0.8, 0.275, 'V*', fontsize=20, weight='bold', ha='center')

# Arrows: E and I
# Top right: E (evaluate π)
ax.annotate('', xy=(0.7, 0.775), xytext=(0.35, 0.775),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(0.525, 0.82, 'E: evaluate π', fontsize=11, ha='center', weight='bold', color='green')

# Right side down: I (improve on V^π)
ax.annotate('', xy=(0.8, 0.4), xytext=(0.8, 0.68),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
ax.text(0.88, 0.54, 'I: improve', fontsize=11, ha='left', weight='bold', color='purple')

# Bottom left: E (evaluate π*)
ax.annotate('', xy=(0.35, 0.275), xytext=(0.7, 0.275),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(0.525, 0.22, 'E: evaluate π*', fontsize=11, ha='center', weight='bold', color='green')

# Left side up: I (improve on V*)
ax.annotate('', xy=(0.2, 0.68), xytext=(0.2, 0.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
ax.text(0.12, 0.54, 'I:', fontsize=11, ha='right', weight='bold', color='purple')
ax.text(0.12, 0.50, 'improve', fontsize=11, ha='right', weight='bold', color='purple')

# Fixed point label
ax.text(0.5, 0.075, 'Fixed Point: π* is greedy-in-self, V* = V^(π*)', 
        fontsize=12, ha='center', weight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Generalized Policy Iteration (GPI) Diagram', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('/tmp/gpi_diagram.png', dpi=150)
```

---

## 🔗 후속 레포와의 연결

- **Ch6-01+ Model-Free RL Deep Dive**: GPI 의 sampling 버전 (MC, TD, Q-learning, SARSA)
- **Ch7-01+ Policy Gradient Deep Dive**: Actor-Critic 의 GPI 프레임
- **Ch8-01+ Deep RL Deep Dive**: DQN, PPO, A3C 등의 GPI 해석

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 확장 |
|------|-------------|
| Tabular | FA: convergence guarantee 약화 (deadly triad) |
| Model-known | Model-free: E 를 sampling 으로 추정 (convergence 느림) |
| Finite state/action | Continuous: discretization 또는 deep FA |
| Stationary | Non-stationary: episodic, restart-able |

---

## 📌 핵심 정리

$$\boxed{\text{GPI: 임의의 E-I interleaving} \to V^*, \pi^*}$$

**통합 프레임**:

| 알고리즘 | E 방식 | I 방식 | 특징 |
|---------|---------|---------|------|
| Policy Iteration | Exact (iterate to conv) | Greedy | 느림, 정확 |
| Value Iteration | 1-step | Implicit (via max) | 빠름, 근사 |
| Monte Carlo | Full rollout | Greedy | Sample 기반, high var |
| TD/SARSA | 1-step bootstrap | Greedy/ε-greedy | On-policy, low var |
| Q-learning | 1-step off-policy | Implicit greedy | Off-policy, 불안정 가능 |
| Actor-Critic | Critic bootstraps | Actor improves | 병렬 학습 |
| Deep RL | NN approximation | NN policy | FA + deadly triad 주의 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GPI 사각 다이어그램에서 네 개의 경로 (순시계) 가 모두 같은 고정점에 도달한다는 것을 보이라. 어느 경로가 가장 빠른가?

<details>
<summary>해설</summary>

네 경로:
1. $\pi \xrightarrow{E} V^\pi \xrightarrow{I} \pi'$ 
2. $\pi \xrightarrow{I} \pi' \xrightarrow{E} V^{\pi'}$ (I 먼저)
3. $\pi' \xrightarrow{E} V^{\pi'} \xrightarrow{I} \pi''$ 
4. $\pi' \xrightarrow{I} \pi'' \xrightarrow{E} V^{\pi''}$ 

모두 같은 고정점 ($V^*, \pi^*$) 로 수렴 — 이것이 GPI 의 정의.

**가장 빠른 경로**: E 를 충분히 (정확함) 하면서도 I 를 자주 (빠름). Trade-off 는 문제 구조에 따라 다름.

예: PI (E fully) 는 iteration 수는 적지만 각 iteration 이 비쌈. VI (E minimal) 는 각 iteration 이 싸지만 많음 $\square$

</details>

**문제 2** (심화): Q-learning 이 왜 GPI 인가? Exploration (ε-greedy) 가 PI 에 어떻게 영향을 미치는가?

<details>
<summary>해설</summary>

Q-learning 의 update:
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**E part**: TD target $r + \gamma \max_{a'} Q(s', a')$ — 1-step evaluation of some (off-policy) behavior.

**I part**: $\max_{a'}$ operator — implicit greedy improvement.

**Exploration**: ε-greedy 는 PI 가 충분히 모든 action 을 보게 함 (off-policy 보정).

Without exploration: Q-learning 이 suboptimal action 에 stuck (incomplete sampling).

이것이 "exploration vs exploitation" 의 정체 — GPI 가 충분히 일어나게 하기 위한 필요조건 $\square$

</details>

**문제 3** (실전): Deep RL (DQN, PPO) 에서 "Deadly Triad" (Off-policy + Bootstrapping + FA) 가 왜 발산하는가? GPI 틀에서 어디가 깨지는가?

<details>
<summary>해설</summary>

Deadly Triad (Sutton 2015, Szepesvári & Scherrer 2010):

세 조건이 동시에 만족되면 발산 가능:

1. **Off-policy** (Q-learning): E 가 다른 behavior policy 에서 sample
2. **Bootstrapping** (TD): E 가 다른 estimate 에 의존 (target Q 가 exact 아님)
3. **Function Approximation** (Deep): Linear 또는 nonlinear FA

**이유**: 
- On-policy TD + FA 는 안정 (projected Bellman equation)
- Off-policy + Bootstrapping: 비정상성 (non-stationarity) 유입
- FA: Approximation error + overestimation bias

**GPI 틀에서의 문제**:
- E 가 정확하지 않음 (FA error)
- I 가 biased (off-policy error)
- 동시 발생 ⟹ 오차 누적 ⟹ divergence 가능

**해결**:
- Experience replay (distribution 정상화)
- Frozen target network (bootstrapping 안정화)
- Gradient TD (approximation error 제한)

GPI 는 여전히 유효하지만, 실무에서는 Deadly Triad 를 피해야 함 $\square$

</details>

---

<div align="center">

[◀ 이전: 04. Value Iteration 완성](./04-value-iteration.md) | [📚 README](../README.md) | [다음 ▶: Ch6-01. State Distribution 과 Stationary Distribution](../ch6-mdp-properties/01-state-distribution.md)

</div>
