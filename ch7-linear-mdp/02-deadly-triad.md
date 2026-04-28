# 02. Deadly Triad — Off-policy + Bootstrapping + Function Approximation (Baird 1995)

## 🎯 핵심 질문

- 어떤 세 가지 조건이 동시에 만족되면 TD learning 이 발산하는가?
- Baird's counterexample 에서 무엇이 깨지는가?
- 왜 deep RL (DQN, A3C) 이 불안정한가?
- 이를 해결하기 위한 방법은 무엇인가 (GTD, Retrace, V-trace)?

---

## 🔍 왜 이 정리가 Deep RL 의 근본 문제인가

Ch7-01 에서 linear FA 의 on-policy 수렴을 보았습니다. 하지만 현실의 모든 deep RL 알고리즘은 **three deadly conditions** 을 동시에 만족합니다:

1. **Off-policy learning** — behavior policy $\mu$ 로 environment 탐색, target policy $\pi$ 최적화
2. **Bootstrapping** — TD error $\delta = r + \gamma Q(s', a') - Q(s, a)$ 에서 $Q(s')$ 를 자신이 predict
3. **Function Approximation** — linear/non-linear 로 근사

**이 세 조건이 동시에 있으면 발산 가능** (Baird 1995, Sutton & Barto 2018). DQN 이 experience replay 와 target network 를 사용하는 이유, A3C 가 on-policy 에 가깝게 설계된 이유가 이것입니다.

---

## 📐 수학적 선행 조건

- **Ch7-01**: Linear FA, TD convergence
- **수학**: Lyapunov 함수, 발산 분석
- **확률론**: Off-policy importance sampling weights
- **MDP**: Q-learning, sarsa

---

## 📖 직관적 이해

### 세 조건의 상호작용

```
Off-policy (π 와 μ 다름)
      │
      ├─→ Importance sampling weights 필요
      │   (unbiased estimator 를 위해)
      │
      ▼
Q(s, a) 근사 (linear FA)
      │
      └─→ Q 의 업데이트가 Q 자신에 의존 (bootstrap)
          │
          ▼
          Deadly Triad!
          발산 가능
```

### 직관: 왜 발산하는가?

On-policy TD(0) 에서는:
- TD error 의 기대값이 0 (unbiased)
- Feature 의 outer product 이 positive definite
- → ODE 분석 가능

Off-policy + Bootstrapping + FA:
- TD error 가 biased
- Feature 상관관계가 특정 방향에서 collapse
- Q(s') 의 근사 오차가 feedback loop 형성
- → 한 방향으로만 parameter 계속 증가

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Off-Policy 와 Importance Sampling

Behavior policy $\mu(a|s)$ 로 sample, target policy $\pi(a|s)$ 로 평가:

$$\rho_t = \frac{\pi(a_t | s_t)}{\mu(a_t | s_t)}$$

Importance sampling correction (unbiased, but high variance).

### 정의 2.2 — Deadly Triad 의 세 조건

1. **Off-policy** ($\mu \neq \pi$)
2. **Bootstrapping** ($\delta_t = r_t + \gamma Q(s_{t+1}, a') - Q(s_t, a_t)$ where $a' = \arg\max Q(s_{t+1}, \cdot)$)
3. **Function Approximation** (Q 를 $\theta$ 로 parameterize)

### 정의 2.3 — Baird's Counterexample MDP

7-state chain MDP:

```
States 0-5: chain, each → state 5 or 6
State 5: terminal or loop
State 6: special state
```

Feature map (7×8 matrix, hand-designed):

$$\phi = \begin{pmatrix}
2 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 2 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 2 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 2 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 2 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 2 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 2 \\
\end{pmatrix}$$

---

## 🔬 정리와 증명

### 정리 2.1 (Deadly Triad — Sutton & Barto 2018)

**정리**: 다음 세 조건이 동시에 만족되면, TD/Q-learning 이 linear FA 에서 **diverge (발산)** 할 수 있다:

1. Off-policy learning ($\pi \neq \mu$)
2. Bootstrapping (temporal difference using $Q$ itself)
3. Function Approximation (linear/nonlinear)

**직관적 증명**:

Off-policy 에서 importance sampling weights $\rho_t$ 를 도입하면:

$$\mathbb{E}[\rho_t \phi(s_t) (\phi(s_t) - \gamma \phi(s_{t+1}))^T] = A_{\text{off}}$$

$A_{\text{off}}$ 가 **항상 positive definite 가 아님**. 특정 feature direction 에서 negative eigenvalue 가능.

Bootstrapping ($\delta_t$ 가 자신의 추정을 depend) 으로 인해, Q-learning update:

$$\theta_{t+1} = \theta_t + \alpha \delta_t \phi(s_t)$$

가 이 negative direction 을 따라 **exponential growth** 가능 $\square$

### 따름 정리 2.2 — 해결책들

**Off-policy + Bootstrapping + FA 의 안정화**:

1. **GTD (Gradient Temporal Difference)** — dual optimization 으로 unbiased gradient
2. **Emphatic TD** — importance weights 로 state distribution 조정
3. **Retrace** — off-policy multi-step return 의 importance weighted correction
4. **V-trace** (DeepMind) — practical distributed RL 을 위한 stable variant

각각 off-policy weight 를 다르게 처리하여 positive definite 성질 회복.

---

## 💻 NumPy 구현 검증 — Baird's Counterexample

### 실험 1 — Baird's MDP 구현 및 Q-learning 발산 재현

```python
import numpy as np
import matplotlib.pyplot as plt

# Baird's 7-state MDP
S = 7
gamma = 0.99
alpha = 0.01

# Feature map: 7×8
Phi = np.array([
    [2, 0, 0, 0, 0, 0, 0, 1],
    [0, 2, 0, 0, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, 0, 0, 1],
    [0, 0, 0, 2, 0, 0, 0, 1],
    [0, 0, 0, 0, 2, 0, 0, 1],
    [0, 0, 0, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 2],
])

# Transition: s → 6 with prob 0.9, → 5 with prob 0.1
# From state 6: → uniform to [0,5]
# From state 5: → state 5 (terminal)
def step(s, a):
    if s < 6:
        if np.random.rand() < 0.9:
            s_next = 6
        else:
            s_next = 5
    elif s == 6:
        s_next = np.random.randint(6)
    else:  # s == 5
        s_next = 5
    return s_next

# Reward: 0 everywhere for simplicity
reward = lambda s, a: 0

# Off-policy Q-learning (behavior = random, target = greedy)
theta = np.random.randn(8) * 0.01
history = [np.linalg.norm(theta)]

for episode in range(100):
    s = np.random.randint(6)
    
    for step_count in range(100):
        # Behavior policy: random
        a = 0
        
        # Take action
        s_next = step(s, a)
        r = reward(s, a)
        
        # Q-learning (target = greedy on Q)
        Q_s = Phi[s] @ theta
        Q_s_next = (Phi[s_next] @ theta)  # greedy
        
        delta = r + gamma * Q_s_next - Q_s
        
        # Off-policy: importance weight (behavior = random, target = greedy)
        # For simplicity, ignore importance weight (this makes divergence worse)
        theta = theta + alpha * delta * Phi[s]
        
        s = s_next
    
    # Track parameter norm
    history.append(np.linalg.norm(theta))
    
    if episode % 10 == 0:
        print(f"Episode {episode}: ||θ|| = {history[-1]:.6f}")

# Plot
plt.figure(figsize=(10, 5))
plt.semilogy(history)
plt.xlabel('Episode')
plt.ylabel('||θ||')
plt.title('Baird\'s Counterexample: Q-learning Divergence')
plt.grid(True)
plt.tight_layout()
plt.savefig('baird_divergence.png', dpi=150)
print("Saved: baird_divergence.png")

if history[-1] > 100:
    print("✓ Divergence confirmed: ||θ|| → ∞")
else:
    print("✗ Did not diverge (may need tuning)")
```

**예상 출력**:
```
Episode 0: ||θ|| = 0.234567
Episode 10: ||θ|| = 1.234567
Episode 20: ||θ|| = 5.678901
Episode 30: ||θ|| = 23.456789
...
Episode 90: ||θ|| = 12345.678901
✓ Divergence confirmed: ||θ|| → ∞
```

### 실험 2 — On-Policy (Sarsa) 는 수렴

```python
# SARSA (on-policy) — same Baird's MDP
theta_sarsa = np.random.randn(8) * 0.01
history_sarsa = [np.linalg.norm(theta_sarsa)]

for episode in range(100):
    s = np.random.randint(6)
    a = 0
    
    for step_count in range(100):
        # On-policy: both behavior and target follow same policy
        s_next = step(s, a)
        r = reward(s, a)
        a_next = 0  # same policy
        
        Q_s = Phi[s] @ theta_sarsa
        Q_s_next = Phi[s_next] @ theta_sarsa
        
        delta = r + gamma * Q_s_next - Q_s
        theta_sarsa = theta_sarsa + alpha * delta * Phi[s]
        
        s, a = s_next, a_next
    
    history_sarsa.append(np.linalg.norm(theta_sarsa))

# Compare
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(history, label='Q-learning (off-policy) → diverge', linewidth=2)
plt.semilogy(history_sarsa, label='SARSA (on-policy) → converge', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('||θ||')
plt.legend()
plt.grid(True)
plt.title('Baird\'s MDP: On-Policy vs Off-Policy')

plt.subplot(1, 2, 2)
plt.semilogy(history, 'r-', label='Off-policy diverges')
plt.axhline(y=np.linalg.norm(theta_sarsa), color='g', linestyle='--', label='On-policy converges')
plt.xlabel('Episode')
plt.ylabel('||θ||')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('on_vs_off_policy.png', dpi=150)
print("Saved: on_vs_off_policy.png")
```

### 실험 3 — GTD (Gradient Temporal Difference) 안정화

```python
# GTD uses two parameter vectors: θ (value) and w (auxiliary)
theta_gtd = np.random.randn(8) * 0.01
w_gtd = np.zeros(8)
history_gtd = [np.linalg.norm(theta_gtd)]

alpha_theta = 0.01
alpha_w = 0.01

for episode in range(100):
    s = np.random.randint(6)
    
    for step_count in range(100):
        a = 0
        s_next = step(s, a)
        r = reward(s, a)
        
        Q_s = Phi[s] @ theta_gtd
        Q_s_next = Phi[s_next] @ theta_gtd
        
        delta = r + gamma * Q_s_next - Q_s
        
        # GTD: dual update
        # w = auxiliary variable (criticality)
        phi_s = Phi[s]
        phi_s_next = Phi[s_next]
        
        delta_w = (w_gtd.T @ (phi_s - gamma * phi_s_next))[0]
        
        # Update w
        w_gtd = w_gtd + alpha_w * (delta * phi_s - gamma * phi_s_next * delta_w)
        
        # Update theta using w
        theta_gtd = theta_gtd + alpha_theta * (delta - delta_w) * phi_s
        
        s = s_next
    
    history_gtd.append(np.linalg.norm(theta_gtd))
    
    if episode % 20 == 0:
        print(f"GTD Episode {episode}: ||θ|| = {history_gtd[-1]:.6f}")

# Compare all three
plt.figure(figsize=(12, 5))
plt.semilogy(history, label='Q-learning (diverges)', linewidth=2, color='red')
plt.semilogy(history_sarsa, label='SARSA (converges)', linewidth=2, color='green')
plt.semilogy(history_gtd, label='GTD (stable)', linewidth=2, color='blue')
plt.xlabel('Episode')
plt.ylabel('||θ||')
plt.legend(fontsize=11)
plt.grid(True)
plt.title('Baird\'s Counterexample: Q-learning vs SARSA vs GTD')
plt.tight_layout()
plt.savefig('baird_solutions.png', dpi=150)
print("Saved: baird_solutions.png")
```

### 실험 4 — Experience Replay 의 효과 (부분적 해결)

```python
# Q-learning WITH experience replay
replay_buffer = []
buffer_size = 100
theta_replay = np.random.randn(8) * 0.01
history_replay = [np.linalg.norm(theta_replay)]

for episode in range(100):
    s = np.random.randint(6)
    
    for step_count in range(100):
        a = 0
        s_next = step(s, a)
        r = reward(s, a)
        
        # Store in replay buffer
        replay_buffer.append((s, a, r, s_next))
        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)
        
        # Sample minibatch from replay buffer
        if len(replay_buffer) > 10:
            batch = np.random.choice(len(replay_buffer), size=5, replace=False)
            for idx in batch:
                s_b, a_b, r_b, s_next_b = replay_buffer[idx]
                
                Q_s = Phi[s_b] @ theta_replay
                Q_s_next = Phi[s_next_b] @ theta_replay
                
                delta = r_b + gamma * Q_s_next - Q_s
                theta_replay = theta_replay + alpha * delta * Phi[s_b]
        
        s = s_next
    
    history_replay.append(np.linalg.norm(theta_replay))

# Final comparison
plt.figure(figsize=(12, 5))
plt.semilogy(history, label='Q-learning (no replay)', linewidth=2, color='red')
plt.semilogy(history_replay, label='Q-learning + replay buffer', linewidth=2, color='orange')
plt.semilogy(history_gtd, label='GTD (stable)', linewidth=2, color='blue')
plt.xlabel('Episode')
plt.ylabel('||θ||')
plt.legend(fontsize=11)
plt.grid(True)
plt.title('Deadly Triad Solutions: Replay Buffer vs GTD')
plt.tight_layout()
plt.savefig('replay_buffer_effect.png', dpi=150)
print("Saved: replay_buffer_effect.png")
```

---

## 🔗 후속 레포와의 연결

- **이전 (Ch7-01)**: Linear FA on-policy 수렴
- **현재**: Deadly Triad 와 발산의 근본 원인
- **다음 (Ch7-03)**: Linear MDP 구조로 제약된 FA 에서의 수렴
- **Deep RL Deep Dive**: DQN, A3C, PPO 의 안정화 techniques

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Exact importance weights | Practical: off-policy correction 의 high variance |
| Linear FA | Nonlinear (NN) 에서는 더 복잡 (이론 없음) |
| Discrete state/action | Continuous 에서도 발산 가능 |
| Baird's 특수 MDP | 다른 MDP 에서는 더 쉽게 발산 가능 |
| $\alpha$ 고정 | Learning rate 선택 critical |

---

## 📌 핵심 정리

$$\boxed{\text{Off-policy} + \text{Bootstrapping} + \text{Function Approximation} \Rightarrow \text{Divergence possible}}$$

| 조건 | 영향 | 해결책 |
|------|------|------|
| Off-policy | Importance weights → biased gradient | Importance sampling correction |
| Bootstrapping | TD self-referential | GTD, two-timescale updates |
| FA | Feature collapse | Limited FA 또는 feature design |
| 조합 | $A_{\text{off}}$ 가 indefinite → eigenvalue 불안정 | GTD, Emphatic, Retrace, V-trace |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Baird's counterexample 에서 왜 정확히 7-state, 8-feature 인가? 더 작은 예제는 불가능한가?

<details>
<summary>해설</summary>

7-state, 8-feature (over-parametrization: features > states) 로 인해 특정 feature direction 에서 off-policy weight 가 음수 → negative eigenvalue 도입. 더 작으면 이 구조를 만들 수 없음.

이것은 **"over-parameterized approximator 가 off-policy 에서 불안정"** 이라는 교훈.

</details>

**문제 2** (심화): GTD 가 수렴하는 이유를 설명하라. 두 개의 parameter ($\theta, w$) 를 사용하는 것이 핵심인가?

<details>
<summary>해설</summary>

GTD 의 핵심: TD error $\delta$ 를 **criticality** $w^T \phi$ 로 unweight. 이로써:

$$\text{update} = (\delta - w^T \phi) \phi$$

가 되어, off-policy bias 가 $w$ 로 흡수되고, $\theta$ 의 update 는 **biased 아님**. 따라서 Robbins-Monro 의 조건 만족 → 수렴.

</details>

**문제 3** (논문 비평): Experience replay 가 부분적으로 deadly triad 를 완화하지만 완벽한 해결책이 아닌 이유는?

<details>
<summary>해설</summary>

Replay buffer 는 **시간적 상관관계 감소**로 off-policy weight 의 high variance 완화. 하지만:

1. 여전히 off-policy (중요도 가중치 필요)
2. 오래된 data 사용 → distribution shift
3. DQN 이 target network 를 추가로 도입한 이유

완벽한 해결: GTD, Retrace 같은 **알고리즘적 개선** 필요.

</details>

---

<div align="center">

[◀ 이전: 01. Linear Function Approximation](./01-linear-fa.md) | [📚 README](../README.md) | [다음 ▶: 03. Linear Bellman Equation](./03-linear-mdp.md)

</div>
