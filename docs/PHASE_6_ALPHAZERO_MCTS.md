# PHASE 6 — AlphaZero Architecture for Snake: MCTS + Reinforcement Learning

> This document consolidates and formalizes the transition from **Reactive Reinforcement Learning (Pure DQN)** to a **Planning-Augmented Learning System (AlphaZero-style)**. It integrates theoretical foundations, algorithmic mechanisms, and system-level engineering considerations to ensure conceptual completeness and architectural clarity.

---

# Table of Contents

1. [Limitations of Pure Q-Learning in Long-Horizon Snake](#1-limitations-of-pure-q-learning-in-long-horizon-snake)
2. [Monte Carlo Tree Search (MCTS): Formal Decomposition](#2-monte-carlo-tree-search-mcts-formal-decomposition)
3. [AlphaZero Synergy: Neural Network + Tree Search](#3-alphazero-synergy-neural-network--tree-search)
4. [Resolving the Greedy Trapper Failure Mode](#4-resolving-the-greedy-trapper-failure-mode)
5. [System Architecture & Engineering Constraints](#5-system-architecture--engineering-constraints)
6. [Theoretical Conclusion](#6-theoretical-conclusion)

---

# 1. Limitations of Pure Q-Learning in Long-Horizon Snake

## 1.1 Bellman Optimality vs Practical Approximation

The theoretical foundation of Q-learning is the **Bellman Optimality Equation**:

Q^_(s,a)=\mathbb{E}[r+\gamma \max\_{a'} Q^_(s',a')\mid s,a]

Where:

- ( s ): current state
- ( a ): action
- ( r ): immediate reward
- ( \gamma \in (0,1) ): discount factor
- ( Q^\* ): optimal action-value function

In practice, **Deep Q-Networks (DQN)** approximate this function using a neural network ( Q\_\theta(s,a) ).

### Core Limitation

The issue is not the Bellman equation itself, but:

- **Function approximation error**
- **High-dimensional, non-linear state topology**
- **Long-term dependency chains**

---

## 1.2 Topological Complexity in Snake

Snake introduces a **dynamic topological constraint system**:

- The snake body acts as a **moving obstacle field**
- Valid paths depend on **future body positions**
- The environment is **non-Markovian in practice** under imperfect approximation

### Consequence

The neural network must implicitly learn:

- Spatial reasoning
- Long-term path feasibility
- Avoidance of self-trapping configurations

This is extremely difficult for a pure value function approximator.

---

## 1.3 Discount Factor and Temporal Credit Assignment

Future rewards are exponentially discounted:

\gamma^k

For example:

- ( \gamma = 0.99 )
- ( k = 200 )

→ Contribution ≈ 0.13

### Implication

- Long-term penalties (e.g., self-trap) are **weakly propagated**
- Immediate rewards dominate optimization

This leads to **temporal credit assignment failure**, where:

> The model cannot correctly attribute future failure to earlier decisions

---

## 1.4 The Greedy Trapper Phenomenon

### Definition

A **Greedy Trapper** is a policy that:

- Maximizes short-term reward
- Ignores long-term spatial feasibility
- Enters irreversible dead-end states

### Root Causes

| Phenomenon            | Cause                                      |
| --------------------- | ------------------------------------------ |
| Myopic behavior       | Discounted reward bias                     |
| Dead-end blindness    | Lack of explicit planning                  |
| Delayed failure       | Weak gradient signal from distant outcomes |
| Local optimum lock-in | Policy converges prematurely               |

---

# 2. Monte Carlo Tree Search (MCTS): Formal Decomposition

MCTS is an **online planning algorithm** that constructs a partial search tree rooted at the current state.

## Node Statistics

Each edge ((s,a)) maintains:

- ( N(s,a) ): visit count
- ( W(s,a) ): total accumulated value
- ( Q(s,a) = W/N ): mean value
- ( P(s,a) ): prior probability (from policy network)

---

## 2.1 Selection Phase

Tree traversal is guided by a balance between:

- **Exploitation** (high-value actions)
- **Exploration** (under-explored actions)

### PUCT Formula (AlphaZero)

a*t=\arg\max_a\left(Q(s,a)+c*{puct}\cdot P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}\right)

### Interpretation

- ( Q(s,a) ): empirical value
- ( P(s,a) ): prior guidance
- Exploration term encourages breadth

---

## 2.2 Expansion Phase

When reaching a leaf node:

- Generate all valid actions
- Initialize child nodes
- Assign prior probabilities from neural network

---

## 2.3 Evaluation Phase (Value Approximation)

Instead of random rollout:

- Use **value network**:

[
v\_\theta(s) \in [-1,1]
]

### Benefit

- Lower variance
- Faster convergence
- More stable learning

---

## 2.4 Backpropagation Phase

Update statistics along the path:

[
N \leftarrow N+1,\quad
W \leftarrow W+v,\quad
Q \leftarrow W/N
]

### Key Insight

The tree gradually becomes a **statistical estimator of action quality**

---

# 3. AlphaZero Synergy: Neural Network + Tree Search

## 3.1 Unified Model

The neural network outputs:

[
(\mathbf{p}, v) = f_\theta(s)
]

- ( \mathbf{p} ): policy prior
- ( v ): value estimate

---

## 3.2 Closed Feedback Loop

```text
Neural Network → guides MCTS
MCTS → generates improved policy targets
Targets → train Neural Network
```

This creates a **self-improving system**.

---

## 3.3 Policy Improvement via Search

Instead of greedy action selection, training uses:

[
\pi(a|s) \propto N(s,a)^{1/\tau}
]

Where:

- ( \pi ): improved policy
- ( \tau ): temperature parameter controlling exploration

---

## 3.4 Training Objective

[
\mathcal{L} = (z - v)^2 - \pi^T \log p + \lambda |\theta|^2
]

### Components

| Term           | Meaning                            |
| -------------- | ---------------------------------- |
| Value loss     | Accuracy of state evaluation       |
| Policy loss    | Match improved search distribution |
| Regularization | Prevent overfitting                |

---

# 4. Resolving the Greedy Trapper Failure Mode

## 4.1 Reactive vs Planning Paradigm

| Approach  | Decision Strategy               |
| --------- | ------------------------------- |
| DQN       | Immediate value maximization    |
| AlphaZero | Structured lookahead via search |

---

## 4.2 Why MCTS Works

MCTS transforms:

> Implicit future estimation → Explicit trajectory evaluation

Instead of guessing long-term outcomes, it:

- Simulates them directly
- Observes structural constraints
- Evaluates survivability

---

## 4.3 Topological Awareness Emerges

Through repeated simulation:

- Dead-ends are discovered early
- Safe paths are reinforced
- Space coverage strategies emerge

---

# 5. System Architecture & Engineering Constraints

AlphaZero is not just an algorithm—it is a **distributed computation system**.

---

## 5.1 Layered Architecture

```text
[Environment Layer]
Snake simulation (state transition)

[Planning Layer]
MCTS (tree search, decision engine)

[Learning Layer]
Neural Network (policy + value)

[Infrastructure Layer]
- Parallel execution
- GPU inference
- Memory optimization
```

---

## 5.2 Lightweight State Representation

### Requirement

MCTS requires:

- Thousands of state clones per decision

### Solution

Use a **minimal state structure**:

- Positions only
- No rendering data
- Cache-efficient layout

---

## 5.3 Batched Inference

### Problem

Cross-language calls (e.g., Rust ↔ Python) incur overhead and are constrained by:

- Global Interpreter Lock (GIL)

### Solution

Batch multiple states:

- Input tensor:
  [
  [B, C, H, W]
  ]

- Single GPU inference pass

### Result

- Significant throughput improvement
- Reduced latency per simulation

---

## 5.4 Parallel MCTS with Virtual Loss

### Problem

Threads converge on same high-value nodes

### Solution: Virtual Loss

- Temporarily penalize selected node
- Force exploration of alternative branches
- Restore value after computation

---

# 6. Theoretical Conclusion

To achieve near-perfect Snake performance:

## Transition Required

From:

- **Reactive Value Approximation**

To:

- **Planning-Augmented Learning**

---

## System Roles

| Component              | Role                                            |
| ---------------------- | ----------------------------------------------- |
| Reinforcement Learning | Learn general priors                            |
| MCTS                   | Perform context-specific reasoning              |
| Neural Network         | Compress experience into function approximation |
| Simulation             | Generate high-quality training data             |

---

## Final Insight

AlphaZero succeeds because it separates concerns:

- **Learning** → builds intuition
- **Planning** → enforces correctness

This combination enables:

- Long-horizon reasoning
- Robust handling of delayed consequences
- Emergence of globally optimal strategies

---

This document provides a complete conceptual, mathematical, and architectural foundation for implementing an AlphaZero-style system for Snake in a production-grade environment (e.g., Rust + Python + GPU inference).
