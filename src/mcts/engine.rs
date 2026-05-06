use std::sync::atomic::Ordering;

use crate::domain::action::RelativeAction;
use crate::lite_state::dynamics::SnakeDynamics;
use crate::lite_state::types::SnakeStateLite;

use super::arena::TreeArena;
use super::evaluator::BatchEvaluator;
use super::puct::calculate_puct;
use super::types::{MCTSConfig, MCTSResult};

pub struct MCTSEngine<E: BatchEvaluator> {
    arena: TreeArena,
    config: MCTSConfig,
    evaluator: E,
}

impl<E: BatchEvaluator> MCTSEngine<E> {
    pub fn new(config: MCTSConfig, evaluator: E) -> Self {
        // We add an extra margin for num_threads * 3 abandoned node allocations.
        let capacity = (config.num_simulations + config.num_threads) * 3 + 1;
        Self {
            arena: TreeArena::new(capacity),
            config,
            evaluator,
        }
    }

    pub fn search(&self, root_state: &SnakeStateLite) -> MCTSResult {
        self.arena.clear();

        // 1. Evaluate root sequentially
        let root_state_clone = root_state.clone();
        let (_root_value, root_policy) = self.evaluator.evaluate(&root_state_clone);

        // 2. Create root node
        let root_idx = self.arena.alloc_node();
        self.arena.get(root_idx).init(None, 1.0, None);

        // Expand root immediately if not terminal
        if !root_state_clone.is_terminal() {
            self.expand(root_idx, &root_policy);
        }

        // 3. Spawn threads for concurrent search loops
        let sims_per_thread = self.config.num_simulations / self.config.num_threads;
        let remainder = self.config.num_simulations % self.config.num_threads;

        std::thread::scope(|s| {
            for t in 0..self.config.num_threads {
                let sims = sims_per_thread + if t == 0 { remainder } else { 0 };
                let thread_root_state = root_state.clone();
                s.spawn(move || {
                    // Pre-allocate scratch state once per thread to reuse VecDeque buffer
                    let mut scratch_state = thread_root_state.clone();
                    for _ in 0..sims {
                        scratch_state.clone_from_root(&thread_root_state);
                        self.single_simulation(root_idx, &mut scratch_state);
                    }
                });
            }
        });

        // Calculate improved policy from root
        let root = self.arena.get(root_idx);
        let mut improved_policy = [0.0; 3];
        let root_visits = root.visit_count.load(Ordering::Relaxed) as f32;
        
        if root_visits > 0.0 {
            for i in 0..3 {
                let child_idx = root.children[i].load(Ordering::Relaxed);
                if child_idx != u32::MAX {
                    improved_policy[i] = self.arena.get(child_idx).visit_count.load(Ordering::Relaxed) as f32 / root_visits;
                }
            }
        } else {
            improved_policy = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        }

        MCTSResult {
            policy: improved_policy,
            root_value: root.q_value(self.config.virtual_loss),
        }
    }

    fn single_simulation(&self, root_idx: u32, current_state: &mut SnakeStateLite) {
        let mut current_idx = root_idx;
        let mut search_path = vec![(current_idx, 0.0)];

        // Phase 1: Selection
        while self.arena.get(current_idx).is_expanded() {
            let next_idx = self.select_best_child(current_idx);
            
            // Apply virtual loss
            self.arena.get(next_idx).virtual_loss_active.fetch_add(1, Ordering::Relaxed);
            
            current_idx = next_idx;
            
            let action = self.arena.get(current_idx).action_taken().unwrap();
            let outcome = current_state.apply(action);
            
            search_path.push((current_idx, outcome.reward));
        }

        // Phase 2 & 3: Expansion & Evaluation
        let leaf_value = if current_state.is_terminal() {
            0.0
        } else {
            let (value, policy) = self.evaluator.evaluate(&current_state);
            self.expand(current_idx, &policy);
            value
        };

        // Phase 4: Backpropagation
        self.backpropagate(&search_path, leaf_value);
    }

    fn select_best_child(&self, node_idx: u32) -> u32 {
        let node = self.arena.get(node_idx);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_child = u32::MAX;

        let node_visits = node.visit_count.load(Ordering::Relaxed) + node.virtual_loss_active.load(Ordering::Relaxed);

        for child_atomic in &node.children {
            let child_idx = child_atomic.load(Ordering::Acquire);
            if child_idx != u32::MAX {
                let child = self.arena.get(child_idx);
                let child_visits = child.visit_count.load(Ordering::Relaxed) + child.virtual_loss_active.load(Ordering::Relaxed);
                
                let score = calculate_puct(
                    node_visits,
                    child_visits,
                    child.prior_prob(),
                    child.q_value(self.config.virtual_loss),
                    self.config.c_puct,
                );

                if score > best_score {
                    best_score = score;
                    best_child = child_idx;
                }
            }
        }

        if best_child == u32::MAX {
            panic!("Node is expanded but has no valid children");
        }
        best_child
    }

    fn expand(&self, node_idx: u32, policy: &[f32; 3]) {
        let node = self.arena.get(node_idx);

        // Fast path check
        if node.is_expanded() {
            return;
        }

        let actions = [
            RelativeAction::Straight,
            RelativeAction::TurnLeft,
            RelativeAction::TurnRight,
        ];

        let mut child_indices = [0; 3];
        for i in 0..3 {
            let child_idx = self.arena.alloc_node();
            let child_node = self.arena.get(child_idx);
            child_node.init(Some(node_idx), policy[i], Some(actions[i]));
            child_indices[i] = child_idx;
        }

        // Try to link children using optimistic concurrency control
        for i in 0..3 {
            let res = node.children[i].compare_exchange(
                u32::MAX, 
                child_indices[i], 
                Ordering::Release, 
                Ordering::Relaxed
            );
            
            if res.is_err() {
                // Another thread won and expanded this node already.
                // Our allocated nodes will be abandoned and unused, which is safe.
            }
        }
    }

    fn backpropagate(&self, search_path: &[(u32, f32)], leaf_value: f32) {
        let mut current_value = leaf_value;
        let root_idx = search_path[0].0;
        
        for &(node_idx, reward) in search_path.iter().rev() {
            let node = self.arena.get(node_idx);
            
            if node_idx != root_idx {
                node.virtual_loss_active.fetch_sub(1, Ordering::Relaxed);
            }
            
            node.add_visit();
            node.add_value(current_value);
            
            current_value = reward + self.config.discount_factor * current_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::evaluator::StubEvaluator;
    use crate::lite_state::types::LiteStateConfig;

    #[test]
    fn test_mcts_engine_expansion() {
        let config = MCTSConfig {
            num_simulations: 1000,
            c_puct: 1.0,
            discount_factor: 0.99,
            num_threads: 4,
            virtual_loss: 1.0,
        };
        
        let evaluator = StubEvaluator;
        let engine = MCTSEngine::new(config, evaluator);
        
        let state_config = LiteStateConfig {
            width: 10,
            height: 10,
            max_steps: 100,
            starvation_limit: 200,
            seed: Some(42),
        };
        let state = SnakeStateLite::new(state_config).unwrap();
        
        let result = engine.search(&state);
        
        let root_visits = engine.arena.get(0).visit_count.load(Ordering::Relaxed);
        assert_eq!(root_visits, 1000, "Root visit count should match num_simulations exactly");
        
        // Ensure virtual loss is completely reverted for all nodes
        for i in 0..engine.arena.len() {
            let active_loss = engine.arena.get(i as u32).virtual_loss_active.load(Ordering::Relaxed);
            assert_eq!(active_loss, 0, "Virtual loss active count should be 0 after search");
        }
        
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Policy probabilities should sum to 1.0");
    }
}
