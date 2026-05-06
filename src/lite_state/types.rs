use std::collections::VecDeque;

use rand::rngs::StdRng;

use crate::domain::{
    action::{Direction, RelativeAction},
    state::Position,
};

#[derive(Debug, Clone, Copy)]
pub struct LiteStateConfig {
    pub width: usize,
    pub height: usize,
    pub max_steps: u32,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub ate_food: bool,
}

#[derive(Debug, Clone)]
pub struct SnakeStateLite {
    pub width: usize,
    pub height: usize,
    pub body: VecDeque<Position>,
    pub dir: Direction,
    pub food: Position,
    pub pending_growth: u32,
    pub steps: u32,
    pub score: u32,
    pub terminated: bool,
    pub truncated: bool,
    pub max_steps: u32,
    pub pending_action: RelativeAction,
    pub rng: StdRng,
}
