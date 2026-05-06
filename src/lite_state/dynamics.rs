use std::collections::VecDeque;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::domain::{
    action::{Direction, RelativeAction},
    rules::{REWARD_DEATH, REWARD_FOOD, REWARD_STEP},
    state::Position,
};

use super::types::{LiteStateConfig, SnakeStateLite, StepOutcome};

pub trait SnakeDynamics {
    fn legal_actions(&self) -> [bool; 3];
    fn apply(&mut self, action: RelativeAction) -> StepOutcome;
    fn is_terminal(&self) -> bool;
}

impl SnakeStateLite {
    pub fn new(config: LiteStateConfig) -> Result<Self, String> {
        if config.width < 5 || config.height < 5 {
            return Err("width and height must both be >= 5".to_string());
        }
        if config.max_steps == 0 {
            return Err("max_steps must be > 0".to_string());
        }

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let mut state = Self {
            width: config.width,
            height: config.height,
            body: VecDeque::new(),
            dir: Direction::Right,
            food: Position::new(0, 0),
            pending_growth: 0,
            steps: 0,
            score: 0,
            terminated: false,
            truncated: false,
            max_steps: config.max_steps,
            starvation_limit: config.starvation_limit,
            steps_since_food: 0,
            pending_action: RelativeAction::Straight,
            rng,
        };
        state.reset();
        Ok(state)
    }

    pub fn reset(&mut self) {
        self.body = initial_body(self.width, self.height);
        self.dir = Direction::Right;
        self.pending_growth = 0;
        self.steps = 0;
        self.score = 0;
        self.terminated = false;
        self.truncated = false;
        self.steps_since_food = 0;
        self.pending_action = RelativeAction::Straight;
        self.food = sample_free_position(
            self.width,
            self.height,
            &self.body,
            self.pending_growth,
            &mut self.rng,
        );
    }

    pub fn head(&self) -> Position {
        *self.body.front().expect("snake body is never empty")
    }

    pub fn apply_in_place(&mut self, action: RelativeAction) -> StepOutcome {
        if self.terminated || self.truncated {
            return StepOutcome {
                reward: 0.0,
                terminated: self.terminated,
                truncated: self.truncated,
                ate_food: false,
            };
        }

        self.pending_action = action;
        self.dir = self.dir.apply_relative(action);

        let head = self.head();
        let (dx, dy) = self.dir.delta();
        let next_head = Position::new(head.x + dx, head.y + dy);

        let mut reward = REWARD_STEP;
        if is_out_of_bounds(next_head, self.width, self.height)
            || collides_with_body(next_head, &self.body, self.pending_growth)
        {
            reward += REWARD_DEATH;
            self.terminated = true;
            self.steps = self.steps.saturating_add(1);
            return StepOutcome {
                reward,
                terminated: self.terminated,
                truncated: self.truncated,
                ate_food: false,
            };
        }

        self.body.push_front(next_head);
        let ate_food = next_head == self.food;
        if ate_food {
            reward += REWARD_FOOD;
            self.score = self.score.saturating_add(1);
            self.pending_growth = self.pending_growth.saturating_add(1);
            self.steps_since_food = 0;
            self.food = sample_free_position(
                self.width,
                self.height,
                &self.body,
                self.pending_growth,
                &mut self.rng,
            );
        } else {
            self.steps_since_food = self.steps_since_food.saturating_add(1);
        }

        if self.pending_growth > 0 {
            self.pending_growth -= 1;
        } else {
            self.body.pop_back();
        }

        self.steps = self.steps.saturating_add(1);
        if self.steps >= self.max_steps || self.steps_since_food >= self.starvation_limit {
            self.truncated = true;
        }

        StepOutcome {
            reward,
            terminated: self.terminated,
            truncated: self.truncated,
            ate_food,
        }
    }
}

impl SnakeDynamics for SnakeStateLite {
    fn legal_actions(&self) -> [bool; 3] {
        [true, true, true]
    }

    fn apply(&mut self, action: RelativeAction) -> StepOutcome {
        self.apply_in_place(action)
    }

    fn is_terminal(&self) -> bool {
        self.terminated || self.truncated
    }
}

impl SnakeStateLite {
    /// Optimized clone that reuses existing heap allocations (VecDeque buffer).
    /// Avoids repeated heap allocs during MCTS simulations.
    pub fn clone_from_root(&mut self, other: &Self) {
        self.body.clone_from(&other.body);
        self.width = other.width;
        self.height = other.height;
        self.dir = other.dir;
        self.food = other.food;
        self.pending_growth = other.pending_growth;
        self.steps = other.steps;
        self.score = other.score;
        self.terminated = other.terminated;
        self.truncated = other.truncated;
        self.max_steps = other.max_steps;
        self.starvation_limit = other.starvation_limit;
        self.steps_since_food = other.steps_since_food;
        self.pending_action = other.pending_action;
        self.rng = other.rng.clone();
    }
}

fn initial_body(width: usize, height: usize) -> VecDeque<Position> {
    let center_x = (width / 2) as i32;
    let center_y = (height / 2) as i32;

    let mut body = VecDeque::new();
    body.push_back(Position::new(center_x, center_y));
    body.push_back(Position::new(center_x - 1, center_y));
    body.push_back(Position::new(center_x - 2, center_y));
    body
}

fn is_out_of_bounds(position: Position, width: usize, height: usize) -> bool {
    position.x < 0 || position.y < 0 || position.x >= width as i32 || position.y >= height as i32
}

fn collides_with_body(next_head: Position, body: &VecDeque<Position>, pending_growth: u32) -> bool {
    let body_to_check = if pending_growth == 0 {
        body.len().saturating_sub(1)
    } else {
        body.len()
    };
    body.iter().take(body_to_check).any(|segment| *segment == next_head)
}

fn sample_free_position(
    width: usize,
    height: usize,
    body: &VecDeque<Position>,
    _pending_growth: u32,
    rng: &mut StdRng,
) -> Position {
    let total_cells = width * height;
    if body.len() >= total_cells {
        return *body.front().expect("snake body is never empty");
    }

    loop {
        let x = rng.gen_range(0..width) as i32;
        let y = rng.gen_range(0..height) as i32;
        let pos = Position::new(x, y);
        if !body.iter().any(|segment| *segment == pos) {
            return pos;
        }
    }
}
