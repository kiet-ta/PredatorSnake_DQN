use crate::lite_state::types::SnakeStateLite;

pub trait BatchEvaluator: Sync {
    /// Evaluates a state and returns (value, policy)
    /// policy is an array of probabilities for [Straight, TurnLeft, TurnRight]
    fn evaluate(&self, state: &SnakeStateLite) -> (f32, [f32; 3]);
}

pub struct StubEvaluator;

impl BatchEvaluator for StubEvaluator {
    fn evaluate(&self, _state: &SnakeStateLite) -> (f32, [f32; 3]) {
        (0.0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    }
}
