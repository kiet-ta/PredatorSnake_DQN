use crate::domain::{
    config::EnvConfig,
    state::{Position, SnakeState},
};

pub const OBS_CHANNELS: usize = 5;
pub const CHANNEL_EMPTY: usize = 0;
pub const CHANNEL_HEAD: usize = 1;
pub const CHANNEL_BODY: usize = 2;
pub const CHANNEL_FOOD: usize = 3;
pub const CHANNEL_REACHABLE: usize = 4;

pub const REWARD_FOOD: f32 = 10.0;
pub const REWARD_DEATH: f32 = -10.0;
pub const REWARD_STEP: f32 = -0.01;

pub fn is_out_of_bounds(position: Position, config: &EnvConfig) -> bool {
    position.x < 0
        || position.y < 0
        || position.x >= config.width as i32
        || position.y >= config.height as i32
}

pub fn collides_with_body(next_head: Position, snake: &SnakeState) -> bool {
    let body_to_check = if snake.pending_growth == 0 {
        snake.body.len().saturating_sub(1)
    } else {
        snake.body.len()
    };

    snake
        .body
        .iter()
        .take(body_to_check)
        .any(|segment| *segment == next_head)
}
