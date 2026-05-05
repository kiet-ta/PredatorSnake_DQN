use ndarray::{Array3, Axis};

use bevy::prelude::{Res, ResMut};

use crate::{
    domain::{
        rules::{
            CHANNEL_BODY, CHANNEL_EMPTY, CHANNEL_FOOD, CHANNEL_HEAD, OBS_CHANNELS, REWARD_DEATH,
            REWARD_FOOD, REWARD_STEP, collides_with_body, is_out_of_bounds,
        },
        state::Position,
    },
    ecs::resources::{
        EnvConfigResource, FoodResource, ObservationBuffer, PendingAction, ProposedHead,
        RandomResource, SnakeResource, StepMetrics, sample_free_position,
    },
};

pub fn apply_action_system(mut snake: ResMut<'_, SnakeResource>, action: Res<'_, PendingAction>) {
    snake.0.direction = snake.0.direction.apply_relative(action.0);
}

pub fn move_system(snake: Res<'_, SnakeResource>, mut proposed: ResMut<'_, ProposedHead>) {
    let head = snake.0.head();
    let (dx, dy) = snake.0.direction.delta();
    proposed.0 = Position::new(head.x + dx, head.y + dy);
}

pub fn resolve_step_system(
    config: Res<'_, EnvConfigResource>,
    mut snake: ResMut<'_, SnakeResource>,
    mut food: ResMut<'_, FoodResource>,
    proposed: Res<'_, ProposedHead>,
    mut rng: ResMut<'_, RandomResource>,
    mut metrics: ResMut<'_, StepMetrics>,
) {
    metrics.reward = REWARD_STEP;
    metrics.terminated = false;
    metrics.truncated = false;

    let next_head = proposed.0;
    let collision =
        is_out_of_bounds(next_head, &config.0) || collides_with_body(next_head, &snake.0);
    if collision {
        metrics.reward += REWARD_DEATH;
        metrics.terminated = true;
        metrics.steps = metrics.steps.saturating_add(1);
        return;
    }

    snake.0.body.push_front(next_head);
    let ate_food = next_head == food.0;
    if ate_food {
        metrics.reward += REWARD_FOOD;
        metrics.score = metrics.score.saturating_add(1);
        snake.0.pending_growth = snake.0.pending_growth.saturating_add(1);
        food.0 = sample_free_position(&config.0, &snake.0, &mut rng.0);
    }

    if snake.0.pending_growth > 0 {
        snake.0.pending_growth -= 1;
    } else {
        snake.0.body.pop_back();
    }

    metrics.steps = metrics.steps.saturating_add(1);
    if metrics.steps >= config.0.max_steps {
        metrics.truncated = true;
    }
}

pub fn write_observation_system(
    config: Res<'_, EnvConfigResource>,
    snake: Res<'_, SnakeResource>,
    food: Res<'_, FoodResource>,
    mut buffer: ResMut<'_, ObservationBuffer>,
) {
    write_observation_grid(&mut buffer.grid, &config.0, &snake.0, food.0);
}

pub fn write_observation_grid(
    grid: &mut Array3<u8>,
    config: &crate::domain::config::EnvConfig,
    snake: &crate::domain::state::SnakeState,
    food: Position,
) {
    if grid.dim() != (OBS_CHANNELS, config.height, config.width) {
        *grid = Array3::zeros((OBS_CHANNELS, config.height, config.width));
    } else {
        grid.fill(0);
    }

    {
        let mut empty_channel = grid.index_axis_mut(Axis(0), CHANNEL_EMPTY);
        empty_channel.fill(1);
    }

    let food_y = food.y as usize;
    let food_x = food.x as usize;
    grid[(CHANNEL_FOOD, food_y, food_x)] = 1;
    grid[(CHANNEL_EMPTY, food_y, food_x)] = 0;

    for (idx, segment) in snake.body.iter().enumerate() {
        let channel = if idx == 0 { CHANNEL_HEAD } else { CHANNEL_BODY };
        let y = segment.y as usize;
        let x = segment.x as usize;

        grid[(channel, y, x)] = 1;
        grid[(CHANNEL_EMPTY, y, x)] = 0;
    }
}
