use ndarray::{Array3, Axis};
use std::collections::VecDeque;

use bevy::prelude::{Res, ResMut};

#[cfg(feature = "gui")]
use bevy::prelude::{
    Camera2dBundle, Color, Commands, Component, Query, Sprite, SpriteBundle,
    Transform, Vec2,
};

use crate::{
    domain::{
        rules::{
            CHANNEL_BODY, CHANNEL_EMPTY, CHANNEL_FOOD, CHANNEL_HEAD, CHANNEL_REACHABLE,
            OBS_CHANNELS, REWARD_DEATH, REWARD_FOOD, REWARD_STEP, collides_with_body,
            is_out_of_bounds,
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

    // Channel 4: Reachable area via BFS flood-fill from head
    write_reachable_channel(grid, config.width, config.height, snake);
}

/// BFS flood-fill from snake head, writing 1s into CHANNEL_REACHABLE.
/// Mirrors the lite_state encoding logic exactly for parity.
fn write_reachable_channel(
    grid: &mut Array3<u8>,
    width: usize,
    height: usize,
    snake: &crate::domain::state::SnakeState,
) {
    if snake.body.is_empty() {
        return;
    }

    let head = snake.head();

    // Build occupancy grid: true = blocked (body segment)
    let mut blocked = vec![false; width * height];
    for seg in &snake.body {
        if seg.x >= 0 && seg.y >= 0 && (seg.x as usize) < width && (seg.y as usize) < height {
            blocked[seg.y as usize * width + seg.x as usize] = true;
        }
    }

    let mut visited = vec![false; width * height];
    let mut queue = VecDeque::new();

    let head_flat = head.y as usize * width + head.x as usize;
    visited[head_flat] = true;
    queue.push_back((head.x as usize, head.y as usize));
    grid[(CHANNEL_REACHABLE, head.y as usize, head.x as usize)] = 1;

    while let Some((cx, cy)) = queue.pop_front() {
        for (dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;

            if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                continue;
            }

            let nux = nx as usize;
            let nuy = ny as usize;
            let flat = nuy * width + nux;

            if !visited[flat] && !blocked[flat] {
                visited[flat] = true;
                queue.push_back((nux, nuy));
                grid[(CHANNEL_REACHABLE, nuy, nux)] = 1;
            }
        }
    }
}

#[cfg(feature = "gui")]
#[derive(Component)]
pub struct RenderCell {
    pub x: usize,
    pub y: usize,
}

#[cfg(feature = "gui")]
pub fn setup_render_scene(mut commands: Commands<'_, '_>, config: Res<'_, EnvConfigResource>) {
    commands.spawn(Camera2dBundle::default());

    let width = config.0.width;
    let height = config.0.height;
    let tile_size = 24.0f32;
    let board_width = width as f32 * tile_size;
    let board_height = height as f32 * tile_size;

    for y in 0..height {
        for x in 0..width {
            let world_x = x as f32 * tile_size - board_width / 2.0 + tile_size / 2.0;
            let world_y = board_height / 2.0 - y as f32 * tile_size - tile_size / 2.0;
            commands.spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: color_for_channels(true, false, false, false),
                        custom_size: Some(Vec2::splat(tile_size - 1.0)),
                        ..Default::default()
                    },
                    transform: Transform::from_xyz(world_x, world_y, 0.0),
                    ..Default::default()
                },
                RenderCell { x, y },
            ));
        }
    }
}

#[cfg(feature = "gui")]
pub fn sync_render_scene(
    observation: Res<'_, ObservationBuffer>,
    mut cells: Query<'_, '_, (&RenderCell, &mut Sprite)>,
) {
    let (channels, height, width) = observation.grid.dim();
    if channels != OBS_CHANNELS {
        return;
    }

    for (cell, mut sprite) in &mut cells {
        if cell.x >= width || cell.y >= height {
            continue;
        }

        let is_empty = observation.grid[(CHANNEL_EMPTY, cell.y, cell.x)] == 1;
        let is_head = observation.grid[(CHANNEL_HEAD, cell.y, cell.x)] == 1;
        let is_body = observation.grid[(CHANNEL_BODY, cell.y, cell.x)] == 1;
        let is_food = observation.grid[(CHANNEL_FOOD, cell.y, cell.x)] == 1;
        sprite.color = color_for_channels(is_empty, is_head, is_body, is_food);
    }
}

#[cfg(feature = "gui")]
fn color_for_channels(is_empty: bool, is_head: bool, is_body: bool, is_food: bool) -> Color {
    if is_head {
        Color::srgb(0.2, 0.9, 0.2)
    } else if is_body {
        Color::srgb(0.1, 0.45, 0.1)
    } else if is_food {
        Color::srgb(0.95, 0.15, 0.15)
    } else if is_empty {
        Color::srgb(0.08, 0.08, 0.08)
    } else {
        Color::srgb(0.2, 0.2, 0.2)
    }
}
