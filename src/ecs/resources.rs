use std::collections::VecDeque;

use bevy::prelude::{Resource, World};
use ndarray::Array3;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::domain::{
    action::{Direction, RelativeAction},
    config::EnvConfig,
    rules::{CHANNEL_EMPTY, OBS_CHANNELS},
    state::{Position, SnakeState},
};
use crate::ecs::systems::write_observation_grid;

#[derive(Resource, Clone)]
pub struct EnvConfigResource(pub EnvConfig);

#[derive(Resource, Debug, Clone)]
pub struct SnakeResource(pub SnakeState);

#[derive(Resource, Debug, Clone, Copy)]
pub struct FoodResource(pub Position);

#[derive(Resource, Debug, Clone, Copy)]
pub struct PendingAction(pub RelativeAction);

#[derive(Resource, Debug, Clone, Copy)]
pub struct ProposedHead(pub Position);

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct StepMetrics {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub steps: u32,
    pub score: u32,
}

#[derive(Resource, Debug)]
pub struct ObservationBuffer {
    pub grid: Array3<u8>,
}

#[derive(Resource)]
pub struct RandomResource(pub StdRng);

pub fn initialize_world(world: &mut World, config: EnvConfig) {
    let rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut grid = Array3::zeros((OBS_CHANNELS, config.height, config.width));
    for y in 0..config.height {
        for x in 0..config.width {
            grid[(CHANNEL_EMPTY, y, x)] = 1;
        }
    }

    world.insert_resource(EnvConfigResource(config));
    world.insert_resource(RandomResource(rng));
    world.insert_resource(ObservationBuffer { grid });
    world.insert_resource(PendingAction(RelativeAction::Straight));
    world.insert_resource(ProposedHead(Position::new(0, 0)));
    world.insert_resource(StepMetrics::default());
    world.insert_resource(FoodResource(Position::new(0, 0)));
    world.insert_resource(SnakeResource(empty_snake()));

    reset_episode(world);
}

pub fn reset_episode(world: &mut World) {
    let config = world.resource::<EnvConfigResource>().0.clone();
    let snake = initial_snake(&config);
    let head = snake.head();

    world.resource_mut::<SnakeResource>().0 = snake;
    *world.resource_mut::<StepMetrics>() = StepMetrics::default();
    world.resource_mut::<PendingAction>().0 = RelativeAction::Straight;
    world.resource_mut::<ProposedHead>().0 = head;

    let snake_snapshot = world.resource::<SnakeResource>().0.clone();
    let food_position = {
        let mut rng = world.resource_mut::<RandomResource>();
        sample_free_position(&config, &snake_snapshot, &mut rng.0)
    };
    world.resource_mut::<FoodResource>().0 = food_position;

    {
        let snake = world.resource::<SnakeResource>().0.clone();
        let food = world.resource::<FoodResource>().0;
        let mut observation = world.resource_mut::<ObservationBuffer>();
        write_observation_grid(&mut observation.grid, &config, &snake, food);
    }
}

pub fn sample_free_position(config: &EnvConfig, snake: &SnakeState, rng: &mut StdRng) -> Position {
    let total_cells = config.width * config.height;
    if snake.body.len() >= total_cells {
        return snake.head();
    }

    loop {
        let x = rng.gen_range(0..config.width) as i32;
        let y = rng.gen_range(0..config.height) as i32;
        let position = Position::new(x, y);
        if !snake.contains(position) {
            return position;
        }
    }
}

fn empty_snake() -> SnakeState {
    SnakeState {
        body: VecDeque::new(),
        direction: Direction::Right,
        pending_growth: 0,
    }
}

fn initial_snake(config: &EnvConfig) -> SnakeState {
    let center_x = (config.width / 2) as i32;
    let center_y = (config.height / 2) as i32;

    let mut body = VecDeque::new();
    body.push_back(Position::new(center_x, center_y));
    body.push_back(Position::new(center_x - 1, center_y));
    body.push_back(Position::new(center_x - 2, center_y));

    SnakeState {
        body,
        direction: Direction::Right,
        pending_growth: 0,
    }
}
