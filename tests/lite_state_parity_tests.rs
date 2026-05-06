use bevy::prelude::App;
use rand::{Rng, SeedableRng, rngs::StdRng};

use predator_snake_dqn::{
    app::{builder::build_app, schedules::SnakeStepSchedule},
    domain::{action::RelativeAction, config::EnvConfig, state::Position},
    ecs::resources::{FoodResource, ObservationBuffer, PendingAction, SnakeResource, StepMetrics},
    lite_state::{
        dynamics::SnakeDynamics,
        encoding::SnakeEncoding,
        types::{LiteStateConfig, SnakeStateLite},
    },
};

#[derive(Debug)]
struct EcsSnapshot {
    body: Vec<Position>,
    dir: predator_snake_dqn::domain::action::Direction,
    pending_growth: u32,
    food: Position,
    steps: u32,
    score: u32,
    terminated: bool,
    truncated: bool,
    reward: f32,
    observation: Vec<u8>,
}

#[test]
fn golden_parity_wall_collision_script() {
    let seed = 42;
    let width = 20;
    let height = 20;
    let max_steps = 1_000;

    let (mut app, mut lite) = make_pair(width, height, max_steps, seed);
    assert_pair_parity(&lite, &snapshot_ecs(&app));

    let scripted_actions = vec![RelativeAction::Straight; 64];
    for (idx, action) in scripted_actions.into_iter().enumerate() {
        if lite.is_terminal() {
            break;
        }
        let lite_outcome = lite.apply(action);
        step_ecs(&mut app, action);
        let ecs = snapshot_ecs(&app);

        assert_step_parity(&lite, &lite_outcome, &ecs, seed, idx);
    }
}

#[test]
fn golden_parity_truncation_script() {
    let seed = 7;
    let width = 20;
    let height = 20;
    let max_steps = 5;

    let (mut app, mut lite) = make_pair(width, height, max_steps, seed);
    assert_pair_parity(&lite, &snapshot_ecs(&app));

    for step_idx in 0..max_steps as usize {
        let action = RelativeAction::Straight;
        let lite_outcome = lite.apply(action);
        step_ecs(&mut app, action);
        let ecs = snapshot_ecs(&app);
        assert_step_parity(&lite, &lite_outcome, &ecs, seed, step_idx);
    }

    assert!(lite.truncated, "lite state should be truncated at max_steps");
    assert!(
        snapshot_ecs(&app).truncated,
        "ecs state should be truncated at max_steps"
    );
}

#[test]
fn golden_parity_random_fuzz_episodes() {
    let width = 20;
    let height = 20;
    let max_steps = 250;
    let episodes = 64;

    for episode_seed in 0..episodes {
        let (mut app, mut lite) =
            make_pair(width, height, max_steps, episode_seed as u64 + 1_000);
        assert_pair_parity(&lite, &snapshot_ecs(&app));

        let mut action_rng = StdRng::seed_from_u64(episode_seed as u64 + 99_999);
        let mut step_idx = 0usize;
        while !lite.is_terminal() {
            let action_idx = action_rng.gen_range(0..3) as u8;
            let action = RelativeAction::try_from(action_idx).expect("valid action");

            let lite_outcome = lite.apply(action);
            step_ecs(&mut app, action);
            let ecs = snapshot_ecs(&app);
            assert_step_parity(
                &lite,
                &lite_outcome,
                &ecs,
                episode_seed as u64 + 1_000,
                step_idx,
            );
            step_idx += 1;
        }

        let ecs_final = snapshot_ecs(&app);
        assert_eq!(
            lite.terminated || lite.truncated,
            ecs_final.terminated || ecs_final.truncated,
            "terminal mismatch in episode seed {episode_seed}"
        );
    }
}

fn make_pair(width: usize, height: usize, max_steps: u32, seed: u64) -> (App, SnakeStateLite) {
    let env_cfg =
        EnvConfig::new(width, height, max_steps, false, Some(seed)).expect("valid env config");
    let app = build_app(&env_cfg);

    let lite_cfg = LiteStateConfig {
        width,
        height,
        max_steps,
        seed: Some(seed),
    };
    let lite = SnakeStateLite::new(lite_cfg).expect("valid lite config");
    (app, lite)
}

fn step_ecs(app: &mut App, action: RelativeAction) {
    app.world_mut().resource_mut::<PendingAction>().0 = action;
    app.world_mut().run_schedule(SnakeStepSchedule);
}

fn snapshot_ecs(app: &App) -> EcsSnapshot {
    let world = app.world();
    let snake = &world.resource::<SnakeResource>().0;
    let food = world.resource::<FoodResource>().0;
    let metrics = *world.resource::<StepMetrics>();
    let observation = world
        .resource::<ObservationBuffer>()
        .grid
        .iter()
        .copied()
        .collect::<Vec<u8>>();

    EcsSnapshot {
        body: snake.body.iter().copied().collect(),
        dir: snake.direction,
        pending_growth: snake.pending_growth,
        food,
        steps: metrics.steps,
        score: metrics.score,
        terminated: metrics.terminated,
        truncated: metrics.truncated,
        reward: metrics.reward,
        observation,
    }
}

fn assert_step_parity(
    lite: &SnakeStateLite,
    lite_outcome: &predator_snake_dqn::lite_state::types::StepOutcome,
    ecs: &EcsSnapshot,
    seed: u64,
    step_idx: usize,
) {
    assert_pair_parity(lite, ecs);
    assert!(
        (lite_outcome.reward - ecs.reward).abs() < 1e-6,
        "reward mismatch at seed={seed}, step={step_idx}: lite={} ecs={}",
        lite_outcome.reward,
        ecs.reward
    );
    assert_eq!(
        lite_outcome.terminated, ecs.terminated,
        "terminated mismatch at seed={seed}, step={step_idx}"
    );
    assert_eq!(
        lite_outcome.truncated, ecs.truncated,
        "truncated mismatch at seed={seed}, step={step_idx}"
    );
}

fn assert_pair_parity(lite: &SnakeStateLite, ecs: &EcsSnapshot) {
    assert_eq!(
        lite.body.iter().copied().collect::<Vec<_>>(),
        ecs.body,
        "snake body mismatch"
    );
    assert_eq!(lite.dir, ecs.dir, "snake direction mismatch");
    assert_eq!(
        lite.pending_growth, ecs.pending_growth,
        "pending growth mismatch"
    );
    assert_eq!(lite.food, ecs.food, "food position mismatch");
    assert_eq!(lite.steps, ecs.steps, "step counter mismatch");
    assert_eq!(lite.score, ecs.score, "score mismatch");
    assert_eq!(lite.terminated, ecs.terminated, "terminated mismatch");
    assert_eq!(lite.truncated, ecs.truncated, "truncated mismatch");

    let mut encoded = vec![0u8; lite.encoded_len()];
    lite.encode_onehot_chw(&mut encoded);
    assert_eq!(encoded, ecs.observation, "observation tensor mismatch");
}
