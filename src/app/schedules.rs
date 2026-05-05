use bevy::{
    ecs::schedule::ScheduleLabel,
    prelude::{App, IntoSystemConfigs, Startup, Update},
};

use crate::ecs::systems::{
    apply_action_system, move_system, resolve_step_system, setup_render_scene, sync_render_scene,
    write_observation_system,
};

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnakeStepSchedule;

pub fn configure_schedules(app: &mut App, render_enabled: bool) {
    app.init_schedule(SnakeStepSchedule);
    app.add_systems(
        SnakeStepSchedule,
        (
            apply_action_system,
            move_system,
            resolve_step_system,
            write_observation_system,
        )
            .chain(),
    );

    if render_enabled {
        app.add_systems(Startup, setup_render_scene);
        app.add_systems(Update, sync_render_scene);
    }
}
