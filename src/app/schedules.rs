use bevy::prelude::{App, IntoSystemConfigs, Update};

use crate::ecs::systems::{
    apply_action_system, move_system, resolve_step_system, write_observation_system,
};

pub fn configure_schedules(app: &mut App) {
    app.add_systems(
        Update,
        (
            apply_action_system,
            move_system,
            resolve_step_system,
            write_observation_system,
        )
            .chain(),
    );
}
