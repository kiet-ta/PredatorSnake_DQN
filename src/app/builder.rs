use bevy::{
    core::TaskPoolPlugin,
    prelude::{App, DefaultPlugins, MinimalPlugins, PluginGroup, TaskPoolOptions},
};

use crate::{app::schedules, domain::config::EnvConfig, ecs::resources};

pub fn build_app(config: &EnvConfig) -> App {
    let mut app = App::new();

    if config.render {
        app.add_plugins(DefaultPlugins);
    } else {
        app.add_plugins(MinimalPlugins.set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(1),
        }));
    }

    schedules::configure_schedules(&mut app);
    resources::initialize_world(app.world_mut(), config.clone());
    app.finish();
    app.cleanup();

    app
}
