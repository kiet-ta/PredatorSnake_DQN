use bevy::{
    core::TaskPoolPlugin,
    prelude::{App, DefaultPlugins, MinimalPlugins, PluginGroup, TaskPoolOptions, Window},
    window::{PresentMode, WindowPlugin},
};

use crate::{app::schedules, domain::config::EnvConfig, ecs::resources};

pub fn build_app(config: &EnvConfig) -> App {
    let mut app = App::new();

    if config.render {
        let window_width = (config.width as f32 * 24.0).max(320.0);
        let window_height = (config.height as f32 * 24.0).max(320.0);
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Snake DQN Play".to_string(),
                resolution: (window_width, window_height).into(),
                resizable: false,
                present_mode: PresentMode::AutoVsync,
                ..Default::default()
            }),
            ..Default::default()
        }));
    } else {
        app.add_plugins(MinimalPlugins.set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(1),
        }));
    }

    schedules::configure_schedules(&mut app, config.render);
    resources::initialize_world(app.world_mut(), config.clone());
    app.finish();
    app.cleanup();

    app
}
