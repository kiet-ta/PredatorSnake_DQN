use bevy::prelude::World;
use ndarray::Array3;
use numpy::PyArray3;
use pyo3::prelude::*;

use crate::{
    domain::rules::OBS_CHANNELS,
    ecs::resources::{EnvConfigResource, ObservationBuffer},
};

pub fn take_observation_array<'py>(py: Python<'py>, world: &mut World) -> Bound<'py, PyArray3<u8>> {
    let (channels, height, width) = {
        let config = &world.resource::<EnvConfigResource>().0;
        (OBS_CHANNELS, config.height, config.width)
    };

    let array = {
        let mut observation = world.resource_mut::<ObservationBuffer>();
        std::mem::replace(
            &mut observation.grid,
            Array3::zeros((channels, height, width)),
        )
    };

    PyArray3::from_owned_array_bound(py, array)
}
