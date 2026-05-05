use bevy::prelude::App;
use numpy::PyArray3;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use crate::{
    app::builder::build_app,
    bridge::numpy_view::take_observation_array,
    domain::{action::RelativeAction, config::EnvConfig},
    ecs::resources::{PendingAction, StepMetrics, reset_episode},
};

#[pyclass(unsendable)]
pub struct PySnakeCore {
    app: App,
}

#[pymethods]
impl PySnakeCore {
    #[new]
    #[pyo3(signature = (width=20, height=20, max_steps=1_000, render=false, seed=None))]
    pub fn new(
        width: usize,
        height: usize,
        max_steps: u32,
        render: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let config = EnvConfig::new(width, height, max_steps, render, seed)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            app: build_app(&config),
        })
    }

    pub fn reset<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyDict>)> {
        reset_episode(slf.app.world_mut());
        let observation = take_observation_array(py, slf.app.world_mut());
        let info = Self::build_info_dict(py, slf.app.world().resource::<StepMetrics>())?;
        Ok((observation, info))
    }

    pub fn step<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        action: u8,
    ) -> PyResult<(
        Bound<'py, PyArray3<u8>>,
        f32,
        bool,
        bool,
        Bound<'py, PyDict>,
    )> {
        {
            let metrics = slf.app.world().resource::<StepMetrics>();
            if metrics.terminated || metrics.truncated {
                return Err(PyRuntimeError::new_err(
                    "episode is complete; call reset() before step()",
                ));
            }
        }

        let parsed_action = RelativeAction::try_from(action).map_err(PyValueError::new_err)?;
        slf.app.world_mut().resource_mut::<PendingAction>().0 = parsed_action;
        slf.app.update();

        let (reward, terminated, truncated) = {
            let metrics = slf.app.world().resource::<StepMetrics>();
            (metrics.reward, metrics.terminated, metrics.truncated)
        };
        let info = Self::build_info_dict(py, slf.app.world().resource::<StepMetrics>())?;
        let observation = take_observation_array(py, slf.app.world_mut());

        Ok((observation, reward, terminated, truncated, info))
    }
}

impl PySnakeCore {
    fn build_info_dict<'py>(
        py: Python<'py>,
        metrics: &StepMetrics,
    ) -> PyResult<Bound<'py, PyDict>> {
        let info = PyDict::new_bound(py);
        info.set_item("score", metrics.score)?;
        info.set_item("steps", metrics.steps)?;
        Ok(info)
    }
}
