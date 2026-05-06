#![allow(unsafe_op_in_unsafe_fn)]

pub mod app;
mod bridge;
pub mod domain;
pub mod ecs;
mod ffi;
pub mod lite_state;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ffi::env_api::PySnakeCore>()?;
    Ok(())
}
pub mod mcts;
