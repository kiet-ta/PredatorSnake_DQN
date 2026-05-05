#![allow(unsafe_op_in_unsafe_fn)]

mod app;
mod bridge;
mod domain;
mod ecs;
mod ffi;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ffi::env_api::PySnakeCore>()?;
    Ok(())
}
