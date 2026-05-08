use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use super::coordinator::InferenceBackend;
use crate::domain::rules::OBS_CHANNELS;

/// A PyO3-based backend that invokes a Python callable for evaluation.
///
/// Converts a contiguous Rust byte slice representing a batch of observations
/// into a zero-copy PyArray (which Python should convert/cast to float32).
/// It expects the Python callable to return a tuple of `(logits, values)`.
pub struct PyO3Backend {
    pub model: PyObject,
    pub width: usize,
    pub height: usize,
}

impl InferenceBackend for PyO3Backend {
    fn infer(&mut self, batch_obs: &[u8], batch_size: usize) -> (Vec<f32>, Vec<[f32; 3]>) {
        Python::with_gil(|py| {
            // Reconstruct the shape: [B, OBS_CHANNELS, H, W]
            let shape = [batch_size, OBS_CHANNELS, self.height, self.width];
            
            // Create a numpy array directly from the slice. 
            // `IntoPyArray::into_pyarray` does a copy if we use `into_pyarray_bound` or `to_pyarray`.
            // Wait, since batch_obs is a slice and we don't own it (it's borrowed from the coordinator),
            // we must copy the data into a new PyArray, or use `PyArray::from_slice_bound` (which copies).
            // This is safer and avoids lifetimes tying the PyArray to the coordinator's loop.
            let py_array = numpy::PyArray::from_slice_bound(py, batch_obs).reshape(shape).unwrap();

            // Invoke the Python model
            let args = (py_array,);
            let result_tuple = self.model.call1(py, args)
                .expect("Failed to call Python inference model");

            // Extract the (logits, values) tuple
            let tuple = result_tuple.bind(py).downcast::<pyo3::types::PyTuple>()
                .expect("Model must return a tuple (logits, values)");

            let py_logits_any = tuple.get_item(0).unwrap();
            let py_logits = py_logits_any.downcast::<PyArray2<f32>>()
                .expect("Logits must be a 2D numpy array of float32");
                
            let py_values_any = tuple.get_item(1).unwrap();
            let py_values = py_values_any.downcast::<PyArray1<f32>>()
                .expect("Values must be a 1D numpy array of float32");

            // Convert to Rust native types
            // To ensure safe access without blocking Python GC, we use `as_slice` or `readonly()`.
            let logits_readonly = py_logits.readonly();
            let values_readonly = py_values.readonly();
            let logits_slice = logits_readonly.as_slice().unwrap();
            let values_slice = values_readonly.as_slice().unwrap();

            let mut values_out = Vec::with_capacity(batch_size);
            let mut policies_out = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                values_out.push(values_slice[i]);
                
                let logit_start = i * 3;
                let logit_slice = &logits_slice[logit_start..logit_start + 3];
                let logits_array = [logit_slice[0], logit_slice[1], logit_slice[2]];
                
                let probs = softmax_row(&logits_array);
                policies_out.push(probs);
            }

            (values_out, policies_out)
        })
    }
}

/// Numerically stable softmax: computes exp(x - max(x)) / sum(exp(x - max(x)))
#[inline]
fn softmax_row(logits: &[f32; 3]) -> [f32; 3] {
    let max_val = logits[0].max(logits[1]).max(logits[2]);
    let mut exp_vals = [0.0; 3];
    let mut sum_exp = 0.0;

    for i in 0..3 {
        exp_vals[i] = (logits[i] - max_val).exp();
        sum_exp += exp_vals[i];
    }

    if sum_exp > 0.0 {
        [
            exp_vals[0] / sum_exp,
            exp_vals[1] / sum_exp,
            exp_vals[2] / sum_exp,
        ]
    } else {
        // Fallback for uniform distribution in case of extreme underflow/NaN
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}
