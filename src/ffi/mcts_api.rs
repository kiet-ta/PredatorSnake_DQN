use numpy::{PyArray3, PyArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use crate::{
    inference_bridge::{
        bridged_evaluator::BridgedEvaluator,
        coordinator::{Coordinator, CoordinatorConfig},
        messages::LeafEvalJob,
        py_evaluator::PyO3Backend,
    },
    lite_state::{
        dynamics::SnakeDynamics,
        encoding::SnakeEncoding,
        types::{LiteStateConfig, SnakeStateLite},
    },
    mcts::{
        engine::MCTSEngine,
        types::MCTSConfig,
    },
};

#[pyclass(unsendable)]
pub struct PyAlphaZeroEngine {
    state: SnakeStateLite,
    mcts_config: MCTSConfig,
    coordinator_config: CoordinatorConfig,
    model: Option<PyObject>,
}

#[pymethods]
impl PyAlphaZeroEngine {
    #[new]
    #[pyo3(signature = (width=20, height=20, max_steps=1_000, num_simulations=800, num_threads=4, c_puct=1.0, discount_factor=0.99, virtual_loss=1.0, batch_size=32, max_batch_wait_us=5000, seed=None))]
    pub fn new(
        width: usize,
        height: usize,
        max_steps: u32,
        num_simulations: usize,
        num_threads: usize,
        c_puct: f32,
        discount_factor: f32,
        virtual_loss: f32,
        batch_size: usize,
        max_batch_wait_us: u64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let state_config = LiteStateConfig {
            width,
            height,
            max_steps,
            seed,
        };

        let state = SnakeStateLite::new(state_config)
            .map_err(|e| PyValueError::new_err(e))?;

        let mcts_config = MCTSConfig {
            num_simulations,
            c_puct,
            discount_factor,
            num_threads,
            virtual_loss,
        };

        let obs_size = state.encoded_len();
        let coordinator_config = CoordinatorConfig {
            batch_size,
            max_batch_wait_us,
            obs_size,
        };

        Ok(Self {
            state,
            mcts_config,
            coordinator_config,
            model: None,
        })
    }

    pub fn set_model(&mut self, model: PyObject) {
        self.model = Some(model);
    }

    pub fn reset<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyDict>)> {
        slf.state.reset();
        
        let obs = slf.get_observation(py);
        let info = Self::build_info_dict(py, &slf.state)?;
        
        Ok((obs, info))
    }

    pub fn mcts_search<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if slf.state.is_terminal() {
            return Err(PyRuntimeError::new_err("Episode is terminal; call reset()"));
        }

        let model = slf.model.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model not set. Call set_model() first.")
        })?.clone_ref(py);

        let (job_tx, job_rx) = crossbeam_channel::unbounded::<LeafEvalJob>();

        let backend = PyO3Backend {
            model,
            width: slf.state.width,
            height: slf.state.height,
        };

        let mut coordinator = Coordinator::new(job_rx, slf.coordinator_config, backend);

        let mcts_config = slf.mcts_config;
        let root_state = slf.state.clone();

        // Critical Section: Release the GIL before waiting on threads.
        // If we don't do this, the Coordinator will deadlock trying to acquire the GIL.
        let result = py.allow_threads(|| {
            std::thread::scope(|s| {
                // Spawn Coordinator
                s.spawn(move || {
                    coordinator.run();
                });

                // Run MCTS search using BridgedEvaluator
                let evaluator = BridgedEvaluator::new(job_tx.clone());
                let engine = MCTSEngine::new(mcts_config, evaluator);
                
                let result = engine.search(&root_state);
                
                // Drop the job_tx to signal the Coordinator to exit
                drop(job_tx);
                
                result
            })
        });

        let result_dict = PyDict::new_bound(py);
        let policy_list: Vec<f32> = result.policy.to_vec();
        result_dict.set_item("policy", policy_list)?;
        result_dict.set_item("root_value", result.root_value)?;

        Ok(result_dict)
    }

    pub fn step<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        action: u8,
    ) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyDict>, bool)> {
        use std::convert::TryFrom;
        let action_enum = crate::domain::action::RelativeAction::try_from(action)
            .map_err(|_| PyValueError::new_err("Invalid action"))?;
        
        slf.state.apply(action_enum);

        let obs = slf.get_observation(py);
        let info_dict = Self::build_info_dict(py, &slf.state)?;
        let is_terminal = slf.state.is_terminal();

        Ok((obs, info_dict, is_terminal))
    }

    pub fn get_observation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        let obs_len = self.state.encoded_len();
        let mut buffer = vec![0u8; obs_len];
        self.state.encode_onehot_chw(&mut buffer);

        let shape = [4, self.state.height, self.state.width];
        numpy::PyArray::from_slice_bound(py, &buffer).reshape(shape).unwrap()
    }
}

impl PyAlphaZeroEngine {
    fn build_info_dict<'py>(
        py: Python<'py>,
        state: &SnakeStateLite,
    ) -> PyResult<Bound<'py, PyDict>> {
        let info = PyDict::new_bound(py);
        info.set_item("score", state.score)?;
        info.set_item("steps", state.steps)?;
        Ok(info)
    }
}
