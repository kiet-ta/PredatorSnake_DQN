use numpy::{PyArray3, PyArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use crate::{
    domain::rules::OBS_CHANNELS,
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

/// Holds the persistent Coordinator thread and its job channel.
/// Dropping this struct disconnects the channel, causing the Coordinator
/// to exit cleanly — acting as an implicit poison pill.
struct CoordinatorState {
    /// Wrapped in Option so Drop can take it before joining the thread
    job_tx: Option<crossbeam_channel::Sender<LeafEvalJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CoordinatorState {
    fn new(
        job_tx: crossbeam_channel::Sender<LeafEvalJob>,
        handle: std::thread::JoinHandle<()>,
    ) -> Self {
        Self {
            job_tx: Some(job_tx),
            handle: Some(handle),
        }
    }

    fn job_tx(&self) -> &crossbeam_channel::Sender<LeafEvalJob> {
        self.job_tx.as_ref().expect("CoordinatorState used after shutdown")
    }
}

impl Drop for CoordinatorState {
    fn drop(&mut self) {
        // Step 1: Drop sender → channel disconnects → Coordinator sees RecvDisconnected and exits
        self.job_tx.take();
        // Step 2: Join the coordinator thread for clean shutdown
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[pyclass(unsendable)]
pub struct PyAlphaZeroEngine {
    state: SnakeStateLite,
    mcts_config: MCTSConfig,
    coordinator_config: CoordinatorConfig,
    model: Option<PyObject>,
    /// Persistent coordinator — lives across multiple mcts_search() calls
    coordinator_state: Option<CoordinatorState>,
}

impl Drop for PyAlphaZeroEngine {
    fn drop(&mut self) {
        // Take the coordinator state — its Drop impl handles clean shutdown
        self.coordinator_state.take();
    }
}

#[pymethods]
impl PyAlphaZeroEngine {
    #[new]
    #[pyo3(signature = (width=20, height=20, max_steps=1_000, starvation_limit=None, num_simulations=800, num_threads=4, c_puct=1.0, discount_factor=0.99, virtual_loss=1.0, batch_size=32, max_batch_wait_us=5000, seed=None))]
    pub fn new(
        width: usize,
        height: usize,
        max_steps: u32,
        starvation_limit: Option<u32>,
        num_simulations: usize,
        num_threads: usize,
        c_puct: f32,
        discount_factor: f32,
        virtual_loss: f32,
        batch_size: usize,
        max_batch_wait_us: u64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Default starvation heuristic: W * H * 2
        let effective_starvation = starvation_limit
            .unwrap_or((width * height * 2) as u32);

        let state_config = LiteStateConfig {
            width,
            height,
            max_steps,
            starvation_limit: effective_starvation,
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
            coordinator_state: None,
        })
    }

    pub fn set_model(&mut self, model: PyObject) {
        // Drop existing coordinator — its Drop impl handles clean shutdown
        self.coordinator_state.take();
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

        // Lazy-init: start coordinator on first search (or after set_model restart)
        if slf.coordinator_state.is_none() {
            slf.start_coordinator(py)?;
        }

        let job_tx = slf.coordinator_state.as_ref().unwrap().job_tx().clone();
        let mcts_config = slf.mcts_config;
        let root_state = slf.state.clone();

        // Release GIL so the Coordinator thread can acquire it for NN inference
        let result = py.allow_threads(|| {
            let evaluator = BridgedEvaluator::new(job_tx);
            let engine = MCTSEngine::new(mcts_config, evaluator);
            engine.search(&root_state)
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

        let shape = [OBS_CHANNELS, self.state.height, self.state.width];
        numpy::PyArray::from_slice_bound(py, &buffer).reshape(shape).unwrap()
    }
}

impl PyAlphaZeroEngine {
    /// Spawn the persistent Coordinator thread with the current model.
    fn start_coordinator(&mut self, py: Python<'_>) -> PyResult<()> {
        let model = self.model.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model not set. Call set_model() first.")
        })?.clone_ref(py);

        let backend = PyO3Backend {
            model,
            width: self.state.width,
            height: self.state.height,
        };

        let (job_tx, job_rx) = crossbeam_channel::unbounded::<LeafEvalJob>();
        let config = self.coordinator_config;

        let handle = std::thread::spawn(move || {
            let mut coordinator = Coordinator::new(job_rx, config, backend);
            coordinator.run();
        });

        self.coordinator_state = Some(CoordinatorState::new(job_tx, handle));

        Ok(())
    }

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
