use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;

use super::messages::{EvalReply, LeafEvalJob};

/// Configuration for the batching Coordinator.
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorConfig {
    pub batch_size: usize,
    pub max_batch_wait_us: u64,
    /// Size of a single state observation in bytes: `OBS_CHANNELS * H * W`.
    pub obs_size: usize,
}

/// Trait abstracting the neural network backend call.
///
/// In Phase D this is a mock; Phase E replaces it with a real PyO3 call.
pub trait InferenceBackend: Send {
    /// Receives a contiguous batch buffer of shape `[B, 4, H, W]` (as flat `u8`)
    /// and returns `(values, policies)` for each item in the batch.
    fn infer(&mut self, batch_obs: &[u8], batch_size: usize) -> (Vec<f32>, Vec<[f32; 3]>);
}

/// A mock backend that returns uniform policy and zero value.
pub struct StubBackend;

impl InferenceBackend for StubBackend {
    fn infer(&mut self, _batch_obs: &[u8], batch_size: usize) -> (Vec<f32>, Vec<[f32; 3]>) {
        let values = vec![0.0; batch_size];
        let policies = vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; batch_size];
        (values, policies)
    }
}

/// The Coordinator runs on a dedicated background thread. It collects
/// `LeafEvalJob`s from MCTS workers, dynamically batches them by count or
/// timeout, dispatches a single backend inference call, and fans results
/// back to each worker via their one-shot reply channels.
pub struct Coordinator<B: InferenceBackend> {
    job_rx: Receiver<LeafEvalJob>,
    config: CoordinatorConfig,
    backend: B,
}

impl<B: InferenceBackend> Coordinator<B> {
    pub fn new(job_rx: Receiver<LeafEvalJob>, config: CoordinatorConfig, backend: B) -> Self {
        Self {
            job_rx,
            config,
            backend,
        }
    }

    /// Main coordinator loop. Runs until the job channel is disconnected
    /// (i.e. all senders are dropped), which signals the end of the search.
    pub fn run(&mut self) {
        let mut pending_jobs: Vec<LeafEvalJob> = Vec::with_capacity(self.config.batch_size);
        let mut batch_obs: Vec<u8> =
            Vec::with_capacity(self.config.batch_size * self.config.obs_size);

        loop {
            pending_jobs.clear();
            batch_obs.clear();

            // Block until the first job arrives (or channel disconnects)
            match self.job_rx.recv() {
                Ok(job) => pending_jobs.push(job),
                Err(_) => return, // Channel disconnected, all workers are done
            }

            let deadline = Instant::now() + Duration::from_micros(self.config.max_batch_wait_us);

            // Collect more jobs until batch is full or deadline expires
            while pending_jobs.len() < self.config.batch_size {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match self.job_rx.recv_timeout(remaining) {
                    Ok(job) => pending_jobs.push(job),
                    Err(_) => break, // Timeout or disconnected
                }
            }

            // Concatenate all observations into a contiguous batch buffer
            for job in &pending_jobs {
                batch_obs.extend_from_slice(&job.state_observation);
            }

            // Dispatch batched inference
            let current_batch_size = pending_jobs.len();
            let (values, policies) = self.backend.infer(&batch_obs, current_batch_size);

            // Fan results back to each waiting worker
            for (i, job) in pending_jobs.drain(..).enumerate() {
                let reply = EvalReply {
                    value: values[i],
                    policy: policies[i],
                };
                // If the worker dropped its receiver, we silently ignore the error
                let _ = job.reply_tx.send(reply);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::bounded;

    /// Verifies that 4 worker threads can concurrently submit 100 total
    /// evaluation jobs and the Coordinator dynamically batches and replies
    /// to every single one.
    #[test]
    fn test_coordinator_multi_threaded_batching() {
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<LeafEvalJob>();

        let coordinator_config = CoordinatorConfig {
            batch_size: 8,
            max_batch_wait_us: 5000,
            obs_size: 4 * 10 * 10,
        };

        // Spawn coordinator on a dedicated background thread
        let coordinator_handle = std::thread::spawn(move || {
            let mut coordinator = Coordinator::new(job_rx, coordinator_config, StubBackend);
            coordinator.run();
        });

        let num_threads = 4;
        let jobs_per_thread = 25;

        // Use std::thread::scope so worker threads can share `job_tx`
        // and we can collect all reply receivers safely.
        let all_replies: Vec<EvalReply> = std::thread::scope(|s| {
            let mut worker_handles = Vec::with_capacity(num_threads);

            for _t in 0..num_threads {
                let tx = job_tx.clone();
                let handle = s.spawn(move || {
                    let mut replies = Vec::with_capacity(jobs_per_thread);
                    for j in 0..jobs_per_thread {
                        let (reply_tx, reply_rx) = bounded::<EvalReply>(1);
                        let fake_obs = vec![0u8; 4 * 10 * 10];
                        let job = LeafEvalJob {
                            node_id: j as u32,
                            state_observation: fake_obs,
                            reply_tx,
                        };
                        tx.send(job).unwrap();
                        // Block until coordinator replies
                        let reply = reply_rx.recv().unwrap();
                        replies.push(reply);
                    }
                    replies
                });
                worker_handles.push(handle);
            }

            // Drop our copy of the sender so coordinator can observe
            // disconnect after all scoped workers finish.
            drop(job_tx);

            // Collect all replies from all workers
            let mut all = Vec::with_capacity(num_threads * jobs_per_thread);
            for handle in worker_handles {
                all.extend(handle.join().unwrap());
            }
            all
        });

        // Coordinator should exit cleanly after all senders are dropped
        coordinator_handle.join().unwrap();

        // Validate every reply
        assert_eq!(all_replies.len(), 100);
        for reply in &all_replies {
            assert_eq!(reply.value, 0.0, "Stub backend should return 0.0 value");
            let policy_sum: f32 = reply.policy.iter().sum();
            assert!(
                (policy_sum - 1.0).abs() < 1e-4,
                "Policy should sum to 1.0"
            );
        }
    }
}
