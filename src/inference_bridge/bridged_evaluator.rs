use crossbeam_channel::{bounded, Sender};

use crate::lite_state::encoding::SnakeEncoding;
use crate::lite_state::types::SnakeStateLite;
use crate::mcts::evaluator::BatchEvaluator;

use super::messages::{EvalReply, LeafEvalJob};

/// A `BatchEvaluator` implementation that bridges MCTS workers to the
/// Coordinator's batched inference pipeline.
///
/// When an MCTS worker calls `evaluate()`, this evaluator:
/// 1. Encodes the state into a flat observation buffer.
/// 2. Creates a one-shot reply channel.
/// 3. Sends a `LeafEvalJob` to the Coordinator's job queue.
/// 4. Blocks on the reply channel until the Coordinator dispatches the result.
pub struct BridgedEvaluator {
    job_tx: Sender<LeafEvalJob>,
}

impl BridgedEvaluator {
    pub fn new(job_tx: Sender<LeafEvalJob>) -> Self {
        Self { job_tx }
    }
}

impl BatchEvaluator for BridgedEvaluator {
    fn evaluate(&self, state: &SnakeStateLite) -> (f32, [f32; 3]) {
        // Encode the state into a flat CHW observation buffer
        let obs_len = state.encoded_len();
        let mut observation = vec![0u8; obs_len];
        state.encode_onehot_chw(&mut observation);

        // Create a one-shot reply channel (bounded to 1)
        let (reply_tx, reply_rx) = bounded::<EvalReply>(1);

        let job = LeafEvalJob {
            node_id: 0, // Node ID is informational; not used by the coordinator
            state_observation: observation,
            reply_tx,
        };

        // Submit the job to the coordinator queue
        self.job_tx
            .send(job)
            .expect("Coordinator channel disconnected unexpectedly");

        // Block until the coordinator replies with the batched inference result
        let reply = reply_rx
            .recv()
            .expect("Coordinator dropped reply channel unexpectedly");

        (reply.value, reply.policy)
    }
}
