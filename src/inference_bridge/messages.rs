use crossbeam_channel::Sender;

/// Result returned by the neural network for a single state evaluation.
#[derive(Debug, Clone)]
pub struct EvalReply {
    pub value: f32,
    pub policy: [f32; 3],
}

/// A leaf evaluation job submitted by an MCTS worker thread.
///
/// The worker encodes its state into a flat observation buffer and sends this
/// job to the Coordinator via the shared job queue. It then blocks on
/// `reply_tx`'s paired receiver until the Coordinator dispatches the batched
/// inference result back.
pub struct LeafEvalJob {
    pub node_id: u32,
    pub state_observation: Vec<u8>,
    pub reply_tx: Sender<EvalReply>,
}
