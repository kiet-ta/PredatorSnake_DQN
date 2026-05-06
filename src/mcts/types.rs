#[derive(Debug, Clone, Copy)]
pub struct MCTSConfig {
    pub num_simulations: usize,
    pub c_puct: f32,
    pub discount_factor: f32,
    pub num_threads: usize,
    pub virtual_loss: f32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            c_puct: 1.0,
            discount_factor: 0.99,
            num_threads: 4,
            virtual_loss: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MCTSResult {
    pub policy: [f32; 3],
    pub root_value: f32,
}
