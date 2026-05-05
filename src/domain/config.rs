#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub width: usize,
    pub height: usize,
    pub max_steps: u32,
    pub render: bool,
    pub seed: Option<u64>,
}

impl EnvConfig {
    pub fn new(
        width: usize,
        height: usize,
        max_steps: u32,
        render: bool,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if width < 5 || height < 5 {
            return Err("width and height must both be >= 5".to_string());
        }
        if max_steps == 0 {
            return Err("max_steps must be > 0".to_string());
        }

        Ok(Self {
            width,
            height,
            max_steps,
            render,
            seed,
        })
    }
}
