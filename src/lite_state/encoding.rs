use crate::domain::rules::{
    CHANNEL_BODY, CHANNEL_EMPTY, CHANNEL_FOOD, CHANNEL_HEAD, OBS_CHANNELS,
};

use super::types::SnakeStateLite;

pub trait SnakeEncoding {
    fn encode_onehot_chw(&self, out: &mut [u8]);
    fn encoded_len(&self) -> usize;
}

impl SnakeEncoding for SnakeStateLite {
    fn encode_onehot_chw(&self, out: &mut [u8]) {
        let expected = self.encoded_len();
        assert!(
            out.len() == expected,
            "output buffer length mismatch: expected {expected}, got {}",
            out.len()
        );

        out.fill(0);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = chw_index(CHANNEL_EMPTY, y, x, self.height, self.width);
                out[idx] = 1;
            }
        }

        let food_idx = chw_index(
            CHANNEL_FOOD,
            self.food.y as usize,
            self.food.x as usize,
            self.height,
            self.width,
        );
        let empty_food_idx = chw_index(
            CHANNEL_EMPTY,
            self.food.y as usize,
            self.food.x as usize,
            self.height,
            self.width,
        );
        out[food_idx] = 1;
        out[empty_food_idx] = 0;

        for (idx, segment) in self.body.iter().enumerate() {
            let channel = if idx == 0 { CHANNEL_HEAD } else { CHANNEL_BODY };
            let y = segment.y as usize;
            let x = segment.x as usize;
            let seg_idx = chw_index(channel, y, x, self.height, self.width);
            let empty_seg_idx = chw_index(CHANNEL_EMPTY, y, x, self.height, self.width);
            out[seg_idx] = 1;
            out[empty_seg_idx] = 0;
        }
    }

    fn encoded_len(&self) -> usize {
        OBS_CHANNELS * self.height * self.width
    }
}

fn chw_index(channel: usize, y: usize, x: usize, height: usize, width: usize) -> usize {
    (channel * height + y) * width + x
}
