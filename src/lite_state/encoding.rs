use std::collections::VecDeque;

use crate::domain::rules::{
    CHANNEL_BODY, CHANNEL_EMPTY, CHANNEL_FOOD, CHANNEL_HEAD, CHANNEL_REACHABLE,
    OBS_CHANNELS,
};

use super::types::SnakeStateLite;

pub trait SnakeEncoding {
    /// Encodes the full game state into a one-hot CHW tensor buffer.
    /// Shape: [OBS_CHANNELS, H, W] — channels: empty, head, body, food, reachable.
    fn encode_onehot_chw(&self, out: &mut [u8]);

    /// Returns the total byte length of the encoded observation.
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

        // Channel 0: Empty (all cells start as empty, then overwritten)
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = chw_index(CHANNEL_EMPTY, y, x, self.height, self.width);
                out[idx] = 1;
            }
        }

        // Channel 3: Food
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

        // Channel 1 (head) and Channel 2 (body)
        for (idx, segment) in self.body.iter().enumerate() {
            let channel = if idx == 0 { CHANNEL_HEAD } else { CHANNEL_BODY };
            let y = segment.y as usize;
            let x = segment.x as usize;
            let seg_idx = chw_index(channel, y, x, self.height, self.width);
            let empty_seg_idx = chw_index(CHANNEL_EMPTY, y, x, self.height, self.width);
            out[seg_idx] = 1;
            out[empty_seg_idx] = 0;
        }

        // Channel 4: Reachable area via BFS flood-fill from head
        self.write_reachable_channel(out);
    }

    fn encoded_len(&self) -> usize {
        OBS_CHANNELS * self.height * self.width
    }
}

impl SnakeStateLite {
    /// BFS flood-fill from head, writing 1s into CHANNEL_REACHABLE.
    /// Treats walls and body segments as obstacles. Food is passable.
    /// Complexity: O(W × H) worst-case, trivial for 20×20 boards.
    fn write_reachable_channel(&self, out: &mut [u8]) {
        let head = *self.body.front().expect("snake body is never empty");
        let w = self.width;
        let h = self.height;

        // Build occupancy grid: true = blocked (body segment)
        let mut blocked = vec![false; w * h];
        for seg in &self.body {
            if seg.x >= 0
                && seg.y >= 0
                && (seg.x as usize) < w
                && (seg.y as usize) < h
            {
                blocked[seg.y as usize * w + seg.x as usize] = true;
            }
        }

        let mut visited = vec![false; w * h];
        let mut queue = VecDeque::new();

        // Seed from head position (head is blocked but we start BFS from it)
        let head_flat = head.y as usize * w + head.x as usize;
        visited[head_flat] = true;
        queue.push_back((head.x as usize, head.y as usize));
        out[chw_index(CHANNEL_REACHABLE, head.y as usize, head.x as usize, h, w)] = 1;

        while let Some((cx, cy)) = queue.pop_front() {
            for (dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;

                if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                    continue;
                }

                let nux = nx as usize;
                let nuy = ny as usize;
                let flat = nuy * w + nux;

                if !visited[flat] && !blocked[flat] {
                    visited[flat] = true;
                    queue.push_back((nux, nuy));
                    out[chw_index(CHANNEL_REACHABLE, nuy, nux, h, w)] = 1;
                }
            }
        }
    }
}

fn chw_index(channel: usize, y: usize, x: usize, height: usize, width: usize) -> usize {
    (channel * height + y) * width + x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_state::types::LiteStateConfig;

    #[test]
    fn test_encoded_len_matches_5_channels() {
        let config = LiteStateConfig {
            width: 10,
            height: 10,
            max_steps: 100,
            starvation_limit: 200,
            seed: Some(42),
        };
        let state = SnakeStateLite::new(config).unwrap();
        assert_eq!(state.encoded_len(), 5 * 10 * 10);
    }

    #[test]
    fn test_flood_fill_open_board() {
        let config = LiteStateConfig {
            width: 10,
            height: 10,
            max_steps: 100,
            starvation_limit: 200,
            seed: Some(42),
        };
        let state = SnakeStateLite::new(config).unwrap();
        let mut buf = vec![0u8; state.encoded_len()];
        state.encode_onehot_chw(&mut buf);

        // Count reachable cells in channel 4
        let reachable_start = CHANNEL_REACHABLE * 10 * 10;
        let reachable_count: usize = buf[reachable_start..reachable_start + 100]
            .iter()
            .filter(|&&v| v == 1)
            .count();

        // Snake body has 3 segments → reachable should be 100 - 3 + 1 (head is reachable)
        // Head is in the reachable set, the other 2 body segments are not.
        // So reachable = total_cells - body_segments_excluding_head = 100 - 2 = 98
        // (head itself is marked reachable, food cell is passable)
        assert!(
            reachable_count >= 95,
            "On an open 10x10 board with 3-segment snake, reachable should be ~98, got {reachable_count}"
        );
    }

    #[test]
    fn test_reachable_channel_head_is_always_reachable() {
        let config = LiteStateConfig {
            width: 10,
            height: 10,
            max_steps: 100,
            starvation_limit: 200,
            seed: Some(42),
        };
        let state = SnakeStateLite::new(config).unwrap();
        let mut buf = vec![0u8; state.encoded_len()];
        state.encode_onehot_chw(&mut buf);

        let head = state.head();
        let head_reachable_idx = chw_index(
            CHANNEL_REACHABLE,
            head.y as usize,
            head.x as usize,
            10,
            10,
        );
        assert_eq!(buf[head_reachable_idx], 1, "Head must always be reachable");
    }
}
