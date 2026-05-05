use std::collections::VecDeque;

use crate::domain::action::Direction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone)]
pub struct SnakeState {
    pub body: VecDeque<Position>,
    pub direction: Direction,
    pub pending_growth: u32,
}

impl SnakeState {
    pub fn head(&self) -> Position {
        *self.body.front().expect("snake body is never empty")
    }

    pub fn contains(&self, pos: Position) -> bool {
        self.body.iter().any(|segment| *segment == pos)
    }
}
