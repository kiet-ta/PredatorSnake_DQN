#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeAction {
    Straight,
    TurnLeft,
    TurnRight,
}

impl TryFrom<u8> for RelativeAction {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Straight),
            1 => Ok(Self::TurnLeft),
            2 => Ok(Self::TurnRight),
            _ => Err(format!("invalid action {value}; expected 0, 1, or 2")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    pub fn apply_relative(self, action: RelativeAction) -> Self {
        match action {
            RelativeAction::Straight => self,
            RelativeAction::TurnLeft => self.turn_left(),
            RelativeAction::TurnRight => self.turn_right(),
        }
    }

    pub fn turn_left(self) -> Self {
        match self {
            Self::Up => Self::Left,
            Self::Left => Self::Down,
            Self::Down => Self::Right,
            Self::Right => Self::Up,
        }
    }

    pub fn turn_right(self) -> Self {
        match self {
            Self::Up => Self::Right,
            Self::Right => Self::Down,
            Self::Down => Self::Left,
            Self::Left => Self::Up,
        }
    }

    pub fn delta(self) -> (i32, i32) {
        match self {
            Self::Up => (0, -1),
            Self::Right => (1, 0),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
        }
    }
}
