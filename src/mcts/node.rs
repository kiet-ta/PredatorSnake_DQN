use std::sync::atomic::{AtomicU32, Ordering};
use crate::domain::action::RelativeAction;

pub struct Node {
    pub visit_count: AtomicU32,
    pub value_sum: AtomicU32,
    pub virtual_loss_active: AtomicU32,
    pub prior_prob: AtomicU32,
    
    pub parent: AtomicU32,
    pub children: [AtomicU32; 3],
    
    pub action_taken: AtomicU32,
}

impl Node {
    pub fn new() -> Self {
        Self {
            visit_count: AtomicU32::new(0),
            value_sum: AtomicU32::new(0.0f32.to_bits()),
            virtual_loss_active: AtomicU32::new(0),
            prior_prob: AtomicU32::new(0.0f32.to_bits()),
            parent: AtomicU32::new(u32::MAX),
            children: [
                AtomicU32::new(u32::MAX),
                AtomicU32::new(u32::MAX),
                AtomicU32::new(u32::MAX),
            ],
            action_taken: AtomicU32::new(3),
        }
    }

    pub fn init(&self, parent: Option<u32>, prior_prob: f32, action_taken: Option<RelativeAction>) {
        self.visit_count.store(0, Ordering::Relaxed);
        self.value_sum.store(0.0f32.to_bits(), Ordering::Relaxed);
        self.virtual_loss_active.store(0, Ordering::Relaxed);
        self.prior_prob.store(prior_prob.to_bits(), Ordering::Relaxed);
        self.parent.store(parent.unwrap_or(u32::MAX), Ordering::Relaxed);
        for c in &self.children {
            c.store(u32::MAX, Ordering::Relaxed);
        }
        let action_val = match action_taken {
            Some(RelativeAction::Straight) => 0,
            Some(RelativeAction::TurnLeft) => 1,
            Some(RelativeAction::TurnRight) => 2,
            None => 3,
        };
        self.action_taken.store(action_val, Ordering::Relaxed);
    }

    pub fn prior_prob(&self) -> f32 {
        f32::from_bits(self.prior_prob.load(Ordering::Relaxed))
    }

    pub fn action_taken(&self) -> Option<RelativeAction> {
        match self.action_taken.load(Ordering::Relaxed) {
            0 => Some(RelativeAction::Straight),
            1 => Some(RelativeAction::TurnLeft),
            2 => Some(RelativeAction::TurnRight),
            _ => None,
        }
    }

    pub fn parent(&self) -> Option<u32> {
        let p = self.parent.load(Ordering::Relaxed);
        if p == u32::MAX { None } else { Some(p) }
    }

    pub fn q_value(&self, virtual_loss: f32) -> f32 {
        let visits = self.visit_count.load(Ordering::Relaxed);
        let active_loss = self.virtual_loss_active.load(Ordering::Relaxed);
        let total_visits = visits + active_loss;

        if total_visits == 0 {
            0.0
        } else {
            let val = f32::from_bits(self.value_sum.load(Ordering::Relaxed));
            let adjusted_val = val - (active_loss as f32 * virtual_loss);
            adjusted_val / total_visits as f32
        }
    }

    pub fn add_visit(&self) {
        self.visit_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_value(&self, val: f32) {
        let mut current_bits = self.value_sum.load(Ordering::Relaxed);
        loop {
            let current_val = f32::from_bits(current_bits);
            let new_val = current_val + val;
            let new_bits = new_val.to_bits();
            
            match self.value_sum.compare_exchange_weak(
                current_bits,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(b) => current_bits = b,
            }
        }
    }

    pub fn is_expanded(&self) -> bool {
        self.children[0].load(Ordering::Acquire) != u32::MAX
    }
}
