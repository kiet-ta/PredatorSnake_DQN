use std::sync::atomic::{AtomicUsize, Ordering};
use super::node::Node;

pub struct TreeArena {
    nodes: Vec<Node>,
    alloc_index: AtomicUsize,
}

impl TreeArena {
    pub fn new(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            nodes.push(Node::new());
        }
        Self {
            nodes,
            alloc_index: AtomicUsize::new(0),
        }
    }

    pub fn alloc_node(&self) -> u32 {
        let idx = self.alloc_index.fetch_add(1, Ordering::Relaxed);
        if idx >= self.nodes.len() {
            panic!("TreeArena out of memory! Capacity was {}", self.nodes.len());
        }
        idx as u32
    }

    pub fn get(&self, index: u32) -> &Node {
        &self.nodes[index as usize]
    }

    pub fn clear(&self) {
        self.alloc_index.store(0, Ordering::Relaxed);
    }

    pub fn len(&self) -> usize {
        self.alloc_index.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.alloc_index.load(Ordering::Relaxed) == 0
    }
}
