pub fn calculate_puct(
    parent_visits: u32,
    child_visits: u32,
    prior: f32,
    q_value: f32,
    c_puct: f32,
) -> f32 {
    let exploration_term = c_puct * prior * ((parent_visits as f32).sqrt() / (1.0 + child_visits as f32));
    q_value + exploration_term
}
