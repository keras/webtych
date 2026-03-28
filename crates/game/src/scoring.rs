use std::collections::{HashMap, HashSet, VecDeque};

use rapier2d::prelude::*;

use crate::physics::PhysicsWorld;
use crate::trimino::ColorId;

/// A group of same-color cells that are in mutual contact (match ≥ 3).
#[derive(Debug, Clone)]
pub struct MatchGroup {
    pub color: ColorId,
    pub handles: Vec<RigidBodyHandle>,
}

/// Detects match-3+ groups from the physics narrow-phase contacts.
///
/// Algorithm:
/// 1. Iterate all active contact pairs in the narrow phase.
/// 2. Resolve collider → body → color for each side.
/// 3. Build an adjacency graph between same-color bodies.
/// 4. Flood-fill connected components.
/// 5. Return components with size ≥ 3.
pub fn detect_matches(physics: &PhysicsWorld) -> Vec<MatchGroup> {
    let narrow = physics.narrow_phase();
    let colliders = physics.colliders();
    let bodies = physics.bodies();

    // Adjacency: body_handle → set of adjacent same-color body handles.
    let mut adjacency: HashMap<RigidBodyHandle, HashSet<RigidBodyHandle>> = HashMap::new();
    // Track color per body handle.
    let mut body_colors: HashMap<RigidBodyHandle, ColorId> = HashMap::new();

    narrow.contact_pairs().for_each(|pair| {
        if !pair.has_any_active_contact {
            return;
        }

        let (Some(parent_a), Some(parent_b)) = (
            colliders
                .get(pair.collider1)
                .and_then(|c| c.parent()),
            colliders
                .get(pair.collider2)
                .and_then(|c| c.parent()),
        ) else {
            return;
        };

        // Skip wall bodies.
        if physics.is_wall(parent_a) || physics.is_wall(parent_b) {
            return;
        }

        let (Some(body_a), Some(body_b)) = (bodies.get(parent_a), bodies.get(parent_b)) else {
            return;
        };

        let color_a = crate::physics::color_from_user_data(body_a.user_data);
        let color_b = crate::physics::color_from_user_data(body_b.user_data);

        body_colors.insert(parent_a, color_a);
        body_colors.insert(parent_b, color_b);

        if color_a == color_b {
            adjacency.entry(parent_a).or_default().insert(parent_b);
            adjacency.entry(parent_b).or_default().insert(parent_a);
        }
    });

    // Flood-fill connected components.
    let mut visited: HashSet<RigidBodyHandle> = HashSet::new();
    let mut groups = Vec::new();

    for &start in adjacency.keys() {
        if visited.contains(&start) {
            continue;
        }

        let color = body_colors[&start];
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);
            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if component.len() >= 3 {
            groups.push(MatchGroup {
                color,
                handles: component,
            });
        }
    }

    groups
}

/// Scoring state: tracks score, level, and clears.
#[derive(Debug, Clone)]
pub struct ScoreState {
    pub score: u32,
    pub level: u32,
    pub total_clears: u32,
    pub chain: u32,
}

/// Number of clears required to advance one level.
const CLEARS_PER_LEVEL: u32 = 10;

/// Base points per cell cleared.
const BASE_POINTS_PER_CELL: u32 = 100;

impl Default for ScoreState {
    fn default() -> Self {
        Self {
            score: 0,
            level: 1,
            total_clears: 0,
            chain: 0,
        }
    }
}

impl ScoreState {
    /// Award points for a match. Returns true if a level-up occurred.
    pub fn award_match(&mut self, cells_cleared: u32) -> bool {
        let chain_multiplier = self.chain.max(1);
        let points = BASE_POINTS_PER_CELL * cells_cleared * chain_multiplier;
        self.score += points;
        self.total_clears += cells_cleared;

        let new_level = 1 + self.total_clears / CLEARS_PER_LEVEL;
        if new_level > self.level {
            self.level = new_level;
            return true;
        }
        false
    }

    /// Start a new chain (called at the beginning of a clear sequence).
    pub fn begin_chain(&mut self) {
        self.chain = 1;
    }

    /// Increment the chain multiplier (called on successive matches in the same sequence).
    pub fn continue_chain(&mut self) {
        self.chain += 1;
    }

    /// Reset the chain (called when no more matches are found).
    pub fn end_chain(&mut self) {
        self.chain = 0;
    }

    /// Drop speed multiplier — pieces fall faster at higher levels.
    /// Returns a multiplier applied to the base drop interval.
    pub fn drop_speed_multiplier(&self) -> f32 {
        // Each level reduces the interval by 10%, capping at 3× speed.
        (1.0 - (self.level.saturating_sub(1) as f32) * 0.1).max(0.33)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_state_defaults() {
        let s = ScoreState::default();
        assert_eq!(s.score, 0);
        assert_eq!(s.level, 1);
        assert_eq!(s.total_clears, 0);
        assert_eq!(s.chain, 0);
    }

    #[test]
    fn award_match_adds_points() {
        let mut s = ScoreState::default();
        s.begin_chain();
        s.award_match(3);
        assert_eq!(s.score, 300); // 100 * 3 * 1
        assert_eq!(s.total_clears, 3);
    }

    #[test]
    fn chain_multiplier_increases_score() {
        let mut s = ScoreState::default();
        s.begin_chain();
        s.award_match(3); // 300
        s.continue_chain();
        s.award_match(3); // 100 * 3 * 2 = 600
        assert_eq!(s.score, 900);
    }

    #[test]
    fn level_up_after_enough_clears() {
        let mut s = ScoreState::default();
        s.begin_chain();
        let leveled = s.award_match(10);
        assert!(leveled);
        assert_eq!(s.level, 2);
    }

    #[test]
    fn drop_speed_increases_with_level() {
        let mut s = ScoreState::default();
        let speed_1 = s.drop_speed_multiplier();
        s.level = 5;
        let speed_5 = s.drop_speed_multiplier();
        assert!(speed_5 < speed_1);
    }

    #[test]
    fn drop_speed_has_minimum() {
        let mut s = ScoreState::default();
        s.level = 100;
        assert!(s.drop_speed_multiplier() >= 0.33);
    }
}
