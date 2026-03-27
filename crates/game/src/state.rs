use glam::Vec2;

use crate::board::Board;
use crate::events::GameEvent;
use crate::physics::{BlockInstance, PhysicsWorld, PieceId};
use crate::scoring::{detect_matches, ScoreState};
use crate::trimino::PieceBag;

/// Actions the player can take each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputAction {
    MoveLeft,
    MoveRight,
    RotateCW,
    RotateCCW,
    SoftDrop,
    HardDrop,
}

/// Phases of the game loop state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    /// About to spawn the next piece.
    Spawning,
    /// Active piece is falling — player has control.
    Falling,
    /// Piece has landed; waiting for physics to settle.
    Settling,
    /// Checking for match-3+ groups.
    Checking,
    /// Matched cells are being removed; waiting for remaining cells to settle.
    Clearing,
    /// Stack reached the top — game over.
    GameOver,
}

/// Tunable constants for the game feel.
const BASE_DROP_INTERVAL: f32 = 1.0; // seconds between auto-drops at level 1
const SETTLE_TIME: f32 = 0.4; // seconds to wait after landing before checking
const CLEARING_SETTLE_TIME: f32 = 0.6; // seconds to wait after clearing before re-check
const MOVE_IMPULSE: f32 = 5.0;
const ROTATION_IMPULSE: f32 = 3.0;
const SOFT_DROP_IMPULSE: f32 = -8.0;
const HARD_DROP_VELOCITY: f32 = -40.0;
const SETTLE_VELOCITY_THRESHOLD: f32 = 1.0;
const PHYSICS_DT: f32 = 1.0 / 60.0;
const IMPACT_VELOCITY_THRESHOLD: f32 = 5.0;

/// Top-level game state.
pub struct GameState {
    pub phase: GamePhase,
    pub physics: PhysicsWorld,
    pub board: Board,
    pub score: ScoreState,
    pub events: Vec<GameEvent>,

    active_piece: Option<PieceId>,
    bag: PieceBag,
    rng: rand::rngs::ThreadRng,
    drop_timer: f32,
    settle_timer: f32,
    accumulator: f32,
}

impl GameState {
    pub fn new() -> Self {
        let board = Board::default();
        let physics = PhysicsWorld::new(&board);
        let mut rng = rand::thread_rng();
        let bag = PieceBag::new(&mut rng);

        Self {
            phase: GamePhase::Spawning,
            physics,
            board,
            score: ScoreState::default(),
            events: Vec::new(),
            active_piece: None,
            bag,
            rng,
            drop_timer: BASE_DROP_INTERVAL,
            settle_timer: 0.0,
            accumulator: 0.0,
        }
    }

    /// Main game tick. Call once per frame with real delta time and collected inputs.
    pub fn update(&mut self, dt: f32, inputs: &[InputAction]) {
        // Fixed-timestep physics accumulator.
        let dt = dt.min(0.1); // cap to prevent spiral of death
        self.accumulator += dt;

        while self.accumulator >= PHYSICS_DT {
            self.physics.step(PHYSICS_DT);
            self.accumulator -= PHYSICS_DT;
        }

        match self.phase {
            GamePhase::Spawning => self.phase_spawning(),
            GamePhase::Falling => self.phase_falling(dt, inputs),
            GamePhase::Settling => self.phase_settling(dt),
            GamePhase::Checking => self.phase_checking(),
            GamePhase::Clearing => self.phase_clearing(dt),
            GamePhase::GameOver => {}
        }
    }

    /// Get block instances for rendering.
    pub fn block_instances(&self) -> Vec<BlockInstance> {
        self.physics.block_instances()
    }

    /// Drain all pending events (consumed by the renderer each frame).
    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.events)
    }

    // ── Phase implementations ─────────────────────────────────────────────

    fn phase_spawning(&mut self) {
        // Check game over: any settled cell above the board.
        if self.physics.any_cell_above(&self.board, SETTLE_VELOCITY_THRESHOLD) {
            self.phase = GamePhase::GameOver;
            self.events.push(GameEvent::GameOver);
            return;
        }

        let descriptor = self.bag.next(&mut self.rng);
        let center = self.board.spawn_position();
        let (piece_id, _) = self.physics.spawn_piece(&descriptor, center);
        self.active_piece = Some(piece_id);
        self.drop_timer = BASE_DROP_INTERVAL * self.score.drop_speed_multiplier();

        self.events.push(GameEvent::Spawn {
            shape: descriptor.shape,
            color: descriptor.color,
        });

        self.phase = GamePhase::Falling;
    }

    fn phase_falling(&mut self, dt: f32, inputs: &[InputAction]) {
        let Some(piece_id) = self.active_piece else {
            self.phase = GamePhase::Spawning;
            return;
        };

        // Process player input.
        for &action in inputs {
            match action {
                InputAction::MoveLeft => {
                    self.physics.apply_move_impulse(piece_id, -MOVE_IMPULSE);
                }
                InputAction::MoveRight => {
                    self.physics.apply_move_impulse(piece_id, MOVE_IMPULSE);
                }
                InputAction::RotateCW => {
                    self.physics
                        .apply_rotation_impulse(piece_id, -ROTATION_IMPULSE);
                }
                InputAction::RotateCCW => {
                    self.physics
                        .apply_rotation_impulse(piece_id, ROTATION_IMPULSE);
                }
                InputAction::SoftDrop => {
                    self.physics.apply_down_impulse(piece_id, SOFT_DROP_IMPULSE);
                }
                InputAction::HardDrop => {
                    self.physics.apply_down_impulse(piece_id, HARD_DROP_VELOCITY);
                    // Immediately transition to settling.
                    self.settle_timer = 0.0;
                    self.phase = GamePhase::Settling;
                    self.physics.detach_piece(piece_id);
                    return;
                }
            }
        }

        // Auto-drop timer.
        self.drop_timer -= dt;
        if self.drop_timer <= 0.0 {
            self.physics.apply_down_impulse(piece_id, SOFT_DROP_IMPULSE);
            self.drop_timer = BASE_DROP_INTERVAL * self.score.drop_speed_multiplier();
        }

        // Detect if piece has come to rest (contact with something below + low velocity).
        if self.physics.is_piece_settled(piece_id, SETTLE_VELOCITY_THRESHOLD) {
            // Check that the piece is actually resting on something
            // (not just momentarily slow at the peak of a bounce).
            // We do this by checking if the piece y-velocity is near zero or negative.
            let handles = self.physics.get_piece_handles(piece_id).cloned();
            if let Some(handles) = handles {
                let avg_y_vel: f32 = handles
                    .iter()
                    .filter_map(|&h| {
                        self.physics
                            .bodies()
                            .get(h)
                            .map(|b| b.linvel().y)
                    })
                    .sum::<f32>()
                    / handles.len() as f32;

                if avg_y_vel <= 0.5 {
                    self.detect_impacts(piece_id);
                    self.physics.detach_piece(piece_id);
                    self.settle_timer = 0.0;
                    self.phase = GamePhase::Settling;
                }
            }
        }
    }

    fn phase_settling(&mut self, dt: f32) {
        self.settle_timer += dt;
        if self.settle_timer >= SETTLE_TIME
            && self
                .physics
                .is_world_settled(SETTLE_VELOCITY_THRESHOLD)
        {
            self.phase = GamePhase::Checking;
        }
    }

    fn phase_checking(&mut self) {
        let matches = detect_matches(&self.physics);

        if matches.is_empty() {
            self.score.end_chain();
            self.active_piece = None;
            self.phase = GamePhase::Spawning;
            return;
        }

        // First match in a sequence starts the chain.
        if self.score.chain == 0 {
            self.score.begin_chain();
        } else {
            self.score.continue_chain();
        }

        for group in &matches {
            let positions: Vec<[f32; 2]> = group
                .handles
                .iter()
                .map(|&h| {
                    let p = self.physics.body_position(h);
                    [p.x, p.y]
                })
                .collect();

            let leveled = self.score.award_match(group.handles.len() as u32);

            self.events.push(GameEvent::Destroy {
                cells: positions,
                color: group.color,
            });

            if leveled {
                self.events.push(GameEvent::LevelUp {
                    level: self.score.level,
                });
            }

            // Remove the matched bodies from the physics world.
            self.physics.remove_bodies(&group.handles);
        }

        self.settle_timer = 0.0;
        self.phase = GamePhase::Clearing;
    }

    fn phase_clearing(&mut self, dt: f32) {
        self.settle_timer += dt;
        if self.settle_timer >= CLEARING_SETTLE_TIME
            && self
                .physics
                .is_world_settled(SETTLE_VELOCITY_THRESHOLD)
        {
            // Re-check for chain reactions.
            self.phase = GamePhase::Checking;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn detect_impacts(&mut self, piece_id: PieceId) {
        if let Some(handles) = self.physics.get_piece_handles(piece_id) {
            for &h in handles {
                let vel = self.physics.bodies().get(h).map(|b| {
                    let v = b.linvel();
                    Vec2::new(v.x, v.y)
                });
                if let Some(v) = vel {
                    if v.length() > IMPACT_VELOCITY_THRESHOLD {
                        let pos = self.physics.body_position(h);
                        self.events.push(GameEvent::Impact {
                            position: [pos.x, pos.y],
                            velocity: v.length(),
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_state_starts_in_spawning() {
        let state = GameState::new();
        assert_eq!(state.phase, GamePhase::Spawning);
        assert_eq!(state.score.level, 1);
    }

    #[test]
    fn first_update_spawns_piece() {
        let mut state = GameState::new();
        state.update(1.0 / 60.0, &[]);
        assert_eq!(state.phase, GamePhase::Falling);
        assert!(state.active_piece.is_some());
    }

    #[test]
    fn spawn_emits_event() {
        let mut state = GameState::new();
        state.update(1.0 / 60.0, &[]);
        let events = state.drain_events();
        assert!(events.iter().any(|e| matches!(e, GameEvent::Spawn { .. })));
    }

    #[test]
    fn hard_drop_transitions_to_settling() {
        let mut state = GameState::new();
        state.update(1.0 / 60.0, &[]); // spawn
        state.update(1.0 / 60.0, &[InputAction::HardDrop]);
        assert_eq!(state.phase, GamePhase::Settling);
    }

    #[test]
    fn multiple_updates_without_panic() {
        let mut state = GameState::new();
        for _ in 0..300 {
            state.update(1.0 / 60.0, &[]);
            let _ = state.drain_events();
        }
    }

    #[test]
    fn block_instances_populated_after_spawn() {
        let mut state = GameState::new();
        state.update(1.0 / 60.0, &[]);
        let instances = state.block_instances();
        assert!(!instances.is_empty());
    }
}
