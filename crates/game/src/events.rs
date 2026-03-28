use crate::trimino::{ColorId, TriminoShape};

/// Events emitted by the game state machine each frame.
/// Drained by the renderer / app layer for visual effects and UI.
#[derive(Debug, Clone)]
pub enum GameEvent {
    /// A group of same-color cells was matched and destroyed.
    Destroy {
        /// World-space positions of the destroyed cells.
        cells: Vec<[f32; 2]>,
        /// Color of the destroyed group.
        color: ColorId,
    },
    /// A piece impacted something (landing, collision).
    Impact {
        /// World-space position of the impact.
        position: [f32; 2],
        /// Magnitude of the impact velocity.
        velocity: f32,
    },
    /// A new piece was spawned.
    Spawn {
        shape: TriminoShape,
        colors: [ColorId; 3],
    },
    /// Player reached a new level.
    LevelUp { level: u32 },
    /// The board is full — game over.
    GameOver,
}
