use glam::Vec2;

/// Board dimensions in cell units.
/// Origin is at bottom-left of the play area.
pub const BOARD_WIDTH: u32 = 10;
pub const BOARD_HEIGHT: u32 = 20;

/// Size of each cell in physics-world units.
pub const CELL_SIZE: f32 = 1.0;

/// Thickness of the wall bodies (outside the play area).
pub const WALL_THICKNESS: f32 = 1.0;

/// How far above the top of the board a piece spawns.
pub const SPAWN_Y_OFFSET: f32 = 2.0;

/// Board encapsulates the play-field geometry and coordinate helpers.
#[derive(Debug, Clone)]
pub struct Board {
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
}

impl Default for Board {
    fn default() -> Self {
        Self {
            width: BOARD_WIDTH,
            height: BOARD_HEIGHT,
            cell_size: CELL_SIZE,
        }
    }
}

impl Board {
    /// World-space width of the play area.
    pub fn world_width(&self) -> f32 {
        self.width as f32 * self.cell_size
    }

    /// World-space height of the play area.
    pub fn world_height(&self) -> f32 {
        self.height as f32 * self.cell_size
    }

    /// Center of the play area in world-space.
    pub fn center(&self) -> Vec2 {
        Vec2::new(self.world_width() / 2.0, self.world_height() / 2.0)
    }

    /// Spawn position: top-center, slightly above the board.
    pub fn spawn_position(&self) -> Vec2 {
        Vec2::new(
            self.world_width() / 2.0,
            self.world_height() + SPAWN_Y_OFFSET,
        )
    }

    /// Convert a grid coordinate (col, row) to world-space center of that cell.
    pub fn grid_to_world(&self, col: u32, row: u32) -> Vec2 {
        Vec2::new(
            (col as f32 + 0.5) * self.cell_size,
            (row as f32 + 0.5) * self.cell_size,
        )
    }

    /// Convert a world-space position to the nearest grid coordinate.
    /// Returns None if the position is outside the board.
    pub fn world_to_grid(&self, pos: Vec2) -> Option<(u32, u32)> {
        let col = (pos.x / self.cell_size).floor() as i32;
        let row = (pos.y / self.cell_size).floor() as i32;
        if col >= 0 && col < self.width as i32 && row >= 0 && row < self.height as i32 {
            Some((col as u32, row as u32))
        } else {
            None
        }
    }

    /// Check if a position is above the game-over threshold (top of board).
    pub fn is_above_board(&self, pos: Vec2) -> bool {
        pos.y > self.world_height()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dimensions() {
        let board = Board::default();
        assert_eq!(board.world_width(), 10.0);
        assert_eq!(board.world_height(), 20.0);
    }

    #[test]
    fn grid_to_world_center_of_cell() {
        let board = Board::default();
        let pos = board.grid_to_world(0, 0);
        assert_eq!(pos, Vec2::new(0.5, 0.5));

        let pos = board.grid_to_world(9, 19);
        assert_eq!(pos, Vec2::new(9.5, 19.5));
    }

    #[test]
    fn world_to_grid_roundtrip() {
        let board = Board::default();
        let world = board.grid_to_world(3, 7);
        let grid = board.world_to_grid(world).unwrap();
        assert_eq!(grid, (3, 7));
    }

    #[test]
    fn world_to_grid_out_of_bounds() {
        let board = Board::default();
        assert!(board.world_to_grid(Vec2::new(-1.0, 5.0)).is_none());
        assert!(board.world_to_grid(Vec2::new(5.0, 25.0)).is_none());
    }

    #[test]
    fn spawn_position_above_board() {
        let board = Board::default();
        let spawn = board.spawn_position();
        assert!(board.is_above_board(spawn));
    }
}
