use glam::Vec2;
use rand::seq::SliceRandom;
use rand::Rng;

/// Block colors — 4 colors fit in one RGBA density texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Color {
    Red = 0,
    Blue = 1,
    Green = 2,
    Yellow = 3,
}

impl Color {
    pub const ALL: [Color; 4] = [Color::Red, Color::Blue, Color::Green, Color::Yellow];

    pub fn id(self) -> u32 {
        self as u32
    }
}

/// The three trimino shapes — each piece has 3 cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriminoShape {
    /// Straight line: ■■■
    I,
    /// L-shape: ■■
    ///           ■
    L,
    /// T-shape: ■■■
    ///           ■  (center cell below)
    T,
}

impl TriminoShape {
    pub const ALL: [TriminoShape; 3] = [TriminoShape::I, TriminoShape::L, TriminoShape::T];

    /// Cell offsets relative to the piece pivot (in cell units) for rotation index 0.
    /// Each shape has exactly 3 cells.
    pub fn base_offsets(self) -> [Vec2; 3] {
        match self {
            TriminoShape::I => [
                Vec2::new(-1.0, 0.0),
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
            ],
            TriminoShape::L => [
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, -1.0),
            ],
            TriminoShape::T => [
                Vec2::new(-1.0, 0.0),
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
            ],
        }
    }

    /// Returns the cell offsets rotated by the given number of 90° clockwise steps.
    pub fn offsets(self, rotation: u8) -> [Vec2; 3] {
        let base = self.base_offsets();
        let steps = rotation % 4;
        base.map(|v| rotate_90_cw(v, steps))
    }
}

/// Rotate a 2D vector by `steps` × 90° clockwise.
fn rotate_90_cw(v: Vec2, steps: u8) -> Vec2 {
    match steps % 4 {
        0 => v,
        1 => Vec2::new(v.y, -v.x),
        2 => Vec2::new(-v.x, -v.y),
        3 => Vec2::new(-v.y, v.x),
        _ => unreachable!(),
    }
}

/// A piece to be spawned: shape + per-cell colors + rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceDescriptor {
    pub shape: TriminoShape,
    /// Each of the 3 cells gets its own color.
    pub colors: [Color; 3],
    pub rotation: u8,
}

/// Randomizer that generates pieces with random shapes and per-cell random colors.
pub struct PieceBag {
    shapes: Vec<TriminoShape>,
}

impl PieceBag {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let mut bag = Self {
            shapes: Vec::new(),
        };
        bag.refill(rng);
        bag
    }

    /// Draw the next piece from the bag. Refills automatically when empty.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> PieceDescriptor {
        if self.shapes.is_empty() {
            self.refill(rng);
        }
        let shape = self.shapes.pop().expect("bag was just refilled");
        let colors = [
            *Color::ALL.choose(rng).unwrap(),
            *Color::ALL.choose(rng).unwrap(),
            *Color::ALL.choose(rng).unwrap(),
        ];
        PieceDescriptor {
            shape,
            colors,
            rotation: 0,
        }
    }

    /// Peek at the next shape (color is random on draw, so not available).
    pub fn peek_shape(&self) -> Option<&TriminoShape> {
        self.shapes.last()
    }

    fn refill<R: Rng>(&mut self, rng: &mut R) {
        self.shapes.clear();
        // Two copies of each shape for a balanced bag.
        for &shape in &TriminoShape::ALL {
            self.shapes.push(shape);
            self.shapes.push(shape);
        }
        self.shapes.shuffle(rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_ids_are_sequential() {
        assert_eq!(Color::Red.id(), 0);
        assert_eq!(Color::Blue.id(), 1);
        assert_eq!(Color::Green.id(), 2);
        assert_eq!(Color::Yellow.id(), 3);
    }

    #[test]
    fn i_shape_has_three_cells() {
        let offsets = TriminoShape::I.offsets(0);
        assert_eq!(offsets.len(), 3);
    }

    #[test]
    fn rotation_360_is_identity() {
        for &shape in &TriminoShape::ALL {
            let base = shape.offsets(0);
            let rotated = shape.offsets(4);
            for i in 0..3 {
                assert!(
                    (base[i] - rotated[i]).length() < 1e-5,
                    "360° rotation should be identity for {:?}",
                    shape,
                );
            }
        }
    }

    #[test]
    fn rotation_produces_perpendicular_vectors() {
        let v = Vec2::new(1.0, 0.0);
        let r1 = rotate_90_cw(v, 1);
        assert!((r1 - Vec2::new(0.0, -1.0)).length() < 1e-5);
    }

    #[test]
    fn piece_bag_yields_all_shapes() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        // Bag has 2 copies of each shape = 6 pieces.
        let mut pieces = Vec::new();
        for _ in 0..6 {
            pieces.push(bag.next(&mut rng));
        }

        // Every shape should appear exactly twice.
        for &shape in &TriminoShape::ALL {
            let count = pieces.iter().filter(|p| p.shape == shape).count();
            assert_eq!(count, 2, "Expected two {:?} pieces in bag", shape);
        }
    }

    #[test]
    fn piece_bag_per_cell_colors() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        let piece = bag.next(&mut rng);
        // Each cell should have a valid color.
        for color in &piece.colors {
            assert!(Color::ALL.contains(color));
        }
    }

    #[test]
    fn piece_bag_auto_refills() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        // Draw more than one bag's worth.
        for _ in 0..20 {
            let _ = bag.next(&mut rng);
        }
    }

    #[test]
    fn peek_shape_returns_next_without_consuming() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        let peeked_shape = *bag.peek_shape().unwrap();
        let drawn = bag.next(&mut rng);
        assert_eq!(peeked_shape, drawn.shape);
    }
}
