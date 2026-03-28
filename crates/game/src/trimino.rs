use glam::Vec2;
use rand::seq::SliceRandom;
use rand::Rng;

/// A color index referencing a slot in the active [`ColorPalette`].
///
/// The value is opaque — its meaning depends on the palette in use.
/// Use `palette.rgba(color_id)` to resolve to an actual RGBA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorId(pub u32);

impl ColorId {
    pub fn id(self) -> u32 {
        self.0
    }
}

/// A configurable color palette.
///
/// The default palette has 4 colors (Red, Blue, Green, Yellow).
/// Higher difficulties can add more colors.
#[derive(Debug, Clone)]
pub struct ColorPalette {
    /// RGBA values for each color slot.
    colors: Vec<[f32; 4]>,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self {
            colors: vec![
                [0.90, 0.22, 0.22, 1.0], // red
                [0.22, 0.42, 0.90, 1.0], // blue
                [0.22, 0.80, 0.35, 1.0], // green
                [0.92, 0.82, 0.12, 1.0], // yellow
                [0.9, 0.82, 0.9, 1.0],   // light grey
            ],
        }
    }
}

impl ColorPalette {
    /// Create a palette from a list of RGBA colors.
    pub fn new(colors: Vec<[f32; 4]>) -> Self {
        assert!(!colors.is_empty(), "palette must have at least one color");
        Self { colors }
    }

    /// Number of colors in the palette.
    pub fn len(&self) -> u32 {
        self.colors.len() as u32
    }

    /// Whether the palette is empty (always false after construction).
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Get the RGBA value for a color index.
    pub fn rgba(&self, id: ColorId) -> [f32; 4] {
        self.colors[id.0 as usize % self.colors.len()]
    }

    /// Get all RGBA values (for uploading to the GPU as a uniform).
    pub fn as_slice(&self) -> &[[f32; 4]] {
        &self.colors
    }

    /// Return a random [`ColorId`] from this palette.
    pub fn random_color<R: Rng>(&self, rng: &mut R) -> ColorId {
        ColorId(rng.gen_range(0..self.len()))
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
    pub colors: [ColorId; 3],
    pub rotation: u8,
}

/// Randomizer that generates pieces with random shapes and per-cell random colors.
pub struct PieceBag {
    shapes: Vec<TriminoShape>,
}

impl PieceBag {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let mut bag = Self { shapes: Vec::new() };
        bag.refill(rng);
        bag
    }

    /// Draw the next piece from the bag. Refills automatically when empty.
    /// Colors are picked randomly from the given palette.
    pub fn next<R: Rng>(&mut self, rng: &mut R, palette: &ColorPalette) -> PieceDescriptor {
        if self.shapes.is_empty() {
            self.refill(rng);
        }
        let shape = self.shapes.pop().expect("bag was just refilled");
        let colors = [
            palette.random_color(rng),
            palette.random_color(rng),
            palette.random_color(rng),
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
    fn color_palette_default_has_four_colors() {
        let p = ColorPalette::default();
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn color_id_roundtrip() {
        assert_eq!(ColorId(0).id(), 0);
        assert_eq!(ColorId(3).id(), 3);
    }

    #[test]
    fn palette_rgba_wraps_on_overflow() {
        let p = ColorPalette::default();
        // Index 4 should wrap to 0.
        assert_eq!(p.rgba(ColorId(4)), p.rgba(ColorId(0)));
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
        let palette = ColorPalette::default();
        let mut pieces = Vec::new();
        for _ in 0..6 {
            pieces.push(bag.next(&mut rng, &palette));
        }

        for &shape in &TriminoShape::ALL {
            let count = pieces.iter().filter(|p| p.shape == shape).count();
            assert_eq!(count, 2, "Expected two {:?} pieces in bag", shape);
        }
    }

    #[test]
    fn piece_bag_per_cell_colors() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        let palette = ColorPalette::default();
        let piece = bag.next(&mut rng, &palette);
        for color in &piece.colors {
            assert!(color.id() < palette.len());
        }
    }

    #[test]
    fn piece_bag_auto_refills() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        let palette = ColorPalette::default();
        for _ in 0..20 {
            let _ = bag.next(&mut rng, &palette);
        }
    }

    #[test]
    fn peek_shape_returns_next_without_consuming() {
        let mut rng = rand::thread_rng();
        let mut bag = PieceBag::new(&mut rng);
        let palette = ColorPalette::default();
        let peeked_shape = *bag.peek_shape().unwrap();
        let drawn = bag.next(&mut rng, &palette);
        assert_eq!(peeked_shape, drawn.shape);
    }
}
