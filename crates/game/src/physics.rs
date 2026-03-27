use glam::Vec2;
use rapier2d::prelude::*;

use crate::board::Board;
use crate::trimino::{Color, PieceDescriptor};

/// Identifier for a piece (group of cells joined together).
pub type PieceId = u64;

/// Data packed into a rigid body's user_data field.
/// Low 32 bits: color_id. Next 32 bits: piece_id.
fn pack_user_data(color: Color, piece_id: PieceId) -> u128 {
    (color.id() as u128) | ((piece_id as u128) << 32)
}

fn unpack_color(user_data: u128) -> Color {
    match (user_data & 0xFFFF_FFFF) as u32 {
        0 => Color::Red,
        1 => Color::Blue,
        2 => Color::Green,
        3 => Color::Yellow,
        _ => Color::Red,
    }
}

/// Public accessor used by the scoring module to read color from body user_data.
pub fn color_from_user_data(user_data: u128) -> Color {
    unpack_color(user_data)
}

fn unpack_piece_id(user_data: u128) -> PieceId {
    ((user_data >> 32) & 0xFFFF_FFFF) as PieceId
}

/// GPU-uploadable per-block instance data.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockInstance {
    pub position: [f32; 2],
    pub rotation: f32,
    pub color_id: u32,
    pub alive: u32,
    pub _pad: [f32; 3],
}

/// Information about a cell body in the physics world.
#[derive(Debug, Clone)]
pub struct CellInfo {
    pub handle: RigidBodyHandle,
    pub position: Vec2,
    pub rotation: f32,
    pub color: Color,
    pub piece_id: PieceId,
    pub linvel: Vec2,
}

/// Spring-damper joint parameters for connecting trimino cells.
/// Cells flex and wobble on impact, producing a bouncy "jello" feel.
/// Using AccelerationBased model so spring behavior is mass-independent.
const SPRING_STIFFNESS: f32 = 8000.0; // high stiffness keeps cells visually connected
const SPRING_DAMPING: f32 = 120.0; // enough damping to settle wobble within ~0.3s

/// Maximum distance between cell centers (in cell-size units) to be considered adjacent.
const ADJACENCY_THRESHOLD: f32 = 1.1;

/// Wraps the Rapier2D physics pipeline and body sets.
pub struct PhysicsWorld {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    integration_params: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseMultiSap,
    narrow_phase: NarrowPhase,
    pipeline: PhysicsPipeline,
    ccd_solver: CCDSolver,
    gravity: Vector<f32>,
    next_piece_id: PieceId,
    /// Handles of cell bodies belonging to each piece.
    piece_cells: Vec<(PieceId, Vec<RigidBodyHandle>)>,
    /// Joints belonging to each piece (so we can break them).
    piece_joints: Vec<(PieceId, Vec<ImpulseJointHandle>)>,
    /// Handles of wall bodies (excluded from game queries).
    wall_handles: Vec<RigidBodyHandle>,
}

impl PhysicsWorld {
    /// Create a new physics world with gravity and board walls.
    pub fn new(board: &Board) -> Self {
        let gravity = vector![0.0, -20.0];

        let mut world = Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            integration_params: IntegrationParameters::default(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseMultiSap::new(),
            narrow_phase: NarrowPhase::new(),
            pipeline: PhysicsPipeline::new(),
            ccd_solver: CCDSolver::new(),
            gravity,
            next_piece_id: 1,
            piece_cells: Vec::new(),
            piece_joints: Vec::new(),
            wall_handles: Vec::new(),
        };

        world.create_walls(board);
        world
    }

    fn create_walls(&mut self, board: &Board) {
        let w = board.world_width();
        let h = board.world_height();
        let t = crate::board::WALL_THICKNESS;
        let half_t = t / 2.0;

        // Floor
        let floor = self
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(vector![w / 2.0, -half_t]));
        self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(w / 2.0, half_t)
                .friction(0.5)
                .restitution(0.2),
            floor,
            &mut self.bodies,
        );
        self.wall_handles.push(floor);

        // Left wall
        let left = self
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(vector![-half_t, h / 2.0]));
        self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(half_t, h / 2.0 + t)
                .friction(0.3)
                .restitution(0.1),
            left,
            &mut self.bodies,
        );
        self.wall_handles.push(left);

        // Right wall
        let right = self
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(vector![w + half_t, h / 2.0]));
        self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(half_t, h / 2.0 + t)
                .friction(0.3)
                .restitution(0.1),
            right,
            &mut self.bodies,
        );
        self.wall_handles.push(right);
    }

    /// Spawn a new piece. Returns the assigned PieceId and the handles of its cells.
    pub fn spawn_piece(
        &mut self,
        descriptor: &PieceDescriptor,
        center: Vec2,
    ) -> (PieceId, Vec<RigidBodyHandle>) {
        let piece_id = self.next_piece_id;
        self.next_piece_id += 1;

        let offsets = descriptor.shape.offsets(descriptor.rotation);
        let half_cell = crate::board::CELL_SIZE / 2.0 * 0.95; // slight gap for visual clarity

        let mut handles = Vec::with_capacity(3);
        for (idx, offset) in offsets.iter().enumerate() {
            let pos = center + *offset * crate::board::CELL_SIZE;
            let user_data = pack_user_data(descriptor.colors[idx], piece_id);
            let body = self.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(vector![pos.x, pos.y])
                    .user_data(user_data)
                    .ccd_enabled(true)
                    .linear_damping(0.5),
            );
            self.colliders.insert_with_parent(
                ColliderBuilder::cuboid(half_cell, half_cell)
                    .friction(0.5)
                    .restitution(0.3)
                    .density(1.0),
                body,
                &mut self.bodies,
            );
            handles.push(body);
        }

        // Connect adjacent cells with 4 spring-damper joints per pair:
        // 2 perpendicular (corner-to-corner across shared edge) +
        // 2 crossing (corner-to-opposite-corner), resisting both translation and rotation.
        let mut joint_handles = Vec::new();
        let cell_size = crate::board::CELL_SIZE;
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let pos_i = self.body_position(handles[i]);
                let pos_j = self.body_position(handles[j]);
                let dist = (pos_j - pos_i).length();

                if dist <= cell_size * ADJACENCY_THRESHOLD {
                    let diff = pos_j - pos_i;
                    let dir = diff.normalize();
                    let perp = Vec2::new(-dir.y, dir.x);

                    // Corner anchors on each cell's shared edge (in local frame).
                    let i_corner_p = dir * half_cell + perp * half_cell;
                    let i_corner_m = dir * half_cell - perp * half_cell;
                    let j_corner_p = -dir * half_cell + perp * half_cell;
                    let j_corner_m = -dir * half_cell - perp * half_cell;

                    // Rest lengths: distance between anchor pairs in equilibrium.
                    let gap = cell_size - 2.0 * half_cell;
                    let perp_rest = gap;
                    let cross_rest = (gap * gap + (2.0 * half_cell).powi(2)).sqrt();

                    // Helper to insert one spring joint.
                    let mut add_spring =
                        |a1: Vec2, a2: Vec2, rest: f32| {
                            let joint = SpringJointBuilder::new(
                                rest,
                                SPRING_STIFFNESS,
                                SPRING_DAMPING,
                            )
                            .local_anchor1(point![a1.x, a1.y])
                            .local_anchor2(point![a2.x, a2.y])
                            .spring_model(MotorModel::AccelerationBased);

                            let jh =
                                self.joints
                                    .insert(handles[i], handles[j], joint, true);
                            joint_handles.push(jh);
                        };

                    // 2 perpendicular springs (straight across shared edge).
                    add_spring(i_corner_p, j_corner_p, perp_rest);
                    add_spring(i_corner_m, j_corner_m, perp_rest);

                    // 2 crossing springs (diagonal).
                    add_spring(i_corner_p, j_corner_m, cross_rest);
                    add_spring(i_corner_m, j_corner_p, cross_rest);
                }
            }
        }

        self.piece_cells.push((piece_id, handles.clone()));
        self.piece_joints.push((piece_id, joint_handles));

        (piece_id, handles)
    }

    /// Detach a piece's cells so they become independent bodies.
    /// Removes the joints connecting them.
    pub fn detach_piece(&mut self, piece_id: PieceId) {
        if let Some(idx) = self.piece_joints.iter().position(|(id, _)| *id == piece_id) {
            let (_, joint_handles) = self.piece_joints.remove(idx);
            for jh in joint_handles {
                self.joints.remove(jh, true);
            }
        }
    }

    /// Remove a single cell body from the world.
    /// Rapier automatically removes attached joints; we clean up our tracking.
    pub fn remove_body(&mut self, handle: RigidBodyHandle) {
        self.bodies.remove(
            handle,
            &mut self.islands,
            &mut self.colliders,
            &mut self.joints,
            &mut self.multibody_joints,
            true,
        );
        // Clean up from piece_cells tracking.
        for (_, cells) in &mut self.piece_cells {
            cells.retain(|h| *h != handle);
        }
        // Clean up stale joint handles (Rapier already removed them,
        // but our tracking still references them).
        for (_, joints) in &mut self.piece_joints {
            joints.retain(|jh| self.joints.get(*jh).is_some());
        }
    }

    /// Remove all cells belonging to a set of handles.
    pub fn remove_bodies(&mut self, handles: &[RigidBodyHandle]) {
        for &h in handles {
            self.remove_body(h);
        }
    }

    /// Step the physics simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        self.integration_params.dt = dt;
        self.pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
    }

    /// Apply a horizontal velocity impulse to all cells of a piece.
    pub fn apply_move_impulse(&mut self, piece_id: PieceId, impulse_x: f32) {
        if let Some(handles) = self.get_piece_handles(piece_id) {
            let handles = handles.clone();
            for h in handles {
                if let Some(body) = self.bodies.get_mut(h) {
                    body.apply_impulse(vector![impulse_x, 0.0], true);
                }
            }
        }
    }

    /// Apply a torque impulse to the first cell of a piece (rotates the group via joints).
    pub fn apply_rotation_impulse(&mut self, piece_id: PieceId, torque: f32) {
        if let Some(handles) = self.get_piece_handles(piece_id) {
            if let Some(&first) = handles.first() {
                if let Some(body) = self.bodies.get_mut(first) {
                    body.apply_torque_impulse(torque, true);
                }
            }
        }
    }

    /// Apply a downward velocity boost (soft drop).
    pub fn apply_down_impulse(&mut self, piece_id: PieceId, impulse_y: f32) {
        if let Some(handles) = self.get_piece_handles(piece_id) {
            let handles = handles.clone();
            for h in handles {
                if let Some(body) = self.bodies.get_mut(h) {
                    body.apply_impulse(vector![0.0, impulse_y], true);
                }
            }
        }
    }

    /// Get all cell bodies (excluding walls).
    pub fn cell_infos(&self) -> Vec<CellInfo> {
        let mut infos = Vec::new();
        for (handle, body) in self.bodies.iter() {
            if self.wall_handles.contains(&handle) {
                continue;
            }
            let pos = body.translation();
            let rot = body.rotation().angle();
            let vel = body.linvel();
            let color = unpack_color(body.user_data);
            let piece_id = unpack_piece_id(body.user_data);
            infos.push(CellInfo {
                handle,
                position: Vec2::new(pos.x, pos.y),
                rotation: rot,
                color,
                piece_id,
                linvel: Vec2::new(vel.x, vel.y),
            });
        }
        infos
    }

    /// Get BlockInstance data for GPU rendering.
    pub fn block_instances(&self) -> Vec<BlockInstance> {
        self.cell_infos()
            .iter()
            .map(|info| BlockInstance {
                position: info.position.into(),
                rotation: info.rotation,
                color_id: info.color.id(),
                alive: 1,
                _pad: [0.0; 3],
            })
            .collect()
    }

    /// Access bodies for a given piece.
    pub fn get_piece_handles(&self, piece_id: PieceId) -> Option<&Vec<RigidBodyHandle>> {
        self.piece_cells
            .iter()
            .find(|(id, _)| *id == piece_id)
            .map(|(_, handles)| handles)
    }

    /// Check if all cells of a piece have velocity below the given threshold.
    pub fn is_piece_settled(&self, piece_id: PieceId, vel_threshold: f32) -> bool {
        if let Some(handles) = self.get_piece_handles(piece_id) {
            handles.iter().all(|&h| {
                self.bodies
                    .get(h)
                    .map(|b| b.linvel().norm() < vel_threshold)
                    .unwrap_or(true)
            })
        } else {
            true
        }
    }

    /// Check if all non-wall bodies in the world are below the velocity threshold.
    pub fn is_world_settled(&self, vel_threshold: f32) -> bool {
        for (handle, body) in self.bodies.iter() {
            if self.wall_handles.contains(&handle) {
                continue;
            }
            if body.linvel().norm() >= vel_threshold {
                return false;
            }
        }
        true
    }

    /// Get body position as Vec2.
    pub fn body_position(&self, handle: RigidBodyHandle) -> Vec2 {
        let t = self.bodies[handle].translation();
        Vec2::new(t.x, t.y)
    }

    /// Provide read access to the narrow phase for contact queries.
    pub fn narrow_phase(&self) -> &NarrowPhase {
        &self.narrow_phase
    }

    /// Provide read access to colliders for contact resolution.
    pub fn colliders(&self) -> &ColliderSet {
        &self.colliders
    }

    /// Provide read access to bodies.
    pub fn bodies(&self) -> &RigidBodySet {
        &self.bodies
    }

    /// Is a handle a wall body?
    pub fn is_wall(&self, handle: RigidBodyHandle) -> bool {
        self.wall_handles.contains(&handle)
    }

    /// Check if any settled cell is above the board top.
    pub fn any_cell_above(&self, board: &Board, vel_threshold: f32) -> bool {
        for info in self.cell_infos() {
            if info.linvel.length() < vel_threshold && board.is_above_board(info.position) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::trimino::{Color, PieceDescriptor, TriminoShape};

    #[test]
    fn user_data_roundtrip() {
        for &color in &Color::ALL {
            let packed = pack_user_data(color, 42);
            assert_eq!(unpack_color(packed), color);
            assert_eq!(unpack_piece_id(packed), 42);
        }
    }

    #[test]
    fn physics_world_creates_walls() {
        let board = Board::default();
        let world = PhysicsWorld::new(&board);
        // Floor + left + right = 3 wall bodies.
        assert_eq!(world.wall_handles.len(), 3);
    }

    #[test]
    fn spawn_piece_creates_three_bodies() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::I,
            colors: [Color::Red; 3],
            rotation: 0,
        };
        let (_, handles) = world.spawn_piece(&desc, board.spawn_position());
        assert_eq!(handles.len(), 3);
    }

    #[test]
    fn detach_piece_removes_joints() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::L,
            colors: [Color::Blue; 3],
            rotation: 0,
        };
        let (piece_id, _) = world.spawn_piece(&desc, board.spawn_position());
        assert!(!world.piece_joints.is_empty());

        world.detach_piece(piece_id);
        // The piece_joints entry should be removed.
        assert!(world
            .piece_joints
            .iter()
            .find(|(id, _)| *id == piece_id)
            .is_none());
    }

    #[test]
    fn cell_infos_excludes_walls() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::T,
            colors: [Color::Green; 3],
            rotation: 0,
        };
        world.spawn_piece(&desc, board.spawn_position());
        let infos = world.cell_infos();
        assert_eq!(infos.len(), 3);
    }

    #[test]
    fn block_instances_match_cell_count() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::I,
            colors: [Color::Yellow; 3],
            rotation: 0,
        };
        world.spawn_piece(&desc, board.spawn_position());
        let instances = world.block_instances();
        assert_eq!(instances.len(), 3);
        assert!(instances.iter().all(|b| b.color_id == Color::Yellow.id()));
    }

    #[test]
    fn step_does_not_panic() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::I,
            colors: [Color::Red; 3],
            rotation: 0,
        };
        world.spawn_piece(&desc, board.spawn_position());
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }
    }

    #[test]
    fn remove_body_cleans_up() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::I,
            colors: [Color::Red; 3],
            rotation: 0,
        };
        let (_, handles) = world.spawn_piece(&desc, board.spawn_position());
        let to_remove = handles[0];
        world.remove_body(to_remove);
        let infos = world.cell_infos();
        assert_eq!(infos.len(), 2);
    }

    #[test]
    fn spring_joints_pull_displaced_cells_back() {
        // Spawn an I-piece in zero gravity, displace one cell, and verify
        // the spring pulls it back toward its rest position.
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        // Disable gravity so we measure spring force in isolation.
        world.gravity = vector![0.0, 0.0];

        let desc = PieceDescriptor {
            shape: TriminoShape::I,
            colors: [Color::Red; 3],
            rotation: 0,
        };
        let (_, handles) = world.spawn_piece(&desc, board.spawn_position());

        // Record initial position of cell 2 (rightmost).
        let initial_pos = world.body_position(handles[2]);

        // Displace cell 2 to the right by 2 units.
        if let Some(body) = world.bodies.get_mut(handles[2]) {
            let t = body.translation();
            body.set_translation(vector![t.x + 2.0, t.y], true);
        }
        let displaced_pos = world.body_position(handles[2]);
        assert!((displaced_pos.x - initial_pos.x - 2.0).abs() < 0.01);

        // Step physics — the spring should pull cell 2 back.
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }

        let final_pos = world.body_position(handles[2]);
        let recovery = (displaced_pos.x - final_pos.x).abs();
        assert!(
            recovery > 0.5,
            "Spring should pull displaced cell back significantly. \
             Displaced: {:.2}, Final: {:.2}, Recovery: {:.2}",
            displaced_pos.x,
            final_pos.x,
            recovery,
        );
    }

    #[test]
    fn spring_joints_are_created_for_adjacent_cells() {
        let board = Board::default();
        let mut world = PhysicsWorld::new(&board);
        let desc = PieceDescriptor {
            shape: TriminoShape::I,
            colors: [Color::Red; 3],
            rotation: 0,
        };
        let (piece_id, _) = world.spawn_piece(&desc, board.spawn_position());

        // I-shape has 3 cells in a line. Cells 0-1 and 1-2 are adjacent (distance 1.0),
        // but 0-2 are not (distance 2.0). So we expect 2 pairs × 4 springs = 8 joints.
        let joints = world.piece_joints.iter().find(|(id, _)| *id == piece_id);
        assert!(joints.is_some());
        let (_, joint_handles) = joints.unwrap();
        assert_eq!(
            joint_handles.len(),
            8,
            "I-shape should have 8 spring joints (4 per adjacent pair)"
        );
    }
}
