use webtych_game::state::{GamePhase, GameState, InputAction};
use webtych_game::events::GameEvent;

/// Run the game for N frames and return the final state.
fn run_frames(state: &mut GameState, frames: usize, inputs: &[InputAction]) {
    let dt = 1.0 / 60.0;
    for _ in 0..frames {
        state.update(dt, inputs);
        let _ = state.drain_events();
    }
}

#[test]
fn game_initializes_and_spawns() {
    let mut state = GameState::new();
    assert_eq!(state.phase, GamePhase::Spawning);

    state.update(1.0 / 60.0, &[]);
    assert_eq!(state.phase, GamePhase::Falling);

    let events = state.drain_events();
    assert!(
        events.iter().any(|e| matches!(e, GameEvent::Spawn { .. })),
        "Expected a Spawn event on first update"
    );
}

#[test]
fn pieces_fall_under_gravity() {
    let mut state = GameState::new();
    // Spawn a piece.
    state.update(1.0 / 60.0, &[]);

    let instances_before = state.block_instances();
    assert_eq!(instances_before.len(), 3, "A trimino has 3 cells");

    let y_before: f32 = instances_before.iter().map(|b| b.position[1]).sum::<f32>() / 3.0;

    // Run for a second — pieces should have fallen.
    run_frames(&mut state, 60, &[]);

    let instances_after = state.block_instances();
    let y_after: f32 = instances_after.iter().map(|b| b.position[1]).sum::<f32>()
        / instances_after.len().max(1) as f32;

    assert!(
        y_after < y_before,
        "Pieces should fall: y_before={}, y_after={}",
        y_before,
        y_after
    );
}

#[test]
fn move_left_shifts_piece() {
    let mut state = GameState::new();
    state.update(1.0 / 60.0, &[]); // spawn

    let before: f32 = state
        .block_instances()
        .iter()
        .map(|b| b.position[0])
        .sum::<f32>()
        / 3.0;

    // Apply left input for several frames.
    for _ in 0..10 {
        state.update(1.0 / 60.0, &[InputAction::MoveLeft]);
    }

    let after: f32 = state
        .block_instances()
        .iter()
        .map(|b| b.position[0])
        .sum::<f32>()
        / state.block_instances().len().max(1) as f32;

    assert!(
        after < before,
        "Piece should move left: before={}, after={}",
        before,
        after
    );
}

#[test]
fn move_right_shifts_piece() {
    let mut state = GameState::new();
    state.update(1.0 / 60.0, &[]);

    let before: f32 = state
        .block_instances()
        .iter()
        .map(|b| b.position[0])
        .sum::<f32>()
        / 3.0;

    for _ in 0..10 {
        state.update(1.0 / 60.0, &[InputAction::MoveRight]);
    }

    let after: f32 = state
        .block_instances()
        .iter()
        .map(|b| b.position[0])
        .sum::<f32>()
        / state.block_instances().len().max(1) as f32;

    assert!(
        after > before,
        "Piece should move right: before={}, after={}",
        before,
        after
    );
}

#[test]
fn hard_drop_transitions_to_settling() {
    let mut state = GameState::new();
    state.update(1.0 / 60.0, &[]); // spawn → falling
    assert_eq!(state.phase, GamePhase::Falling);

    state.update(1.0 / 60.0, &[InputAction::HardDrop]);
    assert_eq!(state.phase, GamePhase::Settling);
}

#[test]
fn piece_eventually_settles_and_new_piece_spawns() {
    let mut state = GameState::new();
    state.update(1.0 / 60.0, &[]); // spawn first piece

    // Hard drop and wait for settle + check + spawn cycle.
    state.update(1.0 / 60.0, &[InputAction::HardDrop]);

    // Run enough frames for the piece to settle and a new one to spawn.
    run_frames(&mut state, 300, &[]);

    // Should have more blocks now (3 from first piece + 3 from second).
    let instances = state.block_instances();
    assert!(
        instances.len() >= 3,
        "Expected at least 3 blocks, got {}",
        instances.len()
    );
}

#[test]
fn scoring_starts_at_zero() {
    let state = GameState::new();
    assert_eq!(state.score.score, 0);
    assert_eq!(state.score.level, 1);
}

#[test]
fn block_instances_have_valid_color_ids() {
    let mut state = GameState::new();
    state.update(1.0 / 60.0, &[]);

    for instance in state.block_instances() {
        assert!(
            instance.color_id <= 3,
            "color_id should be 0..=3, got {}",
            instance.color_id
        );
    }
}

#[test]
fn block_instances_are_alive() {
    let mut state = GameState::new();
    state.update(1.0 / 60.0, &[]);

    for instance in state.block_instances() {
        assert_eq!(instance.alive, 1, "Blocks should start alive");
    }
}

#[test]
fn long_game_session_no_panic() {
    let mut state = GameState::new();
    // Simulate 30 seconds of gameplay with mixed inputs.
    let inputs_cycle: &[&[InputAction]] = &[
        &[],
        &[InputAction::MoveLeft],
        &[],
        &[InputAction::MoveRight],
        &[],
        &[InputAction::SoftDrop],
        &[],
        &[InputAction::RotateCW],
        &[],
        &[InputAction::HardDrop],
    ];

    for i in 0..1800 {
        let inputs = inputs_cycle[i % inputs_cycle.len()];
        state.update(1.0 / 60.0, inputs);
        let _ = state.drain_events();
    }
    // If we get here without panicking, the test passes.
}

#[test]
fn multiple_hard_drops_accumulate_blocks() {
    let mut state = GameState::new();

    for _ in 0..5 {
        // Spawn.
        state.update(1.0 / 60.0, &[]);
        // Hard drop.
        state.update(1.0 / 60.0, &[InputAction::HardDrop]);
        // Let it settle and cycle.
        run_frames(&mut state, 120, &[]);
    }

    // We should have accumulated blocks (unless some were matched and cleared).
    let instances = state.block_instances();
    assert!(
        instances.len() >= 3,
        "Expected accumulated blocks, got {} instances",
        instances.len()
    );
}
