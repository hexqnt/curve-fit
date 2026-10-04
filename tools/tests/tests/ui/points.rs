use super::*;

#[test]
fn points_editor_paste_preserves_unicode_history_and_clears_redo_on_new_edit() {
    let mut harness = harness();
    let original = "0 1\n1 2\n# заметка 🦀\n";
    let edited = "0 3\n1 5\n# другая заметка λ\n";
    let replacement = "0 4\n1 8\n# ещё 🦀\n";
    points(&mut harness, original);
    let history_len = state(&harness).selected_layer.undo.len();
    harness.run_steps(3);
    assert_eq!(state(&harness).selected_layer.undo.len(), history_len);
    assert_eq!(state(&harness).selected_layer.points_text, original);

    for pasted in [edited, replacement] {
        harness
            .get_by_role_and_label(Role::MultilineTextInput, "Input points")
            .click();
        harness.run();
        assert!(
            harness
                .get_by_role_and_label(Role::MultilineTextInput, "Input points")
                .is_focused()
        );
        harness.key_press_modifiers(Modifiers::COMMAND, Key::A);
        harness.run();
        harness
            .input_mut()
            .events
            .push(egui::Event::Paste(pasted.into()));
        harness.run();
        let snapshot = state(&harness);
        assert_eq!(snapshot.selected_layer.points_text, pasted);
        assert_eq!(
            snapshot.selected_layer.undo.last().map(String::as_str),
            Some(original)
        );
        assert!(snapshot.selected_layer.redo.is_empty());
        click(&mut harness, "Undo");
        assert_eq!(state(&harness).selected_layer.points_text, original);
        click(&mut harness, "Redo");
        assert_eq!(state(&harness).selected_layer.points_text, pasted);
        click(&mut harness, "Undo");
        assert_eq!(state(&harness).selected_layer.points_text, original);
        assert!(!state(&harness).selected_layer.redo.is_empty());
    }
}

#[test]
fn move_points_to_positive_xy_pushes_undo_and_clears_redo() {
    let mut harness = harness();
    points(&mut harness, "-1 0\n1 2\n");
    clear_and_undo(&mut harness);
    click(&mut harness, "Actions");
    click(&mut harness, "Move to positive x/y");
    assert_eq!(
        state(&harness)
            .selected_layer
            .undo
            .last()
            .map(String::as_str),
        Some("-1 0\n1 2\n")
    );
    assert!(state(&harness).selected_layer.redo.is_empty());
    let moved_text = state(&harness).selected_layer.points_text.to_owned();
    assert_eq!(moved_text, "0.00000100 0.00000100\n2.00000100 2.00000100\n");
    click(&mut harness, "Undo");
    assert_eq!(state(&harness).selected_layer.points_text, "-1 0\n1 2\n");
    click(&mut harness, "Redo");
    assert_eq!(state(&harness).selected_layer.points_text, moved_text);
}

#[test]
fn keyboard_shortcuts_undo_and_redo_points_edits() {
    let mut harness = harness();
    points(&mut harness, "0 1\n1 2\n");
    click(&mut harness, "Clear points");
    harness.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    harness.run();
    assert_eq!(state(&harness).selected_layer.points_text, "0 1\n1 2\n");
    harness.key_press_modifiers(Modifiers::COMMAND, Key::Y);
    harness.run();
    assert!(state(&harness).selected_layer.points_text.is_empty());
}

#[test]
fn points_data_change_clears_stale_fit_outputs() {
    let mut harness = harness();
    points(&mut harness, "0 1\n1 3\n2 5\n3 7\n");
    fit_and_wait(&mut harness);
    assert!(state(&harness).replay_frames > 0);
    points(&mut harness, "10 20\n30 40\n");
    assert!(state(&harness).fit_result.is_none());
    assert_eq!(state(&harness).replay_frames, 0);
    assert!(state(&harness).replay_iteration.is_none());
}

#[test]
fn fill_points_with_residuals_replaces_points_text_and_pushes_undo() {
    let mut harness = harness();
    let original = "0 1\n1 3\n2 4.9\n3 7\n";
    points(&mut harness, original);
    fit_and_wait(&mut harness);
    let expected = {
        let snapshot = state(&harness);
        let params = &snapshot.fit_result.unwrap().params;
        [(0.0, 1.0), (1.0, 3.0), (2.0, 4.9), (3.0, 7.0)].map(|(x, y)| (x, params.evaluate(x) - y))
    };
    click(&mut harness, "Actions");
    click(&mut harness, "Fill with residuals");
    let text = state(&harness).selected_layer.points_text;
    assert_ne!(text, original);
    assert_eq!(text.lines().count(), 4);
    for (line, (x, expected_residual)) in text.lines().zip(expected) {
        let mut values = line
            .split_whitespace()
            .map(|value| value.parse::<f64>().unwrap());
        assert_eq!(values.next(), Some(x));
        let residual = values.next().unwrap();
        assert!((residual - expected_residual).abs() < 1e-7);
        assert_eq!(values.next(), None);
    }
    assert_eq!(
        state(&harness)
            .selected_layer
            .undo
            .last()
            .map(String::as_str),
        Some(original)
    );
    assert!(state(&harness).selected_layer.redo.is_empty());
    click(&mut harness, "Undo");
    assert_eq!(state(&harness).selected_layer.points_text, original);
}

#[test]
fn point_actions_are_disabled_without_data() {
    let mut harness = harness();
    click(&mut harness, "Actions");
    assert!(
        harness
            .get_by_label("Move to positive x/y")
            .accesskit_node()
            .is_disabled()
    );
    assert!(
        harness
            .get_by_label("Fill with residuals")
            .accesskit_node()
            .is_disabled()
    );
}
