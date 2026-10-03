use super::*;

#[test]
fn default_app_has_one_visible_selected_layer() {
    let harness = harness();
    let snapshot = state(&harness);
    assert_eq!(snapshot.layers().len(), 1);
    assert_eq!(snapshot.selected_layer_index, 0);
    assert_eq!(snapshot.selected_layer.name, "Layer 1");
    assert!(snapshot.selected_layer.visible);
    assert!(snapshot.selected_layer.points_text.is_empty());
    assert!(
        !harness
            .get_by_label("Input points")
            .accesskit_node()
            .is_disabled()
    );
}

#[test]
fn creating_layer_selects_it_and_deleting_last_layer_resets_default() {
    let mut harness = harness();
    click(&mut harness, "New layer");
    assert_eq!(state(&harness).layers().len(), 2);
    assert_eq!(state(&harness).selected_layer_index, 1);
    assert_eq!(state(&harness).selected_layer.name, "Layer 2");
    click(&mut harness, "Delete layer");
    assert_eq!(state(&harness).layers().len(), 1);
    assert_eq!(state(&harness).selected_layer.name, "Layer 1");

    replace_text(&mut harness, Role::TextInput, "Layer name", "Custom");
    points(&mut harness, "0 1\n1 2\n");
    click(&mut harness, "Layer visibility");
    assert!(!state(&harness).selected_layer.visible);
    click(&mut harness, "Delete layer");
    assert_eq!(state(&harness).layers().len(), 1);
    assert_eq!(state(&harness).selected_layer.name, "Layer 1");
    assert!(state(&harness).selected_layer.visible);
    assert!(state(&harness).selected_layer.points_text.is_empty());
}

#[test]
fn selected_layer_only_editing_preserves_other_layers() {
    let mut harness = harness();
    points(&mut harness, "0 1\n1 2\n");
    click(&mut harness, "New layer");
    points(&mut harness, "10 20\n30 40\n");
    assert_eq!(
        state(&harness).layers().next().unwrap().points_text,
        "0 1\n1 2\n"
    );
    assert_eq!(state(&harness).selected_layer.points_text, "10 20\n30 40\n");

    click(&mut harness, "Layer 1");
    assert_eq!(state(&harness).selected_layer_index, 0);
    assert_eq!(
        harness.get_by_label("Input points").value().as_deref(),
        Some("0 1\n1 2\n")
    );
}

#[test]
fn duplicate_layer_copies_points_and_selects_copy() {
    let mut harness = harness();
    replace_text(&mut harness, Role::TextInput, "Layer name", "Source");
    points(&mut harness, "0 1\n1 2\n");
    click(&mut harness, "Layer visibility");
    click(&mut harness, "Duplicate layer");
    assert_eq!(state(&harness).layers().len(), 2);
    assert_eq!(state(&harness).selected_layer_index, 1);
    assert_eq!(state(&harness).selected_layer.name, "Source copy");
    assert!(!state(&harness).selected_layer.visible);
    assert!(!state(&harness).layers().next().unwrap().visible);
    assert_ne!(
        state(&harness).layers().next().unwrap().id,
        state(&harness).selected_layer.id
    );
    assert_eq!(state(&harness).selected_layer.points_text, "0 1\n1 2\n");
    points(&mut harness, "10 20\n30 40\n");
    assert_eq!(
        state(&harness).layers().next().unwrap().points_text,
        "0 1\n1 2\n"
    );
}

#[test]
fn clipboard_import_creates_selected_layer_and_preserves_existing_points() {
    let mut harness = harness();
    points(&mut harness, "0 1\n1 2\n");
    clear_and_undo(&mut harness);
    let original_history_len = state(&harness).selected_layer.undo.len();
    let original_id = state(&harness).selected_layer.id;
    paste_layer(&mut harness, "10\t20\n30;40");
    {
        let snapshot = state(&harness);
        let original = snapshot.layers().next().unwrap();
        assert_eq!(original.id, original_id);
        assert_eq!(original.undo.len(), original_history_len);
        assert_eq!(original.redo, [String::new()]);
    }
    assert_eq!(state(&harness).layers().len(), 2);
    assert_eq!(state(&harness).selected_layer_index, 1);
    assert_eq!(
        state(&harness).layers().next().unwrap().points_text,
        "0 1\n1 2\n"
    );
    assert_eq!(
        state(&harness).selected_layer.points_text,
        "10.00000000 20.00000000\n30.00000000 40.00000000\n"
    );
    assert!(state(&harness).selected_layer.undo.is_empty());
    assert!(state(&harness).selected_layer.redo.is_empty());
    assert!(state(&harness).error.is_none());
}

#[test]
fn clipboard_import_error_keeps_existing_points_text() {
    let mut harness = harness();
    points(&mut harness, "0 1\n1 2\n");
    clear_and_undo(&mut harness);
    let original_history_len = state(&harness).selected_layer.undo.len();
    let original_id = state(&harness).selected_layer.id;
    paste_layer(&mut harness, "1 2 3");
    assert_eq!(state(&harness).selected_layer.id, original_id);
    assert_eq!(
        state(&harness).selected_layer.undo.len(),
        original_history_len
    );
    assert_eq!(state(&harness).selected_layer.redo, [String::new()]);
    assert_eq!(state(&harness).layers().len(), 1);
    assert_eq!(state(&harness).selected_layer.points_text, "0 1\n1 2\n");
    assert!(
        state(&harness)
            .error
            .unwrap()
            .starts_with("Clipboard import error: ")
    );
    assert!(
        harness
            .query_by_label_contains("Clipboard import error")
            .is_some()
    );
}
