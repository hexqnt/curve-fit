use super::*;

#[test]
fn move_points_to_positive_xy_rebases_minimums_and_preserves_offsets() {
    let original = [(-2.0, -1.0), (0.0, 3.0), (5.0, -4.0)];
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: "-2 -1\n0 3\n5 -4\n".to_string(),
        ..Default::default()
    });
    app.invalidate_points_cache();

    app.move_points_to_positive_xy();

    let shifted = parsed_point_pairs(&mut app);
    assert_eq!(shifted.len(), original.len());

    let min_x = shifted
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let min_y = shifted
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    assert_approx_eq(min_x, super::POINTS_POSITIVE_AXIS_EPS, 1e-12);
    assert_approx_eq(min_y, super::POINTS_POSITIVE_AXIS_EPS, 1e-12);

    for index in 1..shifted.len() {
        assert_approx_eq(
            shifted[index].0 - shifted[0].0,
            original[index].0 - original[0].0,
            1e-12,
        );
        assert_approx_eq(
            shifted[index].1 - shifted[0].1,
            original[index].1 - original[0].1,
            1e-12,
        );
    }
}

#[test]
fn move_points_to_positive_xy_keeps_already_positive_points_unchanged() {
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: "2 3\n4 5\n".to_string(),
        ..Default::default()
    });
    app.invalidate_points_cache();

    app.move_points_to_positive_xy();

    let shifted = parsed_point_pairs(&mut app);
    assert_eq!(shifted.len(), 2);
    assert_approx_eq(shifted[0].0, 2.0, 1e-12);
    assert_approx_eq(shifted[0].1, 3.0, 1e-12);
    assert_approx_eq(shifted[1].0, 4.0, 1e-12);
    assert_approx_eq(shifted[1].1, 5.0, 1e-12);
}

#[test]
fn move_points_to_positive_xy_moves_only_axis_that_needs_it() {
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: "-2 3\n1 5\n".to_string(),
        ..Default::default()
    });
    app.invalidate_points_cache();

    app.move_points_to_positive_xy();

    let shifted = parsed_point_pairs(&mut app);
    assert_eq!(shifted.len(), 2);
    assert_approx_eq(shifted[0].0, super::POINTS_POSITIVE_AXIS_EPS, 1e-12);
    assert_approx_eq(shifted[0].1, 3.0, 1e-12);
    assert_approx_eq(shifted[1].0, 3.000001, 1e-12);
    assert_approx_eq(shifted[1].1, 5.0, 1e-12);
}

#[test]
fn move_points_to_positive_xy_pushes_undo_and_clears_redo() {
    let previous_text = "-1 0\n1 2\n";
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: previous_text.to_string(),
        redo_stack: vec!["stale redo entry".to_string()],
        ..Default::default()
    });
    app.invalidate_points_cache();

    app.move_points_to_positive_xy();

    assert_eq!(
        app.selected_points_editor().undo_stack,
        vec![previous_text.to_string()]
    );
    assert!(app.selected_points_editor().redo_stack.is_empty());
    assert_ne!(app.selected_points_editor().text, previous_text);

    let moved_text = app.selected_points_editor().text.clone();
    app.undo_points_edit();
    assert_eq!(app.selected_points_editor().text, previous_text);
    app.redo_points_edit();
    assert_eq!(app.selected_points_editor().text, moved_text);

    let shifted = parsed_point_pairs(&mut app);
    let min_x = shifted
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let min_y = shifted
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    assert_approx_eq(min_x, super::POINTS_POSITIVE_AXIS_EPS, 1e-12);
    assert_approx_eq(min_y, super::POINTS_POSITIVE_AXIS_EPS, 1e-12);
}

#[test]
fn can_move_points_to_positive_xy_requires_non_empty_valid_points() {
    let mut empty = app_with_points_editor_state(super::PointsEditorState {
        text: String::new(),
        ..Default::default()
    });
    empty.invalidate_points_cache();
    assert!(!empty.can_move_points_to_positive_xy());

    let mut invalid = app_with_points_editor_state(super::PointsEditorState {
        text: "1 2 3\n".to_string(),
        ..Default::default()
    });
    invalid.invalidate_points_cache();
    assert!(!invalid.can_move_points_to_positive_xy());

    let mut valid = app_with_points_editor_state(super::PointsEditorState {
        text: "0 0\n1 1\n".to_string(),
        ..Default::default()
    });
    valid.invalidate_points_cache();
    assert!(valid.can_move_points_to_positive_xy());
}

#[test]
fn default_app_has_one_visible_selected_layer() {
    let app = CurveFitApp::default();

    assert_eq!(app.point_layers.layers.len(), 1);
    assert_eq!(app.point_layers.selected_index(), 0);
    assert_eq!(app.selected_layer().name, "Layer 1");
    assert!(app.selected_layer().visible);
    assert!(app.selected_points_editor().text.is_empty());
}

#[test]
fn creating_layer_selects_it_and_deleting_last_layer_resets_default() {
    let mut app = CurveFitApp::default();

    app.create_empty_point_layer();
    assert_eq!(app.point_layers.layers.len(), 2);
    assert_eq!(app.point_layers.selected_index(), 1);
    assert_eq!(app.selected_layer().name, "Layer 2");

    app.delete_selected_point_layer();
    assert_eq!(app.point_layers.layers.len(), 1);
    assert_eq!(app.selected_layer().name, "Layer 1");

    app.selected_layer_mut().name = "Custom".to_string();
    app.selected_layer_mut().visible = false;
    app.write_points_text(
        &[
            Point::try_new(0.0, 1.0).unwrap(),
            Point::try_new(1.0, 2.0).unwrap(),
        ],
        false,
    );
    app.delete_selected_point_layer();

    assert_eq!(app.point_layers.layers.len(), 1);
    assert_eq!(app.selected_layer().name, "Layer 1");
    assert!(app.selected_layer().visible);
    assert!(app.selected_points_editor().text.is_empty());
}

#[test]
fn selected_layer_only_editing_preserves_other_layers() {
    let mut app = CurveFitApp::default();
    app.write_points_text(
        &[
            Point::try_new(0.0, 1.0).unwrap(),
            Point::try_new(1.0, 2.0).unwrap(),
        ],
        false,
    );
    app.create_empty_point_layer();
    app.write_points_text(
        &[
            Point::try_new(10.0, 20.0).unwrap(),
            Point::try_new(30.0, 40.0).unwrap(),
        ],
        false,
    );

    assert_eq!(
        app.point_layers.layers[0].points.text,
        "0.00000000 1.00000000\n1.00000000 2.00000000\n"
    );
    assert_eq!(
        app.selected_points_editor().text,
        "10.00000000 20.00000000\n30.00000000 40.00000000\n"
    );
}

#[test]
fn points_data_change_clears_stale_fit_outputs() {
    let mut app = CurveFitApp {
        fit_result: Some(FitResult {
            family: CurveFamily::Linear,
            params: CurveParams::Linear { a: 1.0, b: 0.0 },
            mse: 0.0,
            rmse: 0.0,
            iterations: 1,
        }),
        last_fit_duration: Some(std::time::Duration::from_millis(42)),
        status: Some(StatusMessage::FitCompleted),
        ..Default::default()
    };

    app.write_points_text(
        &[
            Point::try_new(10.0, 20.0).unwrap(),
            Point::try_new(30.0, 40.0).unwrap(),
        ],
        false,
    );

    assert!(app.fit_result.is_none());
    assert!(app.last_fit_duration.is_none());
    assert!(matches!(app.status, Some(StatusMessage::Ready)));
}

#[test]
fn duplicate_layer_copies_points_and_selects_copy() {
    let mut app = CurveFitApp::default();
    app.selected_layer_mut().name = "Source".to_string();
    app.write_points_text(
        &[
            Point::try_new(0.0, 1.0).unwrap(),
            Point::try_new(1.0, 2.0).unwrap(),
        ],
        false,
    );

    app.duplicate_selected_point_layer();

    assert_eq!(app.point_layers.layers.len(), 2);
    assert_eq!(app.point_layers.selected_index(), 1);
    assert_eq!(app.selected_layer().name, "Source copy");
    assert_eq!(
        app.selected_points_editor().text,
        "0.00000000 1.00000000\n1.00000000 2.00000000\n"
    );
    assert_ne!(app.point_layers.layers[0].id, app.point_layers.layers[1].id);
}

#[test]
fn visible_point_aggregate_excludes_hidden_layers() {
    let mut app = CurveFitApp::default();
    app.write_points_text(
        &[
            Point::try_new(0.0, 1.0).unwrap(),
            Point::try_new(1.0, 2.0).unwrap(),
        ],
        false,
    );
    app.create_empty_point_layer();
    app.write_points_text(
        &[
            Point::try_new(10.0, 20.0).unwrap(),
            Point::try_new(30.0, 40.0).unwrap(),
        ],
        false,
    );
    app.selected_layer_mut().visible = false;

    let points = app
        .parse_visible_points_strict()
        .expect("visible aggregate must parse");

    assert_eq!(points.len(), 2);
    assert_approx_eq(points[0].x(), 0.0, 1e-12);
    assert_approx_eq(points[1].x(), 1.0, 1e-12);
}

#[test]
fn show_only_layer_makes_target_visible_and_hides_others() {
    let mut app = CurveFitApp::default();
    let first_id = app.selected_layer().id;
    app.create_empty_point_layer();
    let second_id = app.selected_layer().id;
    app.create_empty_point_layer();
    let third_id = app.selected_layer().id;
    app.point_layers.layers[1].visible = false;

    assert!(app.point_layers.show_only(second_id));

    assert!(!app.point_layers.layers[0].visible);
    assert!(app.point_layers.layers[1].visible);
    assert!(!app.point_layers.layers[2].visible);
    assert!(!app.point_layers.show_only(second_id));
    assert!(app.point_layers.show_only(first_id));
    assert!(app.point_layers.layers[0].visible);
    assert!(!app.point_layers.layers[1].visible);
    assert!(!app.point_layers.layers[2].visible);
    assert!(app.point_layers.show_only(third_id));
}

#[test]
fn visible_invalid_layer_blocks_aggregate_with_layer_name() {
    let mut app = CurveFitApp::default();
    app.selected_layer_mut().name = "Bad input".to_string();
    set_selected_points_text(&mut app, "1 2 3\n");

    let error = app
        .parse_visible_points_strict()
        .expect_err("visible invalid layer must block fitting input");

    assert!(
        error.contains("Layer 'Bad input':"),
        "unexpected error: {error}"
    );
}

#[test]
fn hidden_invalid_layer_does_not_block_aggregate() {
    let mut app = CurveFitApp::default();
    set_selected_points_text(&mut app, "0 1\n1 2\n");
    app.create_empty_point_layer();
    set_selected_points_text(&mut app, "1 2 3\n");
    app.selected_layer_mut().visible = false;

    let points = app
        .parse_visible_points_strict()
        .expect("hidden invalid layer must be ignored");

    assert_eq!(points.len(), 2);
    assert_approx_eq(points[0].x(), 0.0, 1e-12);
    assert_approx_eq(points[1].x(), 1.0, 1e-12);
}

#[test]
fn points_edit_status_reports_visible_non_selected_layer_error() {
    let mut app = CurveFitApp::default();
    set_selected_points_text(&mut app, "0 1\n1 2\n");
    app.create_empty_point_layer();
    app.selected_layer_mut().name = "Bad input".to_string();
    set_selected_points_text(&mut app, "1 2 3\n");
    app.point_layers.select(app.point_layers.layers[0].id);

    app.refresh_status_after_points_edit();

    assert!(matches!(
        app.status.as_ref(),
        Some(StatusMessage::Error(message))
            if message.contains("Layer 'Bad input': Line 1")
    ));
}

#[test]
fn points_edit_status_ignores_hidden_invalid_layer() {
    let mut app = CurveFitApp::default();
    set_selected_points_text(&mut app, "0 1\n1 2\n");
    app.create_empty_point_layer();
    set_selected_points_text(&mut app, "1 2 3\n");
    app.selected_layer_mut().visible = false;

    app.refresh_status_after_points_edit();

    assert!(matches!(app.status, Some(StatusMessage::Ready)));
}

#[test]
fn editing_selected_layer_keeps_other_visible_layer_parse_error() {
    let mut app = CurveFitApp::default();
    set_selected_points_text(&mut app, "0 1\n1 2\n");
    let valid_layer_id = app.selected_layer().id;
    app.create_empty_point_layer();
    app.selected_layer_mut().name = "Bad input".to_string();
    set_selected_points_text(&mut app, "1 2 3\n");
    app.point_layers.select(valid_layer_id);
    app.refresh_status_after_points_edit();

    app.write_points_text(
        &[
            Point::try_new(2.0, 3.0).unwrap(),
            Point::try_new(3.0, 4.0).unwrap(),
        ],
        true,
    );

    assert!(matches!(
        app.status.as_ref(),
        Some(StatusMessage::Error(message))
            if message.contains("Layer 'Bad input': Line 1")
    ));
}

#[test]
fn empty_layer_name_uses_stable_display_name_in_errors() {
    let mut app = CurveFitApp::default();
    app.selected_layer_mut().name = "   ".to_string();
    set_selected_points_text(&mut app, "1 2 3\n");

    let error = app
        .parse_visible_points_strict()
        .expect_err("visible invalid layer must block fitting input");

    assert!(
        error.contains("Layer 'Unnamed layer':"),
        "unexpected error: {error}"
    );
}
