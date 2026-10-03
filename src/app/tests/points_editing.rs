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
