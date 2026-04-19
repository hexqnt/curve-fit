use super::*;

#[test]
fn move_points_to_positive_xy_rebases_minimums_and_preserves_offsets() {
    let original = [(-2.0, -1.0), (0.0, 3.0), (5.0, -4.0)];
    let mut app = CurveFitApp {
        points: super::PointsEditorState {
            text: "-2 -1\n0 3\n5 -4\n".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
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
    let mut app = CurveFitApp {
        points: super::PointsEditorState {
            text: "2 3\n4 5\n".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
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
    let mut app = CurveFitApp {
        points: super::PointsEditorState {
            text: "-2 3\n1 5\n".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
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
    let mut app = CurveFitApp {
        points: super::PointsEditorState {
            text: previous_text.to_string(),
            redo_stack: vec!["stale redo entry".to_string()],
            ..Default::default()
        },
        ..Default::default()
    };
    app.invalidate_points_cache();

    app.move_points_to_positive_xy();

    assert_eq!(app.points.undo_stack, vec![previous_text.to_string()]);
    assert!(app.points.redo_stack.is_empty());
    assert_ne!(app.points.text, previous_text);

    let moved_text = app.points.text.clone();
    app.undo_points_edit();
    assert_eq!(app.points.text, previous_text);
    app.redo_points_edit();
    assert_eq!(app.points.text, moved_text);

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
    let mut empty = CurveFitApp {
        points: super::PointsEditorState {
            text: String::new(),
            ..Default::default()
        },
        ..Default::default()
    };
    empty.invalidate_points_cache();
    assert!(!empty.can_move_points_to_positive_xy());

    let mut invalid = CurveFitApp {
        points: super::PointsEditorState {
            text: "1 2 3\n".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    invalid.invalidate_points_cache();
    assert!(!invalid.can_move_points_to_positive_xy());

    let mut valid = CurveFitApp {
        points: super::PointsEditorState {
            text: "0 0\n1 1\n".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    valid.invalidate_points_cache();
    assert!(valid.can_move_points_to_positive_xy());
}
