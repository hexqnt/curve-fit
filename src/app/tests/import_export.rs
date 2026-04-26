use super::*;

#[test]
fn format_fit_duration_uses_expected_units_at_boundary() {
    assert_eq!(
        CurveFitApp::format_fit_duration(std::time::Duration::from_millis(999)),
        "999 ms"
    );
    assert_eq!(
        CurveFitApp::format_fit_duration(std::time::Duration::from_millis(1_000)),
        "1.00 s"
    );
}

#[test]
fn fill_points_with_residuals_replaces_points_text_and_pushes_undo() {
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: "0 1\n1 2\n".to_string(),
        ..Default::default()
    });
    app.residual_plot_points = vec![PlotPoint::new(0.0, -0.5), PlotPoint::new(1.0, 0.25)];

    app.fill_points_with_residuals();

    assert_eq!(
        app.selected_points_editor().text,
        "0.00000000 -0.50000000\n1.00000000 0.25000000\n"
    );
    assert_eq!(
        app.selected_points_editor().undo_stack,
        vec!["0 1\n1 2\n".to_string()]
    );
    assert!(app.selected_points_editor().redo_stack.is_empty());
}

#[test]
fn fill_points_with_residuals_is_noop_when_residuals_are_absent() {
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: "0 1\n1 2\n".to_string(),
        ..Default::default()
    });

    app.fill_points_with_residuals();

    assert_eq!(app.selected_points_editor().text, "0 1\n1 2\n");
    assert!(app.selected_points_editor().undo_stack.is_empty());
    assert!(app.selected_points_editor().redo_stack.is_empty());
}

#[test]
fn clipboard_import_creates_selected_layer_and_preserves_existing_points() {
    let previous_text = "0 1\n1 2\n";
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: previous_text.to_string(),
        redo_stack: vec!["stale redo entry".to_string()],
        ..Default::default()
    });
    app = CurveFitApp {
        status: Some(StatusMessage::Error(format!(
            "{}previous error",
            super::CLIPBOARD_IMPORT_ERROR_PREFIX
        ))),
        ..app
    };

    app.handle_points_clipboard_import_result(Ok("10\t20\n30;40".to_string()));

    assert_eq!(app.point_layers.layers.len(), 2);
    assert_eq!(app.point_layers.selected_index(), 1);
    assert_eq!(app.point_layers.layers[0].points.text, previous_text);
    assert_eq!(
        app.selected_points_editor().text,
        "10.00000000 20.00000000\n30.00000000 40.00000000\n"
    );
    assert!(app.point_layers.layers[0].points.undo_stack.is_empty());
    assert!(app.selected_points_editor().undo_stack.is_empty());
    assert!(app.selected_points_editor().redo_stack.is_empty());
    assert!(matches!(app.status, Some(StatusMessage::Ready)));
}

#[test]
fn clipboard_import_error_keeps_existing_points_text() {
    let previous_text = "0 1\n1 2\n";
    let mut app = app_with_points_editor_state(super::PointsEditorState {
        text: previous_text.to_string(),
        ..Default::default()
    });

    app.handle_points_clipboard_import_result(Ok("1 2 3".to_string()));

    assert_eq!(app.selected_points_editor().text, previous_text);
    assert!(app.selected_points_editor().undo_stack.is_empty());
    assert!(matches!(
        app.status.as_ref(),
        Some(StatusMessage::Error(message)) if message.starts_with(super::CLIPBOARD_IMPORT_ERROR_PREFIX)
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn points_file_import_remembers_last_directory_from_selected_file() {
    let mut app = CurveFitApp::default();
    let path = write_temp_points_csv(b"1;2\n3;4\n");
    let expected_directory = path
        .parent()
        .expect("temporary test file must have parent directory")
        .to_path_buf();

    app.handle_points_file_import_path(&path);
    cleanup_temp_file(&path);

    assert_eq!(
        app.points_file_import_last_directory,
        Some(expected_directory)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn points_file_import_dialog_uses_remembered_directory() {
    let mut app = CurveFitApp::default();
    let remembered_directory = std::env::temp_dir();
    app.points_file_import_last_directory = Some(remembered_directory.clone());

    app.request_points_file_import();

    assert_eq!(
        app.points_file_import_dialog.config_mut().initial_directory,
        remembered_directory
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fit_export_save_dialog_uses_remembered_directory() {
    let mut app = CurveFitApp::default();
    let remembered_directory = std::env::temp_dir();
    let result = FitResult {
        family: CurveFamily::Linear,
        params: CurveParams::Linear { a: 2.0, b: 1.0 },
        mse: 0.01,
        rmse: 0.1,
        iterations: 42,
    };
    app.store_parametric_fit_export_record(&result, 2);
    app.fit_export_last_directory = Some(remembered_directory.clone());

    app.request_fit_export_save_json();

    assert_eq!(
        app.fit_export_file_dialog.config_mut().initial_directory,
        remembered_directory
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn dialog_directory_from_path_returns_parent_for_file_path() {
    let mut output_path = std::env::temp_dir();
    output_path.push("curve-fit-export.json");
    let expected_directory = output_path
        .parent()
        .expect("temporary output path must have parent")
        .to_path_buf();

    let actual_directory = super::dialog_directory_from_path(&output_path);

    assert_eq!(actual_directory, Some(expected_directory));
}
