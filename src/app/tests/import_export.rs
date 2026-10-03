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
