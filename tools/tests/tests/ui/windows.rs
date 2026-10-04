use super::*;

#[test]
fn formula_window_opens_and_closes() {
    let mut harness = harness();
    click(&mut harness, "Formula");
    assert!(state(&harness).show_formula_window);
    assert!(
        harness
            .query_by_role_and_label(Role::Window, "Model Formula")
            .is_some()
    );
    click(&mut harness, "Close window");
    assert!(!state(&harness).show_formula_window);
    assert!(
        harness
            .query_by_role_and_label(Role::Window, "Model Formula")
            .is_none()
    );
}

#[test]
fn panels_menu_changes_panel_visibility() {
    let mut harness = harness();
    click(&mut harness, "Panels");
    click(&mut harness, "Left panel");
    assert!(!state(&harness).show_left_panel);
    assert!(harness.query_by_label("Input points").is_none());
    click(&mut harness, "Panels");
    click(&mut harness, "Right panel");
    assert!(!state(&harness).show_right_panel);
    assert!(
        harness
            .query_by_role_and_label(Role::ComboBox, "Method")
            .is_none()
    );
    click(&mut harness, "Panels");
    click(&mut harness, "Diagnostics");
    assert!(!state(&harness).show_diagnostics);
    assert!(
        harness
            .query_by_role_and_label(Role::Button, "Residuals")
            .is_none()
    );
    click(&mut harness, "Panels");
    click(&mut harness, "Left panel");
    assert!(state(&harness).show_left_panel);
    assert!(harness.query_by_label("Input points").is_some());
}

#[test]
fn language_menu_changes_visible_labels() {
    let mut harness = harness();
    click(&mut harness, "English");
    click(&mut harness, "Русский");
    assert!(harness.query_by_label("Формула").is_some());
    assert!(harness.query_by_label("Новый слой").is_some());
    assert!(harness.query_by_label("Ввод точек").is_some());
    assert!(harness.query_by_label("Input points").is_none());
    click(&mut harness, "Русский");
    click(&mut harness, "English");
    assert!(harness.query_by_label("Input points").is_some());
    assert!(harness.query_by_label("Ввод точек").is_none());
}

#[test]
fn themes_preserve_editor_errors_and_formula_window_controls() {
    let mut harness = harness();
    for theme in ["☀ Light", "🌙 Dark"] {
        click(&mut harness, theme);
        points(&mut harness, "0 1\ninvalid\n");
        click(&mut harness, "Fit");
        assert!(
            harness
                .query_by_label_contains("Points parse error:")
                .is_some()
        );
        points(&mut harness, "0 1\n1 3\n");
        fit_and_wait(&mut harness);
        assert!(
            harness
                .query_by_label_contains("Points parse error:")
                .is_none()
        );
        click(&mut harness, "Formula");
        assert!(state(&harness).show_formula_window);
        assert!(
            harness
                .query_by_role_and_label(Role::Button, "Copy model formula")
                .is_some()
        );
        click(&mut harness, "Close window");
        assert!(!state(&harness).show_formula_window);
    }
}

#[test]
fn tooltips_show_toolbar_details_and_single_line_help() {
    let mut harness = harness();
    harness.get_by_label("New layer").hover();
    harness.run_steps(60);
    assert!(
        harness
            .query_by_label("- Add an empty point layer and select it")
            .is_some()
    );
    harness.get_by_label("Formula").hover();
    harness.run_steps(60);
    assert!(
        harness
            .query_by_label("Toggle model formula window")
            .is_some()
    );
    harness.get_by_label("Basic").hover();
    harness.run_steps(60);
    assert!(harness.query_by_label("Basic mode").is_some());
    assert!(
        harness
            .query_by_label("- Choose a preset to balance speed vs stability")
            .is_some()
    );
}

#[test]
fn keyboard_toggles_panels_and_moves_focus_between_switches() {
    let mut harness = harness();
    click(&mut harness, "Panels");
    harness.get_by_label("Left panel").focus();
    harness.run();
    harness.key_press(Key::Space);
    harness.run();
    assert!(!state(&harness).show_left_panel);
    harness.key_press(Key::Space);
    harness.run();
    assert!(state(&harness).show_left_panel);
    harness.key_press(Key::Tab);
    harness.run();
    assert!(harness.get_by_label("Right panel").is_focused());
    harness.key_press(Key::Space);
    harness.run();
    assert!(!state(&harness).show_right_panel);
}

#[test]
fn narrow_window_keeps_russian_controls_reachable() {
    let mut harness = harness_with_size(egui::vec2(900.0, 700.0));
    click(&mut harness, "English");
    click(&mut harness, "Русский");
    click(&mut harness, "Панели");
    click(&mut harness, "Левая панель");
    assert!(!state(&harness).show_left_panel);
    assert!(
        harness
            .query_by_role_and_label(Role::ComboBox, "Метод")
            .is_some()
    );
    harness.get_by_label("Панели").click();
    harness.run();
    click(&mut harness, "Правая панель");
    assert!(!state(&harness).show_right_panel);
    harness.get_by_label("Формула").scroll_to_me();
    harness.run();
    click(&mut harness, "Формула");
    assert!(state(&harness).show_formula_window);
    click(&mut harness, "Close window");
    assert!(!state(&harness).show_formula_window);
}
