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
