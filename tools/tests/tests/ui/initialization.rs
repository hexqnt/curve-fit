use super::*;

fn cubic(harness: &mut AppHarness) {
    harness
        .get_by_role_and_label(Role::Slider, "Degree")
        .focus();
    harness.run();
    for _ in 0..2 {
        harness.key_press(Key::ArrowRight);
        harness.run();
    }
    assert_eq!(state(harness).parameter_inputs.len(), 4);
}

fn initialize(harness: &mut AppHarness, method: &str) {
    click(harness, "+ Initialize");
    click(harness, method);
}

fn assert_fit_cleared(harness: &AppHarness) {
    let snapshot = state(harness);
    assert!(snapshot.fit_result.is_none());
    assert!(snapshot.replay_iteration.is_none());
    assert_eq!(snapshot.replay_frames, 0);
    assert_eq!(snapshot.diagnostic_iterations, 0);
    assert_eq!(
        harness
            .get_by_role_and_label(Role::Button, "Residuals")
            .accesskit_node()
            .toggled(),
        Some(egui::accesskit::Toggled::True)
    );
}

#[test]
fn apply_param_init_updates_inputs_and_clears_fit_state() {
    let mut harness = harness();
    cubic(&mut harness);
    points(&mut harness, "0 1\n1 3\n2 5\n3 7\n");
    fit_and_wait(&mut harness);
    click(&mut harness, "Residuals");
    initialize(&mut harness, "Data-based");
    let parameters = state(&harness).parameter_inputs;
    assert_eq!(parameters.len(), 4);
    for (text, expected) in parameters.iter().zip([0.0, 0.0, 2.0, 1.0]) {
        assert_eq!(text.parse::<f64>().unwrap(), expected);
    }
    assert_fit_cleared(&harness);
}

#[test]
fn apply_fitted_param_init_updates_inputs_and_clears_fit_state() {
    let mut harness = harness();
    cubic(&mut harness);
    points(&mut harness, "0 3\n1 6.25\n2 18\n3 47.25\n4 103\n");
    fit_and_wait(&mut harness);
    let values = state(&harness).fit_result.unwrap().params.values();
    assert_eq!(values.len(), 4);
    click(&mut harness, "Residuals");
    initialize(&mut harness, "From fitted model");
    for (text, expected) in state(&harness).parameter_inputs.iter().zip(values) {
        assert!((text.parse::<f64>().unwrap() - expected).abs() < 1e-7);
    }
    assert_fit_cleared(&harness);
    assert!(state(&harness).error.is_none());
}

#[test]
fn apply_param_init_sets_error_status_on_failure() {
    let mut harness = harness();
    choose(&mut harness, "Model type", "Power");
    points(&mut harness, "1 0\n2 2\n");
    initialize(&mut harness, "Data-based");
    assert!(state(&harness).error.unwrap().contains("requires y > 0"));
}

#[test]
fn fitted_initialization_is_disabled_without_a_fit() {
    let mut harness = harness();
    click(&mut harness, "+ Initialize");
    assert!(
        harness
            .get_by_label("From fitted model (fit this model first)")
            .accesskit_node()
            .is_disabled()
    );
}

#[test]
fn parameter_inputs_keep_focus_and_values_across_model_controls() {
    let mut harness = harness();
    replace_text(&mut harness, Role::TextInput, "a", "1.25");
    assert_eq!(state(&harness).parameter_inputs[0], "1.25");
    choose(&mut harness, "Model type", "Saturating Trend Basis");
    replace_text(&mut harness, Role::TextInput, "tau1", "0.125");
    assert_eq!(
        harness
            .get_by_role_and_label(Role::TextInput, "tau1")
            .value()
            .as_deref(),
        Some("0.125")
    );
    click(&mut harness, "Reset tau grid");
    assert_eq!(
        harness
            .get_by_role_and_label(Role::TextInput, "tau1")
            .value()
            .as_deref(),
        Some("0.25")
    );
    choose(&mut harness, "Model type", "Linear Spline");
    replace_text(&mut harness, Role::TextInput, "knot_y[0]", "2.5");
    assert_eq!(
        harness
            .get_by_role_and_label(Role::TextInput, "knot_y[0]")
            .value()
            .as_deref(),
        Some("2.5")
    );
    initialize(&mut harness, "Default");
    assert_eq!(
        harness
            .get_by_role_and_label(Role::TextInput, "knot_y[0]")
            .value()
            .as_deref(),
        Some("0")
    );
}
