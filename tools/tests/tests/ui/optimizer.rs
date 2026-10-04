use super::*;

#[test]
fn optimizer_presets_are_stored_per_method() {
    let mut harness = harness();
    let methods = [
        ("LBFGS", "Fast"),
        ("Nelder-Mead", "Precise"),
        ("Steepest Descent", "Fast"),
        ("Newton-CG", "Precise"),
        ("SGD", "Balanced"),
        ("Adam", "Precise"),
    ];
    for (method, preset) in methods {
        choose(&mut harness, "Method", method);
        choose(&mut harness, "Preset", preset);
    }
    for (method, preset) in methods {
        choose(&mut harness, "Method", method);
        assert_eq!(
            harness
                .get_by_role_and_label(Role::ComboBox, "Preset")
                .value()
                .as_deref(),
            Some(preset)
        );
    }
}

fn configs(label: &str, method: OptimizerMethod) -> (OptimizerConfig, OptimizerConfig) {
    let mut harness = harness();
    choose(&mut harness, "Method", label);
    assert_eq!(state(&harness).optimizer_method, method);
    choose(&mut harness, "Preset", "Fast");
    let fast = testing::optimizer_config(harness.state()).expect("Fast preset must be valid");
    choose(&mut harness, "Preset", "Precise");
    (
        fast,
        testing::optimizer_config(harness.state()).expect("Precise preset must be valid"),
    )
}

#[test]
fn optimizer_preset_changes_active_config_values() {
    let (OptimizerConfig::NelderMead(fast), OptimizerConfig::NelderMead(precise)) =
        configs("Nelder-Mead", OptimizerMethod::NelderMead)
    else {
        panic!("Nelder-Mead must remain active")
    };
    assert!(precise.max_iters() > fast.max_iters());
}

#[test]
fn sgd_preset_changes_active_config_values() {
    let (OptimizerConfig::Sgd(fast), OptimizerConfig::Sgd(precise)) =
        configs("SGD", OptimizerMethod::Sgd)
    else {
        panic!("SGD must remain active")
    };
    assert!(precise.max_iters() > fast.max_iters());
    assert!(precise.learning_rate() < fast.learning_rate());
}

#[test]
fn newton_cg_preset_changes_active_config_values() {
    let (OptimizerConfig::NewtonCg(fast), OptimizerConfig::NewtonCg(precise)) =
        configs("Newton-CG", OptimizerMethod::NewtonCg)
    else {
        panic!("Newton-CG must remain active")
    };
    assert!(precise.max_iters() > fast.max_iters());
    assert!(precise.tol() < fast.tol());
}

#[test]
fn adam_preset_changes_active_config_values() {
    let (OptimizerConfig::Adam(fast), OptimizerConfig::Adam(precise)) =
        configs("Adam", OptimizerMethod::Adam)
    else {
        panic!("Adam must remain active")
    };
    assert!(precise.max_iters() > fast.max_iters());
    assert!(precise.learning_rate() < fast.learning_rate());
}

#[test]
fn collapsed_optimizer_runs_fit_and_preserves_result_on_reopen() {
    let mut harness = harness();
    points(&mut harness, "0 1\n1 3\n2 5\n3 7\n");
    click(&mut harness, "Optimizer");
    assert!(
        harness
            .query_by_role_and_label(Role::ComboBox, "Method")
            .is_none()
    );
    fit_and_wait(&mut harness);
    assert!(harness.query_by_label("MSE").is_some());
    click(&mut harness, "Optimizer");
    assert!(
        harness
            .query_by_role_and_label(Role::ComboBox, "Method")
            .is_some()
    );
    assert!(state(&harness).fit_result.is_some());
    assert!(
        harness
            .query_by_role_and_label(Role::Button, "Fit")
            .is_some()
    );
}

fn stop_and_restart_fit(model: Option<&str>, collapsed: bool) {
    let (mut harness, pause) = harness_with_paused_fit();
    if let Some(model) = model {
        choose(&mut harness, "Model type", model);
    }
    let input = if model.is_some() { "knot_y[0]" } else { "a" };
    let before = harness
        .get_by_role_and_label(Role::TextInput, input)
        .value();
    let samples = "0 1\n1 3\n2 5\n3 7\n4 9\n5 11\n6 13\n7 15\n";
    points(&mut harness, samples);
    if collapsed {
        click(&mut harness, "Optimizer");
    }
    start_paused_fit(&mut harness);
    for (role, label) in [
        (Role::TextInput, input),
        (Role::MultilineTextInput, "Input points"),
        (Role::Button, "New layer"),
        (Role::Button, "Clear points"),
    ] {
        assert!(
            harness
                .get_by_role_and_label(role, label)
                .accesskit_node()
                .is_disabled(),
            "{label} must be disabled during fitting"
        );
    }
    harness
        .get_by_role_and_label(Role::TextInput, input)
        .click();
    harness.step();
    harness
        .get_by_role_and_label(Role::TextInput, input)
        .type_text("99");
    harness.step();
    assert_eq!(
        harness
            .get_by_role_and_label(Role::TextInput, input)
            .value(),
        before
    );
    assert_eq!(state(&harness).selected_layer.points_text, samples);

    harness.get_by_role_and_label(Role::Button, "Stop").click();
    harness.step();
    pause.resume();
    wait_for_fit(&mut harness);
    assert!(harness.query_by_label("Fit stopped").is_some());
    let snapshot = state(&harness);
    assert!(snapshot.fit_result.is_none());
    assert!(snapshot.spline_result.is_none());
    assert_eq!(snapshot.replay_frames, 0);
    assert!(
        !harness
            .get_by_role_and_label(Role::TextInput, input)
            .accesskit_node()
            .is_disabled()
    );
    assert!(
        !harness
            .get_by_role_and_label(Role::MultilineTextInput, "Input points")
            .accesskit_node()
            .is_disabled()
    );
    fit_and_wait(&mut harness);
}

#[test]
fn fit_stop_restores_editing_and_allows_restart() {
    stop_and_restart_fit(None, false);
}

#[test]
fn collapsed_fit_stop_restores_editing_and_allows_restart() {
    stop_and_restart_fit(None, true);
}

#[test]
fn spline_fit_stop_restores_editing_and_allows_restart() {
    stop_and_restart_fit(Some("Linear Spline"), false);
}

#[test]
fn auto_refit_runs_after_parameter_edit_and_can_be_disabled() {
    let (mut harness, pause) = harness_with_paused_fit();
    points(&mut harness, "0 1\n1 3\n2 5\n3 7\n");
    click(&mut harness, "Auto-play");
    click(&mut harness, "Auto-refit");
    assert!(state(&harness).auto_refit_enabled);
    harness.get_by_role_and_label(Role::TextInput, "a").click();
    harness.run();
    harness.key_press_modifiers(Modifiers::COMMAND, Key::A);
    harness
        .get_by_role_and_label(Role::TextInput, "a")
        .type_text("0.5");
    harness.run_steps(2);
    assert_eq!(&state(&harness).parameter_inputs[0], "0.5");
    assert!(state(&harness).fit_in_progress);
    pause.resume();
    wait_for_fit(&mut harness);
    assert!(state(&harness).fit_result.is_some());
    click(&mut harness, "Auto-refit");
    assert!(!state(&harness).auto_refit_enabled);
    replace_text(&mut harness, Role::TextInput, "a", "0.25");
    assert!(!state(&harness).fit_in_progress);
}

#[test]
fn quantization_toggle_controls_decimal_places() {
    let mut harness = harness();
    let toggle = "Quantize objective/metrics before residual";
    assert!(
        harness
            .get_by_role_and_label(Role::Slider, "Decimal places")
            .accesskit_node()
            .is_disabled()
    );
    click(&mut harness, toggle);
    assert!(state(&harness).metric_quantization_enabled);
    let slider = harness.get_by_role_and_label(Role::Slider, "Decimal places");
    assert!(!slider.accesskit_node().is_disabled());
    let before = slider.accesskit_node().numeric_value();
    slider.focus();
    harness.run();
    harness.key_press(Key::ArrowRight);
    harness.run();
    assert_ne!(
        harness
            .get_by_role_and_label(Role::Slider, "Decimal places")
            .accesskit_node()
            .numeric_value(),
        before
    );
    click(&mut harness, toggle);
    assert!(!state(&harness).metric_quantization_enabled);
    assert!(
        harness
            .get_by_role_and_label(Role::Slider, "Decimal places")
            .accesskit_node()
            .is_disabled()
    );
}
