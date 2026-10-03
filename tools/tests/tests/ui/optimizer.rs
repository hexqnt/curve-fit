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
