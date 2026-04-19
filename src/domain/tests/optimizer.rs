use super::*;

#[test]
fn lbfgs_config_validates_constraints() {
    let result = LbfgsConfig::try_new(0, 100, 1e-6, 1e-8, 1e-4, 0.9, 1e-6, 10.0, 1e-10);
    assert!(result.is_err());

    let result = LbfgsConfig::try_new(5, 100, 1e-6, 1e-8, 0.95, 0.9, 1e-6, 10.0, 1e-10);
    assert!(result.is_err());

    let result = LbfgsConfig::try_new(5, 100, 1e-6, 1e-8, 1e-4, 0.9, 10.0, 1.0, 1e-10);
    assert!(result.is_err());
}

#[test]
fn nelder_mead_config_validates_constraints() {
    let result = NelderMeadConfig::try_new(0, 0.1, 1e-8, 1.0, 2.0, 0.5, 0.5);
    assert!(result.is_err());

    let result = NelderMeadConfig::try_new(200, 0.0, 1e-8, 1.0, 2.0, 0.5, 0.5);
    assert!(result.is_err());

    let result = NelderMeadConfig::try_new(200, 0.1, 1e-8, 1.0, 1.0, 0.5, 0.5);
    assert!(result.is_err());

    let result = NelderMeadConfig::try_new(200, 0.1, 1e-8, 1.0, 2.0, 0.0, 0.5);
    assert!(result.is_err());
}

#[test]
fn steepest_descent_config_validates_constraints() {
    let result = SteepestDescentConfig::try_new(0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
    assert!(result.is_err());

    let result = SteepestDescentConfig::try_new(100, 0.9, 0.9, 1e-12, 10.0, 1e-10);
    assert!(result.is_err());

    let result = SteepestDescentConfig::try_new(100, 1e-4, 0.9, 10.0, 1.0, 1e-10);
    assert!(result.is_err());
}

#[test]
fn newton_cg_config_validates_constraints() {
    let result = NewtonCgConfig::try_new(0, 1e-8, 0.0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
    assert!(result.is_err());

    let result = NewtonCgConfig::try_new(100, 0.0, 0.0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
    assert!(result.is_err());

    let result = NewtonCgConfig::try_new(100, 1e-8, -1.0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
    assert!(result.is_err());

    let result = NewtonCgConfig::try_new(100, 1e-8, 0.0, 0.9, 0.9, 1e-12, 10.0, 1e-10);
    assert!(result.is_err());

    let result = NewtonCgConfig::try_new(100, 1e-8, 0.0, 1e-4, 0.9, 10.0, 1.0, 1e-10);
    assert!(result.is_err());
}

#[test]
fn sgd_config_validates_constraints() {
    let result = SgdConfig::try_new(0, 1e-2);
    assert!(result.is_err());

    let result = SgdConfig::try_new(100, 0.0);
    assert!(result.is_err());

    let result = SgdConfig::try_new(100, f64::NAN);
    assert!(result.is_err());
}

#[test]
fn adam_config_validates_constraints() {
    let result = AdamConfig::try_new(0, 1e-3);
    assert!(result.is_err());

    let result = AdamConfig::try_new(100, -1e-3);
    assert!(result.is_err());

    let result = AdamConfig::try_new(100, f64::INFINITY);
    assert!(result.is_err());
}
