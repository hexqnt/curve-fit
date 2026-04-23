use super::common::Vf64;
use super::test_support::assert_family_gradient_and_hessian_match_numerical_reference;
use crate::domain::CurveFamily;

fn long_positive_x_values() -> Vec<f64> {
    (0..(Vf64::LEN + 3))
        .map(|index| 0.2 + index as f64 * 0.35)
        .collect()
}

#[test]
fn analytic_raw_hessian_models_match_numerical_reference_on_simd_chunks_and_tail() {
    let x_values = long_positive_x_values();

    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Arrhenius,
        &x_values,
        &[1.2, -0.2],
        &[1.0, -0.1],
        5e-5,
        8e-4,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Inverse,
        &x_values,
        &[1.0, 0.2],
        &[0.9, 0.15],
        5e-5,
        8e-4,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Logistic,
        &x_values,
        &[2.2, 1.1, 0.3],
        &[1.8, 0.8, -0.1],
        5e-5,
        1e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Gompertz,
        &x_values,
        &[1.9, 0.9, 0.2],
        &[1.4, 0.6, -0.2],
        5e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::BiExponential,
        &x_values,
        &[1.2, 0.7, 0.5, 0.25, -0.3],
        &[0.9, 0.4, 0.4, 0.1, -0.1],
        6e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::ExponentialBasic,
        &x_values,
        &[0.8, 1.4, 0.6],
        &[0.5, 1.1, 0.3],
        5e-5,
        8e-4,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::ExponentialLinear,
        &x_values,
        &[1.4, 0.35, -0.4, 0.2],
        &[1.0, 0.2, -0.2, 0.0],
        6e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Softplus,
        &x_values,
        &[1.3, 0.7, 0.2, 0.2],
        &[1.0, 0.5, -0.1, 0.0],
        6e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Linear,
        &x_values,
        &[1.5, -0.25],
        &[0.3, -0.7],
        5e-5,
        8e-4,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::SaturatingTrendBasis6,
        &x_values,
        &[0.1, 1.0, -0.2, 0.4, 0.1, -0.1, 0.05],
        &[0.0, 0.8, -0.1, 0.2, 0.0, -0.1, 0.0],
        6e-5,
        2e-3,
    );
}

#[test]
fn gradient_only_models_match_numerical_reference_on_simd_chunks_and_tail() {
    let x_values = long_positive_x_values();

    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::DampedSinusoid,
        &x_values,
        &[1.2, 0.4, 1.1, 0.2, -0.1],
        &[0.9, 0.3, 0.8, 0.1, -0.2],
        8e-5,
        3e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Lorentzian,
        &x_values,
        &[1.2, 1.5, 0.7, -0.1],
        &[1.0, 1.2, 0.5, 0.0],
        8e-5,
        3e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::NaturalLog,
        &x_values,
        &[0.8, 1.1],
        &[0.6, 0.9],
        6e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::MichaelisMenten,
        &x_values,
        &[1.8, 0.7],
        &[1.3, 0.5],
        6e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::ExponentialHalfLife,
        &x_values,
        &[0.7, 1.2, 0.9],
        &[0.5, 1.0, 0.7],
        6e-5,
        2e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::FallingExponential,
        &x_values,
        &[1.2, 0.8, 0.6],
        &[1.0, 0.6, 0.4],
        8e-5,
        3e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Gaussian,
        &x_values,
        &[1.1, 1.4, 0.8],
        &[0.9, 1.1, 0.6],
        8e-5,
        3e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Rational11,
        &x_values,
        &[0.7, 1.2, 0.3, -0.1],
        &[0.6, 0.9, 0.2, 0.0],
        8e-5,
        3e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Rational22,
        &x_values,
        &[0.4, 0.7, 1.0, 0.2, 0.1],
        &[0.3, 0.6, 0.8, 0.1, 0.05],
        8e-5,
        3e-3,
    );
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::PseudoVoigt,
        &x_values,
        &[1.0, 1.3, 0.7, 0.6, 0.2, -0.2],
        &[0.8, 1.0, 0.6, 0.5, 0.1, -0.1],
        1e-4,
        4e-3,
    );
}
