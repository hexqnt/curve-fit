use super::common::HESSIAN_DIAGONAL_JITTER;
use super::{GradientComputation, accumulate_gradient, analytic_hessian, evaluate_raw};
use crate::domain::CurveFamily;
use ndarray::Array2;

const GRADIENT_REL_STEP: f64 = 1e-6;
const GRADIENT_MIN_STEP: f64 = 1e-7;
const HESSIAN_REL_STEP: f64 = 2e-4;
const HESSIAN_MIN_STEP: f64 = 1e-6;

fn assert_near(actual: f64, expected: f64, epsilon: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= epsilon,
        "expected {expected}, got {actual}, delta={delta}, epsilon={epsilon}"
    );
}

fn soft_l1_value(prediction: f64, target: f64) -> f64 {
    let residual = prediction - target;
    2.0 * ((1.0 + residual * residual).sqrt() - 1.0)
}

fn soft_l1_derivative(prediction: f64, target: f64) -> f64 {
    let residual = prediction - target;
    2.0 * residual / (1.0 + residual * residual).sqrt()
}

fn soft_l1_second_derivative(prediction: f64, target: f64) -> f64 {
    let residual = prediction - target;
    2.0 / (1.0 + residual * residual).powf(1.5)
}

fn mean_soft_l1_cost(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
) -> f64 {
    let total = x_values
        .iter()
        .zip(y_values.iter())
        .map(|(&x, &y)| {
            let prediction = evaluate_raw(family, param, x);
            soft_l1_value(prediction, y)
        })
        .sum::<f64>();
    total / x_values.len() as f64
}

fn numerical_gradient_from_cost<F>(param: &[f64], mut cost: F) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let mut gradient = vec![0.0; param.len()];
    let mut probe = param.to_vec();

    for index in 0..param.len() {
        let step = ((param[index].abs() + 1.0) * GRADIENT_REL_STEP).max(GRADIENT_MIN_STEP);
        probe[index] = param[index] + step;
        let cost_plus = cost(&probe);
        probe[index] = param[index] - step;
        let cost_minus = cost(&probe);
        probe[index] = param[index];
        gradient[index] = (cost_plus - cost_minus) / (2.0 * step);
    }

    gradient
}

fn numerical_hessian_from_cost<F>(param: &[f64], mut cost: F) -> Array2<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let dimension = param.len();
    let mut hessian = Array2::zeros((dimension, dimension));
    let mut probe = param.to_vec();
    let base_cost = cost(param);

    for row in 0..dimension {
        let step_row = ((param[row].abs() + 1.0) * HESSIAN_REL_STEP).max(HESSIAN_MIN_STEP);

        probe[row] = param[row] + step_row;
        let cost_plus = cost(&probe);
        probe[row] = param[row] - step_row;
        let cost_minus = cost(&probe);
        probe[row] = param[row];
        hessian[[row, row]] = (cost_plus - 2.0 * base_cost + cost_minus) / (step_row * step_row);

        for column in (row + 1)..dimension {
            let step_column =
                ((param[column].abs() + 1.0) * HESSIAN_REL_STEP).max(HESSIAN_MIN_STEP);

            probe[row] = param[row] + step_row;
            probe[column] = param[column] + step_column;
            let cost_pp = cost(&probe);

            probe[column] = param[column] - step_column;
            let cost_pm = cost(&probe);

            probe[row] = param[row] - step_row;
            probe[column] = param[column] + step_column;
            let cost_mp = cost(&probe);

            probe[column] = param[column] - step_column;
            let cost_mm = cost(&probe);

            probe[row] = param[row];
            probe[column] = param[column];

            let mixed = (cost_pp - cost_pm - cost_mp + cost_mm) / (4.0 * step_row * step_column);
            hessian[[row, column]] = mixed;
            hessian[[column, row]] = mixed;
        }
    }

    for index in 0..dimension {
        if !hessian[[index, index]].is_finite() {
            hessian[[index, index]] = 0.0;
        }
        hessian[[index, index]] += HESSIAN_DIAGONAL_JITTER;
    }

    hessian
}

fn assert_family_gradient_and_hessian_match_numerical_reference(
    family: CurveFamily,
    x_values: &[f64],
    true_params: &[f64],
    probe_params: &[f64],
    gradient_epsilon: f64,
    hessian_epsilon: f64,
) {
    let y_values = x_values
        .iter()
        .map(|&x| evaluate_raw(family, true_params, x))
        .collect::<Vec<_>>();

    let mut analytic_gradient = vec![0.0; probe_params.len()];
    let mut loss_derivative = |prediction: f64, target: f64| soft_l1_derivative(prediction, target);
    let mode = accumulate_gradient(
        family,
        x_values,
        &y_values,
        probe_params,
        &mut loss_derivative,
        &mut analytic_gradient,
    );
    assert_eq!(mode, GradientComputation::Analytic);

    let sample_scale = 1.0 / x_values.len() as f64;
    for value in &mut analytic_gradient {
        *value *= sample_scale;
    }

    let numerical_gradient = numerical_gradient_from_cost(probe_params, |param| {
        mean_soft_l1_cost(family, x_values, &y_values, param)
    });

    for index in 0..probe_params.len() {
        assert_near(
            analytic_gradient[index],
            numerical_gradient[index],
            gradient_epsilon,
        );
    }

    let mut loss_first = |prediction: f64, target: f64| soft_l1_derivative(prediction, target);
    let mut loss_second =
        |prediction: f64, target: f64| soft_l1_second_derivative(prediction, target);
    let analytic_hessian = analytic_hessian(
        family,
        x_values,
        &y_values,
        probe_params,
        &mut loss_first,
        &mut loss_second,
    )
    .expect("analytic hessian must be available");

    let numerical_hessian = numerical_hessian_from_cost(probe_params, |param| {
        mean_soft_l1_cost(family, x_values, &y_values, param)
    });

    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(
                analytic_hessian[[row, column]],
                numerical_hessian[[row, column]],
                hessian_epsilon,
            );
        }
    }
}

#[test]
fn polynomial_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Linear,
        &[-1.0, 0.0, 2.0, 3.5],
        &[1.5, -0.25],
        &[0.3, -0.7],
        2e-5,
        2e-4,
    );
}

#[test]
fn inverse_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Inverse,
        &[1.0, 2.0, 4.0, 8.0],
        &[1.0, 0.5],
        &[0.9, 0.3],
        2e-5,
        2e-4,
    );
}

#[test]
fn exponential_basic_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::ExponentialBasic,
        &[-1.0, -0.2, 0.3, 1.1, 2.0],
        &[0.8, 1.4, 0.6],
        &[0.5, 1.1, 0.3],
        2e-5,
        3e-4,
    );
}

#[test]
fn exponential_linear_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::ExponentialLinear,
        &[-1.2, -0.5, 0.0, 0.7, 1.4],
        &[1.4, 0.35, -0.4, 0.2],
        &[1.0, 0.2, -0.2, 0.0],
        3e-5,
        6e-4,
    );
}

#[test]
fn arrhenius_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Arrhenius,
        &[0.4, 0.8, 1.4, 2.5, 4.0],
        &[1.5, 0.9],
        &[1.2, 0.5],
        2e-5,
        3e-4,
    );
}

#[test]
fn power_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Power,
        &[0.3, 0.8, 1.2, 2.5, 4.0],
        &[1.1, 0.8],
        &[0.8, 0.5],
        3e-5,
        4e-4,
    );
}

#[test]
fn logistic_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Logistic,
        &[-2.0, -1.0, -0.3, 0.4, 1.1, 2.0],
        &[2.2, 1.1, 0.3],
        &[1.8, 0.8, -0.1],
        3e-5,
        6e-4,
    );
}

#[test]
fn gompertz_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Gompertz,
        &[-1.5, -0.8, -0.2, 0.6, 1.4, 2.3],
        &[1.9, 0.9, 0.2],
        &[1.4, 0.6, -0.2],
        4e-5,
        8e-4,
    );
}

#[test]
fn hyperbolic_tangent_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::HyperbolicTangent,
        &[-2.0, -1.0, -0.2, 0.7, 1.6, 2.4],
        &[1.7, 0.9, 0.4, -0.3],
        &[1.2, 0.6, 0.1, -0.1],
        4e-5,
        1e-3,
    );
}

#[test]
fn arctangent_step_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::ArctangentStep,
        &[-2.2, -1.3, -0.5, 0.2, 1.1, 2.1],
        &[2.1, 0.8, 0.3, 0.4],
        &[1.7, 0.5, -0.2, 0.1],
        4e-5,
        1e-3,
    );
}

#[test]
fn softplus_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::Softplus,
        &[-2.0, -1.1, -0.4, 0.3, 1.0, 1.9],
        &[1.3, 0.7, 0.2, 0.2],
        &[1.0, 0.5, -0.1, 0.0],
        4e-5,
        1e-3,
    );
}

#[test]
fn bi_exponential_derivatives_match_numerical_reference() {
    assert_family_gradient_and_hessian_match_numerical_reference(
        CurveFamily::BiExponential,
        &[-0.8, -0.1, 0.3, 0.9, 1.8, 2.7],
        &[1.2, 0.7, 0.5, 0.25, -0.3],
        &[0.9, 0.4, 0.4, 0.1, -0.1],
        5e-5,
        2e-3,
    );
}
