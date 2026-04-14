use super::common::HESSIAN_DIAGONAL_JITTER;
use super::{
    GradientComputation, PredictionLoss, accumulate_gradient, analytic_hessian, evaluate_raw,
    objective_value, objective_value_grad, objective_value_grad_hessian,
    objective_value_grad_raw_hessian,
};
use crate::domain::CurveFamily;
use ndarray::Array2;

const GRADIENT_REL_STEP: f64 = 1e-6;
const GRADIENT_MIN_STEP: f64 = 1e-7;
const HESSIAN_REL_STEP: f64 = 2e-4;
const HESSIAN_MIN_STEP: f64 = 1e-6;
const OBJECTIVE_VALUE_EPS: f64 = 1e-12;

pub(super) fn assert_near(actual: f64, expected: f64, epsilon: f64) {
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

struct SoftL1Loss;

impl PredictionLoss for SoftL1Loss {
    fn value(&self, prediction: f64, target: f64) -> f64 {
        soft_l1_value(prediction, target)
    }

    fn d_prediction(&self, prediction: f64, target: f64) -> f64 {
        soft_l1_derivative(prediction, target)
    }

    fn d2_prediction(&self, prediction: f64, target: f64) -> f64 {
        soft_l1_second_derivative(prediction, target)
    }
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

pub(super) fn assert_family_gradient_and_hessian_match_numerical_reference(
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

    let loss = SoftL1Loss;
    let objective_value = objective_value(family, x_values, &y_values, probe_params, &loss);

    let mut objective_grad = vec![0.0; probe_params.len()];
    let (objective_value_from_grad, objective_mode_from_grad) = objective_value_grad(
        family,
        x_values,
        &y_values,
        probe_params,
        &loss,
        &mut objective_grad,
    );
    assert_eq!(objective_mode_from_grad, GradientComputation::Analytic);

    let mut objective_grad_with_hessian = vec![0.0; probe_params.len()];
    let (objective_value_from_hessian, objective_mode_from_hessian, objective_hessian) =
        objective_value_grad_hessian(
            family,
            x_values,
            &y_values,
            probe_params,
            &loss,
            &mut objective_grad_with_hessian,
        );
    assert_eq!(objective_mode_from_hessian, GradientComputation::Analytic);
    assert_near(
        objective_value,
        objective_value_from_grad,
        OBJECTIVE_VALUE_EPS,
    );
    assert_near(
        objective_value,
        objective_value_from_hessian,
        OBJECTIVE_VALUE_EPS,
    );
    for index in 0..probe_params.len() {
        assert_near(
            objective_grad[index],
            objective_grad_with_hessian[index],
            gradient_epsilon,
        );
    }

    let mut raw_grad = vec![0.0; probe_params.len()];
    let (_, _, raw_hessian) = objective_value_grad_raw_hessian(
        family,
        x_values,
        &y_values,
        probe_params,
        &loss,
        &mut raw_grad,
    );
    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(
                raw_hessian[[row, column]],
                raw_hessian[[column, row]],
                hessian_epsilon,
            );
            assert_near(
                objective_hessian[[row, column]],
                objective_hessian[[column, row]],
                hessian_epsilon,
            );
        }
    }

    let mut analytic_gradient = vec![0.0; probe_params.len()];
    let mode = accumulate_gradient(
        family,
        x_values,
        &y_values,
        probe_params,
        &loss,
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

    let analytic_hessian = analytic_hessian(family, x_values, &y_values, probe_params, &loss)
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
