use super::{
    CurveObjective, DataTerm, ObjectiveGrad, ObjectiveHessian, ObjectiveValue, PredictionLoss,
    central_diff_gradient_from_value, central_diff_hessian_from_gradient, value_at,
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
            let prediction = value_at(family, param, x);
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
        .map(|&x| value_at(family, true_params, x))
        .collect::<Vec<_>>();

    let loss = SoftL1Loss;
    let term = DataTerm::new(family, x_values, &y_values, None, loss);
    let objective = CurveObjective::new(probe_params.len(), term);

    let value = objective.value(probe_params);
    let (value_from_grad, gradient) = objective.value_grad(probe_params);
    let (value_from_raw_hessian, gradient_from_raw_hessian, raw_hessian) =
        objective.value_grad_raw_hessian(probe_params);
    let (value_from_hessian, gradient_from_hessian, hessian) =
        objective.value_grad_hessian(probe_params);

    assert_near(value, value_from_grad, OBJECTIVE_VALUE_EPS);
    assert_near(value, value_from_raw_hessian, OBJECTIVE_VALUE_EPS);
    assert_near(value, value_from_hessian, OBJECTIVE_VALUE_EPS);

    for index in 0..probe_params.len() {
        assert_near(
            gradient[index],
            gradient_from_raw_hessian[index],
            gradient_epsilon,
        );
        assert_near(
            gradient[index],
            gradient_from_hessian[index],
            gradient_epsilon,
        );
    }

    let numerical_gradient = numerical_gradient_from_cost(probe_params, |param| {
        mean_soft_l1_cost(family, x_values, &y_values, param)
    });
    for index in 0..probe_params.len() {
        assert_near(gradient[index], numerical_gradient[index], gradient_epsilon);
    }

    let mut numerical_gradient_from_value = vec![0.0; probe_params.len()];
    central_diff_gradient_from_value(
        probe_params,
        GRADIENT_REL_STEP,
        GRADIENT_MIN_STEP,
        |param| objective.value(param),
        &mut numerical_gradient_from_value,
    );
    for index in 0..probe_params.len() {
        assert_near(
            gradient[index],
            numerical_gradient_from_value[index],
            gradient_epsilon,
        );
    }

    let numerical_hessian_from_grad = central_diff_hessian_from_gradient(
        probe_params,
        HESSIAN_REL_STEP,
        HESSIAN_MIN_STEP,
        |param, output| {
            let (_, gradient_probe) = objective.value_grad(param);
            output.copy_from_slice(&gradient_probe);
        },
    );

    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(
                raw_hessian[[row, column]],
                raw_hessian[[column, row]],
                hessian_epsilon,
            );
            assert_near(
                hessian[[row, column]],
                hessian[[column, row]],
                hessian_epsilon,
            );
            assert_near(
                raw_hessian[[row, column]],
                numerical_hessian_from_grad[[row, column]],
                hessian_epsilon,
            );
        }
    }

    let mut stabilized_numerical_hessian = numerical_hessian_from_grad.clone();
    super::common::stabilize_hessian(&mut stabilized_numerical_hessian);

    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(
                hessian[[row, column]],
                stabilized_numerical_hessian[[row, column]],
                hessian_epsilon,
            );
        }
    }

    let numerical_hessian_from_value = numerical_hessian_from_cost(probe_params, |param| {
        mean_soft_l1_cost(family, x_values, &y_values, param)
    });
    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(
                raw_hessian[[row, column]],
                numerical_hessian_from_value[[row, column]],
                hessian_epsilon,
            );
        }
    }
}
