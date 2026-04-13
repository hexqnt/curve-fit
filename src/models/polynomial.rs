use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, stabilize_hessian};
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    param
        .iter()
        .copied()
        .fold(0.0, |acc, coefficient| acc * x + coefficient)
}

pub(super) fn accumulate_gradient<L>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_derivative_from_prediction: &mut L,
    gradient: &mut [f64],
) where
    L: FnMut(f64, f64) -> f64,
{
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let model = eval(param, x);
        let residual = loss_derivative_from_prediction(model, y);

        let mut basis = 1.0;
        for gradient_value in gradient.iter_mut().rev() {
            *gradient_value += residual * basis;
            basis *= x;
        }
        index += 1;
    }
}

pub(super) fn analytic_hessian<L1, L2>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    _loss_derivative_from_prediction: &mut L1,
    loss_second_derivative_from_prediction: &mut L2,
) -> Option<Array2<f64>>
where
    L1: FnMut(f64, f64) -> f64,
    L2: FnMut(f64, f64) -> f64,
{
    debug_assert_eq!(x_values.len(), y_values.len());

    let dimension = param.len();
    if dimension == 0 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((dimension, dimension));
    let mut basis = vec![0.0; dimension];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let y = y_values[index];
        let model = eval(param, x);
        if !model.is_finite() {
            return None;
        }

        let weight = loss_second_derivative_from_prediction(model, y);
        if !is_finite_non_negative(weight) {
            return None;
        }

        let mut basis_index = dimension;
        let mut power = 1.0;
        while basis_index > 0 {
            basis_index -= 1;
            basis[basis_index] = power;
            power *= x;
        }

        let mut row = 0;
        while row < dimension {
            let basis_row = basis[row];
            let mut column = row;
            while column < dimension {
                hessian[[row, column]] += weight * basis_row * basis[column];
                column += 1;
            }
            row += 1;
        }

        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    stabilize_hessian(&mut hessian);
    Some(hessian)
}
