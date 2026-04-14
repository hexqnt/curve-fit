use super::common::{
    is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian, stabilize_hessian,
};
use ndarray::Array2;

/// Вычисляет степенную зависимость:
/// `f(x) = scale * x^exponent`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `exponent` — показатель степени.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let scale = param[0];
    let exponent = param[1];
    scale * positive_x(x).powf(exponent)
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
    let scale = param[0];
    let exponent = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let pow = x.powf(exponent);
        let model = scale * pow;
        let residual = loss_derivative_from_prediction(model, y);
        gradient[0] += residual * pow;
        gradient[1] += residual * scale * pow * x.ln();
        index += 1;
    }
}

pub(super) fn analytic_hessian<L1, L2>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_derivative_from_prediction: &mut L1,
    loss_second_derivative_from_prediction: &mut L2,
) -> Option<Array2<f64>>
where
    L1: FnMut(f64, f64) -> f64,
    L2: FnMut(f64, f64) -> f64,
{
    if param.len() != 2 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((2, 2));
    let scale = param[0];
    let exponent = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let log_x = x.ln();
        let pow = x.powf(exponent);
        let model = scale * pow;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss_derivative_from_prediction(model, y);
        let loss_second = loss_second_derivative_from_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a = pow;
        let jac_b = scale * pow * log_x;
        let d2_model_dadb = pow * log_x;
        let d2_model_dbdb = scale * pow * log_x * log_x;

        hessian[[0, 0]] += loss_second * jac_a * jac_a;
        hessian[[0, 1]] += loss_second * jac_a * jac_b + loss_first * d2_model_dadb;
        hessian[[1, 1]] += loss_second * jac_b * jac_b + loss_first * d2_model_dbdb;
        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    stabilize_hessian(&mut hessian);
    Some(hessian)
}
