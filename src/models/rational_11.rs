use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет рациональную функцию порядка (1, 1):
/// `f(x) = offset + (num_slope * x + num_offset) / (1 + den_slope * x)`,
/// где:
/// - `num_slope`, `num_offset` — коэффициенты числителя,
/// - `den_slope` — коэффициент знаменателя,
/// - `offset` — вертикальный сдвиг.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let num_slope = param[0];
    let num_offset = param[1];
    let den_slope = param[2];
    let offset = param[3];
    let numerator = num_slope * x + num_offset;
    let denominator_raw = 1.0 + den_slope * x;
    let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
    offset + numerator / denominator
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 4);

    let num_slope = param[0];
    let num_offset = param[1];
    let den_slope = param[2];
    let offset = param[3];
    let numerator = num_slope * x + num_offset;
    let denominator_raw = 1.0 + den_slope * x;
    let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);

    grad[0] = x / denominator;
    grad[1] = 1.0 / denominator;
    grad[2] = (-numerator * x / (denominator * denominator)) * d_den_raw;
    grad[3] = 1.0;

    offset + numerator / denominator
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 4];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    _x_values: &[f64],
    _param: &[f64],
    _value_first: &[f64],
    _value_second: &[f64],
) -> Option<Array2<f64>> {
    None
}
