use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет рациональную функцию порядка (2, 2):
/// `f(x) = (num_quad * x^2 + num_linear * x + num_const) / (1 + den_linear * x + den_quad * x^2)`,
/// где:
/// - `num_quad`, `num_linear`, `num_const` — коэффициенты числителя,
/// - `den_linear`, `den_quad` — коэффициенты знаменателя.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let num_quad = param[0];
    let num_linear = param[1];
    let num_const = param[2];
    let den_linear = param[3];
    let den_quad = param[4];
    let x2 = x * x;
    let numerator = num_quad * x2 + num_linear * x + num_const;
    let denominator_raw = 1.0 + den_linear * x + den_quad * x2;
    let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
    numerator / denominator
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 5);

    let num_quad = param[0];
    let num_linear = param[1];
    let num_const = param[2];
    let den_linear = param[3];
    let den_quad = param[4];
    let x2 = x * x;
    let numerator = num_quad * x2 + num_linear * x + num_const;
    let denominator_raw = 1.0 + den_linear * x + den_quad * x2;
    let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);

    grad[0] = x2 / denominator;
    grad[1] = x / denominator;
    grad[2] = 1.0 / denominator;
    grad[3] = (-numerator * x / (denominator * denominator)) * d_den_raw;
    grad[4] = (-numerator * x2 / (denominator * denominator)) * d_den_raw;

    numerator / denominator
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 5];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
        gradient[4] += upstream * point_grad[4];
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
