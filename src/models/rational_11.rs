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

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let num_slope = param[0];
    let num_offset = param[1];
    let den_slope = param[2];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let numerator = num_slope * x + num_offset;
        let denominator_raw = 1.0 + den_slope * x;
        let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);
        let residual = value_first[index];

        gradient[0] += residual * (x / denominator);
        gradient[1] += residual * (1.0 / denominator);
        gradient[2] += residual * (-numerator * x / (denominator * denominator)) * d_den_raw;
        gradient[3] += residual;
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
