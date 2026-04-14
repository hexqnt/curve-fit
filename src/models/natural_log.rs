use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;

/// Вычисляет логарифмическую зависимость:
/// `f(x) = scale * ln(x / x_scale)`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `x_scale` — масштаб по оси `x` (параметризован положительным преобразованием).
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let scale = param[0];
    let x_scale_raw = param[1];
    let x = positive_x(x);
    let (x_scale, _) = positive_param_with_derivative(x_scale_raw);
    scale * (x / x_scale).ln()
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let scale = param[0];
    let x_scale_raw = param[1];
    let (x_scale, d_b_raw) = positive_param_with_derivative(x_scale_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let ln_term = (x / x_scale).ln();
        let residual = value_first[index];

        gradient[0] += residual * ln_term;
        gradient[1] += residual * (-scale / x_scale) * d_b_raw;
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
