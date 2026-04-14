use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет модель Михаэлиса-Ментен:
/// `f(x) = vmax * x / (x + km)`,
/// где:
/// - `vmax` — максимальная скорость,
/// - `km` — константа Михаэлиса.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let vmax = param[0];
    let km = param[1];
    let (denominator, _) = non_zero_param_with_derivative(x + km);
    vmax * x / denominator
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let vmax = param[0];
    let km = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let (denominator, d_den_d_km) = non_zero_param_with_derivative(x + km);
        let residual = value_first[index];
        let d_model_d_vmax = x / denominator;
        let d_model_d_km = -vmax * x / (denominator * denominator) * d_den_d_km;

        gradient[0] += residual * d_model_d_vmax;
        gradient[1] += residual * d_model_d_km;
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
