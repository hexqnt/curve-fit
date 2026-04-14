use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;

/// Вычисляет четырёхпараметрическую логистическую кривую (4PL):
/// `f(x) = bottom + (top - bottom) / (1 + (x / ec50)^hill_slope)`,
/// где:
/// - `top` — верхняя асимптота,
/// - `hill_slope` — крутизна,
/// - `ec50` — точка перегиба (параметризована положительным преобразованием),
/// - `bottom` — нижняя асимптота.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let top = param[0];
    let hill_slope = param[1];
    let ec50_raw = param[2];
    let bottom = param[3];
    let x = positive_x(x);
    let (ec50, _) = positive_param_with_derivative(ec50_raw);
    let ratio = x / ec50;
    let pow = ratio.powf(hill_slope);
    bottom + (top - bottom) / (1.0 + pow)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 4);

    let top = param[0];
    let hill_slope = param[1];
    let ec50_raw = param[2];
    let bottom = param[3];
    let x = positive_x(x);
    let (ec50, d_c_raw) = positive_param_with_derivative(ec50_raw);
    let ratio = x / ec50;
    let pow = ratio.powf(hill_slope);
    let den = 1.0 + pow;
    let inv_den = 1.0 / den;
    let d_pow_db = pow * ratio.ln();
    let d_pow_dc = -pow * hill_slope / ec50;
    let d_model_da = inv_den;
    let d_model_dd = 1.0 - inv_den;
    let d_model_db = -(top - bottom) * d_pow_db / (den * den);
    let d_model_dc = -(top - bottom) * d_pow_dc / (den * den);

    grad[0] = d_model_da;
    grad[1] = d_model_db;
    grad[2] = d_model_dc * d_c_raw;
    grad[3] = d_model_dd;

    bottom + (top - bottom) * inv_den
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
