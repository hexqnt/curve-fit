use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;

/// Вычисляет пяти-параметрическую логистическую кривую (5PL):
/// `f(x) = bottom + (top - bottom) * (1 + (x / ec50)^hill_slope)^(-asymmetry)`,
/// где:
/// - `top` — верхняя асимптота,
/// - `hill_slope` — крутизна,
/// - `ec50` — точка перегиба (параметризована положительным преобразованием),
/// - `bottom` — нижняя асимптота,
/// - `asymmetry` — параметр асимметрии (параметризован положительным преобразованием).
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let top = param[0];
    let hill_slope = param[1];
    let ec50_raw = param[2];
    let bottom = param[3];
    let asymmetry_raw = param[4];
    let x = positive_x(x);
    let (ec50, _) = positive_param_with_derivative(ec50_raw);
    let (asymmetry, _) = positive_param_with_derivative(asymmetry_raw);
    let ratio = x / ec50;
    let pow = ratio.powf(hill_slope);
    bottom + (top - bottom) * (1.0 + pow).powf(-asymmetry)
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let top = param[0];
    let hill_slope = param[1];
    let ec50_raw = param[2];
    let bottom = param[3];
    let asymmetry_raw = param[4];
    let (ec50, d_c_raw) = positive_param_with_derivative(ec50_raw);
    let (asymmetry, d_m_raw) = positive_param_with_derivative(asymmetry_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let ratio = x / ec50;
        let pow = ratio.powf(hill_slope);
        let den = 1.0 + pow;
        let inv = den.powf(-asymmetry);
        let residual = value_first[index];
        let d_pow_db = pow * ratio.ln();
        let d_pow_dc = -pow * hill_slope / ec50;
        let d_inv_db = -asymmetry * den.powf(-asymmetry - 1.0) * d_pow_db;
        let d_inv_dc = -asymmetry * den.powf(-asymmetry - 1.0) * d_pow_dc;
        let d_inv_dm = -inv * den.ln();

        gradient[0] += residual * inv;
        gradient[1] += residual * (top - bottom) * d_inv_db;
        gradient[2] += residual * (top - bottom) * d_inv_dc * d_c_raw;
        gradient[3] += residual * (1.0 - inv);
        gradient[4] += residual * (top - bottom) * d_inv_dm * d_m_raw;
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
