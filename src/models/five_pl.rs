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
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
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
        let y = y_values[index];
        let ratio = x / ec50;
        let pow = ratio.powf(hill_slope);
        let den = 1.0 + pow;
        let inv = den.powf(-asymmetry);
        let model = bottom + (top - bottom) * inv;
        let residual = loss_derivative_from_prediction(model, y);
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

pub(super) fn analytic_hessian<L1, L2>(
    _x_values: &[f64],
    _y_values: &[f64],
    _param: &[f64],
    _loss_derivative_from_prediction: &mut L1,
    _loss_second_derivative_from_prediction: &mut L2,
) -> Option<Array2<f64>>
where
    L1: FnMut(f64, f64) -> f64,
    L2: FnMut(f64, f64) -> f64,
{
    None
}
