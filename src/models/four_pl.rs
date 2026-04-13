use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let x = positive_x(x);
    let (c, _) = positive_param_with_derivative(param[2]);
    let ratio = x / c;
    let pow = ratio.powf(param[1]);
    param[3] + (param[0] - param[3]) / (1.0 + pow)
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

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let a = param[0];
        let b = param[1];
        let (c, d_c_raw) = positive_param_with_derivative(param[2]);
        let d = param[3];
        let ratio = x / c;
        let pow = ratio.powf(b);
        let den = 1.0 + pow;
        let inv_den = 1.0 / den;
        let model = d + (a - d) * inv_den;
        let residual = loss_derivative_from_prediction(model, y);
        let d_pow_db = pow * ratio.ln();
        let d_pow_dc = -pow * b / c;
        let d_model_da = inv_den;
        let d_model_dd = 1.0 - inv_den;
        let d_model_db = -(a - d) * d_pow_db / (den * den);
        let d_model_dc = -(a - d) * d_pow_dc / (den * den);

        gradient[0] += residual * d_model_da;
        gradient[1] += residual * d_model_db;
        gradient[2] += residual * d_model_dc * d_c_raw;
        gradient[3] += residual * d_model_dd;
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
