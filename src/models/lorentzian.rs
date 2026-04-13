use super::common::positive_param_with_derivative;
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let (gamma, _) = positive_param_with_derivative(param[2]);
    let u = (x - param[1]) / gamma;
    param[3] + param[0] / (1.0 + u * u)
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
        let x = x_values[index];
        let y = y_values[index];
        let a = param[0];
        let x0 = param[1];
        let (gamma, d_gamma_raw) = positive_param_with_derivative(param[2]);
        let u = (x - x0) / gamma;
        let den = 1.0 + u * u;
        let inv_den = 1.0 / den;
        let model = param[3] + a * inv_den;
        let residual = loss_derivative_from_prediction(model, y);
        let common = 2.0 * a / (den * den * gamma);

        gradient[0] += residual * inv_den;
        gradient[1] += residual * (common * u);
        gradient[2] += residual * (common * u * u) * d_gamma_raw;
        gradient[3] += residual;
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
