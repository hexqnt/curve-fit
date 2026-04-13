use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let x2 = x * x;
    let numerator = param[0] * x2 + param[1] * x + param[2];
    let denominator_raw = 1.0 + param[3] * x + param[4] * x2;
    let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
    numerator / denominator
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
        let x2 = x * x;
        let numerator = param[0] * x2 + param[1] * x + param[2];
        let denominator_raw = 1.0 + param[3] * x + param[4] * x2;
        let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);
        let model = numerator / denominator;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * (x2 / denominator);
        gradient[1] += residual * (x / denominator);
        gradient[2] += residual * (1.0 / denominator);
        gradient[3] += residual * (-numerator * x / (denominator * denominator)) * d_den_raw;
        gradient[4] += residual * (-numerator * x2 / (denominator * denominator)) * d_den_raw;
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
