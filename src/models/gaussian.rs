use super::common::positive_param_with_derivative;
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let (c, _) = positive_param_with_derivative(param[2]);
    let delta = x - param[1];
    param[0] * (-(delta * delta) / (2.0 * c * c)).exp()
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
        let b = param[1];
        let (c, d_c_raw) = positive_param_with_derivative(param[2]);
        let c2 = c * c;
        let delta = x - b;
        let exp_part = (-(delta * delta) / (2.0 * c2)).exp();
        let model = a * exp_part;
        let residual = loss_derivative_from_prediction(model, y);
        let d_model_d_a = exp_part;
        let d_model_d_b = a * exp_part * delta / c2;
        let d_model_d_c = a * exp_part * delta * delta / (c2 * c);

        gradient[0] += residual * d_model_d_a;
        gradient[1] += residual * d_model_d_b;
        gradient[2] += residual * d_model_d_c * d_c_raw;
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
