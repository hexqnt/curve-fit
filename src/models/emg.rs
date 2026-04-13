use super::common::{erfc_approx, positive_param_with_derivative};
use ndarray::Array2;

#[inline]
fn eval_right(a: f64, mu: f64, sigma: f64, tau: f64, c: f64, x: f64) -> f64 {
    let (sigma, _) = positive_param_with_derivative(sigma);
    let (tau, _) = positive_param_with_derivative(tau);
    let delta = x - mu;
    let z = (sigma / tau - delta / sigma) / std::f64::consts::SQRT_2;
    let exponent = sigma * sigma / (2.0 * tau * tau) - delta / tau;
    c + (a / (2.0 * tau)) * exponent.exp() * erfc_approx(z)
}

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    if param[3].is_sign_negative() {
        let reflected_x = 2.0 * param[1] - x;
        eval_right(
            param[0],
            param[1],
            param[2],
            param[3].abs(),
            param[4],
            reflected_x,
        )
    } else {
        eval_right(param[0], param[1], param[2], param[3], param[4], x)
    }
}

pub(super) fn accumulate_gradient<L>(
    _x_values: &[f64],
    _y_values: &[f64],
    _param: &[f64],
    _loss_derivative_from_prediction: &mut L,
    _gradient: &mut [f64],
) where
    L: FnMut(f64, f64) -> f64,
{
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
