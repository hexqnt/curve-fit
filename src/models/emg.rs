use super::common::{erfc_approx, positive_param_with_derivative};
use ndarray::Array2;

#[inline]
fn eval_right(amplitude: f64, mu: f64, sigma: f64, tau: f64, baseline: f64, x: f64) -> f64 {
    let (sigma, _) = positive_param_with_derivative(sigma);
    let (tau, _) = positive_param_with_derivative(tau);
    let delta = x - mu;
    let z = (sigma / tau - delta / sigma) / std::f64::consts::SQRT_2;
    let exponent = sigma * sigma / (2.0 * tau * tau) - delta / tau;
    baseline + (amplitude / (2.0 * tau)) * exponent.exp() * erfc_approx(z)
}

#[inline]
/// Вычисляет экспоненциально-модифицированную гауссиану (EMG):
/// `f(x) = baseline + (amplitude / (2 * tau)) * exp(sigma^2 / (2 * tau^2) - (x - mu) / tau) * erfc(z)`,
/// где:
/// - `amplitude` — амплитуда,
/// - `mu` — центр гауссовой части,
/// - `sigma` — ширина гауссовой части,
/// - `tau` — экспоненциальная постоянная,
/// - `baseline` — вертикальный сдвиг.
///
/// Для `tau < 0` используется отражение по `mu`, что даёт хвост в противоположную сторону.
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let mu = param[1];
    let sigma = param[2];
    let tau = param[3];
    let baseline = param[4];

    if tau.is_sign_negative() {
        let reflected_x = 2.0 * mu - x;
        eval_right(amplitude, mu, sigma, tau.abs(), baseline, reflected_x)
    } else {
        eval_right(amplitude, mu, sigma, tau, baseline, x)
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
