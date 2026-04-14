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
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
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

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 5);
    grad.fill(0.0);
    value_at(param, x)
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 5];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
        gradient[4] += upstream * point_grad[4];
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
