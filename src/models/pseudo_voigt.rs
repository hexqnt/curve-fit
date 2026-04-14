use super::common::{positive_param_with_derivative, sigmoid};
use ndarray::Array2;

/// Вычисляет псевдо-Войт профиль:
/// `f(x) = baseline + amplitude * (eta * G(x) + (1 - eta) * L(x))`,
/// где:
/// - `amplitude` — амплитуда,
/// - `x0` — центр пика,
/// - `sigma` — ширина гауссовой части (положительный параметр),
/// - `gamma` — ширина лоренцевой части (положительный параметр),
/// - `eta` — вес смешивания `G/L` (через сигмоиду),
/// - `baseline` — базовый уровень.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let x0 = param[1];
    let sigma_raw = param[2];
    let gamma_raw = param[3];
    let eta_raw = param[4];
    let baseline = param[5];
    let (sigma, _) = positive_param_with_derivative(sigma_raw);
    let (gamma, _) = positive_param_with_derivative(gamma_raw);
    let eta = sigmoid(eta_raw);
    let delta = x - x0;
    let gaussian = (-(delta * delta) / (2.0 * sigma * sigma)).exp();
    let lorentzian = 1.0 / (1.0 + (delta / gamma).powi(2));
    baseline + amplitude * (eta * gaussian + (1.0 - eta) * lorentzian)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 6);

    let amplitude = param[0];
    let x0 = param[1];
    let sigma_raw = param[2];
    let gamma_raw = param[3];
    let eta_raw = param[4];
    let baseline = param[5];
    let (sigma, d_sigma_raw) = positive_param_with_derivative(sigma_raw);
    let (gamma, d_gamma_raw) = positive_param_with_derivative(gamma_raw);
    let eta = sigmoid(eta_raw);
    let eta_prime = eta * (1.0 - eta);
    let delta = x - x0;

    let sigma2 = sigma * sigma;
    let gaussian = (-(delta * delta) / (2.0 * sigma2)).exp();
    let d_gaussian_dx0 = gaussian * delta / sigma2;
    let d_gaussian_d_sigma = gaussian * delta * delta / (sigma2 * sigma);

    let u = delta / gamma;
    let den = 1.0 + u * u;
    let lorentzian = 1.0 / den;
    let den2 = den * den;
    let d_lorentzian_dx0 = 2.0 * u / (den2 * gamma);
    let d_lorentzian_d_gamma = 2.0 * u * u / (den2 * gamma);

    let mix = eta * gaussian + (1.0 - eta) * lorentzian;

    grad[0] = mix;
    grad[1] = amplitude * (eta * d_gaussian_dx0 + (1.0 - eta) * d_lorentzian_dx0);
    grad[2] = amplitude * eta * d_gaussian_d_sigma * d_sigma_raw;
    grad[3] = amplitude * (1.0 - eta) * d_lorentzian_d_gamma * d_gamma_raw;
    grad[4] = amplitude * (gaussian - lorentzian) * eta_prime;
    grad[5] = 1.0;

    baseline + amplitude * mix
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 6];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
        gradient[4] += upstream * point_grad[4];
        gradient[5] += upstream * point_grad[5];
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
