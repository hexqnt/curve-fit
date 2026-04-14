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

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let amplitude = param[0];
    let x0 = param[1];
    let sigma_raw = param[2];
    let gamma_raw = param[3];
    let eta_raw = param[4];
    let (sigma, d_sigma_raw) = positive_param_with_derivative(sigma_raw);
    let (gamma, d_gamma_raw) = positive_param_with_derivative(gamma_raw);
    let eta = sigmoid(eta_raw);
    let eta_prime = eta * (1.0 - eta);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
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
        let residual = value_first[index];

        gradient[0] += residual * mix;
        gradient[1] +=
            residual * amplitude * (eta * d_gaussian_dx0 + (1.0 - eta) * d_lorentzian_dx0);
        gradient[2] += residual * amplitude * eta * d_gaussian_d_sigma * d_sigma_raw;
        gradient[3] += residual * amplitude * (1.0 - eta) * d_lorentzian_d_gamma * d_gamma_raw;
        gradient[4] += residual * amplitude * (gaussian - lorentzian) * eta_prime;
        gradient[5] += residual;
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
