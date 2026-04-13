use super::common::{positive_param_with_derivative, sigmoid};
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let (sigma, _) = positive_param_with_derivative(param[2]);
    let (gamma, _) = positive_param_with_derivative(param[3]);
    let eta = sigmoid(param[4]);
    let delta = x - param[1];
    let gaussian = (-(delta * delta) / (2.0 * sigma * sigma)).exp();
    let lorentzian = 1.0 / (1.0 + (delta / gamma).powi(2));
    param[5] + param[0] * (eta * gaussian + (1.0 - eta) * lorentzian)
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
        let (sigma, d_sigma_raw) = positive_param_with_derivative(param[2]);
        let (gamma, d_gamma_raw) = positive_param_with_derivative(param[3]);
        let eta = sigmoid(param[4]);
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
        let model = param[5] + a * mix;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * mix;
        gradient[1] += residual * a * (eta * d_gaussian_dx0 + (1.0 - eta) * d_lorentzian_dx0);
        gradient[2] += residual * a * eta * d_gaussian_d_sigma * d_sigma_raw;
        gradient[3] += residual * a * (1.0 - eta) * d_lorentzian_d_gamma * d_gamma_raw;
        gradient[4] += residual * a * (gaussian - lorentzian) * eta_prime;
        gradient[5] += residual;
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
