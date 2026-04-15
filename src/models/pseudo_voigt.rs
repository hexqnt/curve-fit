use super::common::{Vf64, positive_param_with_derivative_simd, sigmoid_simd};
use super::common::{positive_param_with_derivative, sigmoid};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

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

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let x0 = Vf64::splat(param[1]);
    let sigma_raw = Vf64::splat(param[2]);
    let gamma_raw = Vf64::splat(param[3]);
    let eta_raw = Vf64::splat(param[4]);
    let baseline = Vf64::splat(param[5]);
    let (sigma, _) = positive_param_with_derivative_simd(sigma_raw);
    let (gamma, _) = positive_param_with_derivative_simd(gamma_raw);
    let eta = sigmoid_simd(eta_raw);
    let delta = x - x0;
    let gaussian = (-(delta * delta) / (Vf64::splat(2.0) * sigma * sigma)).exp();
    let u = delta / gamma;
    let lorentzian = Vf64::splat(1.0) / (Vf64::splat(1.0) + u * u);
    baseline + amplitude * (eta * gaussian + (Vf64::splat(1.0) - eta) * lorentzian)
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

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 6]) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let x0 = Vf64::splat(param[1]);
    let sigma_raw = Vf64::splat(param[2]);
    let gamma_raw = Vf64::splat(param[3]);
    let eta_raw = Vf64::splat(param[4]);
    let baseline = Vf64::splat(param[5]);
    let (sigma, d_sigma_raw) = positive_param_with_derivative_simd(sigma_raw);
    let (gamma, d_gamma_raw) = positive_param_with_derivative_simd(gamma_raw);
    let eta = sigmoid_simd(eta_raw);
    let eta_prime = eta * (Vf64::splat(1.0) - eta);
    let delta = x - x0;

    let sigma2 = sigma * sigma;
    let gaussian = (-(delta * delta) / (Vf64::splat(2.0) * sigma2)).exp();
    let d_gaussian_dx0 = gaussian * delta / sigma2;
    let d_gaussian_d_sigma = gaussian * delta * delta / (sigma2 * sigma);

    let u = delta / gamma;
    let den = Vf64::splat(1.0) + u * u;
    let lorentzian = Vf64::splat(1.0) / den;
    let den2 = den * den;
    let d_lorentzian_dx0 = Vf64::splat(2.0) * u / (den2 * gamma);
    let d_lorentzian_d_gamma = Vf64::splat(2.0) * u * u / (den2 * gamma);

    let mix = eta * gaussian + (Vf64::splat(1.0) - eta) * lorentzian;

    grad[0] = mix;
    grad[1] = amplitude * (eta * d_gaussian_dx0 + (Vf64::splat(1.0) - eta) * d_lorentzian_dx0);
    grad[2] = amplitude * eta * d_gaussian_d_sigma * d_sigma_raw;
    grad[3] = amplitude * (Vf64::splat(1.0) - eta) * d_lorentzian_d_gamma * d_gamma_raw;
    grad[4] = amplitude * (gaussian - lorentzian) * eta_prime;
    grad[5] = Vf64::splat(1.0);

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

    {
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());

        let mut point_grad = [Vf64::splat(0.0); 6];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);
        let mut gradient_2 = Vf64::splat(0.0);
        let mut gradient_3 = Vf64::splat(0.0);
        let mut gradient_4 = Vf64::splat(0.0);
        let mut gradient_5 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);

            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
            gradient_2 += upstream * point_grad[2];
            gradient_3 += upstream * point_grad[3];
            gradient_4 += upstream * point_grad[4];
            gradient_5 += upstream * point_grad[5];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();
        gradient[2] += gradient_2.reduce_sum();
        gradient[3] += gradient_3.reduce_sum();
        gradient[4] += gradient_4.reduce_sum();
        gradient[5] += gradient_5.reduce_sum();

        let mut point_grad = [0.0; 6];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);

            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
            gradient[2] += upstream * point_grad[2];
            gradient[3] += upstream * point_grad[3];
            gradient[4] += upstream * point_grad[4];
            gradient[5] += upstream * point_grad[5];
        }
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
