use super::common::{Vf64, erfc_approx_simd};
use super::common::{erfc_approx, positive_param_with_derivative};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

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
fn eval_right_simd(amplitude: f64, mu: f64, sigma: f64, tau: f64, baseline: f64, x: Vf64) -> Vf64 {
    let (sigma, _) = positive_param_with_derivative(sigma);
    let (tau, _) = positive_param_with_derivative(tau);
    let delta = x - Vf64::splat(mu);
    let z = (Vf64::splat(sigma / tau) - delta / Vf64::splat(sigma))
        / Vf64::splat(std::f64::consts::SQRT_2);
    let exponent = Vf64::splat(sigma * sigma / (2.0 * tau * tau)) - delta / Vf64::splat(tau);
    Vf64::splat(baseline)
        + Vf64::splat(amplitude / (2.0 * tau)) * exponent.exp() * erfc_approx_simd(z)
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

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let amplitude = param[0];
    let mu = param[1];
    let sigma = param[2];
    let tau = param[3];
    let baseline = param[4];

    if tau.is_sign_negative() {
        let reflected_x = Vf64::splat(2.0 * mu) - x;
        eval_right_simd(amplitude, mu, sigma, tau.abs(), baseline, reflected_x)
    } else {
        eval_right_simd(amplitude, mu, sigma, tau, baseline, x)
    }
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 5);
    grad.fill(0.0);
    value_at(param, x)
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 5]) -> Vf64 {
    grad.fill(Vf64::splat(0.0));
    value_simd_at(param, x)
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

        let mut point_grad = [Vf64::splat(0.0); 5];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);
        let mut gradient_2 = Vf64::splat(0.0);
        let mut gradient_3 = Vf64::splat(0.0);
        let mut gradient_4 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);
            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
            gradient_2 += upstream * point_grad[2];
            gradient_3 += upstream * point_grad[3];
            gradient_4 += upstream * point_grad[4];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();
        gradient[2] += gradient_2.reduce_sum();
        gradient[3] += gradient_3.reduce_sum();
        gradient[4] += gradient_4.reduce_sum();

        let mut point_grad = [0.0; 5];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);

            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
            gradient[2] += upstream * point_grad[2];
            gradient[3] += upstream * point_grad[3];
            gradient[4] += upstream * point_grad[4];
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
