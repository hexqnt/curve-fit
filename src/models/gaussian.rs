use super::common::positive_param_with_derivative;
use super::common::{Vf64, positive_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

/// Вычисляет гауссову кривую:
/// `f(x) = amplitude * exp(-(x - mean)^2 / (2 * sigma^2))`,
/// где:
/// - `amplitude` — амплитуда,
/// - `mean` — центр пика,
/// - `sigma` — ширина (параметризована положительным преобразованием).
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let mean = param[1];
    let sigma_raw = param[2];
    let (sigma, _) = positive_param_with_derivative(sigma_raw);
    let delta = x - mean;
    amplitude * (-(delta * delta) / (2.0 * sigma * sigma)).exp()
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let mean = Vf64::splat(param[1]);
    let sigma_raw = Vf64::splat(param[2]);
    let (sigma, _) = positive_param_with_derivative_simd(sigma_raw);
    let delta = x - mean;
    amplitude * (-(delta * delta) / (Vf64::splat(2.0) * sigma * sigma)).exp()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let amplitude = param[0];
    let mean = param[1];
    let sigma_raw = param[2];
    let (sigma, d_c_raw) = positive_param_with_derivative(sigma_raw);
    let c2 = sigma * sigma;
    let delta = x - mean;
    let exp_part = (-(delta * delta) / (2.0 * c2)).exp();
    let d_model_d_a = exp_part;
    let d_model_d_b = amplitude * exp_part * delta / c2;
    let d_model_d_c = amplitude * exp_part * delta * delta / (c2 * sigma);

    grad[0] = d_model_d_a;
    grad[1] = d_model_d_b;
    grad[2] = d_model_d_c * d_c_raw;

    amplitude * exp_part
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 3]) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let mean = Vf64::splat(param[1]);
    let sigma_raw = Vf64::splat(param[2]);
    let (sigma, d_c_raw) = positive_param_with_derivative_simd(sigma_raw);
    let c2 = sigma * sigma;
    let delta = x - mean;
    let exp_part = (-(delta * delta) / (Vf64::splat(2.0) * c2)).exp();
    let d_model_d_a = exp_part;
    let d_model_d_b = amplitude * exp_part * delta / c2;
    let d_model_d_c = amplitude * exp_part * delta * delta / (c2 * sigma);

    grad[0] = d_model_d_a;
    grad[1] = d_model_d_b;
    grad[2] = d_model_d_c * d_c_raw;

    amplitude * exp_part
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

        let mut point_grad = [Vf64::splat(0.0); 3];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);
        let mut gradient_2 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);

            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
            gradient_2 += upstream * point_grad[2];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();
        gradient[2] += gradient_2.reduce_sum();

        let mut point_grad = [0.0; 3];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);

            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
            gradient[2] += upstream * point_grad[2];
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
