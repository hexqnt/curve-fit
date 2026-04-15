use super::common::positive_param_with_derivative;
use super::common::{Vf64, positive_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::num::SimdFloat;

/// Вычисляет лоренцев пик:
/// `f(x) = baseline + amplitude / (1 + ((x - x0) / gamma)^2)`,
/// где:
/// - `amplitude` — амплитуда пика,
/// - `x0` — центр пика,
/// - `gamma` — полуширина (параметризована положительным преобразованием),
/// - `baseline` — базовый уровень.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let x0 = param[1];
    let gamma_raw = param[2];
    let baseline = param[3];
    let (gamma, _) = positive_param_with_derivative(gamma_raw);
    let u = (x - x0) / gamma;
    baseline + amplitude / (1.0 + u * u)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let x0 = Vf64::splat(param[1]);
    let gamma_raw = Vf64::splat(param[2]);
    let baseline = Vf64::splat(param[3]);
    let (gamma, _) = positive_param_with_derivative_simd(gamma_raw);
    let u = (x - x0) / gamma;
    baseline + amplitude / (Vf64::splat(1.0) + u * u)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 4);

    let amplitude = param[0];
    let x0 = param[1];
    let gamma_raw = param[2];
    let baseline = param[3];
    let (gamma, d_gamma_raw) = positive_param_with_derivative(gamma_raw);
    let u = (x - x0) / gamma;
    let den = 1.0 + u * u;
    let inv_den = 1.0 / den;
    let common = 2.0 * amplitude / (den * den * gamma);

    grad[0] = inv_den;
    grad[1] = common * u;
    grad[2] = common * u * u * d_gamma_raw;
    grad[3] = 1.0;

    baseline + amplitude * inv_den
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 4]) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let x0 = Vf64::splat(param[1]);
    let gamma_raw = Vf64::splat(param[2]);
    let baseline = Vf64::splat(param[3]);
    let (gamma, d_gamma_raw) = positive_param_with_derivative_simd(gamma_raw);
    let u = (x - x0) / gamma;
    let den = Vf64::splat(1.0) + u * u;
    let inv_den = Vf64::splat(1.0) / den;
    let common = Vf64::splat(2.0) * amplitude / (den * den * gamma);

    grad[0] = inv_den;
    grad[1] = common * u;
    grad[2] = common * u * u * d_gamma_raw;
    grad[3] = Vf64::splat(1.0);

    baseline + amplitude * inv_den
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

        let mut point_grad = [Vf64::splat(0.0); 4];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);
        let mut gradient_2 = Vf64::splat(0.0);
        let mut gradient_3 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);

            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
            gradient_2 += upstream * point_grad[2];
            gradient_3 += upstream * point_grad[3];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();
        gradient[2] += gradient_2.reduce_sum();
        gradient[3] += gradient_3.reduce_sum();

        let mut point_grad = [0.0; 4];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);

            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
            gradient[2] += upstream * point_grad[2];
            gradient[3] += upstream * point_grad[3];
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
