use super::common::{Vf64, positive_param_with_derivative_simd, positive_x_simd};
use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

/// Вычисляет логарифмическую зависимость:
/// `f(x) = scale * ln(x / x_scale)`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `x_scale` — масштаб по оси `x` (параметризован положительным преобразованием).
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let scale = param[0];
    let x_scale_raw = param[1];
    let x = positive_x(x);
    let (x_scale, _) = positive_param_with_derivative(x_scale_raw);
    scale * (x / x_scale).ln()
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let scale = Vf64::splat(param[0]);
    let x_scale_raw = Vf64::splat(param[1]);
    let x = positive_x_simd(x);
    let (x_scale, _) = positive_param_with_derivative_simd(x_scale_raw);
    scale * (x / x_scale).ln()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 2);

    let scale = param[0];
    let x_scale_raw = param[1];
    let x = positive_x(x);
    let (x_scale, d_b_raw) = positive_param_with_derivative(x_scale_raw);
    let ln_term = (x / x_scale).ln();

    grad[0] = ln_term;
    grad[1] = (-scale / x_scale) * d_b_raw;

    scale * ln_term
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 2]) -> Vf64 {
    let scale = Vf64::splat(param[0]);
    let x_scale_raw = Vf64::splat(param[1]);
    let x = positive_x_simd(x);
    let (x_scale, d_b_raw) = positive_param_with_derivative_simd(x_scale_raw);
    let ln_term = (x / x_scale).ln();

    grad[0] = ln_term;
    grad[1] = (-scale / x_scale) * d_b_raw;

    scale * ln_term
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

        let mut point_grad = [Vf64::splat(0.0); 2];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);
            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();

        let mut point_grad = [0.0; 2];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);
            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
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
