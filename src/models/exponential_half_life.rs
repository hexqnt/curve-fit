use super::common::positive_param_with_derivative;
use super::common::{Vf64, positive_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

const LN_2: f64 = std::f64::consts::LN_2;

/// Вычисляет экспоненциальную модель через период полураспада:
/// `f(x) = offset + amplitude * exp(-ln(2) * x / half_life)`,
/// где:
/// - `offset` — базовый уровень,
/// - `amplitude` — амплитуда экспоненциальной части,
/// - `half_life` — период полураспада (параметризован положительным преобразованием).
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let amplitude = param[1];
    let half_life_raw = param[2];
    let (half_life, _) = positive_param_with_derivative(half_life_raw);
    let exponent = -LN_2 * x / half_life;
    offset + amplitude * exponent.exp()
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let offset = Vf64::splat(param[0]);
    let amplitude = Vf64::splat(param[1]);
    let half_life_raw = Vf64::splat(param[2]);
    let (half_life, _) = positive_param_with_derivative_simd(half_life_raw);
    let exponent = -Vf64::splat(LN_2) * x / half_life;
    offset + amplitude * exponent.exp()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let offset = param[0];
    let amplitude = param[1];
    let half_life_raw = param[2];
    let (half_life, d_c_raw) = positive_param_with_derivative(half_life_raw);
    let exponent = -LN_2 * x / half_life;
    let pow = exponent.exp();
    let d_model_d_c = amplitude * pow * LN_2 * x / (half_life * half_life);

    grad[0] = 1.0;
    grad[1] = pow;
    grad[2] = d_model_d_c * d_c_raw;

    offset + amplitude * pow
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 3]) -> Vf64 {
    let offset = Vf64::splat(param[0]);
    let amplitude = Vf64::splat(param[1]);
    let half_life_raw = Vf64::splat(param[2]);
    let (half_life, d_c_raw) = positive_param_with_derivative_simd(half_life_raw);
    let exponent = -Vf64::splat(LN_2) * x / half_life;
    let pow = exponent.exp();
    let d_model_d_c = amplitude * pow * Vf64::splat(LN_2) * x / (half_life * half_life);

    grad[0] = Vf64::splat(1.0);
    grad[1] = pow;
    grad[2] = d_model_d_c * d_c_raw;

    offset + amplitude * pow
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
