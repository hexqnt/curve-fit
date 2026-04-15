use super::common::Vf64;
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

/// Вычисляет затухающую синусоиду:
/// `f(x) = amplitude * exp(-damping * x) * sin(omega * x + phi) + offset`,
/// где:
/// - `amplitude` — начальная амплитуда,
/// - `damping` — коэффициент затухания,
/// - `omega` — угловая частота,
/// - `phi` — фазовый сдвиг,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];
    let offset = param[4];
    amplitude * (-damping * x).exp() * (omega * x + phi).sin() + offset
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let damping = Vf64::splat(param[1]);
    let omega = Vf64::splat(param[2]);
    let phi = Vf64::splat(param[3]);
    let offset = Vf64::splat(param[4]);
    amplitude * (-damping * x).exp() * (omega * x + phi).sin() + offset
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 5);

    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];
    let offset = param[4];
    let exp_part = (-damping * x).exp();
    let angle = omega * x + phi;
    let sin_part = angle.sin();
    let cos_part = angle.cos();

    grad[0] = exp_part * sin_part;
    grad[1] = -amplitude * x * exp_part * sin_part;
    grad[2] = amplitude * exp_part * cos_part * x;
    grad[3] = amplitude * exp_part * cos_part;
    grad[4] = 1.0;

    amplitude * exp_part * sin_part + offset
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 5]) -> Vf64 {
    let amplitude = Vf64::splat(param[0]);
    let damping = Vf64::splat(param[1]);
    let omega = Vf64::splat(param[2]);
    let phi = Vf64::splat(param[3]);
    let offset = Vf64::splat(param[4]);
    let exp_part = (-damping * x).exp();
    let angle = omega * x + phi;
    let sin_part = angle.sin();
    let cos_part = angle.cos();

    grad[0] = exp_part * sin_part;
    grad[1] = -amplitude * x * exp_part * sin_part;
    grad[2] = amplitude * exp_part * cos_part * x;
    grad[3] = amplitude * exp_part * cos_part;
    grad[4] = Vf64::splat(1.0);

    amplitude * exp_part * sin_part + offset
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
