use super::common::non_zero_param_with_derivative;
use super::common::{Vf64, non_zero_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::num::SimdFloat;

const MIN_RATIONAL_NN_PARAMS: usize = 7;
const MAX_RATIONAL_NN_PARAMS: usize = 11;

#[inline]
fn rational_degree_from_param_len(param_len: usize) -> usize {
    match param_len {
        7 => 3,
        9 => 4,
        11 => 5,
        _ => panic!("Unsupported Rational n/n parameter count: {param_len}"),
    }
}

#[inline]
fn rational_degree(param: &[f64]) -> usize {
    debug_assert!((MIN_RATIONAL_NN_PARAMS..=MAX_RATIONAL_NN_PARAMS).contains(&param.len()));
    rational_degree_from_param_len(param.len())
}

#[inline]
fn numerator_and_denominator_raw(param: &[f64], x: f64, degree: usize) -> (f64, f64) {
    let numerator = param[..=degree]
        .iter()
        .copied()
        .fold(0.0, |acc, coefficient| acc * x + coefficient);

    let mut denominator_raw = 1.0;
    let mut power = x;
    for &coefficient in &param[(degree + 1)..] {
        denominator_raw += coefficient * power;
        power *= x;
    }

    (numerator, denominator_raw)
}

#[inline]
fn numerator_and_denominator_raw_simd(param: &[f64], x: Vf64, degree: usize) -> (Vf64, Vf64) {
    let numerator = param[..=degree]
        .iter()
        .copied()
        .fold(Vf64::splat(0.0), |acc, coefficient| {
            acc * x + Vf64::splat(coefficient)
        });

    let mut denominator_raw = Vf64::splat(1.0);
    let mut power = x;
    for &coefficient in &param[(degree + 1)..] {
        denominator_raw += Vf64::splat(coefficient) * power;
        power *= x;
    }

    (numerator, denominator_raw)
}

#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let degree = rational_degree(param);
    let (numerator, denominator_raw) = numerator_and_denominator_raw(param, x, degree);
    let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
    numerator / denominator
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let degree = rational_degree(param);
    let (numerator, denominator_raw) = numerator_and_denominator_raw_simd(param, x, degree);
    let (denominator, _) = non_zero_param_with_derivative_simd(denominator_raw);
    numerator / denominator
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), param.len());

    let degree = rational_degree(param);
    let (numerator, denominator_raw) = numerator_and_denominator_raw(param, x, degree);
    let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);

    let inv_denominator = 1.0 / denominator;
    let common_den_grad = (-numerator / (denominator * denominator)) * d_den_raw;

    let mut basis = 1.0;
    for grad_value in grad[..=degree].iter_mut().rev() {
        *grad_value = basis * inv_denominator;
        basis *= x;
    }

    let mut power = x;
    for grad_value in &mut grad[(degree + 1)..] {
        *grad_value = common_den_grad * power;
        power *= x;
    }

    numerator / denominator
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64]) -> Vf64 {
    debug_assert_eq!(grad.len(), param.len());

    let degree = rational_degree(param);
    let (numerator, denominator_raw) = numerator_and_denominator_raw_simd(param, x, degree);
    let (denominator, d_den_raw) = non_zero_param_with_derivative_simd(denominator_raw);

    let inv_denominator = Vf64::splat(1.0) / denominator;
    let common_den_grad = (-numerator / (denominator * denominator)) * d_den_raw;

    let mut basis = Vf64::splat(1.0);
    for grad_value in grad[..=degree].iter_mut().rev() {
        *grad_value = basis * inv_denominator;
        basis *= x;
    }

    let mut power = x;
    for grad_value in &mut grad[(degree + 1)..] {
        *grad_value = common_den_grad * power;
        power *= x;
    }

    numerator / denominator
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

        let mut point_grad = [Vf64::splat(0.0); MAX_RATIONAL_NN_PARAMS];
        let point_grad = &mut point_grad[..gradient.len()];
        let mut gradient_accum = [Vf64::splat(0.0); MAX_RATIONAL_NN_PARAMS];
        let gradient_accum = &mut gradient_accum[..gradient.len()];

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, point_grad);

            for (gradient_value, point_grad_value) in
                gradient_accum.iter_mut().zip(point_grad.iter().copied())
            {
                *gradient_value += upstream * point_grad_value;
            }
        }

        for (gradient_value, accum_value) in gradient.iter_mut().zip(gradient_accum.iter().copied())
        {
            *gradient_value += accum_value.reduce_sum();
        }

        let mut point_grad_tail = [0.0; MAX_RATIONAL_NN_PARAMS];
        let point_grad_tail = &mut point_grad_tail[..gradient.len()];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, point_grad_tail);

            for (gradient_value, point_grad_value) in
                gradient.iter_mut().zip(point_grad_tail.iter())
            {
                *gradient_value += upstream * point_grad_value;
            }
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

#[cfg(test)]
mod tests {
    use super::{value_at, value_grad_at};
    use crate::models::test_support::assert_near;

    fn numerical_gradient(param: &[f64], x: f64) -> Vec<f64> {
        let mut gradient = vec![0.0; param.len()];
        let mut probe = param.to_vec();
        let step_scale = 1e-6;
        let step_floor = 1e-7;

        for index in 0..param.len() {
            let step = ((param[index].abs() + 1.0) * step_scale).max(step_floor);
            probe[index] = param[index] + step;
            let plus = value_at(&probe, x);
            probe[index] = param[index] - step;
            let minus = value_at(&probe, x);
            probe[index] = param[index];
            gradient[index] = (plus - minus) / (2.0 * step);
        }

        gradient
    }

    fn assert_point_gradient_matches_numerical(param: &[f64], x: f64) {
        let mut analytic = vec![0.0; param.len()];
        let _ = value_grad_at(param, x, &mut analytic);
        let numerical = numerical_gradient(param, x);
        for (actual, expected) in analytic.iter().zip(numerical.iter()) {
            assert_near(*actual, *expected, 5e-5);
        }
    }

    #[test]
    fn rational_33_point_gradient_matches_numerical_reference() {
        let param = [0.1, -0.2, 0.9, 0.3, 0.05, -0.01, 0.005];
        for &x in &[-1.4, -0.2, 0.7, 1.6] {
            assert_point_gradient_matches_numerical(&param, x);
        }
    }

    #[test]
    fn rational_55_point_gradient_matches_numerical_reference() {
        let param = [0.0, 0.0, 0.0, 0.1, 0.8, 0.2, 0.02, -0.01, 0.005, 0.0, 0.0];
        for &x in &[-1.8, -0.6, 0.4, 1.3] {
            assert_point_gradient_matches_numerical(&param, x);
        }
    }
}
