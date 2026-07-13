use ndarray::Array2;

use super::common::non_zero_param_with_derivative;
use super::common::{Vf64, non_zero_param_with_derivative_simd};

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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
