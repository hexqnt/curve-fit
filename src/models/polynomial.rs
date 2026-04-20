use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const MAX_POLYNOMIAL_PARAMS: usize = 10;
const MAX_UPPER_HESSIAN_LEN: usize = MAX_POLYNOMIAL_PARAMS * (MAX_POLYNOMIAL_PARAMS + 1) / 2;

/// Вычисляет полином в форме Горнера:
/// `f(x) = p0 * x^n + p1 * x^(n-1) + ... + pn`,
/// где `param = [p0, p1, ..., pn]` — коэффициенты от старшей степени к младшей.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    param
        .iter()
        .copied()
        .fold(0.0, |acc, coefficient| acc * x + coefficient)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    param
        .iter()
        .copied()
        .fold(Vf64::splat(0.0), |acc, coefficient| {
            acc * x + Vf64::splat(coefficient)
        })
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), param.len());

    let value = value_at(param, x);
    let mut basis = 1.0;
    for grad_value in grad.iter_mut().rev() {
        *grad_value = basis;
        basis *= x;
    }

    value
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64]) -> Vf64 {
    debug_assert_eq!(grad.len(), param.len());

    let value = value_simd_at(param, x);
    let mut basis = Vf64::splat(1.0);
    for grad_value in grad.iter_mut().rev() {
        *grad_value = basis;
        basis *= x;
    }

    value
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());
    debug_assert!(gradient.len() <= MAX_POLYNOMIAL_PARAMS);

    {
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());

        let mut accum = [Vf64::splat(0.0); MAX_POLYNOMIAL_PARAMS];
        let accum = &mut accum[..gradient.len()];
        let mut point_grad = [Vf64::splat(0.0); MAX_POLYNOMIAL_PARAMS];
        let point_grad = &mut point_grad[..gradient.len()];

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, point_grad);

            for (accum_value, point_grad_value) in accum.iter_mut().zip(point_grad.iter().copied())
            {
                *accum_value += upstream * point_grad_value;
            }
        }

        for (gradient_value, accum_value) in gradient.iter_mut().zip(accum.iter().copied()) {
            *gradient_value += accum_value.reduce_sum();
        }

        let mut point_grad = [0.0; MAX_POLYNOMIAL_PARAMS];
        let point_grad = &mut point_grad[..gradient.len()];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, point_grad);
            for (gradient_value, point_grad_value) in gradient.iter_mut().zip(point_grad.iter()) {
                *gradient_value += upstream * point_grad_value;
            }
        }
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    _value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    let dimension = param.len();
    if dimension == 0 {
        return None;
    }
    debug_assert!(dimension <= MAX_POLYNOMIAL_PARAMS);

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((dimension, dimension)));
    }
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((dimension, dimension));
    let mut basis = [0.0; MAX_POLYNOMIAL_PARAMS];
    let basis = &mut basis[..dimension];

    {
        let mut basis_simd = [Vf64::splat(0.0); MAX_POLYNOMIAL_PARAMS];
        let basis_simd = &mut basis_simd[..dimension];
        let upper_len = dimension * (dimension + 1) / 2;
        let mut upper = [Vf64::splat(0.0); MAX_UPPER_HESSIAN_LEN];
        let upper = &mut upper[..upper_len];
        let zero = Vf64::splat(0.0);
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_second_chunks, value_second_tail) = value_second.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_second_chunks.len());
        debug_assert_eq!(x_tail.len(), value_second_tail.len());

        for (x_chunk, value_second_chunk) in x_chunks.iter().zip(value_second_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let model = value_simd_at(param, x);
            if !model.is_finite().all() {
                return None;
            }

            let weight = Vf64::from_array(*value_second_chunk);
            if !weight.is_finite().all() || !weight.simd_ge(zero).all() {
                return None;
            }

            let mut basis_index = dimension;
            let mut power = Vf64::splat(1.0);
            while basis_index > 0 {
                basis_index -= 1;
                basis_simd[basis_index] = power;
                power *= x;
            }

            let mut upper_index = 0;
            let mut row = 0;
            while row < dimension {
                let basis_row = basis_simd[row];
                let mut column = row;
                while column < dimension {
                    upper[upper_index] += weight * basis_row * basis_simd[column];
                    upper_index += 1;
                    column += 1;
                }
                row += 1;
            }
        }

        let mut upper_index = 0;
        let mut row = 0;
        while row < dimension {
            let mut column = row;
            while column < dimension {
                hessian[[row, column]] += upper[upper_index].reduce_sum();
                upper_index += 1;
                column += 1;
            }
            row += 1;
        }

        for (&x, &weight) in x_tail.iter().zip(value_second_tail.iter()) {
            let model = value_at(param, x);
            if !model.is_finite() {
                return None;
            }

            if !is_finite_non_negative(weight) {
                return None;
            }

            let mut basis_index = dimension;
            let mut power = 1.0;
            while basis_index > 0 {
                basis_index -= 1;
                basis[basis_index] = power;
                power *= x;
            }

            let mut row = 0;
            while row < dimension {
                let basis_row = basis[row];
                let mut column = row;
                while column < dimension {
                    hessian[[row, column]] += weight * basis_row * basis[column];
                    column += 1;
                }
                row += 1;
            }
        }
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::value_at;
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[2.0, -3.0, 1.0], 4.0);
        assert_near(value, 21.0, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Linear,
            &[-1.0, 0.0, 2.0, 3.5],
            &[1.5, -0.25],
            &[0.3, -0.7],
            2e-5,
            2e-4,
        );
    }
}
