use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use crate::domain::MAX_SATURATING_TREND_TAU_COUNT;
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

pub(crate) const SATURATING_TREND_PARAM_COUNT: usize = MAX_SATURATING_TREND_TAU_COUNT + 1;
const MAX_UPPER_HESSIAN_LEN: usize =
    SATURATING_TREND_PARAM_COUNT * (SATURATING_TREND_PARAM_COUNT + 1) / 2;

#[inline]
fn basis_at(x: f64, taus: &[f64], grad: &mut [f64; SATURATING_TREND_PARAM_COUNT]) {
    grad[0] = 1.0;
    for (index, tau) in taus.iter().copied().enumerate() {
        grad[index + 1] = 1.0 - (-x / tau).exp();
    }
}

#[inline]
fn basis_simd_at(x: Vf64, taus: &[f64], grad: &mut [Vf64; SATURATING_TREND_PARAM_COUNT]) {
    grad[0] = Vf64::splat(1.0);
    for (index, tau) in taus.iter().copied().enumerate() {
        grad[index + 1] = Vf64::splat(1.0) - (-x / Vf64::splat(tau)).exp();
    }
}

/// Вычисляет линейную комбинацию фиксированных насыщаемых трендов:
/// `f(x) = c + Σ_i w_i * (1 - exp(-x / τ_i))`,
/// где набор `τ_i` задается извне и не оптимизируется.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64, taus: &[f64]) -> f64 {
    debug_assert_eq!(param.len(), taus.len() + 1);

    let mut basis = [0.0; SATURATING_TREND_PARAM_COUNT];
    basis_at(x, taus, &mut basis);
    param
        .iter()
        .copied()
        .zip(basis[..param.len()].iter().copied())
        .map(|(weight, basis)| weight * basis)
        .sum()
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64, taus: &[f64]) -> Vf64 {
    debug_assert_eq!(param.len(), taus.len() + 1);

    let mut basis = [Vf64::splat(0.0); SATURATING_TREND_PARAM_COUNT];
    basis_simd_at(x, taus, &mut basis);
    param
        .iter()
        .copied()
        .zip(basis[..param.len()].iter().copied())
        .fold(Vf64::splat(0.0), |acc, (weight, basis)| {
            acc + Vf64::splat(weight) * basis
        })
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, taus: &[f64], grad: &mut [f64]) -> f64 {
    debug_assert_eq!(param.len(), taus.len() + 1);
    debug_assert_eq!(grad.len(), param.len());

    let mut basis = [0.0; SATURATING_TREND_PARAM_COUNT];
    basis_at(x, taus, &mut basis);
    grad.copy_from_slice(&basis[..grad.len()]);
    param
        .iter()
        .copied()
        .zip(basis[..grad.len()].iter().copied())
        .map(|(weight, basis)| weight * basis)
        .sum()
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, taus: &[f64], grad: &mut [Vf64]) -> Vf64 {
    debug_assert_eq!(param.len(), taus.len() + 1);
    debug_assert_eq!(grad.len(), param.len());

    let mut basis = [Vf64::splat(0.0); SATURATING_TREND_PARAM_COUNT];
    basis_simd_at(x, taus, &mut basis);
    grad.copy_from_slice(&basis[..param.len()]);
    param
        .iter()
        .copied()
        .zip(grad.iter().copied())
        .fold(Vf64::splat(0.0), |acc, (weight, basis)| {
            acc + Vf64::splat(weight) * basis
        })
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    taus: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(param.len(), taus.len() + 1);
    debug_assert_eq!(gradient.len(), param.len());

    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
    debug_assert_eq!(x_tail.len(), value_first_tail.len());

    let mut accum = [Vf64::splat(0.0); SATURATING_TREND_PARAM_COUNT];
    let accum = &mut accum[..gradient.len()];
    let mut point_grad = [Vf64::splat(0.0); SATURATING_TREND_PARAM_COUNT];
    let point_grad = &mut point_grad[..gradient.len()];

    for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk);
        let upstream = Vf64::from_array(*value_first_chunk);
        value_grad_simd_at(param, x, taus, point_grad);

        for (accum_value, point_grad_value) in accum.iter_mut().zip(point_grad.iter().copied()) {
            *accum_value += upstream * point_grad_value;
        }
    }

    for (gradient_value, accum_value) in gradient.iter_mut().zip(accum.iter().copied()) {
        *gradient_value += accum_value.reduce_sum();
    }

    let mut point_grad = [0.0; SATURATING_TREND_PARAM_COUNT];
    let point_grad = &mut point_grad[..gradient.len()];
    for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
        value_grad_at(param, x, taus, point_grad);
        for (gradient_value, point_grad_value) in gradient.iter_mut().zip(point_grad.iter()) {
            *gradient_value += upstream * point_grad_value;
        }
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    taus: &[f64],
    _value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    let dimension = param.len();
    if dimension != taus.len() + 1 || !(2..=SATURATING_TREND_PARAM_COUNT).contains(&dimension) {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((dimension, dimension)));
    }

    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((dimension, dimension));

    let zero = Vf64::splat(0.0);
    let mut upper = [Vf64::splat(0.0); MAX_UPPER_HESSIAN_LEN];
    let mut basis_simd = [Vf64::splat(0.0); SATURATING_TREND_PARAM_COUNT];
    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (value_second_chunks, value_second_tail) = value_second.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), value_second_chunks.len());
    debug_assert_eq!(x_tail.len(), value_second_tail.len());

    for (x_chunk, weight_chunk) in x_chunks.iter().zip(value_second_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk);
        let model = value_simd_at(param, x, taus);
        if !model.is_finite().all() {
            return None;
        }

        let weight = Vf64::from_array(*weight_chunk);
        if !weight.is_finite().all() || !weight.simd_ge(zero).all() {
            return None;
        }

        basis_simd_at(x, taus, &mut basis_simd);

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

    let mut basis = [0.0; SATURATING_TREND_PARAM_COUNT];
    for (&x, &weight) in x_tail.iter().zip(value_second_tail.iter()) {
        let model = value_at(param, x, taus);
        if !model.is_finite() {
            return None;
        }

        if !is_finite_non_negative(weight) {
            return None;
        }

        basis_at(x, taus, &mut basis);

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

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::value_at;
    use crate::domain::CurveFamily;
    use crate::domain::DEFAULT_SATURATING_TREND_TAUS_YEARS;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let x = 2.0;
        let taus = &DEFAULT_SATURATING_TREND_TAUS_YEARS;
        let value = value_at(&[0.4, 1.0, -0.3, 0.2, 0.1, -0.2, 0.05], x, taus);
        let expected = 0.4 + 1.0 * (1.0 - (-x / taus[0]).exp())
            - 0.3 * (1.0 - (-x / taus[1]).exp())
            + 0.2 * (1.0 - (-x / taus[2]).exp())
            + 0.1 * (1.0 - (-x / taus[3]).exp())
            - 0.2 * (1.0 - (-x / taus[4]).exp())
            + 0.05 * (1.0 - (-x / taus[5]).exp());
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::SaturatingTrendBasis6,
            &[0.0, 0.25, 0.75, 1.5, 3.0, 5.0, 8.0],
            &[0.2, 1.0, -0.3, 0.4, 0.1, -0.2, 0.05],
            &[0.1, 0.8, -0.1, 0.2, 0.0, -0.1, 0.0],
            4e-5,
            1e-3,
        );
    }
}
