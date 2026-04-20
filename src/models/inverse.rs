use super::common::{Vf64, positive_x_simd};
use super::common::{is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 2;

#[derive(Clone, Copy)]
struct Params<T> {
    offset: T,
    scale: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [offset, scale]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self { offset, scale }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            offset: Vf64::splat(self.offset),
            scale: Vf64::splat(self.scale),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        self.offset + self.scale / positive_x(x)
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let inv_x = 1.0 / positive_x(x);

        grad[0] = 1.0;
        grad[1] = inv_x;

        self.offset + self.scale * inv_x
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let x = positive_x_simd(x);
        self.offset + self.scale / x
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x = positive_x_simd(x);
        let inv_x = Vf64::splat(1.0) / x;

        grad[0] = Vf64::splat(1.0);
        grad[1] = inv_x;

        self.offset + self.scale * inv_x
    }
}

/// Вычисляет обратную зависимость:
/// `f(x) = offset + scale / x`,
/// где:
/// - `offset` — базовый уровень,
/// - `scale` — коэффициент обратной компоненты.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    Params::parse(param).value_at(x)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    Params::parse(param).simd().value_at(x)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    Params::parse(param).value_grad_at(x, grad)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
    Params::parse(param).simd().value_grad_at(x, grad)
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());
    let params = Params::parse(param);
    let params_simd = params.simd();

    {
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());

        let mut point_grad = [Vf64::splat(0.0); PARAM_COUNT];
        let mut gradient_accum = [Vf64::splat(0.0); PARAM_COUNT];

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            params_simd.value_grad_at(x, &mut point_grad);

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

        let mut point_grad = [0.0; PARAM_COUNT];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            params.value_grad_at(x, &mut point_grad);

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
    debug_assert_eq!(x_values.len(), value_second.len());

    if param.len() != PARAM_COUNT {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((PARAM_COUNT, PARAM_COUNT)));
    }

    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((PARAM_COUNT, PARAM_COUNT));
    let params = Params::parse(param);
    let params_simd = params.simd();

    {
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_second_chunks, value_second_tail) = value_second.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_second_chunks.len());
        debug_assert_eq!(x_tail.len(), value_second_tail.len());

        let mut h00 = Vf64::splat(0.0);
        let mut h01 = Vf64::splat(0.0);
        let mut h11 = Vf64::splat(0.0);
        let zero = Vf64::splat(0.0);

        for (x_chunk, value_second_chunk) in x_chunks.iter().zip(value_second_chunks.iter()) {
            let x = positive_x_simd(Vf64::from_array(*x_chunk));
            let inv_x = Vf64::splat(1.0) / x;
            let model = params_simd.offset + params_simd.scale * inv_x;
            if !model.is_finite().all() {
                return None;
            }

            let weight = Vf64::from_array(*value_second_chunk);
            if !weight.is_finite().all() || !weight.simd_ge(zero).all() {
                return None;
            }

            h00 += weight;
            h01 += weight * inv_x;
            h11 += weight * inv_x * inv_x;
        }

        hessian[[0, 0]] += h00.reduce_sum();
        hessian[[0, 1]] += h01.reduce_sum();
        hessian[[1, 1]] += h11.reduce_sum();

        for (&x, &weight) in x_tail.iter().zip(value_second_tail.iter()) {
            let x = positive_x(x);
            let inv_x = 1.0 / x;
            let model = params.offset + params.scale * inv_x;
            if !model.is_finite() {
                return None;
            }

            if !is_finite_non_negative(weight) {
                return None;
            }

            hessian[[0, 0]] += weight;
            hessian[[0, 1]] += weight * inv_x;
            hessian[[1, 1]] += weight * inv_x * inv_x;
        }
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::{add_value_grad_raw_hessian, value_at};
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[1.25, -0.6], 2.0);
        assert_near(value, 0.95, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Inverse,
            &[1.0, 2.0, 4.0, 8.0],
            &[1.0, 0.5],
            &[0.9, 0.3],
            2e-5,
            2e-4,
        );
    }

    #[test]
    fn raw_hessian_is_zero_for_empty_dataset() {
        let hessian = add_value_grad_raw_hessian(&[], &[1.0, 0.5], &[], &[])
            .expect("empty dataset must produce zero hessian");
        assert_eq!(hessian.shape(), &[2, 2]);
        assert!(hessian.iter().all(|&value| value == 0.0));
    }
}
