use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct Params<T> {
    a1: T,
    k1: T,
    a2: T,
    k2: T,
    offset: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [a1, k1, a2, k2, offset]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            a1,
            k1,
            a2,
            k2,
            offset,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            a1: Vf64::splat(self.a1),
            k1: Vf64::splat(self.k1),
            a2: Vf64::splat(self.a2),
            k2: Vf64::splat(self.k2),
            offset: Vf64::splat(self.offset),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        self.a1 * (-self.k1 * x).exp() + self.a2 * (-self.k2 * x).exp() + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let exp1 = (-self.k1 * x).exp();
        let exp2 = (-self.k2 * x).exp();

        grad[0] = exp1;
        grad[1] = -self.a1 * x * exp1;
        grad[2] = exp2;
        grad[3] = -self.a2 * x * exp2;
        grad[4] = 1.0;

        self.a1 * exp1 + self.a2 * exp2 + self.offset
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        self.a1 * (-self.k1 * x).exp() + self.a2 * (-self.k2 * x).exp() + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let exp1 = (-self.k1 * x).exp();
        let exp2 = (-self.k2 * x).exp();

        grad[0] = exp1;
        grad[1] = -self.a1 * x * exp1;
        grad[2] = exp2;
        grad[3] = -self.a2 * x * exp2;
        grad[4] = Vf64::splat(1.0);

        self.a1 * exp1 + self.a2 * exp2 + self.offset
    }
}

/// Вычисляет сумму двух экспонент:
/// `f(x) = a1 * exp(-k1 * x) + a2 * exp(-k2 * x) + offset`,
/// где:
/// - `a1`, `a2` — амплитуды экспоненциальных компонент,
/// - `k1`, `k2` — коэффициенты затухания компонент,
/// - `offset` — вертикальный сдвиг.
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
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
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
        let zero = Vf64::splat(0.0);
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        let (value_second_chunks, value_second_tail) = value_second.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_chunks.len(), value_second_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());
        debug_assert_eq!(x_tail.len(), value_second_tail.len());

        let mut h00 = Vf64::splat(0.0);
        let mut h01 = Vf64::splat(0.0);
        let mut h02 = Vf64::splat(0.0);
        let mut h03 = Vf64::splat(0.0);
        let mut h04 = Vf64::splat(0.0);
        let mut h11 = Vf64::splat(0.0);
        let mut h12 = Vf64::splat(0.0);
        let mut h13 = Vf64::splat(0.0);
        let mut h14 = Vf64::splat(0.0);
        let mut h22 = Vf64::splat(0.0);
        let mut h23 = Vf64::splat(0.0);
        let mut h24 = Vf64::splat(0.0);
        let mut h33 = Vf64::splat(0.0);
        let mut h34 = Vf64::splat(0.0);
        let mut h44 = Vf64::splat(0.0);

        for ((x_chunk, value_first_chunk), value_second_chunk) in x_chunks
            .iter()
            .zip(value_first_chunks.iter())
            .zip(value_second_chunks.iter())
        {
            let x = Vf64::from_array(*x_chunk);
            let exp1 = (-params_simd.k1 * x).exp();
            let exp2 = (-params_simd.k2 * x).exp();
            let model = params_simd.a1 * exp1 + params_simd.a2 * exp2 + params_simd.offset;
            if !model.is_finite().all() {
                return None;
            }

            let value_first = Vf64::from_array(*value_first_chunk);
            let value_second = Vf64::from_array(*value_second_chunk);
            if !value_first.is_finite().all()
                || !value_second.is_finite().all()
                || !value_second.simd_ge(zero).all()
            {
                return None;
            }

            let jac_a1 = exp1;
            let jac_k1 = -params_simd.a1 * x * exp1;
            let jac_a2 = exp2;
            let jac_k2 = -params_simd.a2 * x * exp2;
            let jac_c = Vf64::splat(1.0);
            let d2_model_da1dk1 = -x * exp1;
            let d2_model_dk1dk1 = params_simd.a1 * x * x * exp1;
            let d2_model_da2dk2 = -x * exp2;
            let d2_model_dk2dk2 = params_simd.a2 * x * x * exp2;

            h00 += value_second * jac_a1 * jac_a1;
            h01 += value_second * jac_a1 * jac_k1 + value_first * d2_model_da1dk1;
            h02 += value_second * jac_a1 * jac_a2;
            h03 += value_second * jac_a1 * jac_k2;
            h04 += value_second * jac_a1 * jac_c;
            h11 += value_second * jac_k1 * jac_k1 + value_first * d2_model_dk1dk1;
            h12 += value_second * jac_k1 * jac_a2;
            h13 += value_second * jac_k1 * jac_k2;
            h14 += value_second * jac_k1 * jac_c;
            h22 += value_second * jac_a2 * jac_a2;
            h23 += value_second * jac_a2 * jac_k2 + value_first * d2_model_da2dk2;
            h24 += value_second * jac_a2 * jac_c;
            h33 += value_second * jac_k2 * jac_k2 + value_first * d2_model_dk2dk2;
            h34 += value_second * jac_k2 * jac_c;
            h44 += value_second * jac_c * jac_c;
        }

        hessian[[0, 0]] += h00.reduce_sum();
        hessian[[0, 1]] += h01.reduce_sum();
        hessian[[0, 2]] += h02.reduce_sum();
        hessian[[0, 3]] += h03.reduce_sum();
        hessian[[0, 4]] += h04.reduce_sum();
        hessian[[1, 1]] += h11.reduce_sum();
        hessian[[1, 2]] += h12.reduce_sum();
        hessian[[1, 3]] += h13.reduce_sum();
        hessian[[1, 4]] += h14.reduce_sum();
        hessian[[2, 2]] += h22.reduce_sum();
        hessian[[2, 3]] += h23.reduce_sum();
        hessian[[2, 4]] += h24.reduce_sum();
        hessian[[3, 3]] += h33.reduce_sum();
        hessian[[3, 4]] += h34.reduce_sum();
        hessian[[4, 4]] += h44.reduce_sum();

        for ((&x, &value_first), &value_second) in x_tail
            .iter()
            .zip(value_first_tail.iter())
            .zip(value_second_tail.iter())
        {
            let exp1 = (-params.k1 * x).exp();
            let exp2 = (-params.k2 * x).exp();
            let model = params.a1 * exp1 + params.a2 * exp2 + params.offset;
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a1 = exp1;
            let jac_k1 = -params.a1 * x * exp1;
            let jac_a2 = exp2;
            let jac_k2 = -params.a2 * x * exp2;
            let jac_c = 1.0;
            let d2_model_da1dk1 = -x * exp1;
            let d2_model_dk1dk1 = params.a1 * x * x * exp1;
            let d2_model_da2dk2 = -x * exp2;
            let d2_model_dk2dk2 = params.a2 * x * x * exp2;

            hessian[[0, 0]] += value_second * jac_a1 * jac_a1;
            hessian[[0, 1]] += value_second * jac_a1 * jac_k1 + value_first * d2_model_da1dk1;
            hessian[[0, 2]] += value_second * jac_a1 * jac_a2;
            hessian[[0, 3]] += value_second * jac_a1 * jac_k2;
            hessian[[0, 4]] += value_second * jac_a1 * jac_c;
            hessian[[1, 1]] += value_second * jac_k1 * jac_k1 + value_first * d2_model_dk1dk1;
            hessian[[1, 2]] += value_second * jac_k1 * jac_a2;
            hessian[[1, 3]] += value_second * jac_k1 * jac_k2;
            hessian[[1, 4]] += value_second * jac_k1 * jac_c;
            hessian[[2, 2]] += value_second * jac_a2 * jac_a2;
            hessian[[2, 3]] += value_second * jac_a2 * jac_k2 + value_first * d2_model_da2dk2;
            hessian[[2, 4]] += value_second * jac_a2 * jac_c;
            hessian[[3, 3]] += value_second * jac_k2 * jac_k2 + value_first * d2_model_dk2dk2;
            hessian[[3, 4]] += value_second * jac_k2 * jac_c;
            hessian[[4, 4]] += value_second * jac_c * jac_c;
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
        let value = value_at(&[1.0, 0.4, 0.5, 0.2, -0.1], 1.5);
        let expected = (-0.6_f64).exp() + 0.5 * (-0.3_f64).exp() - 0.1;
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::BiExponential,
            &[-0.8, -0.1, 0.3, 0.9, 1.8, 2.7],
            &[1.2, 0.7, 0.5, 0.25, -0.3],
            &[0.9, 0.4, 0.4, 0.1, -0.1],
            5e-5,
            2e-3,
        );
    }
}
