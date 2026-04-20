use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 4;

#[derive(Clone, Copy)]
struct Params<T> {
    exp_amplitude: T,
    exp_rate: T,
    linear_slope: T,
    offset: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [exp_amplitude, exp_rate, linear_slope, offset]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            exp_amplitude,
            exp_rate,
            linear_slope,
            offset,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            exp_amplitude: Vf64::splat(self.exp_amplitude),
            exp_rate: Vf64::splat(self.exp_rate),
            linear_slope: Vf64::splat(self.linear_slope),
            offset: Vf64::splat(self.offset),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        self.exp_amplitude * (self.exp_rate * x).exp() + self.linear_slope * x + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let exp_part = (self.exp_rate * x).exp();

        grad[0] = exp_part;
        grad[1] = self.exp_amplitude * exp_part * x;
        grad[2] = x;
        grad[3] = 1.0;

        self.exp_amplitude * exp_part + self.linear_slope * x + self.offset
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        self.exp_amplitude * (self.exp_rate * x).exp() + self.linear_slope * x + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let exp_part = (self.exp_rate * x).exp();

        grad[0] = exp_part;
        grad[1] = self.exp_amplitude * exp_part * x;
        grad[2] = x;
        grad[3] = Vf64::splat(1.0);

        self.exp_amplitude * exp_part + self.linear_slope * x + self.offset
    }
}

/// Вычисляет экспоненциально-линейную модель:
/// `f(x) = exp_amplitude * exp(exp_rate * x) + linear_slope * x + offset`,
/// где:
/// - `exp_amplitude` — амплитуда экспоненциальной части,
/// - `exp_rate` — показатель экспоненциального роста/затухания,
/// - `linear_slope` — наклон линейной части,
/// - `offset` — свободный член.
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
        let mut h11 = Vf64::splat(0.0);
        let mut h12 = Vf64::splat(0.0);
        let mut h13 = Vf64::splat(0.0);
        let mut h22 = Vf64::splat(0.0);
        let mut h23 = Vf64::splat(0.0);
        let mut h33 = Vf64::splat(0.0);

        for ((x_chunk, value_first_chunk), value_second_chunk) in x_chunks
            .iter()
            .zip(value_first_chunks.iter())
            .zip(value_second_chunks.iter())
        {
            let x = Vf64::from_array(*x_chunk);
            let exp_part = (params_simd.exp_rate * x).exp();
            let model = params_simd.exp_amplitude * exp_part
                + params_simd.linear_slope * x
                + params_simd.offset;
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

            let jac_a = exp_part;
            let jac_b = params_simd.exp_amplitude * x * exp_part;
            let jac_c = x;
            let jac_d = Vf64::splat(1.0);
            let d2_model_dadb = x * exp_part;
            let d2_model_dbdb = params_simd.exp_amplitude * x * x * exp_part;

            h00 += value_second * jac_a * jac_a;
            h01 += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            h02 += value_second * jac_a * jac_c;
            h03 += value_second * jac_a * jac_d;
            h11 += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
            h12 += value_second * jac_b * jac_c;
            h13 += value_second * jac_b * jac_d;
            h22 += value_second * jac_c * jac_c;
            h23 += value_second * jac_c * jac_d;
            h33 += value_second * jac_d * jac_d;
        }

        hessian[[0, 0]] += h00.reduce_sum();
        hessian[[0, 1]] += h01.reduce_sum();
        hessian[[0, 2]] += h02.reduce_sum();
        hessian[[0, 3]] += h03.reduce_sum();
        hessian[[1, 1]] += h11.reduce_sum();
        hessian[[1, 2]] += h12.reduce_sum();
        hessian[[1, 3]] += h13.reduce_sum();
        hessian[[2, 2]] += h22.reduce_sum();
        hessian[[2, 3]] += h23.reduce_sum();
        hessian[[3, 3]] += h33.reduce_sum();

        for ((&x, &value_first), &value_second) in x_tail
            .iter()
            .zip(value_first_tail.iter())
            .zip(value_second_tail.iter())
        {
            let exp_part = (params.exp_rate * x).exp();
            let model = params.exp_amplitude * exp_part + params.linear_slope * x + params.offset;
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a = exp_part;
            let jac_b = params.exp_amplitude * x * exp_part;
            let jac_c = x;
            let jac_d = 1.0;
            let d2_model_dadb = x * exp_part;
            let d2_model_dbdb = params.exp_amplitude * x * x * exp_part;

            hessian[[0, 0]] += value_second * jac_a * jac_a;
            hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            hessian[[0, 2]] += value_second * jac_a * jac_c;
            hessian[[0, 3]] += value_second * jac_a * jac_d;
            hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
            hessian[[1, 2]] += value_second * jac_b * jac_c;
            hessian[[1, 3]] += value_second * jac_b * jac_d;
            hessian[[2, 2]] += value_second * jac_c * jac_c;
            hessian[[2, 3]] += value_second * jac_c * jac_d;
            hessian[[3, 3]] += value_second * jac_d * jac_d;
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
        let value = value_at(&[1.2, 0.3, -0.4, 0.1], 2.0);
        let expected = 1.2 * 0.6_f64.exp() - 0.8 + 0.1;
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::ExponentialLinear,
            &[-1.2, -0.5, 0.0, 0.7, 1.4],
            &[1.4, 0.35, -0.4, 0.2],
            &[1.0, 0.2, -0.2, 0.0],
            3e-5,
            6e-4,
        );
    }
}
