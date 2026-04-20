use super::common::{Vf64, positive_x_simd};
use super::common::{is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 2;

#[derive(Clone, Copy)]
struct Params<T> {
    prefactor: T,
    temp_coeff: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [prefactor, temp_coeff]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            prefactor,
            temp_coeff,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            prefactor: Vf64::splat(self.prefactor),
            temp_coeff: Vf64::splat(self.temp_coeff),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let x = positive_x(x);
        self.prefactor * (self.temp_coeff / x).exp()
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let x = positive_x(x);
        let exp_term = (self.temp_coeff / x).exp();

        grad[0] = exp_term;
        grad[1] = self.prefactor * exp_term / x;

        self.prefactor * exp_term
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let x = positive_x_simd(x);
        self.prefactor * (self.temp_coeff / x).exp()
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x = positive_x_simd(x);
        let exp_term = (self.temp_coeff / x).exp();

        grad[0] = exp_term;
        grad[1] = self.prefactor * exp_term / x;

        self.prefactor * exp_term
    }
}

/// Вычисляет кривую Аррениуса:
/// `f(x) = prefactor * exp(temp_coeff / x)`,
/// где:
/// - `prefactor` — масштабный коэффициент,
/// - `temp_coeff` — параметр температурной чувствительности.
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
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        let (value_second_chunks, value_second_tail) = value_second.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_chunks.len(), value_second_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());
        debug_assert_eq!(x_tail.len(), value_second_tail.len());

        let mut h00 = Vf64::splat(0.0);
        let mut h01 = Vf64::splat(0.0);
        let mut h11 = Vf64::splat(0.0);
        let zero = Vf64::splat(0.0);

        for ((x_chunk, value_first_chunk), value_second_chunk) in x_chunks
            .iter()
            .zip(value_first_chunks.iter())
            .zip(value_second_chunks.iter())
        {
            let x = positive_x_simd(Vf64::from_array(*x_chunk));
            let exp_term = (params_simd.temp_coeff / x).exp();
            let inv_x = Vf64::splat(1.0) / x;
            let model = params_simd.prefactor * exp_term;
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

            let jac_a = exp_term;
            let jac_b = params_simd.prefactor * exp_term * inv_x;
            let d2_model_dadb = exp_term * inv_x;
            let d2_model_dbdb = params_simd.prefactor * exp_term * inv_x * inv_x;

            h00 += value_second * jac_a * jac_a;
            h01 += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            h11 += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
        }

        hessian[[0, 0]] += h00.reduce_sum();
        hessian[[0, 1]] += h01.reduce_sum();
        hessian[[1, 1]] += h11.reduce_sum();

        for ((&x, &value_first), &value_second) in x_tail
            .iter()
            .zip(value_first_tail.iter())
            .zip(value_second_tail.iter())
        {
            let x = positive_x(x);
            let exp_term = (params.temp_coeff / x).exp();
            let inv_x = 1.0 / x;
            let model = params.prefactor * exp_term;
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a = exp_term;
            let jac_b = params.prefactor * exp_term * inv_x;
            let d2_model_dadb = exp_term * inv_x;
            let d2_model_dbdb = params.prefactor * exp_term * inv_x * inv_x;

            hessian[[0, 0]] += value_second * jac_a * jac_a;
            hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
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
        let value = value_at(&[1.2, 0.5], 2.0);
        assert_near(value, 1.2 * (0.25_f64).exp(), 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Arrhenius,
            &[0.4, 0.8, 1.4, 2.5, 4.0],
            &[1.5, 0.9],
            &[1.2, 0.5],
            2e-5,
            3e-4,
        );
    }
}
