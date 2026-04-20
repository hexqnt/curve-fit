use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 3;

#[derive(Clone, Copy)]
struct Params<T> {
    upper_asymptote: T,
    growth_rate: T,
    x0: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [upper_asymptote, growth_rate, x0]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            upper_asymptote,
            growth_rate,
            x0,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            upper_asymptote: Vf64::splat(self.upper_asymptote),
            growth_rate: Vf64::splat(self.growth_rate),
            x0: Vf64::splat(self.x0),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let inner = (-self.growth_rate * (x - self.x0)).exp();
        self.upper_asymptote * (-inner).exp()
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let x_centered = x - self.x0;
        let exp_inner = (-self.growth_rate * x_centered).exp();
        let exp_outer = (-exp_inner).exp();

        grad[0] = exp_outer;
        grad[1] = self.upper_asymptote * exp_outer * exp_inner * x_centered;
        grad[2] = -self.upper_asymptote * exp_outer * exp_inner * self.growth_rate;

        self.upper_asymptote * exp_outer
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let inner = (-self.growth_rate * (x - self.x0)).exp();
        self.upper_asymptote * (-inner).exp()
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x_centered = x - self.x0;
        let exp_inner = (-self.growth_rate * x_centered).exp();
        let exp_outer = (-exp_inner).exp();

        grad[0] = exp_outer;
        grad[1] = self.upper_asymptote * exp_outer * exp_inner * x_centered;
        grad[2] = -self.upper_asymptote * exp_outer * exp_inner * self.growth_rate;

        self.upper_asymptote * exp_outer
    }
}

/// Вычисляет кривую Гомпертца:
/// `f(x) = upper_asymptote * exp(-exp(-growth_rate * (x - x0)))`,
/// где:
/// - `upper_asymptote` — верхняя асимптота,
/// - `growth_rate` — скорость роста,
/// - `x0` — положение точки перегиба.
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
    debug_assert_eq!(x_values.len(), value_first.len());
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
        let mut h11 = Vf64::splat(0.0);
        let mut h12 = Vf64::splat(0.0);
        let mut h22 = Vf64::splat(0.0);

        for ((x_chunk, value_first_chunk), value_second_chunk) in x_chunks
            .iter()
            .zip(value_first_chunks.iter())
            .zip(value_second_chunks.iter())
        {
            let x = Vf64::from_array(*x_chunk);
            let u = x - params_simd.x0;
            let exp_inner = (-params_simd.growth_rate * u).exp();
            let exp_outer = (-exp_inner).exp();
            let exp_product = exp_outer * exp_inner;
            let d2_shape_dz2 = exp_product * (exp_inner - Vf64::splat(1.0));
            let model = params_simd.upper_asymptote * exp_outer;
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

            let jac_a = exp_outer;
            let jac_b = params_simd.upper_asymptote * exp_product * u;
            let jac_c = -params_simd.upper_asymptote * exp_product * params_simd.growth_rate;

            let d2_model_dadb = exp_product * u;
            let d2_model_dadc = -exp_product * params_simd.growth_rate;
            let d2_model_dbdb = params_simd.upper_asymptote * d2_shape_dz2 * u * u;
            let d2_model_dbdc = -params_simd.upper_asymptote
                * (params_simd.growth_rate * u * d2_shape_dz2 + exp_product);
            let d2_model_dcdc = params_simd.upper_asymptote
                * d2_shape_dz2
                * params_simd.growth_rate
                * params_simd.growth_rate;

            h00 += value_second * jac_a * jac_a;
            h01 += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            h02 += value_second * jac_a * jac_c + value_first * d2_model_dadc;
            h11 += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
            h12 += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
            h22 += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
        }

        hessian[[0, 0]] += h00.reduce_sum();
        hessian[[0, 1]] += h01.reduce_sum();
        hessian[[0, 2]] += h02.reduce_sum();
        hessian[[1, 1]] += h11.reduce_sum();
        hessian[[1, 2]] += h12.reduce_sum();
        hessian[[2, 2]] += h22.reduce_sum();

        for ((&x, &value_first), &value_second) in x_tail
            .iter()
            .zip(value_first_tail.iter())
            .zip(value_second_tail.iter())
        {
            let u = x - params.x0;
            let exp_inner = (-params.growth_rate * u).exp();
            let exp_outer = (-exp_inner).exp();
            let exp_product = exp_outer * exp_inner;
            let d2_shape_dz2 = exp_product * (exp_inner - 1.0);
            let model = params.upper_asymptote * exp_outer;
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a = exp_outer;
            let jac_b = params.upper_asymptote * exp_product * u;
            let jac_c = -params.upper_asymptote * exp_product * params.growth_rate;
            let d2_model_dadb = exp_product * u;
            let d2_model_dadc = -exp_product * params.growth_rate;
            let d2_model_dbdb = params.upper_asymptote * d2_shape_dz2 * u * u;
            let d2_model_dbdc =
                -params.upper_asymptote * (params.growth_rate * u * d2_shape_dz2 + exp_product);
            let d2_model_dcdc =
                params.upper_asymptote * d2_shape_dz2 * params.growth_rate * params.growth_rate;

            hessian[[0, 0]] += value_second * jac_a * jac_a;
            hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            hessian[[0, 2]] += value_second * jac_a * jac_c + value_first * d2_model_dadc;
            hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
            hessian[[1, 2]] += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
            hessian[[2, 2]] += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
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
        let value = value_at(&[1.7, 0.8, -0.2], 0.3);
        let inner = (-0.8_f64 * 0.5).exp();
        assert_near(value, 1.7 * (-inner).exp(), 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Gompertz,
            &[-1.5, -0.8, -0.2, 0.6, 1.4, 2.3],
            &[1.9, 0.9, 0.2],
            &[1.4, 0.6, -0.2],
            3e-5,
            6e-4,
        );
    }

    #[test]
    fn raw_hessian_is_zero_for_empty_dataset() {
        let hessian = add_value_grad_raw_hessian(&[], &[1.0, 0.8, 0.0], &[], &[])
            .expect("empty dataset must produce zero hessian");
        assert_eq!(hessian.shape(), &[3, 3]);
        assert!(hessian.iter().all(|&value| value == 0.0));
    }
}
