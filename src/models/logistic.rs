use super::common::{Vf64, sigmoid_simd};
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, sigmoid};
use ndarray::Array2;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 3;

#[derive(Clone, Copy)]
struct Params {
    upper_asymptote: f64,
    slope: f64,
    x0: f64,
}

impl Params {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [upper_asymptote, slope, x0]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {PARAM_COUNT} params"));
        Self {
            upper_asymptote,
            slope,
            x0,
        }
    }

    #[inline]
    fn simd(self) -> SimdParams {
        SimdParams {
            upper_asymptote: Vf64::splat(self.upper_asymptote),
            slope: Vf64::splat(self.slope),
            x0: Vf64::splat(self.x0),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let z = self.slope * (x - self.x0);
        self.upper_asymptote * sigmoid(z)
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let z = self.slope * (x - self.x0);
        let s = sigmoid(z);
        let ds_dz = s * (1.0 - s);

        grad[0] = s;
        grad[1] = self.upper_asymptote * ds_dz * (x - self.x0);
        grad[2] = -self.upper_asymptote * ds_dz * self.slope;

        self.upper_asymptote * s
    }
}

#[derive(Clone, Copy)]
struct SimdParams {
    upper_asymptote: Vf64,
    slope: Vf64,
    x0: Vf64,
}

impl SimdParams {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let z = self.slope * (x - self.x0);
        self.upper_asymptote * sigmoid_simd(z)
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x_centered = x - self.x0;
        let z = self.slope * x_centered;
        let s = sigmoid_simd(z);
        let ds_dz = s * (Vf64::splat(1.0) - s);

        grad[0] = s;
        grad[1] = self.upper_asymptote * ds_dz * x_centered;
        grad[2] = -self.upper_asymptote * ds_dz * self.slope;

        self.upper_asymptote * s
    }
}

/// Вычисляет значение логистической кривой:
/// `f(x) = upper_asymptote / (1 + exp(-slope * (x - x0)))`,
/// где:
/// - `upper_asymptote` — амплитуда (верхняя асимптота),
/// - `slope` — крутизна перехода,
/// - `x0` — положение точки перегиба по оси `x`.
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

    // Модель должна иметь ровно 3 параметра: upper_asymptote, slope, x0.
    if param.len() != PARAM_COUNT {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((PARAM_COUNT, PARAM_COUNT)));
    }

    let sample_scale = 1.0 / sample_count as f64;
    let params = Params::parse(param);
    let params_simd = params.simd();

    let mut hessian = Array2::zeros((PARAM_COUNT, PARAM_COUNT));

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
            let z = params_simd.slope * u;
            let s = sigmoid_simd(z);
            let ds_dz = s * (Vf64::splat(1.0) - s);
            let d2s_dz2 = ds_dz * (Vf64::splat(1.0) - Vf64::splat(2.0) * s);
            let model = params_simd.upper_asymptote * s;
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

            let jac_a = s;
            let jac_b = params_simd.upper_asymptote * ds_dz * u;
            let jac_c = -params_simd.upper_asymptote * ds_dz * params_simd.slope;

            let d2_model_dadb = ds_dz * u;
            let d2_model_dadc = -ds_dz * params_simd.slope;
            let d2_model_dbdb = params_simd.upper_asymptote * d2s_dz2 * u * u;
            let d2_model_dbdc =
                -params_simd.upper_asymptote * (params_simd.slope * u * d2s_dz2 + ds_dz);
            let d2_model_dcdc =
                params_simd.upper_asymptote * d2s_dz2 * params_simd.slope * params_simd.slope;

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
            let z = params.slope * u;
            let s = sigmoid(z);
            let ds_dz = s * (1.0 - s);
            let d2s_dz2 = ds_dz * (1.0 - 2.0 * s);
            let model = params.upper_asymptote * s;
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a = s;
            let jac_b = params.upper_asymptote * ds_dz * u;
            let jac_c = -params.upper_asymptote * ds_dz * params.slope;
            let d2_model_dadb = ds_dz * u;
            let d2_model_dadc = -ds_dz * params.slope;
            let d2_model_dbdb = params.upper_asymptote * d2s_dz2 * u * u;
            let d2_model_dbdc = -params.upper_asymptote * (params.slope * u * d2s_dz2 + ds_dz);
            let d2_model_dcdc = params.upper_asymptote * d2s_dz2 * params.slope * params.slope;

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
    use super::value_at;
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[2.0, 1.5, 0.5], 1.5);
        assert_near(value, 2.0 / (1.0 + (-1.5_f64).exp()), 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Logistic,
            &[-2.0, -1.0, -0.3, 0.4, 1.1, 2.0],
            &[2.2, 1.1, 0.3],
            &[1.8, 0.8, -0.1],
            3e-5,
            6e-4,
        );
    }
}
