use super::common::{Vf64, ln_1p_simd, sigmoid_simd};
use super::common::{
    is_finite_non_negative, scale_and_mirror_upper_hessian, sigmoid, softplus as math_softplus,
};
use ndarray::Array2;
use std::simd::Select;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 4;

#[inline]
fn softplus_simd(value: Vf64) -> Vf64 {
    let zero = Vf64::splat(0.0);
    let positive_branch = value + ln_1p_simd((-value).exp());
    let negative_branch = ln_1p_simd(value.exp());
    value.simd_gt(zero).select(positive_branch, negative_branch)
}

#[derive(Clone, Copy)]
struct Params<T> {
    amplitude: T,
    slope: T,
    x0: T,
    offset: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [amplitude, slope, x0, offset]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            amplitude,
            slope,
            x0,
            offset,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            amplitude: Vf64::splat(self.amplitude),
            slope: Vf64::splat(self.slope),
            x0: Vf64::splat(self.x0),
            offset: Vf64::splat(self.offset),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        self.amplitude * math_softplus(self.slope * (x - self.x0)) + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let z = self.slope * (x - self.x0);
        let softplus_z = math_softplus(z);
        let sigma_z = sigmoid(z);

        grad[0] = softplus_z;
        grad[1] = self.amplitude * sigma_z * (x - self.x0);
        grad[2] = -self.amplitude * sigma_z * self.slope;
        grad[3] = 1.0;

        self.amplitude * softplus_z + self.offset
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        self.amplitude * softplus_simd(self.slope * (x - self.x0)) + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x_centered = x - self.x0;
        let z = self.slope * x_centered;
        let softplus_z = softplus_simd(z);
        let sigma_z = sigmoid_simd(z);

        grad[0] = softplus_z;
        grad[1] = self.amplitude * sigma_z * x_centered;
        grad[2] = -self.amplitude * sigma_z * self.slope;
        grad[3] = Vf64::splat(1.0);

        self.amplitude * softplus_z + self.offset
    }
}

/// Вычисляет softplus-переход:
/// `f(x) = amplitude * softplus(slope * (x - x0)) + offset`,
/// где:
/// - `amplitude` — масштаб перехода,
/// - `slope` — крутизна перехода,
/// - `x0` — центр перехода по оси `x`,
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
            let u = x - params_simd.x0;
            let z = params_simd.slope * u;
            let softplus_z = softplus_simd(z);
            let sigma_z = sigmoid_simd(z);
            let d2_shape_dz2 = sigma_z * (Vf64::splat(1.0) - sigma_z);
            let model = params_simd.amplitude * softplus_z + params_simd.offset;
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

            let jac_a = softplus_z;
            let jac_b = params_simd.amplitude * sigma_z * u;
            let jac_c = -params_simd.amplitude * sigma_z * params_simd.slope;
            let jac_d = Vf64::splat(1.0);
            let d2_model_dadb = sigma_z * u;
            let d2_model_dadc = -sigma_z * params_simd.slope;
            let d2_model_dbdb = params_simd.amplitude * d2_shape_dz2 * u * u;
            let d2_model_dbdc =
                -params_simd.amplitude * (params_simd.slope * u * d2_shape_dz2 + sigma_z);
            let d2_model_dcdc =
                params_simd.amplitude * d2_shape_dz2 * params_simd.slope * params_simd.slope;

            h00 += value_second * jac_a * jac_a;
            h01 += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            h02 += value_second * jac_a * jac_c + value_first * d2_model_dadc;
            h03 += value_second * jac_a * jac_d;
            h11 += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
            h12 += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
            h13 += value_second * jac_b * jac_d;
            h22 += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
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
            let u = x - params.x0;
            let z = params.slope * u;
            let softplus_z = math_softplus(z);
            let sigma_z = sigmoid(z);
            let d2_shape_dz2 = sigma_z * (1.0 - sigma_z);
            let model = params.amplitude * softplus_z + params.offset;
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a = softplus_z;
            let jac_b = params.amplitude * sigma_z * u;
            let jac_c = -params.amplitude * sigma_z * params.slope;
            let jac_d = 1.0;
            let d2_model_dadb = sigma_z * u;
            let d2_model_dadc = -sigma_z * params.slope;
            let d2_model_dbdb = params.amplitude * d2_shape_dz2 * u * u;
            let d2_model_dbdc = -params.amplitude * (params.slope * u * d2_shape_dz2 + sigma_z);
            let d2_model_dcdc = params.amplitude * d2_shape_dz2 * params.slope * params.slope;

            hessian[[0, 0]] += value_second * jac_a * jac_a;
            hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
            hessian[[0, 2]] += value_second * jac_a * jac_c + value_first * d2_model_dadc;
            hessian[[0, 3]] += value_second * jac_a * jac_d;
            hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
            hessian[[1, 2]] += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
            hessian[[1, 3]] += value_second * jac_b * jac_d;
            hessian[[2, 2]] += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
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
    use super::value_simd_at;
    use crate::domain::CurveFamily;
    use crate::models::common::Vf64;
    use crate::models::test_support::assert_near;
    use crate::models::{
        softplus, test_support::assert_family_gradient_and_hessian_match_numerical_reference,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[1.3, 0.6, -0.4, 0.2], 0.6);
        let expected = 1.3 * softplus(0.6) + 0.2;
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Softplus,
            &[-2.0, -1.1, -0.4, 0.3, 1.0, 1.9],
            &[1.3, 0.7, 0.2, 0.2],
            &[1.0, 0.5, -0.1, 0.0],
            4e-5,
            1e-3,
        );
    }

    #[test]
    fn simd_value_matches_scalar_near_zero_transition() {
        let param = [1.3, 0.6, -0.4, 0.2];
        let [_, _, x0, _] = param;
        let mut x_values = [0.0; Vf64::LEN];
        let center = (Vf64::LEN as f64 - 1.0) * 0.5;
        for (index, x) in x_values.iter_mut().enumerate() {
            *x = x0 + (index as f64 - center) * 1e-9;
        }

        let simd = value_simd_at(&param, Vf64::from_array(x_values)).to_array();
        for (index, &x) in x_values.iter().enumerate() {
            let scalar = value_at(&param, x);
            assert_near(simd[index], scalar, 1e-12);
        }
    }
}
