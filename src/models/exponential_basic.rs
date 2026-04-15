use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

/// Вычисляет базовую экспоненциальную кривую:
/// `f(x) = offset + amplitude * exp(-decay_rate * x)`,
/// где:
/// - `offset` — базовый уровень,
/// - `amplitude` — амплитуда экспоненциальной части,
/// - `decay_rate` — коэффициент затухания.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];
    offset + amplitude * (-decay_rate * x).exp()
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let offset = Vf64::splat(param[0]);
    let amplitude = Vf64::splat(param[1]);
    let decay_rate = Vf64::splat(param[2]);
    offset + amplitude * (-decay_rate * x).exp()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];
    let exp_part = (-decay_rate * x).exp();

    grad[0] = 1.0;
    grad[1] = exp_part;
    grad[2] = -amplitude * x * exp_part;

    offset + amplitude * exp_part
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 3]) -> Vf64 {
    let offset = Vf64::splat(param[0]);
    let amplitude = Vf64::splat(param[1]);
    let decay_rate = Vf64::splat(param[2]);
    let exp_part = (-decay_rate * x).exp();

    grad[0] = Vf64::splat(1.0);
    grad[1] = exp_part;
    grad[2] = -amplitude * x * exp_part;

    offset + amplitude * exp_part
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    {
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());

        let mut point_grad = [Vf64::splat(0.0); 3];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);
        let mut gradient_2 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);

            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
            gradient_2 += upstream * point_grad[2];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();
        gradient[2] += gradient_2.reduce_sum();

        let mut point_grad = [0.0; 3];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);

            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
            gradient[2] += upstream * point_grad[2];
        }
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((3, 3)));
    }
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((3, 3));
    let amplitude = param[1];
    let decay_rate = param[2];

    {
        let amplitude = Vf64::splat(amplitude);
        let decay_rate = Vf64::splat(decay_rate);
        let offset = Vf64::splat(param[0]);
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
            let exp_part = (-decay_rate * x).exp();
            let model = offset + amplitude * exp_part;
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

            let jac_a = Vf64::splat(1.0);
            let jac_b = exp_part;
            let jac_c = -amplitude * x * exp_part;
            let d2_model_dbdc = -x * exp_part;
            let d2_model_dcdc = amplitude * x * x * exp_part;

            h00 += value_second * jac_a * jac_a;
            h01 += value_second * jac_a * jac_b;
            h02 += value_second * jac_a * jac_c;
            h11 += value_second * jac_b * jac_b;
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
            let exp_part = (-param[2] * x).exp();
            let model = value_at(param, x);
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a = 1.0;
            let jac_b = exp_part;
            let jac_c = -param[1] * x * exp_part;
            let d2_model_dbdc = -x * exp_part;
            let d2_model_dcdc = param[1] * x * x * exp_part;

            hessian[[0, 0]] += value_second * jac_a * jac_a;
            hessian[[0, 1]] += value_second * jac_a * jac_b;
            hessian[[0, 2]] += value_second * jac_a * jac_c;
            hessian[[1, 1]] += value_second * jac_b * jac_b;
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
        let value = value_at(&[0.2, 1.5, 0.4], 2.0);
        let expected = 0.2 + 1.5 * (-0.8_f64).exp();
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::ExponentialBasic,
            &[-1.0, -0.2, 0.3, 1.1, 2.0],
            &[0.8, 1.4, 0.6],
            &[0.5, 1.1, 0.3],
            2e-5,
            3e-4,
        );
    }
}
