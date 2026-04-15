use super::common::Vf64;
use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

/// Вычисляет сумму двух экспонент:
/// `f(x) = a1 * exp(-k1 * x) + a2 * exp(-k2 * x) + offset`,
/// где:
/// - `a1`, `a2` — амплитуды экспоненциальных компонент,
/// - `k1`, `k2` — коэффициенты затухания компонент,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let a1 = param[0];
    let k1 = param[1];
    let a2 = param[2];
    let k2 = param[3];
    let offset = param[4];
    a1 * (-k1 * x).exp() + a2 * (-k2 * x).exp() + offset
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let a1 = Vf64::splat(param[0]);
    let k1 = Vf64::splat(param[1]);
    let a2 = Vf64::splat(param[2]);
    let k2 = Vf64::splat(param[3]);
    let offset = Vf64::splat(param[4]);
    a1 * (-k1 * x).exp() + a2 * (-k2 * x).exp() + offset
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 5);

    let a1 = param[0];
    let k1 = param[1];
    let a2 = param[2];
    let k2 = param[3];
    let offset = param[4];
    let exp1 = (-k1 * x).exp();
    let exp2 = (-k2 * x).exp();

    grad[0] = exp1;
    grad[1] = -a1 * x * exp1;
    grad[2] = exp2;
    grad[3] = -a2 * x * exp2;
    grad[4] = 1.0;

    a1 * exp1 + a2 * exp2 + offset
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 5]) -> Vf64 {
    let a1 = Vf64::splat(param[0]);
    let k1 = Vf64::splat(param[1]);
    let a2 = Vf64::splat(param[2]);
    let k2 = Vf64::splat(param[3]);
    let offset = Vf64::splat(param[4]);
    let exp1 = (-k1 * x).exp();
    let exp2 = (-k2 * x).exp();

    grad[0] = exp1;
    grad[1] = -a1 * x * exp1;
    grad[2] = exp2;
    grad[3] = -a2 * x * exp2;
    grad[4] = Vf64::splat(1.0);

    a1 * exp1 + a2 * exp2 + offset
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

        let mut point_grad = [Vf64::splat(0.0); 5];
        let mut gradient_0 = Vf64::splat(0.0);
        let mut gradient_1 = Vf64::splat(0.0);
        let mut gradient_2 = Vf64::splat(0.0);
        let mut gradient_3 = Vf64::splat(0.0);
        let mut gradient_4 = Vf64::splat(0.0);

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            value_grad_simd_at(param, x, &mut point_grad);

            gradient_0 += upstream * point_grad[0];
            gradient_1 += upstream * point_grad[1];
            gradient_2 += upstream * point_grad[2];
            gradient_3 += upstream * point_grad[3];
            gradient_4 += upstream * point_grad[4];
        }

        gradient[0] += gradient_0.reduce_sum();
        gradient[1] += gradient_1.reduce_sum();
        gradient[2] += gradient_2.reduce_sum();
        gradient[3] += gradient_3.reduce_sum();
        gradient[4] += gradient_4.reduce_sum();

        let mut point_grad = [0.0; 5];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            value_grad_at(param, x, &mut point_grad);

            gradient[0] += upstream * point_grad[0];
            gradient[1] += upstream * point_grad[1];
            gradient[2] += upstream * point_grad[2];
            gradient[3] += upstream * point_grad[3];
            gradient[4] += upstream * point_grad[4];
        }
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != 5 {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((5, 5)));
    }
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((5, 5));
    let a1 = param[0];
    let k1 = param[1];
    let a2 = param[2];
    let k2 = param[3];

    {
        let a1 = Vf64::splat(a1);
        let k1 = Vf64::splat(k1);
        let a2 = Vf64::splat(a2);
        let k2 = Vf64::splat(k2);
        let offset = Vf64::splat(param[4]);
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
            let exp1 = (-k1 * x).exp();
            let exp2 = (-k2 * x).exp();
            let model = a1 * exp1 + a2 * exp2 + offset;
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
            let jac_k1 = -a1 * x * exp1;
            let jac_a2 = exp2;
            let jac_k2 = -a2 * x * exp2;
            let jac_c = Vf64::splat(1.0);
            let d2_model_da1dk1 = -x * exp1;
            let d2_model_dk1dk1 = a1 * x * x * exp1;
            let d2_model_da2dk2 = -x * exp2;
            let d2_model_dk2dk2 = a2 * x * x * exp2;

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
            let exp1 = (-param[1] * x).exp();
            let exp2 = (-param[3] * x).exp();
            let model = value_at(param, x);
            if !model.is_finite() {
                return None;
            }

            if !value_first.is_finite() || !is_finite_non_negative(value_second) {
                return None;
            }

            let jac_a1 = exp1;
            let jac_k1 = -param[0] * x * exp1;
            let jac_a2 = exp2;
            let jac_k2 = -param[2] * x * exp2;
            let jac_c = 1.0;
            let d2_model_da1dk1 = -x * exp1;
            let d2_model_dk1dk1 = param[0] * x * x * exp1;
            let d2_model_da2dk2 = -x * exp2;
            let d2_model_dk2dk2 = param[2] * x * x * exp2;

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
