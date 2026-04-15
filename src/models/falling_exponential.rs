use super::common::non_zero_param_with_derivative;
use super::common::{Vf64, exp_m1_simd, non_zero_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

/// Вычисляет кривую экспоненциального спада:
/// `f(x) = y0 - (v0 / k) * (1 - exp(-k * x))`,
/// где:
/// - `y0` — начальный уровень,
/// - `v0` — масштаб скорости спада,
/// - `k` — коэффициент спада (параметризован как ненулевой).
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let y0 = param[0];
    let v0 = param[1];
    let k_raw = param[2];
    let (k, _) = non_zero_param_with_derivative(k_raw);
    let one_minus_exp = -(-k * x).exp_m1();
    y0 - (v0 / k) * one_minus_exp
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    let y0 = Vf64::splat(param[0]);
    let v0 = Vf64::splat(param[1]);
    let k_raw = Vf64::splat(param[2]);
    let (k, _) = non_zero_param_with_derivative_simd(k_raw);
    let one_minus_exp = -exp_m1_simd(-k * x);
    y0 - (v0 / k) * one_minus_exp
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let y0 = param[0];
    let v0 = param[1];
    let k_raw = param[2];
    let (k, d_k_raw) = non_zero_param_with_derivative(k_raw);
    let exp_part = (-k * x).exp();
    let one_minus_exp = -(-k * x).exp_m1();
    let d_model_d_v0 = -one_minus_exp / k;
    let d_model_d_k = v0 * (one_minus_exp - k * x * exp_part) / (k * k);

    grad[0] = 1.0;
    grad[1] = d_model_d_v0;
    grad[2] = d_model_d_k * d_k_raw;

    y0 - (v0 / k) * one_minus_exp
}

#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 3]) -> Vf64 {
    let y0 = Vf64::splat(param[0]);
    let v0 = Vf64::splat(param[1]);
    let k_raw = Vf64::splat(param[2]);
    let (k, d_k_raw) = non_zero_param_with_derivative_simd(k_raw);
    let exp_part = (-k * x).exp();
    let one_minus_exp = -exp_m1_simd(-k * x);
    let d_model_d_v0 = -one_minus_exp / k;
    let d_model_d_k = v0 * (one_minus_exp - k * x * exp_part) / (k * k);

    grad[0] = Vf64::splat(1.0);
    grad[1] = d_model_d_v0;
    grad[2] = d_model_d_k * d_k_raw;

    y0 - (v0 / k) * one_minus_exp
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
    _x_values: &[f64],
    _param: &[f64],
    _value_first: &[f64],
    _value_second: &[f64],
) -> Option<Array2<f64>> {
    None
}

#[cfg(test)]
mod tests {
    use super::value_at;
    use super::value_simd_at;
    use crate::domain::CurveFamily;
    use crate::models::common::Vf64;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[1.2, 0.8, 0.6], 0.5);
        let expected = 1.2 - (0.8 / 0.6) * (1.0 - (-0.3_f64).exp());
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::FallingExponential,
            &[0.2, 0.6, 1.1, 1.9, 2.8],
            &[1.4, 0.9, 0.6],
            &[1.1, 0.6, 0.4],
            6e-5,
            2e-3,
        );
    }

    #[test]
    fn simd_value_matches_scalar_near_zero_exponent() {
        let param = [1.1, 0.7, 0.4];
        let mut x_values = [0.0; Vf64::LEN];
        let center = (Vf64::LEN as f64 - 1.0) * 0.5;
        for (index, x) in x_values.iter_mut().enumerate() {
            *x = (index as f64 - center) * 1e-9;
        }

        let simd = value_simd_at(&param, Vf64::from_array(x_values)).to_array();
        for (index, &x) in x_values.iter().enumerate() {
            let scalar = value_at(&param, x);
            assert_near(simd[index], scalar, 1e-12);
        }
    }
}
