use ndarray::Array2;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Vf64 = std::simd::f64x8;
#[cfg(target_arch = "wasm32")]
pub(crate) type Vf64 = std::simd::f64x2;

use std::simd::Select;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

pub(crate) const PARAM_EPS: f64 = 1e-9;
pub(crate) const HESSIAN_DIAGONAL_JITTER: f64 = 1e-9;

#[inline]
pub(crate) fn positive_x(value: f64) -> f64 {
    value.max(PARAM_EPS)
}

#[inline]
pub(crate) fn positive_x_simd(value: Vf64) -> Vf64 {
    value.simd_max(Vf64::splat(PARAM_EPS))
}

#[inline]
pub(crate) fn positive_param_with_derivative(value: f64) -> (f64, f64) {
    if value.abs() >= PARAM_EPS {
        (value.abs(), value.signum())
    } else {
        (PARAM_EPS, 0.0)
    }
}

#[inline]
pub(crate) fn positive_param_with_derivative_simd(value: Vf64) -> (Vf64, Vf64) {
    let eps = Vf64::splat(PARAM_EPS);
    let abs_value = value.abs();
    let use_value = abs_value.simd_ge(eps);
    let derivative = use_value.select(value.signum(), Vf64::splat(0.0));
    (use_value.select(abs_value, eps), derivative)
}

#[inline]
pub(crate) fn non_zero_param_with_derivative(value: f64) -> (f64, f64) {
    if value.abs() >= PARAM_EPS {
        (value, 1.0)
    } else if value.is_sign_negative() {
        (-PARAM_EPS, 0.0)
    } else {
        (PARAM_EPS, 0.0)
    }
}

#[inline]
pub(crate) fn non_zero_param_with_derivative_simd(value: Vf64) -> (Vf64, Vf64) {
    let eps = Vf64::splat(PARAM_EPS);
    let use_value = value.abs().simd_ge(eps);
    let coerced = value.is_sign_negative().select(-eps, eps);
    let derivative = use_value.select(Vf64::splat(1.0), Vf64::splat(0.0));
    (use_value.select(value, coerced), derivative)
}

#[inline]
pub(crate) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[inline]
pub(crate) fn sigmoid_simd(value: Vf64) -> Vf64 {
    let zero = Vf64::splat(0.0);
    let one = Vf64::splat(1.0);
    let non_negative = value.simd_ge(zero);
    let positive_branch = one / (one + (-value).exp());
    let exp_value = value.exp();
    let negative_branch = exp_value / (one + exp_value);
    non_negative.select(positive_branch, negative_branch)
}

#[inline]
pub(crate) fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

#[inline]
pub(crate) fn ln_1p_simd(value: Vf64) -> Vf64 {
    Vf64::from_array(value.to_array().map(f64::ln_1p))
}

#[inline]
pub(crate) fn exp_m1_simd(value: Vf64) -> Vf64 {
    Vf64::from_array(value.to_array().map(f64::exp_m1))
}

#[inline]
pub(crate) fn erf_approx(value: f64) -> f64 {
    let sign = value.signum();
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
        * t
        + 0.254829592)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[inline]
pub(crate) fn erf_approx_simd(value: Vf64) -> Vf64 {
    let sign = value.signum();
    let x = value.abs();
    let t = Vf64::splat(1.0) / (Vf64::splat(1.0) + Vf64::splat(0.3275911) * x);
    let polynomial = (((((Vf64::splat(1.061405429) * t - Vf64::splat(1.453152027)) * t)
        + Vf64::splat(1.421413741))
        * t
        - Vf64::splat(0.284496736))
        * t
        + Vf64::splat(0.254829592))
        * t;
    sign * (Vf64::splat(1.0) - polynomial * (-(x * x)).exp())
}

#[inline]
pub(crate) fn erfc_approx(value: f64) -> f64 {
    1.0 - erf_approx(value)
}

#[inline]
pub(crate) fn erfc_approx_simd(value: Vf64) -> Vf64 {
    Vf64::splat(1.0) - erf_approx_simd(value)
}

#[inline]
pub(crate) fn is_finite_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

pub(crate) fn scale_and_mirror_upper_hessian(hessian: &mut Array2<f64>, scale: f64) {
    let dimension = hessian.nrows();
    debug_assert_eq!(dimension, hessian.ncols());
    let mut row = 0;
    while row < dimension {
        let mut column = row;
        while column < dimension {
            let value = hessian[[row, column]] * scale;
            hessian[[row, column]] = value;
            hessian[[column, row]] = value;
            column += 1;
        }
        row += 1;
    }
}

pub(crate) fn stabilize_hessian(hessian: &mut Array2<f64>) {
    let dimension = hessian.nrows();
    debug_assert_eq!(dimension, hessian.ncols());
    let mut row = 0;
    while row < dimension {
        let mut column = row + 1;
        while column < dimension {
            let value = 0.5 * (hessian[[row, column]] + hessian[[column, row]]);
            hessian[[row, column]] = value;
            hessian[[column, row]] = value;
            column += 1;
        }
        if !hessian[[row, row]].is_finite() {
            hessian[[row, row]] = 0.0;
        }
        hessian[[row, row]] += HESSIAN_DIAGONAL_JITTER;
        row += 1;
    }
}
