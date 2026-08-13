use std::simd::Select;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

use super::{LARGE_COST, OptimizationLossMetric, positive_x};

const MAX_POLYNOMIAL_PARAMS: usize = 10;

#[cfg(not(target_arch = "wasm32"))]
type Vf64 = std::simd::f64x8;
#[cfg(target_arch = "wasm32")]
type Vf64 = std::simd::f64x2;

#[inline]
fn polynomial_value(param: &[f64], x: f64) -> f64 {
    param
        .iter()
        .copied()
        .fold(0.0, |value, coefficient| value * x + coefficient)
}

#[inline]
fn polynomial_value_simd(param: &[f64], x: Vf64) -> Vf64 {
    param
        .iter()
        .copied()
        .fold(Vf64::splat(0.0), |value, coefficient| {
            value * x + Vf64::splat(coefficient)
        })
}

#[inline]
fn accumulate_polynomial_basis(gradient: &mut [f64], x: f64, derivative: f64) {
    let mut basis = 1.0;
    for gradient_value in gradient.iter_mut().rev() {
        *gradient_value += derivative * basis;
        basis *= x;
    }
}

#[inline]
fn accumulate_polynomial_basis_simd(accum: &mut [Vf64], x: Vf64, derivative: Vf64) {
    let mut basis = Vf64::splat(1.0);
    for accum_value in accum.iter_mut().rev() {
        *accum_value += derivative * basis;
        basis *= x;
    }
}

#[inline]
fn mean_loss_or_large(sum: Vf64, tail_sum: f64, sample_count: usize) -> f64 {
    let total = sum.reduce_sum() + tail_sum;
    if total.is_finite() {
        total / sample_count as f64
    } else {
        LARGE_COST
    }
}

pub(super) fn polynomial_cost(
    param: &[f64],
    x_values: &[f64],
    y_values: &[f64],
    loss_metric: OptimizationLossMetric,
) -> f64 {
    polynomial_cost_simd(param, x_values, y_values, loss_metric)
}

pub(super) fn inverse_cost(
    param: &[f64],
    x_values: &[f64],
    y_values: &[f64],
    loss_metric: OptimizationLossMetric,
) -> f64 {
    inverse_cost_simd(param, x_values, y_values, loss_metric)
}

pub(super) fn polynomial_value_gradient(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) -> f64 {
    polynomial_value_gradient_simd(x_values, y_values, param, loss_metric, gradient)
}

pub(super) fn inverse_value_gradient(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) -> f64 {
    inverse_value_gradient_simd(x_values, y_values, param, loss_metric, gradient)
}

pub(super) fn polynomial_cost_scalar(
    param: &[f64],
    x_values: &[f64],
    y_values: &[f64],
    loss_metric: OptimizationLossMetric,
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let model = polynomial_value(param, x);
        let residual = model - y;
        if !residual.is_finite() {
            return LARGE_COST;
        }
        let value = loss_metric.value_from_residual(residual);
        if !value.is_finite() {
            return LARGE_COST;
        }
        sum += value;
        if !sum.is_finite() {
            return LARGE_COST;
        }
    }

    sum / x_values.len() as f64
}

pub(super) fn inverse_cost_scalar(
    param: &[f64],
    x_values: &[f64],
    y_values: &[f64],
    loss_metric: OptimizationLossMetric,
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let &[a, b, ..] = param else {
        unreachable!("inverse model requires two parameters");
    };

    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let x = positive_x(x);
        let residual = (a + b / x) - y;
        if !residual.is_finite() {
            return LARGE_COST;
        }
        let value = loss_metric.value_from_residual(residual);
        if !value.is_finite() {
            return LARGE_COST;
        }
        sum += value;
        if !sum.is_finite() {
            return LARGE_COST;
        }
    }

    sum / x_values.len() as f64
}

pub(super) fn accumulate_polynomial_gradient_scalar(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert_eq!(gradient.len(), param.len());
    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let derivative = loss_metric.residual_derivative(polynomial_value(param, x) - y);
        accumulate_polynomial_basis(gradient, x, derivative);
    }
}

pub(super) fn accumulate_inverse_gradient_scalar(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert!(gradient.len() >= 2);
    let &[a, b, ..] = param else {
        unreachable!("inverse model requires two parameters");
    };
    let [gradient_0, gradient_1, ..] = gradient else {
        unreachable!("inverse gradient requires two parameters");
    };

    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let x = positive_x(x);
        let residual = loss_metric.residual_derivative((a + b / x) - y);
        *gradient_0 += residual;
        *gradient_1 += residual / x;
    }
}

fn loss_value_simd(loss_metric: OptimizationLossMetric, residual: Vf64) -> Vf64 {
    match loss_metric {
        OptimizationLossMetric::Mse => residual * residual,
        OptimizationLossMetric::Mae | OptimizationLossMetric::Chebyshev => residual.abs(),
        OptimizationLossMetric::SoftL1 => {
            let one = Vf64::splat(1.0);
            Vf64::splat(2.0) * ((one + residual * residual).sqrt() - one)
        }
        OptimizationLossMetric::Msle => {
            let one = Vf64::splat(1.0);
            let log_term = (one + residual.abs()).ln();
            log_term * log_term
        }
    }
}

fn loss_residual_derivative_simd(loss_metric: OptimizationLossMetric, residual: Vf64) -> Vf64 {
    match loss_metric {
        OptimizationLossMetric::Mse => Vf64::splat(2.0) * residual,
        OptimizationLossMetric::Mae | OptimizationLossMetric::Chebyshev => {
            let one = Vf64::splat(1.0);
            let zero = Vf64::splat(0.0);
            let gt_zero = residual.simd_gt(zero);
            let lt_zero = residual.simd_lt(zero);
            lt_zero.select(-one, gt_zero.select(one, zero))
        }
        OptimizationLossMetric::SoftL1 => {
            let one = Vf64::splat(1.0);
            Vf64::splat(2.0) * residual / (one + residual * residual).sqrt()
        }
        OptimizationLossMetric::Msle => {
            let one = Vf64::splat(1.0);
            let abs_residual = residual.abs();
            let log_term = (one + abs_residual).ln();
            let magnitude = Vf64::splat(2.0) * log_term / (one + abs_residual);
            let zero = Vf64::splat(0.0);
            let gt_zero = residual.simd_gt(zero);
            let lt_zero = residual.simd_lt(zero);
            lt_zero.select(-magnitude, gt_zero.select(magnitude, zero))
        }
    }
}

fn loss_value_gradient_simd(loss_metric: OptimizationLossMetric, residual: Vf64) -> (Vf64, Vf64) {
    match loss_metric {
        OptimizationLossMetric::Mse => (residual * residual, Vf64::splat(2.0) * residual),
        OptimizationLossMetric::Mae | OptimizationLossMetric::Chebyshev => {
            let one = Vf64::splat(1.0);
            let zero = Vf64::splat(0.0);
            let gt_zero = residual.simd_gt(zero);
            let lt_zero = residual.simd_lt(zero);
            (
                residual.abs(),
                lt_zero.select(-one, gt_zero.select(one, zero)),
            )
        }
        OptimizationLossMetric::SoftL1 => {
            let one = Vf64::splat(1.0);
            let root = (one + residual * residual).sqrt();
            (
                Vf64::splat(2.0) * (root - one),
                Vf64::splat(2.0) * residual / root,
            )
        }
        OptimizationLossMetric::Msle => {
            let one = Vf64::splat(1.0);
            let abs_residual = residual.abs();
            let log_term = (one + abs_residual).ln();
            let magnitude = Vf64::splat(2.0) * log_term / (one + abs_residual);
            let zero = Vf64::splat(0.0);
            let gt_zero = residual.simd_gt(zero);
            let lt_zero = residual.simd_lt(zero);
            (
                log_term * log_term,
                lt_zero.select(-magnitude, gt_zero.select(magnitude, zero)),
            )
        }
    }
}

pub(super) fn polynomial_cost_simd(
    param: &[f64],
    x_values: &[f64],
    y_values: &[f64],
    loss_metric: OptimizationLossMetric,
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }

    let mut sum = Vf64::splat(0.0);
    let mut tail_sum = 0.0;
    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (y_chunks, y_tail) = y_values.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), y_chunks.len());
    debug_assert_eq!(x_tail.len(), y_tail.len());

    for (x_chunk, y_chunk) in x_chunks.iter().zip(y_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk);
        let y = Vf64::from_array(*y_chunk);

        sum += loss_value_simd(loss_metric, polynomial_value_simd(param, x) - y);
    }

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let model = polynomial_value(param, x);
        let residual = model - y;
        if !residual.is_finite() {
            return LARGE_COST;
        }
        let value = loss_metric.value_from_residual(residual);
        if !value.is_finite() {
            return LARGE_COST;
        }
        tail_sum += value;
        if !tail_sum.is_finite() {
            return LARGE_COST;
        }
    }

    mean_loss_or_large(sum, tail_sum, x_values.len())
}

pub(super) fn inverse_cost_simd(
    param: &[f64],
    x_values: &[f64],
    y_values: &[f64],
    loss_metric: OptimizationLossMetric,
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }
    let &[a_scalar, b_scalar, ..] = param else {
        unreachable!("inverse model requires two parameters");
    };

    let mut sum = Vf64::splat(0.0);
    let mut tail_sum = 0.0;
    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (y_chunks, y_tail) = y_values.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), y_chunks.len());
    debug_assert_eq!(x_tail.len(), y_tail.len());

    let a = Vf64::splat(a_scalar);
    let b = Vf64::splat(b_scalar);
    let eps = Vf64::splat(super::PARAM_EPS);
    for (x_chunk, y_chunk) in x_chunks.iter().zip(y_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk).simd_max(eps);
        let y = Vf64::from_array(*y_chunk);
        sum += loss_value_simd(loss_metric, (a + b / x) - y);
    }

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let x = positive_x(x);
        let residual = (a_scalar + b_scalar / x) - y;
        if !residual.is_finite() {
            return LARGE_COST;
        }
        let value = loss_metric.value_from_residual(residual);
        if !value.is_finite() {
            return LARGE_COST;
        }
        tail_sum += value;
        if !tail_sum.is_finite() {
            return LARGE_COST;
        }
    }

    mean_loss_or_large(sum, tail_sum, x_values.len())
}

pub(super) fn accumulate_polynomial_gradient_simd(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert_eq!(gradient.len(), param.len());
    debug_assert!(gradient.len() <= MAX_POLYNOMIAL_PARAMS);

    let mut accum = [Vf64::splat(0.0); MAX_POLYNOMIAL_PARAMS];
    let accum = &mut accum[..gradient.len()];
    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (y_chunks, y_tail) = y_values.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), y_chunks.len());
    debug_assert_eq!(x_tail.len(), y_tail.len());

    for (x_chunk, y_chunk) in x_chunks.iter().zip(y_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk);
        let y = Vf64::from_array(*y_chunk);

        let derivative =
            loss_residual_derivative_simd(loss_metric, polynomial_value_simd(param, x) - y);
        accumulate_polynomial_basis_simd(accum, x, derivative);
    }

    for (value, accum_value) in gradient.iter_mut().zip(accum.iter().copied()) {
        *value += accum_value.reduce_sum();
    }

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let derivative = loss_metric.residual_derivative(polynomial_value(param, x) - y);
        accumulate_polynomial_basis(gradient, x, derivative);
    }
}

pub(super) fn accumulate_inverse_gradient_simd(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert!(gradient.len() >= 2);
    let &[a_scalar, b_scalar, ..] = param else {
        unreachable!("inverse model requires two parameters");
    };
    let [gradient_scalar_0, gradient_scalar_1, ..] = gradient else {
        unreachable!("inverse gradient requires two parameters");
    };

    let mut gradient_0 = Vf64::splat(0.0);
    let mut gradient_1 = Vf64::splat(0.0);
    let a = Vf64::splat(a_scalar);
    let b = Vf64::splat(b_scalar);
    let eps = Vf64::splat(super::PARAM_EPS);

    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (y_chunks, y_tail) = y_values.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), y_chunks.len());
    debug_assert_eq!(x_tail.len(), y_tail.len());

    for (x_chunk, y_chunk) in x_chunks.iter().zip(y_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk).simd_max(eps);
        let y = Vf64::from_array(*y_chunk);
        let residual_derivative = loss_residual_derivative_simd(loss_metric, (a + b / x) - y);
        gradient_0 += residual_derivative;
        gradient_1 += residual_derivative / x;
    }

    *gradient_scalar_0 += gradient_0.reduce_sum();
    *gradient_scalar_1 += gradient_1.reduce_sum();

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let x = positive_x(x);
        let residual = loss_metric.residual_derivative((a_scalar + b_scalar / x) - y);
        *gradient_scalar_0 += residual;
        *gradient_scalar_1 += residual / x;
    }
}

/// За один проход считает средний loss и накапливает немасштабированный градиент полинома.
pub(super) fn polynomial_value_gradient_simd(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert_eq!(gradient.len(), param.len());
    debug_assert!(gradient.len() <= MAX_POLYNOMIAL_PARAMS);
    if x_values.is_empty() {
        return 0.0;
    }

    let mut sum = Vf64::splat(0.0);
    let mut tail_sum = 0.0;
    let mut accum = [Vf64::splat(0.0); MAX_POLYNOMIAL_PARAMS];
    let accum = &mut accum[..gradient.len()];
    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (y_chunks, y_tail) = y_values.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), y_chunks.len());
    debug_assert_eq!(x_tail.len(), y_tail.len());

    for (x_chunk, y_chunk) in x_chunks.iter().zip(y_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk);
        let y = Vf64::from_array(*y_chunk);

        let (value, residual_derivative) =
            loss_value_gradient_simd(loss_metric, polynomial_value_simd(param, x) - y);
        sum += value;
        accumulate_polynomial_basis_simd(accum, x, residual_derivative);
    }

    for (value, accum_value) in gradient.iter_mut().zip(accum.iter().copied()) {
        *value += accum_value.reduce_sum();
    }

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let residual = polynomial_value(param, x) - y;
        tail_sum += loss_metric.value_from_residual(residual);
        accumulate_polynomial_basis(gradient, x, loss_metric.residual_derivative(residual));
    }

    mean_loss_or_large(sum, tail_sum, x_values.len())
}

/// За один проход считает средний loss и накапливает немасштабированный градиент обратной модели.
pub(super) fn inverse_value_gradient_simd(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    debug_assert!(gradient.len() >= 2);
    if x_values.is_empty() {
        return 0.0;
    }
    let &[a_scalar, b_scalar, ..] = param else {
        unreachable!("inverse model requires two parameters");
    };
    let [gradient_scalar_0, gradient_scalar_1, ..] = gradient else {
        unreachable!("inverse gradient requires two parameters");
    };

    let mut sum = Vf64::splat(0.0);
    let mut tail_sum = 0.0;
    let mut gradient_0 = Vf64::splat(0.0);
    let mut gradient_1 = Vf64::splat(0.0);
    let a = Vf64::splat(a_scalar);
    let b = Vf64::splat(b_scalar);
    let eps = Vf64::splat(super::PARAM_EPS);
    let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
    let (y_chunks, y_tail) = y_values.as_chunks::<{ Vf64::LEN }>();
    debug_assert_eq!(x_chunks.len(), y_chunks.len());
    debug_assert_eq!(x_tail.len(), y_tail.len());

    for (x_chunk, y_chunk) in x_chunks.iter().zip(y_chunks.iter()) {
        let x = Vf64::from_array(*x_chunk).simd_max(eps);
        let y = Vf64::from_array(*y_chunk);
        let inv_x = Vf64::splat(1.0) / x;
        let (value, residual_derivative) =
            loss_value_gradient_simd(loss_metric, (a + b * inv_x) - y);
        sum += value;
        gradient_0 += residual_derivative;
        gradient_1 += residual_derivative * inv_x;
    }

    *gradient_scalar_0 += gradient_0.reduce_sum();
    *gradient_scalar_1 += gradient_1.reduce_sum();

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let inv_x = 1.0 / positive_x(x);
        let residual = (a_scalar + b_scalar * inv_x) - y;
        tail_sum += loss_metric.value_from_residual(residual);
        let residual_derivative = loss_metric.residual_derivative(residual);
        *gradient_scalar_0 += residual_derivative;
        *gradient_scalar_1 += residual_derivative * inv_x;
    }

    mean_loss_or_large(sum, tail_sum, x_values.len())
}
