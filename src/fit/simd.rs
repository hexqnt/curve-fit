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

pub(super) fn accumulate_polynomial_gradient(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) {
    accumulate_polynomial_gradient_simd(x_values, y_values, param, loss_metric, gradient);
}

pub(super) fn accumulate_inverse_gradient(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_metric: OptimizationLossMetric,
    gradient: &mut [f64],
) {
    accumulate_inverse_gradient_simd(x_values, y_values, param, loss_metric, gradient);
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
        let model = param
            .iter()
            .copied()
            .fold(0.0, |acc, coefficient| acc * x + coefficient);
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
        let model = param
            .iter()
            .copied()
            .fold(0.0, |acc, coefficient| acc * x + coefficient);
        let residual = loss_metric.residual_derivative(model - y);

        let mut basis = 1.0;
        for gradient_value in gradient.iter_mut().rev() {
            *gradient_value += residual * basis;
            basis *= x;
        }
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

fn value_from_residual_simd(loss_metric: OptimizationLossMetric, residual: Vf64) -> Vf64 {
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

fn residual_derivative_simd(loss_metric: OptimizationLossMetric, residual: Vf64) -> Vf64 {
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

        let mut model = Vf64::splat(0.0);
        for coefficient in param.iter().copied() {
            model = model * x + Vf64::splat(coefficient);
        }

        sum += value_from_residual_simd(loss_metric, model - y);
    }

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let model = param
            .iter()
            .copied()
            .fold(0.0, |acc, coefficient| acc * x + coefficient);
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

    let total = sum.reduce_sum() + tail_sum;
    if !total.is_finite() {
        LARGE_COST
    } else {
        total / x_values.len() as f64
    }
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
        sum += value_from_residual_simd(loss_metric, (a + b / x) - y);
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

    let total = sum.reduce_sum() + tail_sum;
    if !total.is_finite() {
        LARGE_COST
    } else {
        total / x_values.len() as f64
    }
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

        let mut model = Vf64::splat(0.0);
        for coefficient in param.iter().copied() {
            model = model * x + Vf64::splat(coefficient);
        }
        let residual_derivative = residual_derivative_simd(loss_metric, model - y);

        let mut basis = Vf64::splat(1.0);
        for accum_value in accum.iter_mut().rev() {
            *accum_value += residual_derivative * basis;
            basis *= x;
        }
    }

    for (value, accum_value) in gradient.iter_mut().zip(accum.iter().copied()) {
        *value += accum_value.reduce_sum();
    }

    for (&x, &y) in x_tail.iter().zip(y_tail.iter()) {
        let model = param
            .iter()
            .copied()
            .fold(0.0, |acc, coefficient| acc * x + coefficient);
        let residual = loss_metric.residual_derivative(model - y);

        let mut basis = 1.0;
        for gradient_value in gradient.iter_mut().rev() {
            *gradient_value += residual * basis;
            basis *= x;
        }
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
        let residual_derivative = residual_derivative_simd(loss_metric, (a + b / x) - y);
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
