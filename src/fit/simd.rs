use super::{LARGE_COST, OptimizationLossMetric, positive_x};

const MAX_POLYNOMIAL_PARAMS: usize = 10;

#[cfg(not(target_arch = "wasm32"))]
type Vf64 = std::simd::f64x8;
#[cfg(target_arch = "wasm32")]
type Vf64 = std::simd::f64x2;

use std::simd::Select;
use std::simd::StdFloat;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

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
    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
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
        index += 1;
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
    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let residual = (param[0] + param[1] / x) - y;
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
        index += 1;
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
    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
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
        index += 1;
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
    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let residual = loss_metric.residual_derivative((param[0] + param[1] / x) - y);
        gradient[0] += residual;
        gradient[1] += residual / x;
        index += 1;
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
    let mut index = 0;
    while index + Vf64::LEN <= x_values.len() {
        let x = Vf64::from_slice(&x_values[index..index + Vf64::LEN]);
        let y = Vf64::from_slice(&y_values[index..index + Vf64::LEN]);

        let mut model = Vf64::splat(0.0);
        for coefficient in param.iter().copied() {
            model = model * x + Vf64::splat(coefficient);
        }

        sum += value_from_residual_simd(loss_metric, model - y);
        index += Vf64::LEN;
    }

    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
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
        index += 1;
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

    let mut sum = Vf64::splat(0.0);
    let mut tail_sum = 0.0;
    let mut index = 0;
    let a = Vf64::splat(param[0]);
    let b = Vf64::splat(param[1]);
    let eps = Vf64::splat(super::PARAM_EPS);
    while index + Vf64::LEN <= x_values.len() {
        let x = Vf64::from_slice(&x_values[index..index + Vf64::LEN]).simd_max(eps);
        let y = Vf64::from_slice(&y_values[index..index + Vf64::LEN]);
        sum += value_from_residual_simd(loss_metric, (a + b / x) - y);
        index += Vf64::LEN;
    }

    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let residual = (param[0] + param[1] / x) - y;
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
        index += 1;
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
    let mut index = 0;
    while index + Vf64::LEN <= x_values.len() {
        let x = Vf64::from_slice(&x_values[index..index + Vf64::LEN]);
        let y = Vf64::from_slice(&y_values[index..index + Vf64::LEN]);

        let mut model = Vf64::splat(0.0);
        for coefficient in param.iter().copied() {
            model = model * x + Vf64::splat(coefficient);
        }
        let residual_derivative = residual_derivative_simd(loss_metric, model - y);

        let mut basis = Vf64::splat(1.0);
        for gradient_index in (0..gradient.len()).rev() {
            accum[gradient_index] += residual_derivative * basis;
            basis *= x;
        }
        index += Vf64::LEN;
    }

    for (gradient_index, value) in gradient.iter_mut().enumerate() {
        *value += accum[gradient_index].reduce_sum();
    }

    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
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
        index += 1;
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

    let mut gradient_0 = Vf64::splat(0.0);
    let mut gradient_1 = Vf64::splat(0.0);
    let a = Vf64::splat(param[0]);
    let b = Vf64::splat(param[1]);
    let eps = Vf64::splat(super::PARAM_EPS);

    let mut index = 0;
    while index + Vf64::LEN <= x_values.len() {
        let x = Vf64::from_slice(&x_values[index..index + Vf64::LEN]).simd_max(eps);
        let y = Vf64::from_slice(&y_values[index..index + Vf64::LEN]);
        let residual_derivative = residual_derivative_simd(loss_metric, (a + b / x) - y);
        gradient_0 += residual_derivative;
        gradient_1 += residual_derivative / x;
        index += Vf64::LEN;
    }

    gradient[0] += gradient_0.reduce_sum();
    gradient[1] += gradient_1.reduce_sum();

    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let residual = loss_metric.residual_derivative((param[0] + param[1] / x) - y);
        gradient[0] += residual;
        gradient[1] += residual / x;
        index += 1;
    }
}
