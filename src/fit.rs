//! Численные алгоритмы подгонки параметрических моделей и сплайнов.
//! Модуль инкапсулирует оптимизацию, расчет метрик и дискретизацию кривых.

use std::fmt;

use argmin::core::{
    CostFunction, Gradient, Hessian, IterState, Problem, Solver, State, TerminationReason,
    TerminationStatus,
};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::newton::NewtonCG;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{Array1, Array2};
use stochastic_optimizers::{Adam, Optimizer as StochasticOptimizer, SGD};

use crate::domain::{
    AdamConfig, CurveFamily, CurveParams, FitResult, InputError, LbfgsConfig, NelderMeadConfig,
    NewtonCgConfig, OptimizerConfig, Points, SaturatingTrendTauGrid, SgdConfig,
    SteepestDescentConfig,
};
use crate::models::{
    self, ObjectiveGrad, ObjectiveHessian, ObjectiveValue, PredictionLoss, TermGrad, TermHessian,
    TermValue,
};

mod common;
mod error;
mod finite_diff;
mod metrics;
mod optimizer_engine;
mod parametric;
mod spline_core;

mod curve;
mod simd;
mod spline;

// Общие символы для внутренних `fit/*` модулей (подключаются через `use super::*;`).
use self::common::*;
use self::error::*;
use self::finite_diff::*;
use self::metrics::*;
use self::parametric::*;
use self::spline_core::*;

// Стабильный публичный API.
pub use self::curve::{
    fit_curve, fit_curve_with_optimizer_config, fit_curve_with_progress,
    fit_curve_with_progress_and_optimizer_config,
};
pub use self::error::FitError;
pub use self::metrics::{IterationMetricSnapshot, OptimizationLossMetric, calculate_metrics};
pub use self::optimizer_engine::{IncrementalFitRunner, IncrementalFitStep};
pub use self::parametric::sample_curve;
pub use self::spline::{
    fit_akima_spline, fit_akima_spline_with_config, fit_akima_spline_with_optimizer_config,
    fit_linear_spline, fit_linear_spline_with_config, fit_linear_spline_with_optimizer_config,
    fit_monotone_cubic_spline, fit_monotone_cubic_spline_with_config,
    fit_monotone_cubic_spline_with_optimizer_config, fit_natural_cubic_spline,
    fit_natural_cubic_spline_with_config, fit_natural_cubic_spline_with_optimizer_config,
};
pub use self::spline_core::{
    DEFAULT_SPLINE_KNOTS, DEFAULT_SPLINE_SAMPLES, SplineConfig, SplineDuplicateXPolicy,
    SplineExtrapolation, SplineKnotStrategy, SplineResult,
};

// Внутренние символы для crate/UI/tests.
#[cfg(all(not(target_arch = "wasm32"), test))]
pub(crate) use self::curve::fit_curve_with_progress_and_optimizer_config_and_loss_metric;
#[cfg(test)]
pub(crate) use self::metrics::calculate_iteration_metrics;
pub(crate) use self::metrics::{
    DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES, MetricQuantization,
    MetricQuantizationDecimalPlaces, calculate_iteration_metrics_with_quantization,
    calculate_metrics_with_quantization,
};
pub(crate) use self::optimizer_engine::{IncrementalSplineFitRunner, IncrementalSplineFitStep};
pub(crate) use self::spline::default_spline_initial_knot_y;
pub(crate) use self::spline_core::{SplineFamilyKind, build_spline_initial_curve_from_knot_y};

#[doc(hidden)]
pub mod simd_bench {
    use super::{OptimizationLossMetric, simd};

    pub fn polynomial_cost_scalar(
        param: &[f64],
        x_values: &[f64],
        y_values: &[f64],
        loss_metric: OptimizationLossMetric,
    ) -> f64 {
        simd::polynomial_cost_scalar(param, x_values, y_values, loss_metric)
    }

    pub fn polynomial_cost_simd(
        param: &[f64],
        x_values: &[f64],
        y_values: &[f64],
        loss_metric: OptimizationLossMetric,
    ) -> f64 {
        simd::polynomial_cost_simd(param, x_values, y_values, loss_metric)
    }

    pub fn inverse_cost_scalar(
        param: &[f64],
        x_values: &[f64],
        y_values: &[f64],
        loss_metric: OptimizationLossMetric,
    ) -> f64 {
        simd::inverse_cost_scalar(param, x_values, y_values, loss_metric)
    }

    pub fn inverse_cost_simd(
        param: &[f64],
        x_values: &[f64],
        y_values: &[f64],
        loss_metric: OptimizationLossMetric,
    ) -> f64 {
        simd::inverse_cost_simd(param, x_values, y_values, loss_metric)
    }

    pub fn polynomial_gradient_scalar(
        x_values: &[f64],
        y_values: &[f64],
        param: &[f64],
        loss_metric: OptimizationLossMetric,
        gradient: &mut [f64],
    ) {
        simd::accumulate_polynomial_gradient_scalar(
            x_values,
            y_values,
            param,
            loss_metric,
            gradient,
        );
    }

    pub fn polynomial_gradient_simd(
        x_values: &[f64],
        y_values: &[f64],
        param: &[f64],
        loss_metric: OptimizationLossMetric,
        gradient: &mut [f64],
    ) {
        simd::accumulate_polynomial_gradient_simd(x_values, y_values, param, loss_metric, gradient);
    }

    pub fn inverse_gradient_scalar(
        x_values: &[f64],
        y_values: &[f64],
        param: &[f64],
        loss_metric: OptimizationLossMetric,
        gradient: &mut [f64],
    ) {
        simd::accumulate_inverse_gradient_scalar(x_values, y_values, param, loss_metric, gradient);
    }

    pub fn inverse_gradient_simd(
        x_values: &[f64],
        y_values: &[f64],
        param: &[f64],
        loss_metric: OptimizationLossMetric,
        gradient: &mut [f64],
    ) {
        simd::accumulate_inverse_gradient_simd(x_values, y_values, param, loss_metric, gradient);
    }
}

#[cfg(test)]
mod tests;
