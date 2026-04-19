use super::simd;
use super::{
    CurveProblem, CurveProblemPredictionLoss, DEFAULT_SPLINE_KNOTS, FitError,
    HESSIAN_DIAGONAL_JITTER, HESSIAN_FD_MIN_STEP, HESSIAN_FD_REL_STEP, IncrementalSplineFitRunner,
    IncrementalSplineFitStep, LARGE_COST, MetricQuantization, MetricQuantizationDecimalPlaces,
    OptimizationLossMetric, SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation,
    SplineFamilyKind, SplineFinalizeContext, SplineKnotStrategy, approximate_spline_knots,
    build_spline_initial_curve_from_knot_y, build_spline_result_from_knot_y,
    calculate_iteration_metrics, calculate_iteration_metrics_with_quantization, calculate_metrics,
    evaluate_linear_spline, expanded_spline_curve_x_bounds, fit_akima_spline,
    fit_akima_spline_with_config, fit_curve, fit_curve_with_optimizer_config,
    fit_curve_with_progress, fit_curve_with_progress_and_optimizer_config,
    fit_curve_with_progress_and_optimizer_config_and_loss_metric, fit_linear_spline,
    fit_linear_spline_with_config, fit_monotone_cubic_spline, fit_natural_cubic_spline,
    numerical_hessian_from_gradient, softplus, sorted_points_with_duplicate_policy,
};
use crate::domain::{
    AdamConfig, CurveFamily, CurveParams, InputError, LbfgsConfig, NelderMeadConfig,
    NewtonCgConfig, OptimizerConfig, Point, Points, SgdConfig, SteepestDescentConfig,
};
use crate::models::{self, ObjectiveGrad, ObjectiveHessian, ObjectiveValue, PredictionLoss};
use argmin::core::Gradient;
use ndarray::Array1;

// Общий prelude для тематических подмодулей `fit/tests/*`.
mod finite_diff_simd;
mod metrics_quantization;
mod parametric_optimizers;
mod spline;

#[derive(Clone, Copy)]
struct MsePredictionLoss;

impl PredictionLoss for MsePredictionLoss {
    fn value(&self, prediction: f64, target: f64) -> f64 {
        let residual = prediction - target;
        residual * residual
    }

    fn d_prediction(&self, prediction: f64, target: f64) -> f64 {
        2.0 * (prediction - target)
    }

    fn d2_prediction(&self, _prediction: f64, _target: f64) -> f64 {
        2.0
    }
}

fn build_points<F>(xs: &[f64], f: F) -> Points
where
    F: Fn(f64) -> f64,
{
    let points = xs
        .iter()
        .copied()
        .map(|x| Point::try_new(x, f(x)).unwrap())
        .collect::<Vec<_>>();
    Points::try_from(points).unwrap()
}

fn assert_near(actual: f64, expected: f64, epsilon: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= epsilon,
        "expected {expected}, got {actual}, delta={delta}, epsilon={epsilon}"
    );
}

fn quantization(decimal_places: u8) -> MetricQuantization {
    MetricQuantization::Enabled(
        MetricQuantizationDecimalPlaces::try_new(decimal_places)
            .expect("test decimal places must be valid"),
    )
}

struct RetryGradientProblem {
    center: f64,
    invalid_step: f64,
}

struct AlwaysInvalidGradientProblem;

impl Gradient for RetryGradientProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let delta = (param[0] - self.center).abs();
        if (delta - self.invalid_step).abs() <= 1e-14 {
            return Ok(Array1::from_vec(vec![f64::NAN]));
        }
        Ok(Array1::from_vec(vec![2.0 * param[0]]))
    }
}

impl Gradient for AlwaysInvalidGradientProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        Ok(Array1::from_vec(vec![f64::NAN; param.len()]))
    }
}
