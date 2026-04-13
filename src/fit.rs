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
    NewtonCgConfig, OptimizerConfig, Points, SgdConfig, SteepestDescentConfig,
};
use crate::models::{self, GradientComputation};

mod curve;
mod simd;
mod spline;

#[cfg(all(not(target_arch = "wasm32"), test))]
pub(crate) use curve::fit_curve_with_progress_and_optimizer_config_and_loss_metric;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use curve::fit_curve_with_progress_and_optimizer_config_and_loss_metric_and_metric_quantization;
pub use curve::{
    fit_curve, fit_curve_with_optimizer_config, fit_curve_with_progress,
    fit_curve_with_progress_and_optimizer_config,
};
pub(crate) use spline::default_spline_initial_knot_y;
pub use spline::{
    fit_akima_spline, fit_akima_spline_with_config, fit_akima_spline_with_optimizer_config,
    fit_linear_spline, fit_linear_spline_with_config, fit_linear_spline_with_optimizer_config,
    fit_monotone_cubic_spline, fit_monotone_cubic_spline_with_config,
    fit_monotone_cubic_spline_with_optimizer_config, fit_natural_cubic_spline,
    fit_natural_cubic_spline_with_config, fit_natural_cubic_spline_with_optimizer_config,
};

#[cfg(feature = "portable-simd")]
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

const PARAM_EPS: f64 = models::PARAM_EPS;
const LARGE_COST: f64 = 1e24;
const STEEPEST_DESCENT_GRAD_TOL: f64 = 1e-12;
const HESSIAN_FD_REL_STEP: f64 = 1e-4;
const HESSIAN_FD_MIN_STEP: f64 = 1e-6;
pub(crate) const HESSIAN_DIAGONAL_JITTER: f64 = models::HESSIAN_DIAGONAL_JITTER;
const GRADIENT_FD_REL_STEP: f64 = 1e-5;
const GRADIENT_FD_MIN_STEP: f64 = 1e-7;

fn positive_x(value: f64) -> f64 {
    models::positive_x(value)
}

#[cfg(test)]
fn softplus(value: f64) -> f64 {
    models::softplus(value)
}

fn gradient_l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn vec_to_array1(values: &[f64]) -> Array1<f64> {
    Array1::from_vec(values.to_vec())
}

fn array1_to_vec(values: &Array1<f64>) -> Vec<f64> {
    values.to_vec()
}

fn array1_as_slice(values: &Array1<f64>) -> &[f64] {
    values
        .as_slice()
        .expect("Array1 parameters must have contiguous memory layout")
}

fn array1_as_slice_mut(values: &mut Array1<f64>) -> &mut [f64] {
    values
        .as_slice_mut()
        .expect("Array1 parameters must have contiguous memory layout")
}

fn stabilize_hessian(hessian: &mut Array2<f64>) {
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

fn numerical_hessian_from_gradient<O>(
    problem: &O,
    param: &Array1<f64>,
) -> Result<Array2<f64>, argmin::core::Error>
where
    O: Gradient<Param = Array1<f64>, Gradient = Array1<f64>>,
{
    let param_slice = array1_as_slice(param);
    let dimension = param_slice.len();
    let mut hessian = Array2::zeros((dimension, dimension));
    let mut probe = param.clone();

    for column in 0..dimension {
        let step =
            ((param_slice[column].abs() + 1.0) * HESSIAN_FD_REL_STEP).max(HESSIAN_FD_MIN_STEP);
        probe[column] = param[column] + step;
        let grad_plus = problem.gradient(&probe)?;
        probe[column] = param[column] - step;
        let grad_minus = problem.gradient(&probe)?;
        probe[column] = param[column];

        let denom = 2.0 * step;
        for row in 0..dimension {
            let value = (grad_plus[row] - grad_minus[row]) / denom;
            hessian[[row, column]] = if value.is_finite() { value } else { 0.0 };
        }
    }

    stabilize_hessian(&mut hessian);
    Ok(hessian)
}

#[derive(Debug, Clone, PartialEq)]
/// Ошибки, возникающие при подгонке моделей и сплайнов.
pub enum FitError {
    InvalidInput(InputError),
    InvalidSplineInput(String),
    Cancelled,
    Optimizer(String),
    MissingBestParameters,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(error) => write!(f, "{error}"),
            Self::InvalidSplineInput(message) => write!(f, "{message}"),
            Self::Cancelled => f.write_str("Optimization cancelled by user"),
            Self::Optimizer(error) => write!(f, "Optimization failed: {error}"),
            Self::MissingBestParameters => f.write_str("Optimizer did not return best parameters"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<InputError> for FitError {
    fn from(value: InputError) -> Self {
        Self::InvalidInput(value)
    }
}

fn optimizer_error(error: impl fmt::Display) -> FitError {
    FitError::Optimizer(error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Целевая метрика, по которой оптимизатор минимизирует ошибку.
pub enum OptimizationLossMetric {
    #[default]
    Mse,
    Mae,
    SoftL1,
}

impl OptimizationLossMetric {
    /// Полный список вариантов для UI и переборов.
    pub const ALL: [Self; 3] = [Self::Mse, Self::Mae, Self::SoftL1];

    /// Короткое имя метрики для подписи в легенде.
    pub fn id(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::Mae => "mae",
            Self::SoftL1 => "soft_l1",
        }
    }

    fn value_from_residual(self, residual: f64) -> f64 {
        match self {
            Self::Mse => residual * residual,
            Self::Mae => residual.abs(),
            Self::SoftL1 => 2.0 * ((1.0 + residual * residual).sqrt() - 1.0),
        }
    }

    fn residual_derivative(self, residual: f64) -> f64 {
        match self {
            Self::Mse => 2.0 * residual,
            Self::Mae => {
                if residual > 0.0 {
                    1.0
                } else if residual < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::SoftL1 => 2.0 * residual / (1.0 + residual * residual).sqrt(),
        }
    }

    fn residual_second_derivative(self, residual: f64) -> f64 {
        match self {
            Self::Mse => 2.0,
            Self::Mae => 0.0,
            Self::SoftL1 => 2.0 / (1.0 + residual * residual).powf(1.5),
        }
    }
}

/// Значение по умолчанию для числа знаков после запятой в режиме квантизации метрик.
pub(crate) const DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Число знаков после запятой для квантизации метрик.
pub(crate) struct MetricQuantizationDecimalPlaces(u8);

impl MetricQuantizationDecimalPlaces {
    pub(crate) const MIN: u8 = 0;
    pub(crate) const MAX: u8 = 15;

    pub(crate) fn try_new(value: u8) -> Result<Self, String> {
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(format!(
                "Metric quantization decimal places must be in range {}..={}, got {value}",
                Self::MIN,
                Self::MAX
            ));
        }
        Ok(Self(value))
    }

    pub(crate) fn get(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Конфигурация квантизации для расчета objective и метрик.
pub(crate) enum MetricQuantization {
    #[default]
    Disabled,
    Enabled(MetricQuantizationDecimalPlaces),
}

impl MetricQuantization {
    pub(crate) fn from_ui_state(enabled: bool, decimal_places: u8) -> Result<Self, String> {
        if !enabled {
            return Ok(Self::Disabled);
        }
        Ok(Self::Enabled(MetricQuantizationDecimalPlaces::try_new(
            decimal_places,
        )?))
    }
}

#[derive(Debug, Clone, Copy)]
enum ResidualQuantizer {
    Disabled,
    Enabled { scale: f64 },
}

impl ResidualQuantizer {
    fn new(metric_quantization: MetricQuantization) -> Self {
        match metric_quantization {
            MetricQuantization::Disabled => Self::Disabled,
            MetricQuantization::Enabled(decimal_places) => Self::Enabled {
                scale: 10_f64.powi(decimal_places.get() as i32),
            },
        }
    }

    #[inline]
    fn quantize_value(self, value: f64) -> f64 {
        match self {
            Self::Disabled => value,
            Self::Enabled { scale } => (value * scale).round() / scale,
        }
    }

    #[inline]
    fn residual(self, predicted: f64, observed: f64) -> f64 {
        self.quantize_value(predicted) - self.quantize_value(observed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Снимок метрик на одной итерации оптимизации.
pub struct IterationMetricSnapshot {
    pub loss: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub soft_l1: f64,
    pub r2: f64,
    pub max_abs_error: f64,
}

struct CurveProblem {
    family: CurveFamily,
    point_x: Box<[f64]>,
    point_y: Box<[f64]>,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    residual_quantizer: ResidualQuantizer,
}

impl CurveProblem {
    fn new_with_metric_quantization(
        family: CurveFamily,
        points: &Points,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Self {
        let mut point_x = Vec::with_capacity(points.len());
        let mut point_y = Vec::with_capacity(points.len());
        for point in points.as_slice() {
            point_x.push(point.x());
            point_y.push(point.y());
        }
        Self {
            family,
            point_x: point_x.into_boxed_slice(),
            point_y: point_y.into_boxed_slice(),
            loss_metric,
            metric_quantization,
            residual_quantizer: ResidualQuantizer::new(metric_quantization),
        }
    }

    #[inline]
    fn residual(&self, predicted: f64, observed: f64) -> f64 {
        self.residual_quantizer.residual(predicted, observed)
    }

    #[inline]
    fn loss_value_from_prediction(&self, predicted: f64, observed: f64) -> f64 {
        self.loss_metric
            .value_from_residual(self.residual(predicted, observed))
    }

    #[inline]
    fn loss_derivative_from_prediction(&self, predicted: f64, observed: f64) -> f64 {
        self.loss_metric
            .residual_derivative(self.residual(predicted, observed))
    }

    #[inline]
    fn loss_second_derivative_from_prediction(&self, predicted: f64, observed: f64) -> f64 {
        self.loss_metric
            .residual_second_derivative(self.residual(predicted, observed))
    }

    fn analytic_hessian_for_supported_families(&self, param: &Array1<f64>) -> Option<Array2<f64>> {
        if matches!(self.loss_metric, OptimizationLossMetric::Mae) {
            return None;
        }

        let param = array1_as_slice(param);
        models::analytic_hessian(
            self.family,
            self.point_x.as_ref(),
            self.point_y.as_ref(),
            param,
            |predicted, observed| self.loss_derivative_from_prediction(predicted, observed),
            |predicted, observed| self.loss_second_derivative_from_prediction(predicted, observed),
        )
    }

    fn numerical_gradient_from_cost(
        &self,
        param: &[f64],
        gradient: &mut [f64],
    ) -> Result<(), argmin::core::Error> {
        let mut probe = vec_to_array1(param);
        for index in 0..gradient.len() {
            let step =
                ((param[index].abs() + 1.0) * GRADIENT_FD_REL_STEP).max(GRADIENT_FD_MIN_STEP);
            probe[index] = param[index] + step;
            let cost_plus = CostFunction::cost(self, &probe)?;
            probe[index] = param[index] - step;
            let cost_minus = CostFunction::cost(self, &probe)?;
            probe[index] = param[index];
            gradient[index] = (cost_plus - cost_minus) / (2.0 * step);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Подробный результат подгонки сплайна.
pub struct SplineResult {
    pub knots: Vec<[f64; 2]>,
    pub curve: Vec<[f64; 2]>,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub max_abs_error: f64,
    pub residuals: Vec<[f64; 2]>,
    pub iterations: u64,
}

/// Число узлов сплайна по умолчанию.
pub const DEFAULT_SPLINE_KNOTS: usize = 8;
/// Число сэмплов кривой для визуализации по умолчанию.
pub const DEFAULT_SPLINE_SAMPLES: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Стратегия сокращения исходных точек до фиксированного числа узлов.
pub enum SplineKnotStrategy {
    #[default]
    BinMean,
    BinMedian,
}

impl SplineKnotStrategy {
    /// Полный список стратегий для UI и переборов.
    pub const ALL: [Self; 2] = [Self::BinMean, Self::BinMedian];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Политика экстраполяции сплайна вне диапазона узлов.
pub enum SplineExtrapolation {
    #[default]
    Linear,
    Clamp,
}

impl SplineExtrapolation {
    /// Полный список вариантов экстраполяции.
    pub const ALL: [Self; 2] = [Self::Linear, Self::Clamp];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Политика обработки повторяющихся `x` перед построением сплайна.
pub enum SplineDuplicateXPolicy {
    #[default]
    Error,
    MeanY,
    MedianY,
    FirstY,
}

impl SplineDuplicateXPolicy {
    /// Полный список вариантов обработки повторов.
    pub const ALL: [Self; 4] = [Self::Error, Self::MeanY, Self::MedianY, Self::FirstY];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Конфигурация построения и оптимизации сплайна.
pub struct SplineConfig {
    pub knots: usize,
    pub samples: usize,
    pub knot_strategy: SplineKnotStrategy,
    pub extrapolation: SplineExtrapolation,
    pub duplicate_x_policy: SplineDuplicateXPolicy,
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            knots: DEFAULT_SPLINE_KNOTS,
            samples: DEFAULT_SPLINE_SAMPLES,
            knot_strategy: SplineKnotStrategy::default(),
            extrapolation: SplineExtrapolation::default(),
            duplicate_x_policy: SplineDuplicateXPolicy::default(),
        }
    }
}

impl SplineConfig {
    /// Нормализует конфигурацию, обеспечивая минимально допустимые значения.
    pub fn normalized(self) -> Self {
        Self {
            knots: self.knots.max(2),
            samples: self.samples.max(2),
            knot_strategy: self.knot_strategy,
            extrapolation: self.extrapolation,
            duplicate_x_policy: self.duplicate_x_policy,
        }
    }
}

type GradientState = IterState<Array1<f64>, Array1<f64>, (), (), (), f64>;
type NelderMeadState = IterState<Array1<f64>, (), (), (), (), f64>;
type NewtonCgState = IterState<Array1<f64>, Array1<f64>, (), Array2<f64>, (), f64>;
type LbfgsSolver =
    LBFGS<MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>, Array1<f64>, Array1<f64>, f64>;
type SteepestDescentSolver = SteepestDescent<MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>>;
type NelderMeadSolver = NelderMead<Array1<f64>, f64>;
type NewtonCgSolver = NewtonCG<MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>, f64>;
type SgdSolver = SGD<Vec<f64>>;
type AdamSolver = Adam<Vec<f64>>;

#[derive(Debug, Clone)]
struct StochasticState {
    current_param: Vec<f64>,
    best_param: Vec<f64>,
    best_cost: f64,
    iter: u64,
    max_iters: u64,
}

enum OptimizerSolver {
    Lbfgs(LbfgsSolver),
    NelderMead(NelderMeadSolver),
    SteepestDescent(SteepestDescentSolver),
    NewtonCg(NewtonCgSolver),
    Sgd(SgdSolver),
    Adam(AdamSolver),
}

enum OptimizerState {
    Lbfgs(GradientState),
    NelderMead(NelderMeadState),
    SteepestDescent(GradientState),
    NewtonCg(Box<NewtonCgState>),
    Sgd(StochasticState),
    Adam(StochasticState),
}

#[derive(Debug, Clone, PartialEq)]
/// Шаг инкрементальной подгонки параметрической модели.
pub enum IncrementalFitStep {
    Iteration {
        iteration: u64,
        mse: f64,
        metrics: IterationMetricSnapshot,
        params: CurveParams,
    },
    Finished(FitResult),
    Cancelled,
}

/// Пошаговый раннер оптимизации параметрических семейств.
pub struct IncrementalFitRunner {
    family: CurveFamily,
    points: Points,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    problem: Problem<CurveProblem>,
    solver: OptimizerSolver,
    state: Option<OptimizerState>,
    cancelled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IncrementalSplineFitStep {
    Iteration {
        iteration: u64,
        mse: f64,
        metrics: IterationMetricSnapshot,
        knot_y: Vec<f64>,
        curve: Vec<[f64; 2]>,
    },
    Finished {
        result: SplineResult,
        metrics: IterationMetricSnapshot,
    },
    Cancelled,
}

pub(crate) struct IncrementalSplineFitRunner {
    family: SplineFamilyKind,
    points: Points,
    config: SplineConfig,
    knot_x: Box<[f64]>,
    curve_x_bounds: [f64; 2],
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    problem: Problem<SplineProblem>,
    solver: OptimizerSolver,
    state: Option<OptimizerState>,
    cancelled: bool,
}

fn build_line_search(
    c1: f64,
    c2: f64,
    step_min: f64,
    step_max: f64,
    width_tolerance: f64,
) -> Result<MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>, FitError> {
    MoreThuenteLineSearch::new()
        .with_c(c1, c2)
        .map_err(optimizer_error)?
        .with_bounds(step_min, step_max)
        .map_err(optimizer_error)?
        .with_width_tolerance(width_tolerance)
        .map_err(optimizer_error)
}

fn build_lbfgs_solver(config: &LbfgsConfig) -> Result<LbfgsSolver, FitError> {
    let line_search = build_line_search(
        config.c1,
        config.c2,
        config.step_min,
        config.step_max,
        config.width_tolerance,
    )?;
    LBFGS::new(line_search, config.history_size)
        .with_tolerance_grad(config.tol_grad)
        .map_err(optimizer_error)?
        .with_tolerance_cost(config.tol_cost)
        .map_err(optimizer_error)
}

fn build_steepest_descent_solver(
    config: &SteepestDescentConfig,
) -> Result<SteepestDescentSolver, FitError> {
    let line_search = build_line_search(
        config.c1,
        config.c2,
        config.step_min,
        config.step_max,
        config.width_tolerance,
    )?;
    Ok(SteepestDescent::new(line_search))
}

fn build_newton_cg_solver(config: &NewtonCgConfig) -> Result<NewtonCgSolver, FitError> {
    let line_search = build_line_search(
        config.c1,
        config.c2,
        config.step_min,
        config.step_max,
        config.width_tolerance,
    )?;
    NewtonCG::new(line_search)
        .with_curvature_threshold(config.curvature_threshold)
        .with_tolerance(config.tol)
        .map_err(optimizer_error)
}

fn nelder_mead_simplex(
    initial_param: &[f64],
    simplex_scale: f64,
) -> Result<Vec<Array1<f64>>, FitError> {
    if initial_param.is_empty() {
        return Err(optimizer_error(
            "Nelder-Mead requires at least one optimization parameter",
        ));
    }

    let mut simplex = Vec::with_capacity(initial_param.len() + 1);
    simplex.push(vec_to_array1(initial_param));

    for (index, value) in initial_param.iter().copied().enumerate() {
        let mut vertex = initial_param.to_vec();
        vertex[index] += simplex_scale * (value.abs() + 1.0);
        simplex.push(Array1::from_vec(vertex));
    }

    Ok(simplex)
}

fn build_nelder_mead_solver(
    initial_param: &[f64],
    config: &NelderMeadConfig,
) -> Result<NelderMeadSolver, FitError> {
    let simplex = nelder_mead_simplex(initial_param, config.simplex_scale)?;
    NelderMead::new(simplex)
        .with_sd_tolerance(config.sd_tolerance)
        .map_err(optimizer_error)?
        .with_alpha(config.alpha)
        .map_err(optimizer_error)?
        .with_gamma(config.gamma)
        .map_err(optimizer_error)?
        .with_rho(config.rho)
        .map_err(optimizer_error)?
        .with_sigma(config.sigma)
        .map_err(optimizer_error)
}

fn build_sgd_solver(initial_param: &[f64], config: &SgdConfig) -> SgdSolver {
    SGD::new(initial_param.to_vec(), config.learning_rate)
}

fn build_adam_solver(initial_param: &[f64], config: &AdamConfig) -> AdamSolver {
    Adam::new(initial_param.to_vec(), config.learning_rate)
}

fn build_optimizer_solver(
    initial_param: &[f64],
    config: &OptimizerConfig,
) -> Result<OptimizerSolver, FitError> {
    match config {
        OptimizerConfig::Lbfgs(lbfgs) => Ok(OptimizerSolver::Lbfgs(build_lbfgs_solver(lbfgs)?)),
        OptimizerConfig::NelderMead(nelder_mead) => Ok(OptimizerSolver::NelderMead(
            build_nelder_mead_solver(initial_param, nelder_mead)?,
        )),
        OptimizerConfig::SteepestDescent(steepest_descent) => Ok(OptimizerSolver::SteepestDescent(
            build_steepest_descent_solver(steepest_descent)?,
        )),
        OptimizerConfig::NewtonCg(newton_cg) => Ok(OptimizerSolver::NewtonCg(
            build_newton_cg_solver(newton_cg)?,
        )),
        OptimizerConfig::Sgd(sgd) => Ok(OptimizerSolver::Sgd(build_sgd_solver(initial_param, sgd))),
        OptimizerConfig::Adam(adam) => Ok(OptimizerSolver::Adam(build_adam_solver(
            initial_param,
            adam,
        ))),
    }
}

fn finite_cost_or_large(cost: f64) -> f64 {
    if cost.is_finite() { cost } else { LARGE_COST }
}

fn build_stochastic_state<O>(
    problem: &mut Problem<O>,
    initial_param: Vec<f64>,
    max_iters: u64,
) -> Result<StochasticState, FitError>
where
    O: CostFunction<Param = Array1<f64>, Output = f64>,
{
    let cost = problem
        .cost(&vec_to_array1(&initial_param))
        .map_err(optimizer_error)?;
    Ok(StochasticState {
        current_param: initial_param.clone(),
        best_param: initial_param,
        best_cost: finite_cost_or_large(cost),
        iter: 0,
        max_iters,
    })
}

fn stochastic_state_is_terminated(state: &StochasticState) -> bool {
    state.iter >= state.max_iters
}

fn stochastic_step<O>(
    problem: &mut Problem<O>,
    solver: &mut impl StochasticOptimizer<P = Vec<f64>>,
    state: &mut StochasticState,
) -> Result<(), FitError>
where
    O: CostFunction<Param = Array1<f64>, Output = f64>
        + Gradient<Param = Array1<f64>, Gradient = Array1<f64>>,
{
    let current_param_array = vec_to_array1(&state.current_param);
    let gradient = problem
        .gradient(&current_param_array)
        .map_err(optimizer_error)?;
    solver.step(&array1_to_vec(&gradient));

    let current_param = solver.parameters().clone();
    let current_cost = finite_cost_or_large(
        problem
            .cost(&vec_to_array1(&current_param))
            .map_err(optimizer_error)?,
    );

    if current_cost < state.best_cost {
        state.best_cost = current_cost;
        state.best_param = current_param.clone();
    }
    state.current_param = current_param;

    Ok(())
}

fn optimizer_state_best_param(state: &OptimizerState) -> Option<Array1<f64>> {
    match state {
        OptimizerState::Lbfgs(state) => state
            .get_best_param()
            .or_else(|| state.get_param())
            .cloned(),
        OptimizerState::NelderMead(state) => state
            .get_best_param()
            .or_else(|| state.get_param())
            .cloned(),
        OptimizerState::SteepestDescent(state) => state
            .get_best_param()
            .or_else(|| state.get_param())
            .cloned(),
        OptimizerState::NewtonCg(state) => state
            .get_best_param()
            .or_else(|| state.get_param())
            .cloned(),
        OptimizerState::Sgd(state) => Some(vec_to_array1(&state.best_param)),
        OptimizerState::Adam(state) => Some(vec_to_array1(&state.best_param)),
    }
}

fn optimizer_state_current_param(state: &OptimizerState) -> Option<Array1<f64>> {
    match state {
        OptimizerState::Lbfgs(state) => state.get_param().cloned(),
        OptimizerState::NelderMead(state) => state.get_param().cloned(),
        OptimizerState::SteepestDescent(state) => state.get_param().cloned(),
        OptimizerState::NewtonCg(state) => state.get_param().cloned(),
        OptimizerState::Sgd(state) => Some(vec_to_array1(&state.current_param)),
        OptimizerState::Adam(state) => Some(vec_to_array1(&state.current_param)),
    }
}

fn optimizer_state_iter(state: &OptimizerState) -> u64 {
    match state {
        OptimizerState::Lbfgs(state) => state.get_iter(),
        OptimizerState::NelderMead(state) => state.get_iter(),
        OptimizerState::SteepestDescent(state) => state.get_iter(),
        OptimizerState::NewtonCg(state) => state.get_iter(),
        OptimizerState::Sgd(state) => state.iter,
        OptimizerState::Adam(state) => state.iter,
    }
}

fn optimizer_state_increment_iter(state: &mut OptimizerState) {
    match state {
        OptimizerState::Lbfgs(state) => state.increment_iter(),
        OptimizerState::NelderMead(state) => state.increment_iter(),
        OptimizerState::SteepestDescent(state) => state.increment_iter(),
        OptimizerState::NewtonCg(state) => state.increment_iter(),
        OptimizerState::Sgd(state) => state.iter = state.iter.saturating_add(1),
        OptimizerState::Adam(state) => state.iter = state.iter.saturating_add(1),
    }
}

fn terminate_steepest_descent_on_small_gradient<O>(
    problem: &mut Problem<O>,
    mut state: GradientState,
) -> Result<GradientState, FitError>
where
    O: Gradient<Param = Array1<f64>, Gradient = Array1<f64>>,
{
    if state.terminated() {
        return Ok(state);
    }
    let Some(param) = state.get_param().cloned() else {
        return Ok(state);
    };
    let gradient = problem.gradient(&param).map_err(optimizer_error)?;
    state = state.gradient(gradient.clone());
    if gradient_l2_norm(array1_as_slice(&gradient)) <= STEEPEST_DESCENT_GRAD_TOL {
        state = state.terminate_with(TerminationReason::SolverConverged);
    }
    Ok(state)
}

impl IncrementalFitRunner {
    /// Создает раннер и инициализирует внутреннее состояние оптимизатора.
    pub fn new(
        points: &Points,
        family: CurveFamily,
        initial_params: CurveParams,
        config: &LbfgsConfig,
    ) -> Result<Self, FitError> {
        let optimizer_config = OptimizerConfig::Lbfgs(config.clone());
        Self::new_with_optimizer_config(points, family, initial_params, &optimizer_config)
    }

    /// Создает раннер с произвольной конфигурацией оптимизатора.
    pub fn new_with_optimizer_config(
        points: &Points,
        family: CurveFamily,
        initial_params: CurveParams,
        optimizer_config: &OptimizerConfig,
    ) -> Result<Self, FitError> {
        Self::new_with_optimizer_config_and_loss_metric(
            points,
            family,
            initial_params,
            optimizer_config,
            OptimizationLossMetric::Mse,
        )
    }

    /// Создает раннер с произвольной конфигурацией оптимизатора и явной целевой метрикой.
    pub(crate) fn new_with_optimizer_config_and_loss_metric(
        points: &Points,
        family: CurveFamily,
        initial_params: CurveParams,
        optimizer_config: &OptimizerConfig,
        loss_metric: OptimizationLossMetric,
    ) -> Result<Self, FitError> {
        Self::new_with_optimizer_config_and_loss_metric_and_metric_quantization(
            points,
            family,
            initial_params,
            optimizer_config,
            loss_metric,
            MetricQuantization::Disabled,
        )
    }

    pub(crate) fn new_with_optimizer_config_and_loss_metric_and_metric_quantization(
        points: &Points,
        family: CurveFamily,
        initial_params: CurveParams,
        optimizer_config: &OptimizerConfig,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Result<Self, FitError> {
        if initial_params.family() != family {
            return Err(FitError::InvalidInput(InputError::FamilyMismatch {
                expected: family,
                got: initial_params.family(),
            }));
        }
        family.validate_points(points)?;

        let initial_values = initial_params.values();
        let initial_array = vec_to_array1(&initial_values);
        let max_iters = optimizer_config.max_iters();
        let problem = CurveProblem::new_with_metric_quantization(
            family,
            points,
            loss_metric,
            metric_quantization,
        );
        let mut problem = Problem::new(problem);
        let mut solver = build_optimizer_solver(&initial_values, optimizer_config)?;
        let state = match &mut solver {
            OptimizerSolver::Lbfgs(solver) => {
                let state = IterState::new()
                    .param(initial_array.clone())
                    .max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::Lbfgs(state)
            }
            OptimizerSolver::NelderMead(solver) => {
                let state = IterState::new()
                    .param(initial_array.clone())
                    .max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::NelderMead(state)
            }
            OptimizerSolver::SteepestDescent(solver) => {
                let state = IterState::new()
                    .param(initial_array.clone())
                    .max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::SteepestDescent(state)
            }
            OptimizerSolver::NewtonCg(solver) => {
                let state = IterState::new().param(initial_array).max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::NewtonCg(Box::new(state))
            }
            OptimizerSolver::Sgd(solver) => {
                let state =
                    build_stochastic_state(&mut problem, solver.parameters().clone(), max_iters)?;
                OptimizerState::Sgd(state)
            }
            OptimizerSolver::Adam(solver) => {
                let state =
                    build_stochastic_state(&mut problem, solver.parameters().clone(), max_iters)?;
                OptimizerState::Adam(state)
            }
        };

        Ok(Self {
            family,
            points: points.clone(),
            loss_metric,
            metric_quantization,
            problem,
            solver,
            state: Some(state),
            cancelled: false,
        })
    }

    /// Запрашивает мягкую отмену следующих шагов оптимизации.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Выполняет один шаг оптимизации.
    ///
    /// Возвращает итерацию, финальный результат или признак отмены.
    pub fn step(&mut self) -> Result<IncrementalFitStep, FitError> {
        if self.cancelled {
            return Ok(IncrementalFitStep::Cancelled);
        }

        loop {
            let state = self
                .state
                .take()
                .expect("incremental fit state must be initialized");

            let mut state = match (&mut self.solver, state) {
                (OptimizerSolver::Lbfgs(solver), OptimizerState::Lbfgs(mut state)) => {
                    if !state.terminated() {
                        let termination =
                            <LbfgsSolver as Solver<CurveProblem, GradientState>>::terminate_internal(
                                solver, &state,
                            );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step = self.finalize(OptimizerState::Lbfgs(state))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::Lbfgs(state)
                }
                (OptimizerSolver::NelderMead(solver), OptimizerState::NelderMead(mut state)) => {
                    if !state.terminated() {
                        let termination = <NelderMeadSolver as Solver<
                            CurveProblem,
                            NelderMeadState,
                        >>::terminate_internal(
                            solver, &state
                        );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step = self.finalize(OptimizerState::NelderMead(state))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::NelderMead(state)
                }
                (
                    OptimizerSolver::SteepestDescent(solver),
                    OptimizerState::SteepestDescent(mut state),
                ) => {
                    state = terminate_steepest_descent_on_small_gradient(&mut self.problem, state)?;
                    if !state.terminated() {
                        let termination = <SteepestDescentSolver as Solver<
                            CurveProblem,
                            GradientState,
                        >>::terminate_internal(
                            solver, &state
                        );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step = self.finalize(OptimizerState::SteepestDescent(state))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::SteepestDescent(state)
                }
                (OptimizerSolver::NewtonCg(solver), OptimizerState::NewtonCg(state)) => {
                    let mut state = *state;
                    if !state.terminated() {
                        let termination = <NewtonCgSolver as Solver<
                            CurveProblem,
                            NewtonCgState,
                        >>::terminate_internal(
                            solver, &state
                        );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step =
                            self.finalize(OptimizerState::NewtonCg(Box::new(state)))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::NewtonCg(Box::new(state))
                }
                (OptimizerSolver::Sgd(solver), OptimizerState::Sgd(mut state)) => {
                    if stochastic_state_is_terminated(&state) {
                        let final_step = self.finalize(OptimizerState::Sgd(state))?;
                        return Ok(final_step);
                    }
                    stochastic_step(&mut self.problem, solver, &mut state)?;
                    OptimizerState::Sgd(state)
                }
                (OptimizerSolver::Adam(solver), OptimizerState::Adam(mut state)) => {
                    if stochastic_state_is_terminated(&state) {
                        let final_step = self.finalize(OptimizerState::Adam(state))?;
                        return Ok(final_step);
                    }
                    stochastic_step(&mut self.problem, solver, &mut state)?;
                    OptimizerState::Adam(state)
                }
                _ => {
                    return Err(optimizer_error(
                        "Optimizer solver/state mismatch in incremental fit runner",
                    ));
                }
            };

            let iteration = optimizer_state_iter(&state);
            if let Some(params) = optimizer_state_current_param(&state).and_then(|values| {
                CurveParams::try_from_values(self.family, array1_to_vec(&values)).ok()
            }) {
                let metrics = calculate_iteration_metrics_with_quantization(
                    &self.points,
                    &params,
                    self.loss_metric,
                    self.metric_quantization,
                );
                optimizer_state_increment_iter(&mut state);
                self.state = Some(state);
                return Ok(IncrementalFitStep::Iteration {
                    iteration,
                    mse: metrics.mse,
                    metrics,
                    params,
                });
            }

            // Если параметры недоступны на текущем шаге, продолжаем итерации без рекурсии.
            optimizer_state_increment_iter(&mut state);
            self.state = Some(state);
        }
    }

    fn finalize(&mut self, state: OptimizerState) -> Result<IncrementalFitStep, FitError> {
        let best_param_values =
            optimizer_state_best_param(&state).ok_or(FitError::MissingBestParameters)?;
        let best_params =
            CurveParams::try_from_values(self.family, array1_to_vec(&best_param_values))?;
        let (mse, rmse) = calculate_metrics_with_quantization(
            &self.points,
            &best_params,
            self.metric_quantization,
        );
        let iterations = optimizer_state_iter(&state);
        self.state = Some(state);

        Ok(IncrementalFitStep::Finished(FitResult {
            family: self.family,
            params: best_params,
            mse,
            rmse,
            iterations,
        }))
    }
}

impl IncrementalSplineFitRunner {
    pub(crate) fn new_with_optimizer_config(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
        optimizer_config: &OptimizerConfig,
    ) -> Result<Self, FitError> {
        Self::new_with_initial_knot_y_and_optimizer_config(
            points,
            family,
            config,
            optimizer_config,
            None,
        )
    }

    pub(crate) fn new_with_optimizer_config_and_loss_metric(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
        optimizer_config: &OptimizerConfig,
        loss_metric: OptimizationLossMetric,
    ) -> Result<Self, FitError> {
        Self::new_with_optimizer_config_and_loss_metric_and_metric_quantization(
            points,
            family,
            config,
            optimizer_config,
            loss_metric,
            MetricQuantization::Disabled,
        )
    }

    pub(crate) fn new_with_optimizer_config_and_loss_metric_and_metric_quantization(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
        optimizer_config: &OptimizerConfig,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Result<Self, FitError> {
        Self::new_with_initial_knot_y_and_optimizer_config_and_loss_metric(
            points,
            family,
            config,
            optimizer_config,
            None,
            loss_metric,
            metric_quantization,
        )
    }

    pub(crate) fn new_with_initial_knot_y_and_optimizer_config(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
        optimizer_config: &OptimizerConfig,
        initial_knot_y: Option<&[f64]>,
    ) -> Result<Self, FitError> {
        Self::new_with_initial_knot_y_and_optimizer_config_and_loss_metric(
            points,
            family,
            config,
            optimizer_config,
            initial_knot_y,
            OptimizationLossMetric::Mse,
            MetricQuantization::Disabled,
        )
    }

    pub(crate) fn new_with_initial_knot_y_and_optimizer_config_and_loss_metric(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
        optimizer_config: &OptimizerConfig,
        initial_knot_y: Option<&[f64]>,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Result<Self, FitError> {
        let prepared = prepare_spline_inputs(points, config, family, initial_knot_y)?;
        let max_iters = optimizer_config.max_iters();

        let initial_knots = materialize_spline_knots(prepared.knot_x.as_ref(), &prepared.initial_y);
        let problem = SplineProblem::new(
            family,
            &initial_knots,
            points,
            prepared.config.extrapolation,
            loss_metric,
            metric_quantization,
        );
        let mut problem = Problem::new(problem);
        let initial_knot_y = vec_to_array1(&prepared.initial_y);
        let mut solver = build_optimizer_solver(&prepared.initial_y, optimizer_config)?;
        let state = match &mut solver {
            OptimizerSolver::Lbfgs(solver) => {
                let state = IterState::new()
                    .param(initial_knot_y.clone())
                    .max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::Lbfgs(state)
            }
            OptimizerSolver::NelderMead(solver) => {
                let state = IterState::new()
                    .param(initial_knot_y.clone())
                    .max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::NelderMead(state)
            }
            OptimizerSolver::SteepestDescent(solver) => {
                let state = IterState::new()
                    .param(initial_knot_y.clone())
                    .max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::SteepestDescent(state)
            }
            OptimizerSolver::NewtonCg(solver) => {
                let state = IterState::new().param(initial_knot_y).max_iters(max_iters);
                let (mut state, _) = solver.init(&mut problem, state).map_err(optimizer_error)?;
                state.update();
                state.func_counts(&problem);
                OptimizerState::NewtonCg(Box::new(state))
            }
            OptimizerSolver::Sgd(solver) => {
                let state =
                    build_stochastic_state(&mut problem, solver.parameters().clone(), max_iters)?;
                OptimizerState::Sgd(state)
            }
            OptimizerSolver::Adam(solver) => {
                let state =
                    build_stochastic_state(&mut problem, solver.parameters().clone(), max_iters)?;
                OptimizerState::Adam(state)
            }
        };

        Ok(Self {
            family,
            points: points.clone(),
            config: prepared.config,
            knot_x: prepared.knot_x,
            curve_x_bounds: prepared.curve_x_bounds,
            loss_metric,
            metric_quantization,
            problem,
            solver,
            state: Some(state),
            cancelled: false,
        })
    }

    pub(crate) fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub(crate) fn step(&mut self) -> Result<IncrementalSplineFitStep, FitError> {
        if self.cancelled {
            return Ok(IncrementalSplineFitStep::Cancelled);
        }

        loop {
            let state = self
                .state
                .take()
                .expect("incremental spline fit state must be initialized");

            let mut state = match (&mut self.solver, state) {
                (OptimizerSolver::Lbfgs(solver), OptimizerState::Lbfgs(mut state)) => {
                    if !state.terminated() {
                        let termination = <LbfgsSolver as Solver<
                            SplineProblem,
                            GradientState,
                        >>::terminate_internal(solver, &state);
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step = self.finalize(OptimizerState::Lbfgs(state))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::Lbfgs(state)
                }
                (OptimizerSolver::NelderMead(solver), OptimizerState::NelderMead(mut state)) => {
                    if !state.terminated() {
                        let termination = <NelderMeadSolver as Solver<
                            SplineProblem,
                            NelderMeadState,
                        >>::terminate_internal(
                            solver, &state
                        );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step = self.finalize(OptimizerState::NelderMead(state))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::NelderMead(state)
                }
                (
                    OptimizerSolver::SteepestDescent(solver),
                    OptimizerState::SteepestDescent(mut state),
                ) => {
                    state = terminate_steepest_descent_on_small_gradient(&mut self.problem, state)?;
                    if !state.terminated() {
                        let termination = <SteepestDescentSolver as Solver<
                            SplineProblem,
                            GradientState,
                        >>::terminate_internal(
                            solver, &state
                        );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step = self.finalize(OptimizerState::SteepestDescent(state))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::SteepestDescent(state)
                }
                (OptimizerSolver::NewtonCg(solver), OptimizerState::NewtonCg(state)) => {
                    let mut state = *state;
                    if !state.terminated() {
                        let termination = <NewtonCgSolver as Solver<
                            SplineProblem,
                            NewtonCgState,
                        >>::terminate_internal(
                            solver, &state
                        );
                        if let TerminationStatus::Terminated(reason) = termination {
                            state = state.terminate_with(reason);
                        }
                    }
                    if state.terminated() {
                        let final_step =
                            self.finalize(OptimizerState::NewtonCg(Box::new(state)))?;
                        return Ok(final_step);
                    }
                    let (mut state, _) = solver
                        .next_iter(&mut self.problem, state)
                        .map_err(optimizer_error)?;
                    state.func_counts(&self.problem);
                    state.update();
                    OptimizerState::NewtonCg(Box::new(state))
                }
                (OptimizerSolver::Sgd(solver), OptimizerState::Sgd(mut state)) => {
                    if stochastic_state_is_terminated(&state) {
                        let final_step = self.finalize(OptimizerState::Sgd(state))?;
                        return Ok(final_step);
                    }
                    stochastic_step(&mut self.problem, solver, &mut state)?;
                    OptimizerState::Sgd(state)
                }
                (OptimizerSolver::Adam(solver), OptimizerState::Adam(mut state)) => {
                    if stochastic_state_is_terminated(&state) {
                        let final_step = self.finalize(OptimizerState::Adam(state))?;
                        return Ok(final_step);
                    }
                    stochastic_step(&mut self.problem, solver, &mut state)?;
                    OptimizerState::Adam(state)
                }
                _ => {
                    return Err(optimizer_error(
                        "Optimizer solver/state mismatch in incremental spline runner",
                    ));
                }
            };

            let iteration = optimizer_state_iter(&state);
            if let Some(knot_y) = optimizer_state_current_param(&state) {
                let built = build_spline_curve_from_knot_y(
                    self.family,
                    self.config.extrapolation,
                    self.config.samples,
                    self.knot_x.as_ref(),
                    array1_as_slice(&knot_y),
                    self.curve_x_bounds,
                )?;
                let metrics = calculate_iteration_metrics_from_evaluator(
                    &self.points,
                    self.loss_metric,
                    self.metric_quantization,
                    |x| built.evaluator.evaluate(x),
                );
                let curve = built.curve;

                optimizer_state_increment_iter(&mut state);
                self.state = Some(state);
                return Ok(IncrementalSplineFitStep::Iteration {
                    iteration,
                    mse: metrics.mse,
                    metrics,
                    knot_y: array1_to_vec(&knot_y),
                    curve,
                });
            }

            // Если параметры недоступны на текущем шаге, продолжаем итерации без рекурсии.
            optimizer_state_increment_iter(&mut state);
            self.state = Some(state);
        }
    }

    fn finalize(&mut self, state: OptimizerState) -> Result<IncrementalSplineFitStep, FitError> {
        let best_knot_y =
            optimizer_state_best_param(&state).ok_or(FitError::MissingBestParameters)?;
        let iterations = optimizer_state_iter(&state);
        self.state = Some(state);

        let finalize_context = SplineFinalizeContext {
            points: &self.points,
            family: self.family,
            config: self.config,
            knot_x: self.knot_x.as_ref(),
            curve_x_bounds: self.curve_x_bounds,
            loss_metric: self.loss_metric,
            metric_quantization: self.metric_quantization,
        };
        let (result, metrics) = build_spline_result_from_knot_y(
            &finalize_context,
            array1_as_slice(&best_knot_y),
            iterations,
        )?;

        Ok(IncrementalSplineFitStep::Finished { result, metrics })
    }
}

impl CostFunction for CurveProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let param = array1_as_slice(param);
        if self.family.is_polynomial()
            && matches!(self.metric_quantization, MetricQuantization::Disabled)
        {
            return Ok(simd::polynomial_cost(
                param,
                self.point_x.as_ref(),
                self.point_y.as_ref(),
                self.loss_metric,
            ));
        }

        if self.family == CurveFamily::Inverse
            && matches!(self.metric_quantization, MetricQuantization::Disabled)
        {
            return Ok(simd::inverse_cost(
                param,
                self.point_x.as_ref(),
                self.point_y.as_ref(),
                self.loss_metric,
            ));
        }

        let sample_count = self.point_x.len() as f64;
        let mut sum = 0.0;

        let mut index = 0;
        while index < self.point_x.len() {
            let x = self.point_x[index];
            let y = self.point_y[index];
            let predicted = self.family.evaluate_raw(param, x);
            let residual = self.residual(predicted, y);
            if !residual.is_finite() {
                return Ok(LARGE_COST);
            }
            sum += self.loss_value_from_prediction(predicted, y);
            if !sum.is_finite() {
                return Ok(LARGE_COST);
            }
            index += 1;
        }

        Ok(sum / sample_count)
    }
}

impl Gradient for CurveProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let param = array1_as_slice(param);
        let mut gradient = Array1::zeros(self.family.parameter_count());
        let sample_scale = 1.0 / self.point_x.len() as f64;
        let gradient_slice = array1_as_slice_mut(&mut gradient);

        if self.family.is_polynomial()
            && matches!(self.metric_quantization, MetricQuantization::Disabled)
        {
            simd::accumulate_polynomial_gradient(
                self.point_x.as_ref(),
                self.point_y.as_ref(),
                param,
                self.loss_metric,
                gradient_slice,
            );
        } else if self.family == CurveFamily::Inverse
            && matches!(self.metric_quantization, MetricQuantization::Disabled)
        {
            simd::accumulate_inverse_gradient(
                self.point_x.as_ref(),
                self.point_y.as_ref(),
                param,
                self.loss_metric,
                gradient_slice,
            );
        } else {
            let mode = models::accumulate_gradient(
                self.family,
                self.point_x.as_ref(),
                self.point_y.as_ref(),
                param,
                |predicted, observed| self.loss_derivative_from_prediction(predicted, observed),
                gradient_slice,
            );
            if matches!(mode, GradientComputation::NeedsNumerical) {
                self.numerical_gradient_from_cost(param, gradient_slice)?;
                // Далее применяется общий sample_scale, поэтому здесь возвращается сумма.
                let sample_count = self.point_x.len() as f64;
                for value in gradient_slice.iter_mut() {
                    *value *= sample_count;
                }
            }
        }

        for value in gradient_slice.iter_mut() {
            *value *= sample_scale;
            if !value.is_finite() {
                *value = LARGE_COST;
            }
        }

        Ok(gradient)
    }
}

impl Hessian for CurveProblem {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        if let Some(hessian) = self.analytic_hessian_for_supported_families(param) {
            return Ok(hessian);
        }
        numerical_hessian_from_gradient(self, param)
    }
}

/// Вычисляет базовые метрики качества подгонки: `(MSE, RMSE)`.
pub fn calculate_metrics(points: &Points, params: &CurveParams) -> (f64, f64) {
    calculate_metrics_with_quantization(points, params, MetricQuantization::Disabled)
}

pub(crate) fn calculate_metrics_with_quantization(
    points: &Points,
    params: &CurveParams,
    metric_quantization: MetricQuantization,
) -> (f64, f64) {
    let scalar = calculate_scalar_metrics_from_evaluator(points, metric_quantization, |x| {
        params.evaluate(x)
    });
    (scalar.mse, scalar.rmse)
}

#[cfg(test)]
pub(crate) fn calculate_iteration_metrics(
    points: &Points,
    params: &CurveParams,
    loss_metric: OptimizationLossMetric,
) -> IterationMetricSnapshot {
    calculate_iteration_metrics_with_quantization(
        points,
        params,
        loss_metric,
        MetricQuantization::Disabled,
    )
}

pub(crate) fn calculate_iteration_metrics_with_quantization(
    points: &Points,
    params: &CurveParams,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
) -> IterationMetricSnapshot {
    calculate_iteration_metrics_from_evaluator(points, loss_metric, metric_quantization, |x| {
        params.evaluate(x)
    })
}

fn calculate_iteration_metrics_from_evaluator<F>(
    points: &Points,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    evaluate: F,
) -> IterationMetricSnapshot
where
    F: FnMut(f64) -> f64,
{
    let scalar = calculate_scalar_metrics_from_evaluator(points, metric_quantization, evaluate);
    IterationMetricSnapshot {
        loss: scalar_loss_value(loss_metric, &scalar),
        mse: scalar.mse,
        rmse: scalar.rmse,
        mae: scalar.mae,
        soft_l1: scalar.soft_l1,
        r2: scalar.r2,
        max_abs_error: scalar.max_abs_error,
    }
}

/// Равномерно дискретизирует параметрическую кривую на отрезке `x_min..=x_max`.
pub fn sample_curve(params: &CurveParams, x_min: f64, x_max: f64, samples: usize) -> Vec<[f64; 2]> {
    let sample_count = samples.max(2);
    let family = params.family();
    let mut sample_x_min = x_min;
    let mut sample_x_max = x_max;

    if family.requires_positive_x() {
        sample_x_min = positive_x(sample_x_min);
        sample_x_max = sample_x_max.max(sample_x_min + PARAM_EPS);
    }

    let span = sample_x_max - sample_x_min;
    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count - 1) as f64;
            let x = sample_x_min + span * t;
            [x, params.evaluate(x)]
        })
        .collect()
}

fn aggregate_duplicate_y(values: &[[f64; 2]], policy: SplineDuplicateXPolicy) -> f64 {
    match policy {
        SplineDuplicateXPolicy::Error => {
            unreachable!("Error policy should be handled before aggregation")
        }
        SplineDuplicateXPolicy::MeanY => {
            values.iter().map(|point| point[1]).sum::<f64>() / values.len() as f64
        }
        SplineDuplicateXPolicy::MedianY => {
            let mut sorted = values.iter().map(|point| point[1]).collect::<Vec<_>>();
            sorted.sort_by(|a, b| a.total_cmp(b));
            median_of_sorted(&sorted)
        }
        SplineDuplicateXPolicy::FirstY => values[0][1],
    }
}

fn sorted_points_with_duplicate_policy(
    points: &Points,
    policy: SplineDuplicateXPolicy,
) -> Result<Vec<[f64; 2]>, FitError> {
    let mut sorted = points
        .as_slice()
        .iter()
        .map(|point| [point.x(), point.y()])
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));

    if sorted.len() < 2 {
        return Ok(sorted);
    }

    let mut deduplicated = Vec::with_capacity(sorted.len());
    let mut index = 0;
    while index < sorted.len() {
        let x = sorted[index][0];
        let mut next = index + 1;
        // Считаем почти равные x дублями, чтобы сгладить эффект шумов округления.
        while next < sorted.len() && (sorted[next][0] - x).abs() <= 1e-12 {
            next += 1;
        }

        let duplicate_count = next - index;
        if duplicate_count == 1 {
            deduplicated.push(sorted[index]);
            index = next;
            continue;
        }

        if matches!(policy, SplineDuplicateXPolicy::Error) {
            return Err(FitError::InvalidSplineInput(format!(
                "Spline requires unique x values, but found duplicate x={x}"
            )));
        }

        deduplicated.push([x, aggregate_duplicate_y(&sorted[index..next], policy)]);
        index = next;
    }

    Ok(deduplicated)
}

const MIN_LINEAR_SPLINE_KNOTS: usize = 2;
const MIN_MONOTONE_SPLINE_KNOTS: usize = 2;
const MIN_NATURAL_SPLINE_KNOTS: usize = 3;
const MIN_AKIMA_SPLINE_KNOTS: usize = 5;
const SPLINE_FD_REL_STEP: f64 = 1e-6;
const SPLINE_FD_MIN_STEP: f64 = 1e-7;
const SPLINE_CURVE_DOMAIN_PADDING_RATIO: f64 = 0.1;
const SPLINE_CURVE_DOMAIN_EPS: f64 = 1e-9;
const SPLINE_CURVE_DOMAIN_FALLBACK_PADDING: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplineFamilyKind {
    Linear,
    MonotoneCubic,
    NaturalCubic,
    Akima,
}

impl SplineFamilyKind {
    fn name(self) -> &'static str {
        match self {
            Self::Linear => "Linear spline",
            Self::MonotoneCubic => "Monotone cubic spline",
            Self::NaturalCubic => "Natural cubic spline",
            Self::Akima => "Akima spline",
        }
    }

    fn min_knots(self) -> usize {
        match self {
            Self::Linear => MIN_LINEAR_SPLINE_KNOTS,
            Self::MonotoneCubic => MIN_MONOTONE_SPLINE_KNOTS,
            Self::NaturalCubic => MIN_NATURAL_SPLINE_KNOTS,
            Self::Akima => MIN_AKIMA_SPLINE_KNOTS,
        }
    }
}

enum SplineEvaluator {
    Linear {
        knots: Vec<[f64; 2]>,
        extrapolation: SplineExtrapolation,
    },
    CubicHermite {
        knots: Vec<[f64; 2]>,
        derivatives: Vec<f64>,
        extrapolation: SplineExtrapolation,
    },
    NaturalCubic {
        knots: Vec<[f64; 2]>,
        second_derivatives: Vec<f64>,
        extrapolation: SplineExtrapolation,
    },
}

impl SplineEvaluator {
    fn evaluate(&self, x: f64) -> f64 {
        match self {
            Self::Linear {
                knots,
                extrapolation,
            } => evaluate_linear_spline(knots, x, *extrapolation),
            Self::CubicHermite {
                knots,
                derivatives,
                extrapolation,
            } => evaluate_cubic_hermite_spline(knots, derivatives, x, *extrapolation),
            Self::NaturalCubic {
                knots,
                second_derivatives,
                extrapolation,
            } => evaluate_natural_cubic_spline(knots, second_derivatives, x, *extrapolation),
        }
    }
}

fn build_spline_evaluator(
    family: SplineFamilyKind,
    knots: Vec<[f64; 2]>,
    extrapolation: SplineExtrapolation,
) -> Result<SplineEvaluator, FitError> {
    match family {
        SplineFamilyKind::Linear => Ok(SplineEvaluator::Linear {
            knots,
            extrapolation,
        }),
        SplineFamilyKind::MonotoneCubic => {
            let derivatives = build_monotone_cubic_derivatives(&knots)?;
            Ok(SplineEvaluator::CubicHermite {
                knots,
                derivatives,
                extrapolation,
            })
        }
        SplineFamilyKind::NaturalCubic => {
            let second_derivatives = build_natural_cubic_second_derivatives(&knots)?;
            Ok(SplineEvaluator::NaturalCubic {
                knots,
                second_derivatives,
                extrapolation,
            })
        }
        SplineFamilyKind::Akima => {
            let derivatives = build_akima_derivatives(&knots)?;
            Ok(SplineEvaluator::CubicHermite {
                knots,
                derivatives,
                extrapolation,
            })
        }
    }
}

fn spline_lbfgs_config() -> LbfgsConfig {
    LbfgsConfig::try_new(7, 150, 1e-6, 1e-10, 1e-4, 0.9, 1e-12, 1.0, 1e-10)
        .expect("spline LBFGS config must be valid")
}

fn materialize_spline_knots(knot_x: &[f64], knot_y: &[f64]) -> Vec<[f64; 2]> {
    debug_assert_eq!(knot_x.len(), knot_y.len());
    knot_x
        .iter()
        .copied()
        .zip(knot_y.iter().copied())
        .map(|(x, y)| [x, y])
        .collect()
}

struct SplineProblem {
    family: SplineFamilyKind,
    knot_x: Box<[f64]>,
    points: Points,
    extrapolation: SplineExtrapolation,
    loss_metric: OptimizationLossMetric,
    residual_quantizer: ResidualQuantizer,
}

impl SplineProblem {
    fn new(
        family: SplineFamilyKind,
        initial_knots: &[[f64; 2]],
        points: &Points,
        extrapolation: SplineExtrapolation,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Self {
        let knot_x = initial_knots
            .iter()
            .map(|point| point[0])
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            family,
            knot_x,
            points: points.clone(),
            extrapolation,
            loss_metric,
            residual_quantizer: ResidualQuantizer::new(metric_quantization),
        }
    }

    #[inline]
    fn residual(&self, predicted: f64, observed: f64) -> f64 {
        self.residual_quantizer.residual(predicted, observed)
    }

    fn evaluate_objective(&self, knot_y: &[f64]) -> f64 {
        if knot_y.len() != self.knot_x.len() {
            return LARGE_COST;
        }

        let knots = materialize_spline_knots(self.knot_x.as_ref(), knot_y);
        let evaluator = match build_spline_evaluator(self.family, knots, self.extrapolation) {
            Ok(evaluator) => evaluator,
            Err(_) => return LARGE_COST,
        };

        let mut objective_sum = 0.0;
        for point in self.points.as_slice() {
            let residual = self.residual(evaluator.evaluate(point.x()), point.y());
            if !residual.is_finite() {
                return LARGE_COST;
            }
            objective_sum += self.loss_metric.value_from_residual(residual);
            if !objective_sum.is_finite() {
                return LARGE_COST;
            }
        }
        objective_sum / self.points.len() as f64
    }
}

impl CostFunction for SplineProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.evaluate_objective(array1_as_slice(param)))
    }
}

impl Gradient for SplineProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let param_slice = array1_as_slice(param);
        let mut probe = param.clone();
        let mut gradient = Array1::zeros(param_slice.len());
        for (index, gradient_value) in gradient.iter_mut().enumerate() {
            // Численный градиент по центральной схеме конечной разности.
            let step =
                ((param_slice[index].abs() + 1.0) * SPLINE_FD_REL_STEP).max(SPLINE_FD_MIN_STEP);
            probe[index] = param_slice[index] + step;
            let cost_plus = self.evaluate_objective(array1_as_slice(&probe));
            probe[index] = param_slice[index] - step;
            let cost_minus = self.evaluate_objective(array1_as_slice(&probe));
            probe[index] = param_slice[index];
            let derivative = (cost_plus - cost_minus) / (2.0 * step);
            *gradient_value = if derivative.is_finite() {
                derivative
            } else {
                LARGE_COST
            };
        }
        Ok(gradient)
    }
}

impl Hessian for SplineProblem {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        numerical_hessian_from_gradient(self, param)
    }
}

fn median_of_sorted(values: &[f64]) -> f64 {
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn median_x_of_sorted_window(window: &[[f64; 2]]) -> f64 {
    let mid = window.len() / 2;
    if window.len().is_multiple_of(2) {
        0.5 * (window[mid - 1][0] + window[mid][0])
    } else {
        window[mid][0]
    }
}

fn approximate_spline_knots(
    sorted: &[[f64; 2]],
    max_knots: usize,
    strategy: SplineKnotStrategy,
) -> Vec<[f64; 2]> {
    let max_knots = max_knots.max(2);
    let target = sorted.len().min(max_knots);
    if sorted.len() <= target {
        return sorted.to_vec();
    }

    let n = sorted.len();
    let mut knots = Vec::with_capacity(target);
    let mut y_values = Vec::new();
    for i in 0..target {
        let start = (i * n) / target;
        let end = ((i + 1) * n).div_ceil(target);
        let end = end.min(n);
        let window = &sorted[start..end];

        let knot = match strategy {
            SplineKnotStrategy::BinMean => {
                let count = window.len() as f64;
                let x_mean = window.iter().map(|p| p[0]).sum::<f64>() / count;
                let y_mean = window.iter().map(|p| p[1]).sum::<f64>() / count;
                [x_mean, y_mean]
            }
            SplineKnotStrategy::BinMedian => {
                y_values.clear();
                y_values.extend(window.iter().map(|p| p[1]));
                y_values.sort_by(|a, b| a.total_cmp(b));
                [
                    median_x_of_sorted_window(window),
                    median_of_sorted(&y_values),
                ]
            }
        };
        knots.push(knot);
    }

    // Гарантируем монотонность x после агрегации бинов (защита от округлений float).
    knots.sort_by(|a, b| a[0].total_cmp(&b[0]));
    knots
}

fn slope_between(p0: [f64; 2], p1: [f64; 2]) -> f64 {
    (p1[1] - p0[1]) / (p1[0] - p0[0])
}

fn evaluate_linear_spline(sorted: &[[f64; 2]], x: f64, extrapolation: SplineExtrapolation) -> f64 {
    let last = sorted.len() - 1;
    if x <= sorted[0][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[0][1],
            SplineExtrapolation::Linear => {
                let slope = slope_between(sorted[0], sorted[1]);
                sorted[0][1] + slope * (x - sorted[0][0])
            }
        };
    }
    if x >= sorted[last][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[last][1],
            SplineExtrapolation::Linear => {
                let slope = slope_between(sorted[last - 1], sorted[last]);
                sorted[last][1] + slope * (x - sorted[last][0])
            }
        };
    }

    let upper = sorted.partition_point(|point| point[0] < x);
    let index = upper.saturating_sub(1).min(last - 1);
    let x0 = sorted[index][0];
    let y0 = sorted[index][1];
    let x1 = sorted[index + 1][0];
    let y1 = sorted[index + 1][1];
    let t = (x - x0) / (x1 - x0);
    y0 * (1.0 - t) + y1 * t
}

fn evaluate_cubic_hermite_spline(
    sorted: &[[f64; 2]],
    derivatives: &[f64],
    x: f64,
    extrapolation: SplineExtrapolation,
) -> f64 {
    let last = sorted.len() - 1;
    if x <= sorted[0][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[0][1],
            SplineExtrapolation::Linear => sorted[0][1] + derivatives[0] * (x - sorted[0][0]),
        };
    }
    if x >= sorted[last][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[last][1],
            SplineExtrapolation::Linear => {
                sorted[last][1] + derivatives[last] * (x - sorted[last][0])
            }
        };
    }

    let upper = sorted.partition_point(|point| point[0] < x);
    let index = upper.saturating_sub(1).min(last - 1);
    let x0 = sorted[index][0];
    let y0 = sorted[index][1];
    let x1 = sorted[index + 1][0];
    let y1 = sorted[index + 1][1];
    let h = x1 - x0;
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    h00 * y0 + h10 * h * derivatives[index] + h01 * y1 + h11 * h * derivatives[index + 1]
}

fn natural_left_slope(sorted: &[[f64; 2]], second_derivatives: &[f64]) -> f64 {
    let h = sorted[1][0] - sorted[0][0];
    (sorted[1][1] - sorted[0][1]) / h
        - h * (2.0 * second_derivatives[0] + second_derivatives[1]) / 6.0
}

fn natural_right_slope(sorted: &[[f64; 2]], second_derivatives: &[f64]) -> f64 {
    let n = sorted.len();
    let h = sorted[n - 1][0] - sorted[n - 2][0];
    (sorted[n - 1][1] - sorted[n - 2][1]) / h
        + h * (2.0 * second_derivatives[n - 1] + second_derivatives[n - 2]) / 6.0
}

fn build_monotone_cubic_derivatives(sorted: &[[f64; 2]]) -> Result<Vec<f64>, FitError> {
    let n = sorted.len();
    if n < 2 {
        return Err(FitError::InvalidSplineInput(
            "Monotone cubic spline requires at least 2 points".to_string(),
        ));
    }
    if n == 2 {
        let dx = sorted[1][0] - sorted[0][0];
        let slope = (sorted[1][1] - sorted[0][1]) / dx;
        return Ok(vec![slope, slope]);
    }

    let mut h = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);
    for window in sorted.windows(2) {
        let dx = window[1][0] - window[0][0];
        h.push(dx);
        delta.push((window[1][1] - window[0][1]) / dx);
    }

    let mut derivatives = vec![0.0; n];
    for i in 1..n - 1 {
        let delta_prev = delta[i - 1];
        let delta_next = delta[i];
        if delta_prev * delta_next <= 0.0 {
            derivatives[i] = 0.0;
            continue;
        }

        let h_prev = h[i - 1];
        let h_next = h[i];
        let w1 = 2.0 * h_next + h_prev;
        let w2 = h_next + 2.0 * h_prev;
        derivatives[i] = (w1 + w2) / (w1 / delta_prev + w2 / delta_next);
    }

    let h0 = h[0];
    let h1 = h[1];
    let d0 = delta[0];
    let d1 = delta[1];
    let mut first = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if first.signum() != d0.signum() {
        first = 0.0;
    } else if d0.signum() != d1.signum() && first.abs() > 3.0 * d0.abs() {
        first = 3.0 * d0;
    }
    derivatives[0] = first;

    let hn2 = h[n - 2];
    let hn3 = h[n - 3];
    let dn2 = delta[n - 2];
    let dn3 = delta[n - 3];
    let mut last = ((2.0 * hn2 + hn3) * dn2 - hn2 * dn3) / (hn2 + hn3);
    if last.signum() != dn2.signum() {
        last = 0.0;
    } else if dn2.signum() != dn3.signum() && last.abs() > 3.0 * dn2.abs() {
        last = 3.0 * dn2;
    }
    derivatives[n - 1] = last;

    Ok(derivatives)
}

fn build_akima_derivatives(sorted: &[[f64; 2]]) -> Result<Vec<f64>, FitError> {
    let n = sorted.len();
    if n < 5 {
        return Err(FitError::InvalidSplineInput(
            "Akima spline requires at least 5 points".to_string(),
        ));
    }

    let mut slopes = Vec::with_capacity(n - 1);
    for window in sorted.windows(2) {
        let dx = window[1][0] - window[0][0];
        slopes.push((window[1][1] - window[0][1]) / dx);
    }

    let slope_at = |index: isize| -> f64 {
        let len = slopes.len() as isize;
        if (0..len).contains(&index) {
            return slopes[index as usize];
        }
        if index == -1 {
            return 2.0 * slopes[0] - slopes[1];
        }
        if index == -2 {
            return 3.0 * slopes[0] - 2.0 * slopes[1];
        }
        if index == len {
            return 2.0 * slopes[(len - 1) as usize] - slopes[(len - 2) as usize];
        }
        if index == len + 1 {
            return 3.0 * slopes[(len - 1) as usize] - 2.0 * slopes[(len - 2) as usize];
        }
        unreachable!("Akima slope index must be within extrapolation range");
    };

    let mut derivatives = vec![0.0; n];
    for (i, derivative) in derivatives.iter_mut().enumerate() {
        let i = i as isize;
        let m_im2 = slope_at(i - 2);
        let m_im1 = slope_at(i - 1);
        let m_i = slope_at(i);
        let m_ip1 = slope_at(i + 1);
        let w1 = (m_ip1 - m_i).abs();
        let w2 = (m_im1 - m_im2).abs();
        let weight_sum = w1 + w2;
        *derivative = if weight_sum > 1e-12 {
            (w1 * m_im1 + w2 * m_i) / weight_sum
        } else {
            0.5 * (m_im1 + m_i)
        };
    }

    Ok(derivatives)
}

fn build_natural_cubic_second_derivatives(sorted: &[[f64; 2]]) -> Result<Vec<f64>, FitError> {
    let n = sorted.len();
    if n < 3 {
        return Err(FitError::InvalidSplineInput(
            "Natural cubic spline requires at least 3 points".to_string(),
        ));
    }

    let interior_len = n - 2;
    let mut lower = vec![0.0; interior_len];
    let mut diagonal = vec![0.0; interior_len];
    let mut upper = vec![0.0; interior_len];
    let mut rhs = vec![0.0; interior_len];

    for interior_index in 0..interior_len {
        let i = interior_index + 1;
        let h_prev = sorted[i][0] - sorted[i - 1][0];
        let h_next = sorted[i + 1][0] - sorted[i][0];
        lower[interior_index] = h_prev;
        diagonal[interior_index] = 2.0 * (h_prev + h_next);
        upper[interior_index] = h_next;
        rhs[interior_index] = 6.0
            * ((sorted[i + 1][1] - sorted[i][1]) / h_next
                - (sorted[i][1] - sorted[i - 1][1]) / h_prev);
    }

    for i in 1..interior_len {
        let factor = lower[i] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    // Решаем трехдиагональную систему методом Томаса.
    let mut interior = vec![0.0; interior_len];
    interior[interior_len - 1] = rhs[interior_len - 1] / diagonal[interior_len - 1];
    for i in (0..interior_len - 1).rev() {
        interior[i] = (rhs[i] - upper[i] * interior[i + 1]) / diagonal[i];
    }

    let mut second_derivatives = vec![0.0; n];
    for (index, value) in interior.into_iter().enumerate() {
        second_derivatives[index + 1] = value;
    }
    Ok(second_derivatives)
}

fn evaluate_natural_cubic_spline(
    sorted: &[[f64; 2]],
    second_derivatives: &[f64],
    x: f64,
    extrapolation: SplineExtrapolation,
) -> f64 {
    let last = sorted.len() - 1;
    if x <= sorted[0][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[0][1],
            SplineExtrapolation::Linear => {
                sorted[0][1] + natural_left_slope(sorted, second_derivatives) * (x - sorted[0][0])
            }
        };
    }
    if x >= sorted[last][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[last][1],
            SplineExtrapolation::Linear => {
                sorted[last][1]
                    + natural_right_slope(sorted, second_derivatives) * (x - sorted[last][0])
            }
        };
    }

    let upper = sorted.partition_point(|point| point[0] < x);
    let index = upper.saturating_sub(1).min(last - 1);
    let x0 = sorted[index][0];
    let y0 = sorted[index][1];
    let x1 = sorted[index + 1][0];
    let y1 = sorted[index + 1][1];
    let h = x1 - x0;
    let a = (x1 - x) / h;
    let b = (x - x0) / h;

    a * y0
        + b * y1
        + ((a * a * a - a) * second_derivatives[index]
            + (b * b * b - b) * second_derivatives[index + 1])
            * (h * h / 6.0)
}

fn sample_sorted_curve<F>(samples: usize, x_bounds: [f64; 2], mut evaluate: F) -> Vec<[f64; 2]>
where
    F: FnMut(f64) -> f64,
{
    let sample_count = samples.max(2);
    let x_min = x_bounds[0];
    let x_max = x_bounds[1];
    let span = x_max - x_min;

    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count - 1) as f64;
            let x = x_min + span * t;
            [x, evaluate(x)]
        })
        .collect()
}

fn expanded_spline_curve_x_bounds(x_min: f64, x_max: f64) -> [f64; 2] {
    // Повторяем правило из UI-графика: небольшой запас по X,
    // чтобы линия не "обрубалась" на крайних наблюдениях.
    if (x_max - x_min).abs() < SPLINE_CURVE_DOMAIN_EPS {
        [
            x_min - SPLINE_CURVE_DOMAIN_FALLBACK_PADDING,
            x_max + SPLINE_CURVE_DOMAIN_FALLBACK_PADDING,
        ]
    } else {
        let padding = (x_max - x_min) * SPLINE_CURVE_DOMAIN_PADDING_RATIO;
        [x_min - padding, x_max + padding]
    }
}

struct ScalarMetrics {
    mse: f64,
    rmse: f64,
    mae: f64,
    soft_l1: f64,
    r2: f64,
    max_abs_error: f64,
}

fn scalar_loss_value(loss_metric: OptimizationLossMetric, metrics: &ScalarMetrics) -> f64 {
    match loss_metric {
        OptimizationLossMetric::Mse => metrics.mse,
        OptimizationLossMetric::Mae => metrics.mae,
        OptimizationLossMetric::SoftL1 => metrics.soft_l1,
    }
}

fn calculate_scalar_metrics_from_evaluator<F>(
    points: &Points,
    metric_quantization: MetricQuantization,
    mut evaluate: F,
) -> ScalarMetrics
where
    F: FnMut(f64) -> f64,
{
    let quantizer = ResidualQuantizer::new(metric_quantization);
    let sample_count = points.len() as f64;
    let y_mean = points
        .as_slice()
        .iter()
        .map(|point| quantizer.quantize_value(point.y()))
        .sum::<f64>()
        / sample_count;

    let mut sse = 0.0;
    let mut sae = 0.0;
    let mut soft_l1_sum = 0.0;
    let mut max_abs_error = 0.0_f64;
    for point in points.as_slice() {
        let residual = quantizer.residual(evaluate(point.x()), point.y());
        let abs_residual = residual.abs();
        sse += residual * residual;
        sae += abs_residual;
        soft_l1_sum += OptimizationLossMetric::SoftL1.value_from_residual(residual);
        max_abs_error = max_abs_error.max(abs_residual);
    }

    let sst = points
        .as_slice()
        .iter()
        .map(|point| {
            let centered = quantizer.quantize_value(point.y()) - y_mean;
            centered * centered
        })
        .sum::<f64>();
    let mse = sse / sample_count;
    let rmse = mse.sqrt();
    let mae = sae / sample_count;
    let soft_l1 = soft_l1_sum / sample_count;
    let r2 = if sst <= 1e-15 {
        if sse <= 1e-15 { 1.0 } else { 0.0 }
    } else {
        1.0 - sse / sst
    };

    ScalarMetrics {
        mse,
        rmse,
        mae,
        soft_l1,
        r2,
        max_abs_error,
    }
}

struct EvaluatorMetrics {
    mse: f64,
    rmse: f64,
    mae: f64,
    soft_l1: f64,
    r2: f64,
    max_abs_error: f64,
    residuals: Vec<[f64; 2]>,
}

fn calculate_metrics_from_evaluator<F>(
    points: &Points,
    metric_quantization: MetricQuantization,
    mut evaluate: F,
) -> EvaluatorMetrics
where
    F: FnMut(f64) -> f64,
{
    let scalar =
        calculate_scalar_metrics_from_evaluator(points, metric_quantization, &mut evaluate);

    let mut residuals = Vec::with_capacity(points.len());
    for point in points.as_slice() {
        let residual = evaluate(point.x()) - point.y();
        residuals.push([point.x(), residual]);
    }

    EvaluatorMetrics {
        mse: scalar.mse,
        rmse: scalar.rmse,
        mae: scalar.mae,
        soft_l1: scalar.soft_l1,
        r2: scalar.r2,
        max_abs_error: scalar.max_abs_error,
        residuals,
    }
}

fn ensure_min_knot_count(
    knots: usize,
    min_knots: usize,
    spline_name: &str,
) -> Result<(), FitError> {
    if knots < min_knots {
        return Err(FitError::InvalidSplineInput(format!(
            "{spline_name} requires at least {min_knots} knots"
        )));
    }
    Ok(())
}

struct PreparedSplineInputs {
    config: SplineConfig,
    knot_x: Box<[f64]>,
    initial_y: Vec<f64>,
    curve_x_bounds: [f64; 2],
}

fn prepare_spline_inputs(
    points: &Points,
    config: SplineConfig,
    family: SplineFamilyKind,
    initial_knot_y: Option<&[f64]>,
) -> Result<PreparedSplineInputs, FitError> {
    let config = config.normalized();
    if points.len() >= family.min_knots() {
        ensure_min_knot_count(config.knots, family.min_knots(), family.name())?;
    }

    let sorted = sorted_points_with_duplicate_policy(points, config.duplicate_x_policy)?;
    let initial_knots = approximate_spline_knots(&sorted, config.knots, config.knot_strategy);
    if initial_knots.len() < family.min_knots() {
        return Err(FitError::InvalidSplineInput(format!(
            "{} requires at least {} points",
            family.name(),
            family.min_knots()
        )));
    }
    let curve_x_bounds = expanded_spline_curve_x_bounds(sorted[0][0], sorted[sorted.len() - 1][0]);

    let knot_x = initial_knots
        .iter()
        .map(|point| point[0])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let initial_y_default = initial_knots
        .iter()
        .map(|point| point[1])
        .collect::<Vec<_>>();
    let initial_y = if let Some(values) = initial_knot_y {
        let expected = knot_x.len();
        if values.len() != expected {
            return Err(FitError::InvalidSplineInput(format!(
                "Spline initialization expects {expected} values, got {}",
                values.len()
            )));
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(FitError::InvalidSplineInput(format!(
                    "Spline initialization value at index {index} must be finite, got {value}"
                )));
            }
        }
        values.to_vec()
    } else {
        initial_y_default
    };

    Ok(PreparedSplineInputs {
        config,
        knot_x,
        initial_y,
        curve_x_bounds,
    })
}

struct SplineFinalizeContext<'a> {
    points: &'a Points,
    family: SplineFamilyKind,
    config: SplineConfig,
    knot_x: &'a [f64],
    curve_x_bounds: [f64; 2],
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
}

fn build_spline_result_from_knot_y(
    context: &SplineFinalizeContext<'_>,
    knot_y: &[f64],
    iterations: u64,
) -> Result<(SplineResult, IterationMetricSnapshot), FitError> {
    let built = build_spline_curve_from_knot_y(
        context.family,
        context.config.extrapolation,
        context.config.samples,
        context.knot_x,
        knot_y,
        context.curve_x_bounds,
    )?;
    let metrics =
        calculate_metrics_from_evaluator(context.points, context.metric_quantization, |x| {
            built.evaluator.evaluate(x)
        });

    let result = SplineResult {
        knots: built.knots,
        curve: built.curve,
        mse: metrics.mse,
        rmse: metrics.rmse,
        mae: metrics.mae,
        r2: metrics.r2,
        max_abs_error: metrics.max_abs_error,
        residuals: metrics.residuals,
        iterations,
    };
    let iteration_metrics = IterationMetricSnapshot {
        loss: match context.loss_metric {
            OptimizationLossMetric::Mse => metrics.mse,
            OptimizationLossMetric::Mae => metrics.mae,
            OptimizationLossMetric::SoftL1 => metrics.soft_l1,
        },
        mse: metrics.mse,
        rmse: metrics.rmse,
        mae: metrics.mae,
        soft_l1: metrics.soft_l1,
        r2: metrics.r2,
        max_abs_error: metrics.max_abs_error,
    };

    Ok((result, iteration_metrics))
}

/// Строит стартовую кривую сплайна из пользовательской инициализации.
///
/// Используется UI для формирования replay-кадра `iteration = 0` до запуска оптимизации.
pub(crate) fn build_spline_initial_curve_from_knot_y(
    points: &Points,
    family: SplineFamilyKind,
    config: SplineConfig,
    knot_y: &[f64],
) -> Result<Vec<[f64; 2]>, FitError> {
    let prepared = prepare_spline_inputs(points, config, family, Some(knot_y))?;
    let built = build_spline_curve_from_knot_y(
        family,
        prepared.config.extrapolation,
        prepared.config.samples,
        prepared.knot_x.as_ref(),
        &prepared.initial_y,
        prepared.curve_x_bounds,
    )?;
    Ok(built.curve)
}

struct BuiltSplineCurve {
    knots: Vec<[f64; 2]>,
    evaluator: SplineEvaluator,
    curve: Vec<[f64; 2]>,
}

fn build_spline_curve_from_knot_y(
    family: SplineFamilyKind,
    extrapolation: SplineExtrapolation,
    samples: usize,
    knot_x: &[f64],
    knot_y: &[f64],
    curve_x_bounds: [f64; 2],
) -> Result<BuiltSplineCurve, FitError> {
    let knots = materialize_spline_knots(knot_x, knot_y);
    let evaluator = build_spline_evaluator(family, knots.clone(), extrapolation)?;
    let curve = sample_sorted_curve(samples, curve_x_bounds, |x| evaluator.evaluate(x));
    Ok(BuiltSplineCurve {
        knots,
        evaluator,
        curve,
    })
}

#[cfg(test)]
mod tests;
