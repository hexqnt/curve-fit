//! Численные алгоритмы подгонки параметрических моделей и сплайнов.
//! Модуль инкапсулирует оптимизацию, расчет метрик и дискретизацию кривых.

use std::fmt;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{
    CostFunction, Executor, Gradient, IterState, KV, Problem, Solver, State, TerminationStatus,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use crate::domain::{CurveFamily, CurveParams, FitResult, InputError, LbfgsConfig, Point, Points};

const PARAM_EPS: f64 = 1e-9;
const LARGE_COST: f64 = 1e24;
const LN_2: f64 = std::f64::consts::LN_2;
const CANCELLED_MARKER: &str = "__curve_fit_cancelled__";

fn positive_x(value: f64) -> f64 {
    value.max(PARAM_EPS)
}

fn positive_param_with_derivative(value: f64) -> (f64, f64) {
    if value.abs() >= PARAM_EPS {
        (value.abs(), value.signum())
    } else {
        (PARAM_EPS, 0.0)
    }
}

fn non_zero_param_with_derivative(value: f64) -> (f64, f64) {
    if value.abs() >= PARAM_EPS {
        (value, 1.0)
    } else if value.is_sign_negative() {
        (-PARAM_EPS, 0.0)
    } else {
        (PARAM_EPS, 0.0)
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
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

struct CurveProblem {
    family: CurveFamily,
    points: Points,
}

impl CurveProblem {
    fn new(family: CurveFamily, points: &Points) -> Self {
        Self {
            family,
            points: points.clone(),
        }
    }
}

fn accumulate_polynomial_gradient(points: &[Point], param: &[f64], gradient: &mut [f64]) {
    debug_assert_eq!(gradient.len(), param.len());
    for point in points {
        let x = point.x();
        let model = param
            .iter()
            .copied()
            .fold(0.0, |acc, coefficient| acc * x + coefficient);
        let residual = model - point.y();

        let mut basis = 1.0;
        for gradient_value in gradient.iter_mut().rev() {
            *gradient_value += residual * basis;
            basis *= x;
        }
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
    Clamp,
    Linear,
}

impl SplineExtrapolation {
    /// Полный список вариантов экстраполяции.
    pub const ALL: [Self; 2] = [Self::Clamp, Self::Linear];
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

type LbfgsState = IterState<Vec<f64>, Vec<f64>, (), (), (), f64>;
type LbfgsSolver = LBFGS<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, Vec<f64>, Vec<f64>, f64>;

#[derive(Debug, Clone, PartialEq)]
/// Шаг инкрементальной подгонки параметрической модели.
pub enum IncrementalFitStep {
    Iteration {
        iteration: u64,
        mse: f64,
        params: CurveParams,
    },
    Finished(FitResult),
    Cancelled,
}

/// Пошаговый раннер L-BFGS для параметрических семейств.
pub struct IncrementalFitRunner {
    family: CurveFamily,
    points: Points,
    problem: Problem<CurveProblem>,
    solver: LbfgsSolver,
    state: Option<LbfgsState>,
    cancelled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IncrementalSplineFitStep {
    Iteration {
        iteration: u64,
        mse: f64,
        knot_y: Vec<f64>,
        curve: Vec<[f64; 2]>,
    },
    Finished(SplineResult),
    Cancelled,
}

pub(crate) struct IncrementalSplineFitRunner {
    family: SplineFamilyKind,
    points: Points,
    config: SplineConfig,
    knot_x: Box<[f64]>,
    problem: Problem<SplineProblem>,
    solver: LbfgsSolver,
    state: Option<LbfgsState>,
    cancelled: bool,
}

struct ProgressObserver<F> {
    family: CurveFamily,
    on_iteration: F,
}

impl<F, I> Observe<I> for ProgressObserver<F>
where
    F: FnMut(u64, Option<CurveParams>) -> bool,
    I: State<Param = Vec<f64>, Float = f64>,
{
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), argmin::core::Error> {
        let params = state
            .get_param()
            .and_then(|param_values| CurveParams::try_from_slice(self.family, param_values).ok());
        if !(self.on_iteration)(state.get_iter(), params) {
            return Err(argmin::core::Error::msg(CANCELLED_MARKER));
        }
        Ok(())
    }
}

fn build_line_search(
    config: &LbfgsConfig,
) -> Result<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, FitError> {
    MoreThuenteLineSearch::new()
        .with_c(config.c1, config.c2)
        .map_err(|error| FitError::Optimizer(error.to_string()))?
        .with_bounds(config.step_min, config.step_max)
        .map_err(|error| FitError::Optimizer(error.to_string()))?
        .with_width_tolerance(config.width_tolerance)
        .map_err(|error| FitError::Optimizer(error.to_string()))
}

fn build_lbfgs_solver(config: &LbfgsConfig) -> Result<LbfgsSolver, FitError> {
    let line_search = build_line_search(config)?;
    LBFGS::new(line_search, config.history_size)
        .with_tolerance_grad(config.tol_grad)
        .map_err(|error| FitError::Optimizer(error.to_string()))?
        .with_tolerance_cost(config.tol_cost)
        .map_err(|error| FitError::Optimizer(error.to_string()))
}

impl IncrementalFitRunner {
    /// Создает раннер и инициализирует внутреннее состояние оптимизатора.
    pub fn new(
        points: &Points,
        family: CurveFamily,
        initial_params: CurveParams,
        config: &LbfgsConfig,
    ) -> Result<Self, FitError> {
        if initial_params.family() != family {
            return Err(FitError::InvalidInput(InputError::FamilyMismatch {
                expected: family,
                got: initial_params.family(),
            }));
        }
        family.validate_points(points)?;

        let problem = CurveProblem::new(family, points);
        let mut problem = Problem::new(problem);
        let mut solver = build_lbfgs_solver(config)?;
        let state = IterState::new()
            .param(initial_params.values())
            .max_iters(config.max_iters);
        let (mut state, _) = solver
            .init(&mut problem, state)
            .map_err(|error| FitError::Optimizer(error.to_string()))?;

        state.update();
        state.func_counts(&problem);

        Ok(Self {
            family,
            points: points.clone(),
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
            let mut state = self
                .state
                .take()
                .expect("incremental fit state must be initialized");

            if !state.terminated() {
                let termination =
                    <LbfgsSolver as Solver<CurveProblem, LbfgsState>>::terminate_internal(
                        &mut self.solver,
                        &state,
                    );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }

            if state.terminated() {
                let final_step = self.finalize(state)?;
                return Ok(final_step);
            }

            let (mut state, _) = self
                .solver
                .next_iter(&mut self.problem, state)
                .map_err(|error| FitError::Optimizer(error.to_string()))?;
            state.func_counts(&self.problem);
            state.update();

            let iteration = state.get_iter();
            if let Some(params) = state
                .get_param()
                .and_then(|values| CurveParams::try_from_slice(self.family, values).ok())
            {
                let (mse, _) = calculate_metrics(&self.points, &params);
                state.increment_iter();
                self.state = Some(state);
                return Ok(IncrementalFitStep::Iteration {
                    iteration,
                    mse,
                    params,
                });
            }

            // Если параметры недоступны на текущем шаге, продолжаем итерации без рекурсии.
            state.increment_iter();
            self.state = Some(state);
        }
    }

    fn finalize(&mut self, state: LbfgsState) -> Result<IncrementalFitStep, FitError> {
        let best_param_values = state
            .get_best_param()
            .cloned()
            .ok_or(FitError::MissingBestParameters)?;
        let best_params = CurveParams::try_from_values(self.family, best_param_values)?;
        let (mse, rmse) = calculate_metrics(&self.points, &best_params);
        let iterations = state.get_iter();
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
    pub(crate) fn new(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
    ) -> Result<Self, FitError> {
        Self::new_with_initial_knot_y(points, family, config, None)
    }

    pub(crate) fn new_with_initial_knot_y(
        points: &Points,
        family: SplineFamilyKind,
        config: SplineConfig,
        initial_knot_y: Option<&[f64]>,
    ) -> Result<Self, FitError> {
        let prepared = prepare_spline_inputs(points, config, family, initial_knot_y)?;
        let optimizer_config = spline_lbfgs_config();
        let mut solver = build_lbfgs_solver(&optimizer_config)?;

        let initial_knots = materialize_spline_knots(prepared.knot_x.as_ref(), &prepared.initial_y);
        let problem = SplineProblem::new(
            family,
            &initial_knots,
            points,
            prepared.config.extrapolation,
        );
        let mut problem = Problem::new(problem);
        let state = IterState::new()
            .param(prepared.initial_y)
            .max_iters(optimizer_config.max_iters);
        let (mut state, _) = solver
            .init(&mut problem, state)
            .map_err(|error| FitError::Optimizer(error.to_string()))?;

        state.update();
        state.func_counts(&problem);

        Ok(Self {
            family,
            points: points.clone(),
            config: prepared.config,
            knot_x: prepared.knot_x,
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
            let mut state = self
                .state
                .take()
                .expect("incremental spline fit state must be initialized");

            if !state.terminated() {
                let termination =
                    <LbfgsSolver as Solver<SplineProblem, LbfgsState>>::terminate_internal(
                        &mut self.solver,
                        &state,
                    );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }

            if state.terminated() {
                let final_step = self.finalize(state)?;
                return Ok(final_step);
            }

            let (mut state, _) = self
                .solver
                .next_iter(&mut self.problem, state)
                .map_err(|error| FitError::Optimizer(error.to_string()))?;
            state.func_counts(&self.problem);
            state.update();

            let iteration = state.get_iter();
            if let Some(knot_y) = state.get_param().cloned() {
                let knots = materialize_spline_knots(self.knot_x.as_ref(), &knot_y);
                let evaluator =
                    build_spline_evaluator(self.family, knots.clone(), self.config.extrapolation)?;
                let mse = self
                    .points
                    .as_slice()
                    .iter()
                    .map(|point| {
                        let residual = evaluator.evaluate(point.x()) - point.y();
                        residual * residual
                    })
                    .sum::<f64>()
                    / self.points.len() as f64;
                let curve =
                    sample_sorted_curve(&knots, self.config.samples, |x| evaluator.evaluate(x));

                state.increment_iter();
                self.state = Some(state);
                return Ok(IncrementalSplineFitStep::Iteration {
                    iteration,
                    mse,
                    knot_y,
                    curve,
                });
            }

            // Если параметры недоступны на текущем шаге, продолжаем итерации без рекурсии.
            state.increment_iter();
            self.state = Some(state);
        }
    }

    fn finalize(&mut self, state: LbfgsState) -> Result<IncrementalSplineFitStep, FitError> {
        let best_knot_y = state
            .get_best_param()
            .or_else(|| state.get_param())
            .cloned()
            .ok_or(FitError::MissingBestParameters)?;
        let iterations = state.get_iter();
        self.state = Some(state);

        let result = build_spline_result_from_knot_y(
            &self.points,
            self.family,
            self.config,
            self.knot_x.as_ref(),
            &best_knot_y,
            iterations,
        )?;

        Ok(IncrementalSplineFitStep::Finished(result))
    }
}

impl CostFunction for CurveProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let sample_count = self.points.len() as f64;
        let mut sum = 0.0;

        for point in self.points.as_slice() {
            let predicted = self.family.evaluate_raw(param, point.x());
            let residual = predicted - point.y();
            if !residual.is_finite() {
                return Ok(LARGE_COST);
            }
            sum += residual * residual;
            if !sum.is_finite() {
                return Ok(LARGE_COST);
            }
        }

        Ok(sum / sample_count)
    }
}

impl Gradient for CurveProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let mut gradient = vec![0.0; self.family.parameter_count()];
        let points = self.points.as_slice();
        let scale = 2.0 / points.len() as f64;

        match self.family {
            CurveFamily::Linear
            | CurveFamily::Quadratic
            | CurveFamily::Cubic
            | CurveFamily::Quartic
            | CurveFamily::Quintic
            | CurveFamily::Sextic
            | CurveFamily::Septic
            | CurveFamily::Octic
            | CurveFamily::Nonic => {
                accumulate_polynomial_gradient(points, param, &mut gradient);
            }
            CurveFamily::Arrhenius => {
                for point in points {
                    let x = positive_x(point.x());
                    let exp_term = (param[1] / x).exp();
                    let model = param[0] * exp_term;
                    let residual = model - point.y();
                    gradient[0] += residual * exp_term;
                    gradient[1] += residual * (param[0] * exp_term / x);
                }
            }
            CurveFamily::Inverse => {
                for point in points {
                    let x = positive_x(point.x());
                    let residual = param[0] + param[1] / x - point.y();
                    gradient[0] += residual;
                    gradient[1] += residual / x;
                }
            }
            CurveFamily::Logistic => {
                for point in points {
                    let z = param[1] * (point.x() - param[2]);
                    let s = 1.0 / (1.0 + (-z).exp());
                    let model = param[0] * s;
                    let residual = model - point.y();
                    let ds_dz = s * (1.0 - s);

                    gradient[0] += residual * s;
                    gradient[1] += residual * (param[0] * ds_dz * (point.x() - param[2]));
                    gradient[2] += residual * (param[0] * ds_dz * (-param[1]));
                }
            }
            CurveFamily::Lorentzian => {
                for point in points {
                    let a = param[0];
                    let x0 = param[1];
                    let (gamma, d_gamma_raw) = positive_param_with_derivative(param[2]);
                    let x = point.x();
                    let u = (x - x0) / gamma;
                    let den = 1.0 + u * u;
                    let inv_den = 1.0 / den;
                    let model = param[3] + a * inv_den;
                    let residual = model - point.y();
                    let common = 2.0 * a / (den * den * gamma);

                    gradient[0] += residual * inv_den;
                    gradient[1] += residual * (common * u);
                    gradient[2] += residual * (common * u * u) * d_gamma_raw;
                    gradient[3] += residual;
                }
            }
            CurveFamily::NaturalLog => {
                for point in points {
                    let x = positive_x(point.x());
                    let (b, d_b_raw) = positive_param_with_derivative(param[1]);
                    let ln_term = (x / b).ln();
                    let model = param[0] * ln_term;
                    let residual = model - point.y();

                    gradient[0] += residual * ln_term;
                    gradient[1] += residual * (-param[0] / b) * d_b_raw;
                }
            }
            CurveFamily::FourPl => {
                for point in points {
                    let x = positive_x(point.x());
                    let a = param[0];
                    let b = param[1];
                    let (c, d_c_raw) = positive_param_with_derivative(param[2]);
                    let d = param[3];
                    let ratio = x / c;
                    let pow = ratio.powf(b);
                    let den = 1.0 + pow;
                    let inv_den = 1.0 / den;
                    let model = d + (a - d) * inv_den;
                    let residual = model - point.y();
                    let d_pow_db = pow * ratio.ln();
                    let d_pow_dc = -pow * b / c;
                    let d_model_da = inv_den;
                    let d_model_dd = 1.0 - inv_den;
                    let d_model_db = -(a - d) * d_pow_db / (den * den);
                    let d_model_dc = -(a - d) * d_pow_dc / (den * den);

                    gradient[0] += residual * d_model_da;
                    gradient[1] += residual * d_model_db;
                    gradient[2] += residual * d_model_dc * d_c_raw;
                    gradient[3] += residual * d_model_dd;
                }
            }
            CurveFamily::FivePl => {
                for point in points {
                    let x = positive_x(point.x());
                    let a = param[0];
                    let b = param[1];
                    let (c, d_c_raw) = positive_param_with_derivative(param[2]);
                    let d = param[3];
                    let (m, d_m_raw) = positive_param_with_derivative(param[4]);
                    let ratio = x / c;
                    let pow = ratio.powf(b);
                    let den = 1.0 + pow;
                    let inv = den.powf(-m);
                    let model = d + (a - d) * inv;
                    let residual = model - point.y();
                    let d_pow_db = pow * ratio.ln();
                    let d_pow_dc = -pow * b / c;
                    let d_inv_db = -m * den.powf(-m - 1.0) * d_pow_db;
                    let d_inv_dc = -m * den.powf(-m - 1.0) * d_pow_dc;
                    let d_inv_dm = -inv * den.ln();

                    gradient[0] += residual * inv;
                    gradient[1] += residual * (a - d) * d_inv_db;
                    gradient[2] += residual * (a - d) * d_inv_dc * d_c_raw;
                    gradient[3] += residual * (1.0 - inv);
                    gradient[4] += residual * (a - d) * d_inv_dm * d_m_raw;
                }
            }
            CurveFamily::MichaelisMenten => {
                for point in points {
                    let x = point.x();
                    let vmax = param[0];
                    let (denominator, d_den_d_km) = non_zero_param_with_derivative(x + param[1]);
                    let model = vmax * x / denominator;
                    let residual = model - point.y();
                    let d_model_d_vmax = x / denominator;
                    let d_model_d_km = -vmax * x / (denominator * denominator) * d_den_d_km;

                    gradient[0] += residual * d_model_d_vmax;
                    gradient[1] += residual * d_model_d_km;
                }
            }
            CurveFamily::ExponentialBasic => {
                for point in points {
                    let x = point.x();
                    let exp_part = (-param[2] * x).exp();
                    let residual = param[0] + param[1] * exp_part - point.y();
                    gradient[0] += residual;
                    gradient[1] += residual * exp_part;
                    gradient[2] += residual * (-param[1] * x * exp_part);
                }
            }
            CurveFamily::ExponentialLinear => {
                for point in points {
                    let x = point.x();
                    let exp_part = (param[1] * x).exp();
                    let residual = param[0] * exp_part + param[2] * x + param[3] - point.y();
                    gradient[0] += residual * exp_part;
                    gradient[1] += residual * (param[0] * exp_part * x);
                    gradient[2] += residual * x;
                    gradient[3] += residual;
                }
            }
            CurveFamily::ExponentialHalfLife => {
                for point in points {
                    let x = point.x();
                    let (c, d_c_raw) = positive_param_with_derivative(param[2]);
                    let exponent = -LN_2 * x / c;
                    let pow = exponent.exp();
                    let residual = param[0] + param[1] * pow - point.y();
                    let d_model_d_c = param[1] * pow * LN_2 * x / (c * c);

                    gradient[0] += residual;
                    gradient[1] += residual * pow;
                    gradient[2] += residual * d_model_d_c * d_c_raw;
                }
            }
            CurveFamily::FallingExponential => {
                for point in points {
                    let x = point.x();
                    let y0 = param[0];
                    let v0 = param[1];
                    let (k, d_k_raw) = non_zero_param_with_derivative(param[2]);
                    let exp_part = (-k * x).exp();
                    let one_minus_exp = -(-k * x).exp_m1();
                    let model = y0 - (v0 / k) * one_minus_exp;
                    let residual = model - point.y();
                    let d_model_d_v0 = -one_minus_exp / k;
                    let d_model_d_k = v0 * (one_minus_exp - k * x * exp_part) / (k * k);

                    gradient[0] += residual;
                    gradient[1] += residual * d_model_d_v0;
                    gradient[2] += residual * d_model_d_k * d_k_raw;
                }
            }
            CurveFamily::HyperbolicTangent => {
                for point in points {
                    let x = point.x();
                    let z = param[1] * (x - param[2]);
                    let tanh_z = z.tanh();
                    let sech2_z = 1.0 - tanh_z * tanh_z;
                    let residual = param[0] * tanh_z + param[3] - point.y();

                    gradient[0] += residual * tanh_z;
                    gradient[1] += residual * (param[0] * sech2_z * (x - param[2]));
                    gradient[2] += residual * (-param[0] * sech2_z * param[1]);
                    gradient[3] += residual;
                }
            }
            CurveFamily::ArctangentStep => {
                for point in points {
                    let x = point.x();
                    let z = param[1] * (x - param[2]);
                    let atan_z = z.atan();
                    let inv_den = 1.0 / (1.0 + z * z);
                    let residual = param[0] * atan_z + param[3] - point.y();

                    gradient[0] += residual * atan_z;
                    gradient[1] += residual * (param[0] * (x - param[2]) * inv_den);
                    gradient[2] += residual * (-param[0] * param[1] * inv_den);
                    gradient[3] += residual;
                }
            }
            CurveFamily::Softplus => {
                for point in points {
                    let x = point.x();
                    let z = param[1] * (x - param[2]);
                    let softplus_z = softplus(z);
                    let sigma_z = sigmoid(z);
                    let residual = param[0] * softplus_z + param[3] - point.y();

                    gradient[0] += residual * softplus_z;
                    gradient[1] += residual * (param[0] * sigma_z * (x - param[2]));
                    gradient[2] += residual * (-param[0] * sigma_z * param[1]);
                    gradient[3] += residual;
                }
            }
            CurveFamily::Power => {
                for point in points {
                    let x = positive_x(point.x());
                    let pow = x.powf(param[1]);
                    let residual = param[0] * pow - point.y();
                    gradient[0] += residual * pow;
                    gradient[1] += residual * param[0] * pow * x.ln();
                }
            }
            CurveFamily::Gaussian => {
                for point in points {
                    let x = point.x();
                    let a = param[0];
                    let b = param[1];
                    let (c, d_c_raw) = positive_param_with_derivative(param[2]);
                    let c2 = c * c;
                    let delta = x - b;
                    let exp_part = (-(delta * delta) / (2.0 * c2)).exp();
                    let residual = a * exp_part - point.y();
                    let d_model_d_a = exp_part;
                    let d_model_d_b = a * exp_part * delta / c2;
                    let d_model_d_c = a * exp_part * delta * delta / (c2 * c);

                    gradient[0] += residual * d_model_d_a;
                    gradient[1] += residual * d_model_d_b;
                    gradient[2] += residual * d_model_d_c * d_c_raw;
                }
            }
        }

        for value in &mut gradient {
            *value *= scale;
            if !value.is_finite() {
                *value = LARGE_COST;
            }
        }

        Ok(gradient)
    }
}

/// Вычисляет базовые метрики качества подгонки: `(MSE, RMSE)`.
pub fn calculate_metrics(points: &Points, params: &CurveParams) -> (f64, f64) {
    let mse = points
        .as_slice()
        .iter()
        .map(|point| {
            let residual = params.evaluate(point.x()) - point.y();
            residual * residual
        })
        .sum::<f64>()
        / points.len() as f64;
    (mse, mse.sqrt())
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
}

impl SplineProblem {
    fn new(
        family: SplineFamilyKind,
        initial_knots: &[[f64; 2]],
        points: &Points,
        extrapolation: SplineExtrapolation,
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
        }
    }

    fn evaluate_mse(&self, knot_y: &[f64]) -> f64 {
        if knot_y.len() != self.knot_x.len() {
            return LARGE_COST;
        }

        let knots = materialize_spline_knots(self.knot_x.as_ref(), knot_y);
        let evaluator = match build_spline_evaluator(self.family, knots, self.extrapolation) {
            Ok(evaluator) => evaluator,
            Err(_) => return LARGE_COST,
        };

        let mut sse = 0.0;
        for point in self.points.as_slice() {
            let residual = evaluator.evaluate(point.x()) - point.y();
            if !residual.is_finite() {
                return LARGE_COST;
            }
            sse += residual * residual;
            if !sse.is_finite() {
                return LARGE_COST;
            }
        }
        sse / self.points.len() as f64
    }
}

impl CostFunction for SplineProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.evaluate_mse(param))
    }
}

impl Gradient for SplineProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let base_cost = self.evaluate_mse(param);
        let mut probe = param.clone();
        let mut gradient = vec![0.0; param.len()];
        for (index, gradient_value) in gradient.iter_mut().enumerate() {
            // Численный градиент по прямой схеме конечной разности.
            let step = ((param[index].abs() + 1.0) * SPLINE_FD_REL_STEP).max(SPLINE_FD_MIN_STEP);
            probe[index] = param[index] + step;
            let cost_plus = self.evaluate_mse(&probe);
            probe[index] = param[index];
            let derivative = (cost_plus - base_cost) / step;
            *gradient_value = if derivative.is_finite() {
                derivative
            } else {
                LARGE_COST
            };
        }
        Ok(gradient)
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

fn sample_sorted_curve<F>(sorted: &[[f64; 2]], samples: usize, mut evaluate: F) -> Vec<[f64; 2]>
where
    F: FnMut(f64) -> f64,
{
    let sample_count = samples.max(2);
    let x_min = sorted[0][0];
    let x_max = sorted[sorted.len() - 1][0];
    let span = x_max - x_min;

    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count - 1) as f64;
            let x = x_min + span * t;
            [x, evaluate(x)]
        })
        .collect()
}

struct EvaluatorMetrics {
    mse: f64,
    rmse: f64,
    mae: f64,
    r2: f64,
    max_abs_error: f64,
    residuals: Vec<[f64; 2]>,
}

fn calculate_metrics_from_evaluator<F>(points: &Points, mut evaluate: F) -> EvaluatorMetrics
where
    F: FnMut(f64) -> f64,
{
    let sample_count = points.len() as f64;
    let y_mean = points.as_slice().iter().map(|point| point.y()).sum::<f64>() / sample_count;

    let mut sse = 0.0;
    let mut sae = 0.0;
    let mut max_abs_error = 0.0_f64;
    let mut residuals = Vec::with_capacity(points.len());
    for point in points.as_slice() {
        let residual = evaluate(point.x()) - point.y();
        let abs_residual = residual.abs();
        sse += residual * residual;
        sae += abs_residual;
        max_abs_error = max_abs_error.max(abs_residual);
        residuals.push([point.x(), residual]);
    }

    let sst = points
        .as_slice()
        .iter()
        .map(|point| {
            let centered = point.y() - y_mean;
            centered * centered
        })
        .sum::<f64>();
    let mse = sse / sample_count;
    let rmse = mse.sqrt();
    let mae = sae / sample_count;
    let r2 = if sst <= 1e-15 {
        if sse <= 1e-15 { 1.0 } else { 0.0 }
    } else {
        1.0 - sse / sst
    };

    EvaluatorMetrics {
        mse,
        rmse,
        mae,
        r2,
        max_abs_error,
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
    })
}

fn build_spline_result_from_knot_y(
    points: &Points,
    family: SplineFamilyKind,
    config: SplineConfig,
    knot_x: &[f64],
    knot_y: &[f64],
    iterations: u64,
) -> Result<SplineResult, FitError> {
    let knots = materialize_spline_knots(knot_x, knot_y);
    let evaluator = build_spline_evaluator(family, knots.clone(), config.extrapolation)?;
    let curve = sample_sorted_curve(&knots, config.samples, |x| evaluator.evaluate(x));
    let metrics = calculate_metrics_from_evaluator(points, |x| evaluator.evaluate(x));

    Ok(SplineResult {
        knots,
        curve,
        mse: metrics.mse,
        rmse: metrics.rmse,
        mae: metrics.mae,
        r2: metrics.r2,
        max_abs_error: metrics.max_abs_error,
        residuals: metrics.residuals,
        iterations,
    })
}

fn fit_spline_family_with_config(
    points: &Points,
    config: SplineConfig,
    family: SplineFamilyKind,
) -> Result<SplineResult, FitError> {
    let mut runner = IncrementalSplineFitRunner::new(points, family, config)?;
    loop {
        match runner.step()? {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished(result) => return Ok(result),
            IncrementalSplineFitStep::Cancelled => return Err(FitError::Cancelled),
        }
    }
}

pub(crate) fn default_spline_initial_knot_y(
    points: &Points,
    family: SplineFamilyKind,
    config: SplineConfig,
) -> Result<Vec<f64>, FitError> {
    let prepared = prepare_spline_inputs(points, config, family, None)?;
    Ok(prepared.initial_y)
}

/// Подгоняет линейный сплайн с явными `samples` и `knots`.
pub fn fit_linear_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_linear_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет линейный сплайн с полной конфигурацией.
pub fn fit_linear_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_config(points, config, SplineFamilyKind::Linear)
}

/// Подгоняет монотонный кубический сплайн с явными `samples` и `knots`.
pub fn fit_monotone_cubic_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_monotone_cubic_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет монотонный кубический сплайн с полной конфигурацией.
pub fn fit_monotone_cubic_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_config(points, config, SplineFamilyKind::MonotoneCubic)
}

/// Подгоняет натуральный кубический сплайн с явными `samples` и `knots`.
pub fn fit_natural_cubic_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_natural_cubic_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет натуральный кубический сплайн с полной конфигурацией.
pub fn fit_natural_cubic_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_config(points, config, SplineFamilyKind::NaturalCubic)
}

/// Подгоняет сплайн Акимы с явными `samples` и `knots`.
pub fn fit_akima_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_akima_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет сплайн Акимы с полной конфигурацией.
pub fn fit_akima_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_config(points, config, SplineFamilyKind::Akima)
}

/// Подгоняет параметрическую модель без колбэка прогресса.
pub fn fit_curve(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    config: &LbfgsConfig,
) -> Result<FitResult, FitError> {
    fit_curve_with_progress(
        points,
        family,
        initial_params,
        config,
        |_iteration, _params| true,
    )
}

/// Подгоняет параметрическую модель с колбэком на каждой итерации.
///
/// Возврат `false` из `on_iteration` запрашивает досрочную остановку.
pub fn fit_curve_with_progress<F>(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    config: &LbfgsConfig,
    on_iteration: F,
) -> Result<FitResult, FitError>
where
    F: FnMut(u64, Option<CurveParams>) -> bool + 'static,
{
    if initial_params.family() != family {
        return Err(FitError::InvalidInput(InputError::FamilyMismatch {
            expected: family,
            got: initial_params.family(),
        }));
    }
    family.validate_points(points)?;

    let solver = build_lbfgs_solver(config)?;
    let problem = CurveProblem::new(family, points);
    let progress_observer = ProgressObserver {
        family,
        on_iteration,
    };
    let optimization = Executor::new(problem, solver)
        .configure(|state| {
            state
                .param(initial_params.values())
                .max_iters(config.max_iters)
        })
        .add_observer(progress_observer, ObserverMode::Always)
        .run()
        .map_err(|error| {
            let message = error.to_string();
            if message.contains(CANCELLED_MARKER) {
                FitError::Cancelled
            } else {
                FitError::Optimizer(message)
            }
        })?;

    let best_param_values = optimization
        .state()
        .get_best_param()
        .cloned()
        .ok_or(FitError::MissingBestParameters)?;
    let best_params = CurveParams::try_from_values(family, best_param_values)?;
    let (mse, rmse) = calculate_metrics(points, &best_params);

    Ok(FitResult {
        family,
        params: best_params,
        mse,
        rmse,
        iterations: optimization.state().get_iter(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_SPLINE_KNOTS, FitError, IncrementalSplineFitRunner, IncrementalSplineFitStep,
        SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation, SplineFamilyKind,
        SplineKnotStrategy, approximate_spline_knots, calculate_metrics, evaluate_linear_spline,
        fit_akima_spline, fit_akima_spline_with_config, fit_curve, fit_curve_with_progress,
        fit_linear_spline, fit_monotone_cubic_spline, fit_natural_cubic_spline,
        sorted_points_with_duplicate_policy,
    };
    use crate::domain::{CurveFamily, CurveParams, InputError, LbfgsConfig, Point, Points};

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

    #[test]
    fn metrics_are_computed_correctly() {
        let points = build_points(&[0.0, 1.0, 2.0], |x| x + 1.0);
        let params = CurveParams::Linear { a: 1.0, b: 0.0 };
        let (mse, rmse) = calculate_metrics(&points, &params);

        assert!((mse - 1.0).abs() < 1e-12);
        assert!((rmse - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lbfgs_fits_linear_data() {
        let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Linear,
            CurveParams::Linear { a: 0.2, b: 0.1 },
            &config,
        )
        .expect("linear fit must succeed");

        assert!(result.mse < 1e-10);
    }

    #[test]
    fn lbfgs_fits_cubic_data() {
        let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| {
            0.4 * x * x * x - 0.8 * x * x + 1.2 * x + 0.5
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Cubic,
            CurveParams::Cubic {
                a: 0.1,
                b: 0.1,
                c: 0.1,
                d: 0.1,
            },
            &config,
        )
        .expect("cubic fit must succeed");

        assert!(result.mse < 1e-10);
    }

    #[test]
    fn lbfgs_fits_nonic_data() {
        let points = build_points(
            &[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            |x| {
                0.15 * x.powi(9) - 0.05 * x.powi(8) + 0.12 * x.powi(7) - 0.2 * x.powi(6)
                    + 0.08 * x.powi(5)
                    + 0.1 * x.powi(4)
                    - 0.05 * x.powi(3)
                    + 0.07 * x.powi(2)
                    - 0.03 * x
                    + 0.9
            },
        );
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Nonic,
            CurveParams::Nonic {
                a: 0.1,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
                h: 0.0,
                i: 0.0,
                j: 0.0,
            },
            &config,
        )
        .expect("nonic fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_exponential_basic_data() {
        let points = build_points(&[0.0, 0.5, 1.0, 1.5, 2.0], |x| 0.7 + 2.4 * (-0.9 * x).exp());
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::ExponentialBasic,
            CurveParams::ExponentialBasic {
                a: 0.1,
                b: 1.0,
                c: 0.3,
            },
            &config,
        )
        .expect("exponential basic fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_exponential_linear_data() {
        let points = build_points(&[-1.0, -0.5, 0.0, 0.7, 1.4, 2.0], |x| {
            1.6 * (0.45 * x).exp() - 0.8 * x + 0.3
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::ExponentialLinear,
            CurveParams::ExponentialLinear {
                a: 1.0,
                b: 0.2,
                c: 0.0,
                d: 0.0,
            },
            &config,
        )
        .expect("exponential + linear fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_arrhenius_data() {
        let points = build_points(&[0.5, 0.8, 1.0, 1.4, 2.0, 3.0], |x| 1.8 * (0.9 / x).exp());
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Arrhenius,
            CurveParams::Arrhenius { a: 1.0, b: 0.2 },
            &config,
        )
        .expect("arrhenius fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_inverse_data() {
        let points = build_points(&[0.5, 0.75, 1.0, 1.5, 2.0, 3.0], |x| 1.2 + 2.7 / x);
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Inverse,
            CurveParams::Inverse { a: 0.0, b: 1.0 },
            &config,
        )
        .expect("inverse fit must succeed");

        assert!(result.mse < 1e-10);
    }

    #[test]
    fn lbfgs_fits_logistic_data() {
        let points = build_points(&[-2.0, -1.5, -1.0, -0.2, 0.4, 0.8, 1.2, 1.8], |x| {
            4.0 / (1.0 + (-2.2 * (x - 0.7)).exp())
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Logistic,
            CurveParams::Logistic {
                a: 3.0,
                b: 1.0,
                c: 0.0,
            },
            &config,
        )
        .expect("logistic fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_lorentzian_data() {
        let points = build_points(&[-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0], |x| {
            0.4 + 2.5 / (1.0 + ((x - 0.3) / 0.8).powi(2))
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Lorentzian,
            CurveParams::Lorentzian {
                a: 2.0,
                x0: 0.0,
                gamma: 1.0,
                c: 0.0,
            },
            &config,
        )
        .expect("lorentzian fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_natural_log_data() {
        let points = build_points(&[0.5, 0.8, 1.2, 1.8, 2.5, 3.2], |x| 1.5 * (x / 0.7).ln());
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::NaturalLog,
            CurveParams::NaturalLog { a: 1.0, b: 1.0 },
            &config,
        )
        .expect("natural log fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_michaelis_menten_data() {
        let points = build_points(&[0.5, 1.0, 2.0, 4.0, 8.0], |x| (3.5 * x) / (1.8 + x));
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::MichaelisMenten,
            CurveParams::MichaelisMenten { vmax: 2.0, km: 1.0 },
            &config,
        )
        .expect("michaelis-menten fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_hyperbolic_tangent_data() {
        let points = build_points(&[-2.0, -1.0, -0.4, 0.0, 0.6, 1.1, 1.8], |x| {
            2.2 * (1.3 * (x - 0.35)).tanh() - 0.4
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::HyperbolicTangent,
            CurveParams::HyperbolicTangent {
                a: 1.5,
                b: 0.8,
                c: 0.0,
                d: 0.0,
            },
            &config,
        )
        .expect("hyperbolic tangent fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_arctangent_step_data() {
        let points = build_points(&[-2.0, -1.2, -0.6, 0.0, 0.5, 1.0, 1.8], |x| {
            2.0 * (1.5 * (x - 0.2)).atan() + 0.1
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::ArctangentStep,
            CurveParams::ArctangentStep {
                a: 1.0,
                b: 1.0,
                c: 0.0,
                d: 0.0,
            },
            &config,
        )
        .expect("arctangent step fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_softplus_data() {
        let points = build_points(&[-2.0, -1.0, -0.2, 0.3, 0.8, 1.4, 2.0], |x| {
            1.8 * super::softplus(2.0 * (x - 0.4)) - 0.35
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Softplus,
            CurveParams::Softplus {
                a: 1.0,
                b: 1.0,
                c: 0.0,
                d: 0.0,
            },
            &config,
        )
        .expect("softplus fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_power_data() {
        let points = build_points(&[0.5, 1.0, 1.5, 2.0, 3.0], |x| 1.7 * x.powf(1.35));
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Power,
            CurveParams::Power { a: 1.0, b: 1.0 },
            &config,
        )
        .expect("power fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn lbfgs_fits_gaussian_data() {
        let points = build_points(&[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], |x| {
            2.1 * (-(x - 0.4).powi(2) / (2.0 * 0.7 * 0.7)).exp()
        });
        let config = LbfgsConfig::default();
        let result = fit_curve(
            &points,
            CurveFamily::Gaussian,
            CurveParams::Gaussian {
                a: 1.0,
                b: 0.0,
                c: 1.0,
            },
            &config,
        )
        .expect("gaussian fit must succeed");

        assert!(result.mse < 1e-8);
    }

    #[test]
    fn fit_curve_validates_positive_x_domain() {
        let points = build_points(&[-1.0, 1.0, 2.0], |x| x);
        let config = LbfgsConfig::default();
        let error = fit_curve(
            &points,
            CurveFamily::Power,
            CurveParams::Power { a: 1.0, b: 1.0 },
            &config,
        )
        .expect_err("power family must reject x <= 0");

        assert!(matches!(
            error,
            super::FitError::InvalidInput(InputError::NonPositiveXForFamily {
                family: CurveFamily::Power,
                ..
            })
        ));
    }

    #[test]
    fn linear_spline_builds_curve() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
        let result = fit_linear_spline(&points, 50, DEFAULT_SPLINE_KNOTS)
            .expect("linear spline must succeed");

        assert!(!result.knots.is_empty());
        assert_eq!(result.curve.len(), 50);
        assert!(result.mse < 1e-12);
        assert!(result.iterations > 0);
    }

    #[test]
    fn monotone_cubic_spline_preserves_monotone_curve() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0, 4.0], |x| x * x + 0.5 * x);
        let result = fit_monotone_cubic_spline(&points, 80, DEFAULT_SPLINE_KNOTS)
            .expect("monotone cubic spline must succeed");

        assert_eq!(result.curve.len(), 80);
        assert!(result.mse < 1e-10);
        assert!(result.iterations > 0);
        for window in result.curve.windows(2) {
            assert!(window[1][1] >= window[0][1] - 1e-10);
        }
    }

    #[test]
    fn natural_cubic_spline_builds_curve() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| x * x * x - x + 1.0);
        let result = fit_natural_cubic_spline(&points, 60, DEFAULT_SPLINE_KNOTS)
            .expect("natural cubic spline must succeed");

        assert_eq!(result.curve.len(), 60);
        assert!(result.mse < 1e-8);
        assert!(result.iterations > 0);
    }

    #[test]
    fn akima_spline_builds_curve() {
        let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], |x| {
            x * x * x - 0.5 * x + 1.0
        });
        let result =
            fit_akima_spline(&points, 70, DEFAULT_SPLINE_KNOTS).expect("akima spline must succeed");

        assert_eq!(result.curve.len(), 70);
        assert!(result.mse < 1e-10);
        assert!(result.iterations > 0);
    }

    #[test]
    fn akima_spline_requires_at_least_five_knots() {
        let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], |x| {
            x * x * x - 0.5 * x + 1.0
        });
        let error = fit_akima_spline_with_config(
            &points,
            SplineConfig {
                knots: 4,
                samples: 64,
                knot_strategy: SplineKnotStrategy::BinMean,
                extrapolation: SplineExtrapolation::Clamp,
                duplicate_x_policy: SplineDuplicateXPolicy::Error,
            },
        )
        .expect_err("akima should reject knot count below 5");

        assert!(matches!(
            error,
            FitError::InvalidSplineInput(message) if message.contains("at least 5 knots")
        ));
    }

    #[test]
    fn median_knot_strategy_is_robust_to_single_outlier() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], |x| {
            if (x - 1.0).abs() < 1e-12 { 100.0 } else { 0.0 }
        });
        let sorted = sorted_points_with_duplicate_policy(&points, SplineDuplicateXPolicy::Error)
            .expect("x values are unique");
        let mean_knots = approximate_spline_knots(&sorted, 3, SplineKnotStrategy::BinMean);
        let median_knots = approximate_spline_knots(&sorted, 3, SplineKnotStrategy::BinMedian);

        assert!(mean_knots[0][1] > 30.0);
        assert!(median_knots[0][1].abs() < 1e-12);
    }

    #[test]
    fn duplicate_x_policy_mean_y_merges_points() {
        let points = Points::try_from(vec![
            Point::try_new(1.0, 2.0).unwrap(),
            Point::try_new(1.0, 6.0).unwrap(),
            Point::try_new(2.0, 4.0).unwrap(),
        ])
        .unwrap();

        let sorted = sorted_points_with_duplicate_policy(&points, SplineDuplicateXPolicy::MeanY)
            .expect("duplicate x should be merged with mean");

        assert_eq!(sorted.len(), 2);
        assert!((sorted[0][0] - 1.0).abs() < 1e-12);
        assert!((sorted[0][1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn linear_extrapolation_uses_edge_slope() {
        let knots = [[0.0, 1.0], [2.0, 5.0]];

        let clamped = evaluate_linear_spline(&knots, -1.0, SplineExtrapolation::Clamp);
        let linear = evaluate_linear_spline(&knots, -1.0, SplineExtrapolation::Linear);

        assert!((clamped - 1.0).abs() < 1e-12);
        assert!((linear + 1.0).abs() < 1e-12);
    }

    #[test]
    fn splines_are_approximation_not_exact_interpolation() {
        let points = build_points(&(-20..=20).map(|x| x as f64).collect::<Vec<_>>(), |x| {
            (x * 0.3).sin() + 0.1 * x
        });

        let result = fit_natural_cubic_spline(&points, 60, DEFAULT_SPLINE_KNOTS)
            .expect("natural cubic spline");

        assert_eq!(result.curve.len(), 60);
        assert!(
            result.mse > 1e-6,
            "Smoothing should produce non-zero error on dense input"
        );
        assert!(result.iterations > 0);
    }

    #[test]
    fn incremental_spline_runner_reports_iteration_steps() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
        let mut runner = IncrementalSplineFitRunner::new(
            &points,
            SplineFamilyKind::Linear,
            SplineConfig {
                knots: DEFAULT_SPLINE_KNOTS,
                samples: 48,
                knot_strategy: SplineKnotStrategy::BinMean,
                extrapolation: SplineExtrapolation::Clamp,
                duplicate_x_policy: SplineDuplicateXPolicy::Error,
            },
        )
        .expect("incremental linear spline runner must be created");

        let mut saw_iteration = false;
        loop {
            match runner.step().expect("runner step must succeed") {
                IncrementalSplineFitStep::Iteration { .. } => saw_iteration = true,
                IncrementalSplineFitStep::Finished(result) => {
                    assert!(saw_iteration);
                    assert!(result.iterations > 0);
                    break;
                }
                IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
            }
        }
    }

    #[test]
    fn incremental_spline_runner_rejects_wrong_custom_init_length() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
        let error = IncrementalSplineFitRunner::new_with_initial_knot_y(
            &points,
            SplineFamilyKind::Linear,
            SplineConfig {
                knots: DEFAULT_SPLINE_KNOTS,
                samples: 48,
                knot_strategy: SplineKnotStrategy::BinMean,
                extrapolation: SplineExtrapolation::Clamp,
                duplicate_x_policy: SplineDuplicateXPolicy::Error,
            },
            Some(&[1.0, 2.0, 3.0]),
        );
        let error = match error {
            Ok(_) => panic!("runner must reject mismatched custom initialization length"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            FitError::InvalidSplineInput(message) if message.contains("expects")
        ));
    }

    #[test]
    fn incremental_spline_runner_can_be_cancelled() {
        let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| x * x);
        let mut runner = IncrementalSplineFitRunner::new(
            &points,
            SplineFamilyKind::NaturalCubic,
            SplineConfig::default(),
        )
        .expect("incremental spline runner must be created");

        runner.cancel();
        let step = runner.step().expect("cancelled runner step must succeed");
        assert!(matches!(step, IncrementalSplineFitStep::Cancelled));
    }

    #[test]
    fn fit_curve_can_be_cancelled_via_progress_callback() {
        let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
        let config = LbfgsConfig::default();
        let result = fit_curve_with_progress(
            &points,
            CurveFamily::Linear,
            CurveParams::Linear { a: 0.2, b: 0.1 },
            &config,
            |_iteration, _params| false,
        );

        assert!(matches!(result, Err(FitError::Cancelled)));
    }
}
