//! Инкрементальные раннеры оптимизации поверх `argmin` и стохастических solver-ов.

use super::*;

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

/// Solver и его состояние хранятся вместе, поэтому несовместимую пару нельзя создать.
enum Optimizer {
    Lbfgs {
        solver: LbfgsSolver,
        state: GradientState,
    },
    NelderMead {
        solver: NelderMeadSolver,
        state: NelderMeadState,
    },
    SteepestDescent {
        solver: SteepestDescentSolver,
        state: GradientState,
    },
    NewtonCg {
        solver: NewtonCgSolver,
        state: Box<NewtonCgState>,
    },
    Sgd {
        solver: SgdSolver,
        state: StochasticState,
    },
    Adam {
        solver: AdamSolver,
        state: StochasticState,
    },
}

#[derive(Debug, Clone, PartialEq)]
/// Шаг инкрементальной подгонки параметрической модели.
pub enum IncrementalFitStep {
    Iteration {
        iteration: u64,
        mse: f64,
        metrics: IterationMetricSnapshot,
        gradient_diagnostics: Option<GradientIterationDiagnostics>,
        params: CurveParams,
    },
    Finished(FitResult),
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IncrementalSplineFitStep {
    Iteration {
        iteration: u64,
        mse: f64,
        metrics: IterationMetricSnapshot,
        gradient_diagnostics: Option<GradientIterationDiagnostics>,
        knot_y: Vec<f64>,
        curve: Vec<[f64; 2]>,
    },
    Finished {
        result: SplineResult,
        metrics: IterationMetricSnapshot,
    },
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Диагностика градиента в точке текущей итерации.
pub struct GradientIterationDiagnostics {
    pub gradient_l2_norm: f64,
    pub gradient_cosine: Option<f64>,
}

struct GradientDiagnosticsState {
    parameter_buffer: Array1<f64>,
    previous: Option<Array1<f64>>,
}

impl GradientDiagnosticsState {
    fn new(parameter_count: usize) -> Self {
        Self {
            parameter_buffer: Array1::zeros(parameter_count),
            previous: None,
        }
    }

    fn collect<P>(
        &mut self,
        problem: &mut Problem<P>,
        values: &[f64],
    ) -> Result<GradientIterationDiagnostics, FitError>
    where
        P: Gradient<Param = Array1<f64>, Gradient = Array1<f64>>,
    {
        array1_as_mut_slice(&mut self.parameter_buffer).copy_from_slice(values);
        let gradient = problem
            .gradient(&self.parameter_buffer)
            .map_err(optimizer_error)?;
        let gradient_values = array1_as_slice(&gradient);
        let current_l2_norm = gradient_l2_norm(gradient_values);
        let gradient_cosine = self.previous.as_ref().and_then(|previous| {
            let previous_values = array1_as_slice(previous);
            let previous_l2_norm = gradient_l2_norm(previous_values);
            let dot_product = previous_values
                .iter()
                .zip(gradient_values)
                .map(|(previous, current)| previous * current)
                .sum::<f64>();
            let denominator = previous_l2_norm * current_l2_norm;
            (denominator.is_finite() && denominator > 0.0 && dot_product.is_finite())
                .then(|| (dot_product / denominator).clamp(-1.0, 1.0))
        });
        self.previous = Some(gradient);
        Ok(GradientIterationDiagnostics {
            gradient_l2_norm: current_l2_norm,
            gradient_cosine,
        })
    }
}

enum OptimizerStepOutcome {
    Iterated(Optimizer),
    Terminated(Optimizer),
}

#[derive(Debug, Clone)]
struct StochasticState {
    current_param: Vec<f64>,
    best_param: Vec<f64>,
    gradient_buffer: Vec<f64>,
    param_buffer: Array1<f64>,
    best_cost: f64,
    iter: u64,
    max_iters: u64,
}

/// Пошаговый раннер оптимизации параметрических семейств.
pub struct IncrementalFitRunner {
    family: CurveFamily,
    params_template: CurveParams,
    points: Points,
    loss_metric: OptimizationLossMetric,
    metric_baseline: MetricBaseline,
    problem: Problem<CurveProblem>,
    optimizer: Option<Optimizer>,
    gradient_diagnostics: GradientDiagnosticsState,
    cancelled: bool,
}

impl IncrementalFitRunner {
    /// Создает раннер и инициализирует внутреннее состояние оптимизатора.
    pub fn new(
        points: &Points,
        family: CurveFamily,
        initial_params: CurveParams,
        config: &LbfgsConfig,
    ) -> Result<Self, FitError> {
        let optimizer_config = OptimizerConfig::from(config);
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
        let parameter_count = initial_values.len();
        let metric_baseline = MetricBaseline::new(points, metric_quantization);
        let problem = CurveProblem::new_with_metric_quantization(
            family,
            points,
            initial_params.saturating_trend_tau_grid(),
            loss_metric,
            metric_quantization,
        );
        let mut problem = Problem::new(problem);
        let initial_array = Array1::from_vec(initial_values);
        let optimizer = build_optimizer(&mut problem, &initial_array, optimizer_config)?;

        Ok(Self {
            family,
            params_template: initial_params,
            points: points.clone(),
            loss_metric,
            metric_baseline,
            problem,
            optimizer: Some(optimizer),
            gradient_diagnostics: GradientDiagnosticsState::new(parameter_count),
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
            let Some(optimizer) = self.optimizer.take() else {
                return Err(optimizer_error(
                    "Incremental fit runner state is not initialized",
                ));
            };

            let mut optimizer = match optimizer_step_once(&mut self.problem, optimizer)? {
                OptimizerStepOutcome::Iterated(optimizer) => optimizer,
                OptimizerStepOutcome::Terminated(optimizer) => {
                    let final_step = self.finalize(optimizer)?;
                    return Ok(final_step);
                }
            };

            let iteration = optimizer_iter(&optimizer);
            if let Some(values) = optimizer_current_param(&optimizer)
                && let Ok(params) = CurveParams::try_from_slice_like(&self.params_template, values)
            {
                let metrics = calculate_iteration_metrics_from_evaluator_with_baseline(
                    &self.points,
                    self.loss_metric,
                    self.metric_baseline,
                    |x| params.evaluate(x),
                );
                let gradient_diagnostics = if optimizer_uses_gradient(&optimizer) {
                    Some(
                        self.gradient_diagnostics
                            .collect(&mut self.problem, values)?,
                    )
                } else {
                    None
                };
                optimizer_increment_iter(&mut optimizer);
                self.optimizer = Some(optimizer);
                return Ok(IncrementalFitStep::Iteration {
                    iteration,
                    mse: metrics.mse,
                    metrics,
                    gradient_diagnostics,
                    params,
                });
            }

            // Если параметры недоступны на текущем шаге, продолжаем итерации без рекурсии.
            optimizer_increment_iter(&mut optimizer);
            self.optimizer = Some(optimizer);
        }
    }

    fn finalize(&mut self, optimizer: Optimizer) -> Result<IncrementalFitStep, FitError> {
        let best_param_values =
            optimizer_best_param(&optimizer).ok_or(FitError::MissingBestParameters)?;
        let best_params =
            CurveParams::try_from_slice_like(&self.params_template, best_param_values)?;
        let (mse, rmse) =
            calculate_metrics_with_baseline(&self.points, &best_params, self.metric_baseline);
        let iterations = optimizer_iter(&optimizer);
        self.optimizer = Some(optimizer);

        Ok(IncrementalFitStep::Finished(FitResult {
            family: self.family,
            params: best_params,
            mse,
            rmse,
            iterations,
        }))
    }
}

pub(crate) struct IncrementalSplineFitRunner {
    family: SplineFamilyKind,
    points: Points,
    config: SplineConfig,
    knot_x: Box<[f64]>,
    curve_x_bounds: [f64; 2],
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    metric_baseline: MetricBaseline,
    problem: Problem<SplineProblem>,
    optimizer: Option<Optimizer>,
    gradient_diagnostics: GradientDiagnosticsState,
    cancelled: bool,
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
        let metric_baseline = MetricBaseline::new(points, metric_quantization);
        let PreparedSplineInputs {
            config,
            knot_x,
            initial_y,
            curve_x_bounds,
        } = prepared;
        let initial_knots = materialize_spline_knots(knot_x.as_ref(), &initial_y);
        let problem = SplineProblem::new(
            family,
            &initial_knots,
            points,
            config.extrapolation(),
            loss_metric,
            metric_quantization,
        );
        let mut problem = Problem::new(problem);
        let initial_knot_y_array = Array1::from_vec(initial_y);
        let parameter_count = initial_knot_y_array.len();
        let optimizer = build_optimizer(&mut problem, &initial_knot_y_array, optimizer_config)?;

        Ok(Self {
            family,
            points: points.clone(),
            config,
            knot_x,
            curve_x_bounds,
            loss_metric,
            metric_quantization,
            metric_baseline,
            problem,
            optimizer: Some(optimizer),
            gradient_diagnostics: GradientDiagnosticsState::new(parameter_count),
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
            let Some(optimizer) = self.optimizer.take() else {
                return Err(optimizer_error(
                    "Incremental spline fit runner state is not initialized",
                ));
            };

            let mut optimizer = match optimizer_step_once(&mut self.problem, optimizer)? {
                OptimizerStepOutcome::Iterated(optimizer) => optimizer,
                OptimizerStepOutcome::Terminated(optimizer) => {
                    let final_step = self.finalize(optimizer)?;
                    return Ok(final_step);
                }
            };

            let iteration = optimizer_iter(&optimizer);
            if let Some(knot_y) = optimizer_current_param(&optimizer).map(|knot_y| knot_y.to_vec())
            {
                let built = build_spline_curve_from_knot_y(
                    self.family,
                    self.config.extrapolation(),
                    self.config.samples(),
                    self.knot_x.as_ref(),
                    &knot_y,
                    self.curve_x_bounds,
                )?;
                let metrics = calculate_iteration_metrics_from_evaluator_with_baseline(
                    &self.points,
                    self.loss_metric,
                    self.metric_baseline,
                    |x| built.evaluator.evaluate(&built.knots, x),
                );
                let curve = built.curve;
                let gradient_diagnostics = if optimizer_uses_gradient(&optimizer) {
                    Some(
                        self.gradient_diagnostics
                            .collect(&mut self.problem, &knot_y)?,
                    )
                } else {
                    None
                };

                optimizer_increment_iter(&mut optimizer);
                self.optimizer = Some(optimizer);
                return Ok(IncrementalSplineFitStep::Iteration {
                    iteration,
                    mse: metrics.mse,
                    metrics,
                    gradient_diagnostics,
                    knot_y,
                    curve,
                });
            }

            // Если параметры недоступны на текущем шаге, продолжаем итерации без рекурсии.
            optimizer_increment_iter(&mut optimizer);
            self.optimizer = Some(optimizer);
        }
    }

    fn finalize(&mut self, optimizer: Optimizer) -> Result<IncrementalSplineFitStep, FitError> {
        let best_knot_y = optimizer_best_param(&optimizer)
            .ok_or(FitError::MissingBestParameters)?
            .to_vec();
        let iterations = optimizer_iter(&optimizer);
        self.optimizer = Some(optimizer);

        let finalize_context = SplineFinalizeContext {
            points: &self.points,
            family: self.family,
            config: self.config,
            knot_x: self.knot_x.as_ref(),
            curve_x_bounds: self.curve_x_bounds,
            loss_metric: self.loss_metric,
            metric_quantization: self.metric_quantization,
        };
        let (result, metrics) =
            build_spline_result_from_knot_y(&finalize_context, &best_knot_y, iterations)?;

        Ok(IncrementalSplineFitStep::Finished { result, metrics })
    }
}
fn build_line_search(
    c1: f64,
    c2: f64,
    step_min: f64,
    step_max: f64,
    width_tolerance: f64,
) -> Result<MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>, FitError> {
    // На границе fit-модуля приводим ошибки `argmin` к единому типу `FitError`.
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
        config.c1(),
        config.c2(),
        config.step_min(),
        config.step_max(),
        config.width_tolerance(),
    )?;
    LBFGS::new(line_search, config.history_size())
        .with_tolerance_grad(config.tol_grad())
        .map_err(optimizer_error)?
        .with_tolerance_cost(config.tol_cost())
        .map_err(optimizer_error)
}

fn build_steepest_descent_solver(
    config: &SteepestDescentConfig,
) -> Result<SteepestDescentSolver, FitError> {
    let line_search = build_line_search(
        config.c1(),
        config.c2(),
        config.step_min(),
        config.step_max(),
        config.width_tolerance(),
    )?;
    Ok(SteepestDescent::new(line_search))
}

fn build_newton_cg_solver(config: &NewtonCgConfig) -> Result<NewtonCgSolver, FitError> {
    let line_search = build_line_search(
        config.c1(),
        config.c2(),
        config.step_min(),
        config.step_max(),
        config.width_tolerance(),
    )?;
    NewtonCG::new(line_search)
        .with_curvature_threshold(config.curvature_threshold())
        .with_tolerance(config.tol())
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
        // Масштабируем сдвиг от текущего значения, чтобы симплекс не вырождался возле нуля.
        vertex[index] += simplex_scale * (value.abs() + 1.0);
        simplex.push(Array1::from_vec(vertex));
    }

    Ok(simplex)
}

fn build_nelder_mead_solver(
    initial_param: &[f64],
    config: &NelderMeadConfig,
) -> Result<NelderMeadSolver, FitError> {
    let simplex = nelder_mead_simplex(initial_param, config.simplex_scale())?;
    NelderMead::new(simplex)
        .with_sd_tolerance(config.sd_tolerance())
        .map_err(optimizer_error)?
        .with_alpha(config.alpha())
        .map_err(optimizer_error)?
        .with_gamma(config.gamma())
        .map_err(optimizer_error)?
        .with_rho(config.rho())
        .map_err(optimizer_error)?
        .with_sigma(config.sigma())
        .map_err(optimizer_error)
}

fn build_sgd_solver(initial_param: &[f64], config: &SgdConfig) -> SgdSolver {
    SGD::new(initial_param.to_vec(), config.learning_rate())
}

fn build_adam_solver(initial_param: &[f64], config: &AdamConfig) -> AdamSolver {
    Adam::new(initial_param.to_vec(), config.learning_rate())
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
    let parameter_count = initial_param.len();
    let mut param_buffer = Array1::zeros(parameter_count);
    array1_as_mut_slice(&mut param_buffer).copy_from_slice(&initial_param);
    let cost = problem.cost(&param_buffer).map_err(optimizer_error)?;
    Ok(StochasticState {
        current_param: initial_param.clone(),
        best_param: initial_param,
        gradient_buffer: vec![0.0; parameter_count],
        param_buffer,
        best_cost: finite_cost_or_large(cost),
        iter: 0,
        max_iters,
    })
}

fn stochastic_state_is_terminated(state: &StochasticState) -> bool {
    state.iter >= state.max_iters
}

fn overwrite_fixed_len_vec(target: &mut [f64], source: &[f64]) {
    debug_assert_eq!(target.len(), source.len());
    target.copy_from_slice(source);
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
    array1_as_mut_slice(&mut state.param_buffer).copy_from_slice(&state.current_param);
    let gradient = problem
        .gradient(&state.param_buffer)
        .map_err(optimizer_error)?;
    state
        .gradient_buffer
        .copy_from_slice(array1_as_slice(&gradient));
    solver.step(&state.gradient_buffer);

    let current_param = solver.parameters();
    array1_as_mut_slice(&mut state.param_buffer).copy_from_slice(current_param.as_slice());
    let current_cost =
        finite_cost_or_large(problem.cost(&state.param_buffer).map_err(optimizer_error)?);

    if current_cost < state.best_cost {
        state.best_cost = current_cost;
        overwrite_fixed_len_vec(state.best_param.as_mut_slice(), current_param.as_slice());
    }
    overwrite_fixed_len_vec(state.current_param.as_mut_slice(), current_param.as_slice());

    Ok(())
}

fn optimizer_best_param(optimizer: &Optimizer) -> Option<&[f64]> {
    match optimizer {
        Optimizer::Lbfgs { state, .. } => state
            .get_best_param()
            .or_else(|| state.get_param())
            .map(array1_as_slice),
        Optimizer::NelderMead { state, .. } => state
            .get_best_param()
            .or_else(|| state.get_param())
            .map(array1_as_slice),
        Optimizer::SteepestDescent { state, .. } => state
            .get_best_param()
            .or_else(|| state.get_param())
            .map(array1_as_slice),
        Optimizer::NewtonCg { state, .. } => state
            .get_best_param()
            .or_else(|| state.get_param())
            .map(array1_as_slice),
        Optimizer::Sgd { state, .. } | Optimizer::Adam { state, .. } => {
            Some(state.best_param.as_slice())
        }
    }
}

fn optimizer_current_param(optimizer: &Optimizer) -> Option<&[f64]> {
    match optimizer {
        Optimizer::Lbfgs { state, .. } | Optimizer::SteepestDescent { state, .. } => {
            state.get_param().map(array1_as_slice)
        }
        Optimizer::NelderMead { state, .. } => state.get_param().map(array1_as_slice),
        Optimizer::NewtonCg { state, .. } => state.get_param().map(array1_as_slice),
        Optimizer::Sgd { state, .. } | Optimizer::Adam { state, .. } => {
            Some(state.current_param.as_slice())
        }
    }
}

fn optimizer_uses_gradient(optimizer: &Optimizer) -> bool {
    !matches!(optimizer, Optimizer::NelderMead { .. })
}

fn optimizer_iter(optimizer: &Optimizer) -> u64 {
    match optimizer {
        Optimizer::Lbfgs { state, .. } | Optimizer::SteepestDescent { state, .. } => {
            state.get_iter()
        }
        Optimizer::NelderMead { state, .. } => state.get_iter(),
        Optimizer::NewtonCg { state, .. } => state.get_iter(),
        Optimizer::Sgd { state, .. } | Optimizer::Adam { state, .. } => state.iter,
    }
}

fn optimizer_increment_iter(optimizer: &mut Optimizer) {
    match optimizer {
        Optimizer::Lbfgs { state, .. } | Optimizer::SteepestDescent { state, .. } => {
            state.increment_iter()
        }
        Optimizer::NelderMead { state, .. } => state.increment_iter(),
        Optimizer::NewtonCg { state, .. } => state.increment_iter(),
        Optimizer::Sgd { state, .. } | Optimizer::Adam { state, .. } => {
            state.iter = state.iter.saturating_add(1);
        }
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
    let gradient_norm = gradient_l2_norm(array1_as_slice(&gradient));
    state = state.gradient(gradient);
    if gradient_norm <= STEEPEST_DESCENT_GRAD_TOL {
        state = state.terminate_with(TerminationReason::SolverConverged);
    }
    Ok(state)
}

fn optimizer_step_once<P>(
    problem: &mut Problem<P>,
    optimizer: Optimizer,
) -> Result<OptimizerStepOutcome, FitError>
where
    P: CostFunction<Param = Array1<f64>, Output = f64>
        + Gradient<Param = Array1<f64>, Gradient = Array1<f64>>
        + Hessian<Param = Array1<f64>, Hessian = Array2<f64>>,
{
    let next_optimizer = match optimizer {
        Optimizer::Lbfgs {
            mut solver,
            mut state,
        } => {
            if !state.terminated() {
                let termination = <LbfgsSolver as Solver<P, GradientState>>::terminate_internal(
                    &mut solver,
                    &state,
                );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(Optimizer::Lbfgs {
                    solver,
                    state,
                }));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            Optimizer::Lbfgs { solver, state }
        }
        Optimizer::NelderMead {
            mut solver,
            mut state,
        } => {
            if !state.terminated() {
                let termination =
                    <NelderMeadSolver as Solver<P, NelderMeadState>>::terminate_internal(
                        &mut solver,
                        &state,
                    );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(Optimizer::NelderMead {
                    solver,
                    state,
                }));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            Optimizer::NelderMead { solver, state }
        }
        Optimizer::SteepestDescent {
            mut solver,
            mut state,
        } => {
            state = terminate_steepest_descent_on_small_gradient(problem, state)?;
            if !state.terminated() {
                let termination =
                    <SteepestDescentSolver as Solver<P, GradientState>>::terminate_internal(
                        &mut solver,
                        &state,
                    );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(
                    Optimizer::SteepestDescent { solver, state },
                ));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            Optimizer::SteepestDescent { solver, state }
        }
        Optimizer::NewtonCg { mut solver, state } => {
            let mut state = *state;
            if !state.terminated()
                && let Some(param) = state.get_param().cloned()
            {
                let gradient = problem.gradient(&param).map_err(optimizer_error)?;
                let gradient_norm = gradient_l2_norm(array1_as_slice(&gradient));
                state = state.gradient(gradient);
                if gradient_norm <= STEEPEST_DESCENT_GRAD_TOL {
                    state = state.terminate_with(TerminationReason::SolverConverged);
                }
            }
            if !state.terminated() {
                let termination = <NewtonCgSolver as Solver<P, NewtonCgState>>::terminate_internal(
                    &mut solver,
                    &state,
                );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(Optimizer::NewtonCg {
                    solver,
                    state: Box::new(state),
                }));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            Optimizer::NewtonCg {
                solver,
                state: Box::new(state),
            }
        }
        Optimizer::Sgd {
            mut solver,
            mut state,
        } => {
            if stochastic_state_is_terminated(&state) {
                return Ok(OptimizerStepOutcome::Terminated(Optimizer::Sgd {
                    solver,
                    state,
                }));
            }
            stochastic_step(problem, &mut solver, &mut state)?;
            Optimizer::Sgd { solver, state }
        }
        Optimizer::Adam {
            mut solver,
            mut state,
        } => {
            if stochastic_state_is_terminated(&state) {
                return Ok(OptimizerStepOutcome::Terminated(Optimizer::Adam {
                    solver,
                    state,
                }));
            }
            stochastic_step(problem, &mut solver, &mut state)?;
            Optimizer::Adam { solver, state }
        }
    };

    Ok(OptimizerStepOutcome::Iterated(next_optimizer))
}

fn build_optimizer<P>(
    problem: &mut Problem<P>,
    initial_param: &Array1<f64>,
    config: &OptimizerConfig,
) -> Result<Optimizer, FitError>
where
    P: CostFunction<Param = Array1<f64>, Output = f64>
        + Gradient<Param = Array1<f64>, Gradient = Array1<f64>>
        + Hessian<Param = Array1<f64>, Hessian = Array2<f64>>,
{
    let max_iters = config.max_iters();
    match config {
        OptimizerConfig::Lbfgs(config) => {
            let mut solver = build_lbfgs_solver(config)?;
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(Optimizer::Lbfgs { solver, state })
        }
        OptimizerConfig::NelderMead(config) => {
            let mut solver = build_nelder_mead_solver(array1_as_slice(initial_param), config)?;
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(Optimizer::NelderMead { solver, state })
        }
        OptimizerConfig::SteepestDescent(config) => {
            let mut solver = build_steepest_descent_solver(config)?;
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(Optimizer::SteepestDescent { solver, state })
        }
        OptimizerConfig::NewtonCg(config) => {
            let mut solver = build_newton_cg_solver(config)?;
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(Optimizer::NewtonCg {
                solver,
                state: Box::new(state),
            })
        }
        OptimizerConfig::Sgd(config) => {
            let solver = build_sgd_solver(array1_as_slice(initial_param), config);
            let state = build_stochastic_state(problem, solver.parameters().clone(), max_iters)?;
            Ok(Optimizer::Sgd { solver, state })
        }
        OptimizerConfig::Adam(config) => {
            let solver = build_adam_solver(array1_as_slice(initial_param), config);
            let state = build_stochastic_state(problem, solver.parameters().clone(), max_iters)?;
            Ok(Optimizer::Adam { solver, state })
        }
    }
}
