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

#[derive(Debug, Clone)]
struct StochasticState {
    current_param: Vec<f64>,
    best_param: Vec<f64>,
    gradient_buffer: Vec<f64>,
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
    let parameter_count = initial_param.len();
    let cost = problem
        .cost(&vec_to_array1(&initial_param))
        .map_err(optimizer_error)?;
    Ok(StochasticState {
        current_param: initial_param.clone(),
        best_param: initial_param,
        gradient_buffer: Vec::with_capacity(parameter_count),
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
    state.gradient_buffer.clear();
    state
        .gradient_buffer
        .extend_from_slice(array1_as_slice(&gradient));
    solver.step(&state.gradient_buffer);

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

enum OptimizerStepOutcome {
    Iterated(OptimizerState),
    Terminated(OptimizerState),
}

fn optimizer_step_once<P>(
    solver: &mut OptimizerSolver,
    problem: &mut Problem<P>,
    state: OptimizerState,
) -> Result<OptimizerStepOutcome, FitError>
where
    P: CostFunction<Param = Array1<f64>, Output = f64>
        + Gradient<Param = Array1<f64>, Gradient = Array1<f64>>
        + Hessian<Param = Array1<f64>, Hessian = Array2<f64>>,
{
    let next_state = match (solver, state) {
        (OptimizerSolver::Lbfgs(solver), OptimizerState::Lbfgs(mut state)) => {
            if !state.terminated() {
                let termination =
                    <LbfgsSolver as Solver<P, GradientState>>::terminate_internal(solver, &state);
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(OptimizerState::Lbfgs(
                    state,
                )));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            OptimizerState::Lbfgs(state)
        }
        (OptimizerSolver::NelderMead(solver), OptimizerState::NelderMead(mut state)) => {
            if !state.terminated() {
                let termination =
                    <NelderMeadSolver as Solver<P, NelderMeadState>>::terminate_internal(
                        solver, &state,
                    );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(
                    OptimizerState::NelderMead(state),
                ));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            OptimizerState::NelderMead(state)
        }
        (OptimizerSolver::SteepestDescent(solver), OptimizerState::SteepestDescent(mut state)) => {
            state = terminate_steepest_descent_on_small_gradient(problem, state)?;
            if !state.terminated() {
                let termination =
                    <SteepestDescentSolver as Solver<P, GradientState>>::terminate_internal(
                        solver, &state,
                    );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(
                    OptimizerState::SteepestDescent(state),
                ));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            OptimizerState::SteepestDescent(state)
        }
        (OptimizerSolver::NewtonCg(solver), OptimizerState::NewtonCg(state)) => {
            let mut state = *state;
            if !state.terminated() {
                let termination = <NewtonCgSolver as Solver<P, NewtonCgState>>::terminate_internal(
                    solver, &state,
                );
                if let TerminationStatus::Terminated(reason) = termination {
                    state = state.terminate_with(reason);
                }
            }
            if state.terminated() {
                return Ok(OptimizerStepOutcome::Terminated(OptimizerState::NewtonCg(
                    Box::new(state),
                )));
            }
            let (mut state, _) = solver.next_iter(problem, state).map_err(optimizer_error)?;
            state.func_counts(problem);
            state.update();
            OptimizerState::NewtonCg(Box::new(state))
        }
        (OptimizerSolver::Sgd(solver), OptimizerState::Sgd(mut state)) => {
            if stochastic_state_is_terminated(&state) {
                return Ok(OptimizerStepOutcome::Terminated(OptimizerState::Sgd(state)));
            }
            stochastic_step(problem, solver, &mut state)?;
            OptimizerState::Sgd(state)
        }
        (OptimizerSolver::Adam(solver), OptimizerState::Adam(mut state)) => {
            if stochastic_state_is_terminated(&state) {
                return Ok(OptimizerStepOutcome::Terminated(OptimizerState::Adam(
                    state,
                )));
            }
            stochastic_step(problem, solver, &mut state)?;
            OptimizerState::Adam(state)
        }
        _ => {
            return Err(optimizer_error(
                "Optimizer solver/state mismatch in incremental runner",
            ));
        }
    };

    Ok(OptimizerStepOutcome::Iterated(next_state))
}

fn initialize_optimizer_state<P>(
    solver: &mut OptimizerSolver,
    problem: &mut Problem<P>,
    initial_param: &Array1<f64>,
    max_iters: u64,
) -> Result<OptimizerState, FitError>
where
    P: CostFunction<Param = Array1<f64>, Output = f64>
        + Gradient<Param = Array1<f64>, Gradient = Array1<f64>>
        + Hessian<Param = Array1<f64>, Hessian = Array2<f64>>,
{
    match solver {
        OptimizerSolver::Lbfgs(solver) => {
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(OptimizerState::Lbfgs(state))
        }
        OptimizerSolver::NelderMead(solver) => {
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(OptimizerState::NelderMead(state))
        }
        OptimizerSolver::SteepestDescent(solver) => {
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(OptimizerState::SteepestDescent(state))
        }
        OptimizerSolver::NewtonCg(solver) => {
            let state = IterState::new()
                .param(initial_param.clone())
                .max_iters(max_iters);
            let (mut state, _) = solver.init(problem, state).map_err(optimizer_error)?;
            state.update();
            state.func_counts(problem);
            Ok(OptimizerState::NewtonCg(Box::new(state)))
        }
        OptimizerSolver::Sgd(solver) => {
            let state = build_stochastic_state(problem, solver.parameters().clone(), max_iters)?;
            Ok(OptimizerState::Sgd(state))
        }
        OptimizerSolver::Adam(solver) => {
            let state = build_stochastic_state(problem, solver.parameters().clone(), max_iters)?;
            Ok(OptimizerState::Adam(state))
        }
    }
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
        let state =
            initialize_optimizer_state(&mut solver, &mut problem, &initial_array, max_iters)?;

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
            let Some(state) = self.state.take() else {
                return Err(optimizer_error(
                    "Incremental fit runner state is not initialized",
                ));
            };

            let mut state = match optimizer_step_once(&mut self.solver, &mut self.problem, state)? {
                OptimizerStepOutcome::Iterated(state) => state,
                OptimizerStepOutcome::Terminated(state) => {
                    let final_step = self.finalize(state)?;
                    return Ok(final_step);
                }
            };

            let iteration = optimizer_state_iter(&state);
            if let Some(params) = optimizer_state_current_param(&state).and_then(|values| {
                CurveParams::try_from_slice(self.family, array1_as_slice(&values)).ok()
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
            CurveParams::try_from_slice(self.family, array1_as_slice(&best_param_values))?;
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
        let initial_knot_y_array = vec_to_array1(&prepared.initial_y);
        let mut solver = build_optimizer_solver(&prepared.initial_y, optimizer_config)?;
        let state = initialize_optimizer_state(
            &mut solver,
            &mut problem,
            &initial_knot_y_array,
            max_iters,
        )?;

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
            let Some(state) = self.state.take() else {
                return Err(optimizer_error(
                    "Incremental spline fit runner state is not initialized",
                ));
            };

            let mut state = match optimizer_step_once(&mut self.solver, &mut self.problem, state)? {
                OptimizerStepOutcome::Iterated(state) => state,
                OptimizerStepOutcome::Terminated(state) => {
                    let final_step = self.finalize(state)?;
                    return Ok(final_step);
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
                    knot_y: array1_as_slice(&knot_y).to_vec(),
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
