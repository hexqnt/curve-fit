//! Координация фонового и wasm-раннера фитинга и перенос результатов обратно в UI.

use super::replay::{ReplayFrame, ReplayFramePayload, upsert_replay_frame_in};
use super::*;

#[cfg(not(target_arch = "wasm32"))]
/// Полный снимок входных данных, передаваемых в фоновый поток параметрического фитинга.
struct ParametricFitWorkerInput {
    family: CurveFamily,
    optimization_points: Points,
    display_points: Points,
    optimization_initial_params: CurveParams,
    normalization: Option<ParametricNormalization>,
    optimizer_config: OptimizerConfig,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    cancel_flag: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
const FIT_WORKER_POLL_REPAINT_INTERVAL: Duration = Duration::from_millis(33);

impl CurveFitApp {
    fn plot_points_from_pairs<I>(pairs: I) -> Vec<PlotPoint>
    where
        I: IntoIterator<Item = [f64; 2]>,
    {
        pairs
            .into_iter()
            .map(|point| PlotPoint::new(point[0], point[1]))
            .collect()
    }

    fn reset_fit_runtime_for_new_run(&mut self) {
        self.reset_fit_timer();
        self.clear_fit_export_state();
        self.active_fit_points = None;
        self.fit_run_ui_seed = None;
    }

    fn upsert_buffered_parametric_replay_frame(
        frames: &mut Vec<ReplayFrame>,
        iteration: u64,
        params: CurveParams,
    ) {
        upsert_replay_frame_in(
            frames,
            ReplayFrame {
                iteration,
                payload: ReplayFramePayload::Parametric { params },
            },
        );
    }

    fn upsert_buffered_spline_replay_frame(
        frames: &mut Vec<ReplayFrame>,
        iteration: u64,
        curve: Vec<PlotPoint>,
    ) {
        upsert_replay_frame_in(
            frames,
            ReplayFrame {
                iteration,
                payload: ReplayFramePayload::Spline {
                    curve: curve.into(),
                },
            },
        );
    }

    fn apply_parametric_trace_to_buffers(
        trace: Vec<ParametricIterationTraceEntry>,
        diagnostics: &mut IterationDiagnostics,
        replay_frames: &mut Vec<ReplayFrame>,
    ) {
        for entry in trace {
            diagnostics.append(entry.iteration, entry.metrics, &entry.params);
            Self::upsert_buffered_parametric_replay_frame(
                replay_frames,
                entry.iteration,
                entry.params,
            );
        }
    }

    fn apply_spline_trace_to_buffers(
        trace: Vec<SplineIterationTraceEntry>,
        diagnostics: &mut IterationDiagnostics,
        replay_frames: &mut Vec<ReplayFrame>,
    ) {
        for entry in trace {
            diagnostics.append_spline(entry.iteration, entry.metrics, &entry.knot_y);
            Self::upsert_buffered_spline_replay_frame(
                replay_frames,
                entry.iteration,
                Self::plot_points_from_pairs(entry.curve),
            );
        }
    }

    fn take_parametric_fit_seed(&mut self) -> Option<CurveParams> {
        match self.fit_run_ui_seed.take() {
            Some(FitRunUiSeed::Parametric { initial_params }) => Some(initial_params),
            Some(FitRunUiSeed::Spline { .. }) | None => None,
        }
    }

    fn take_spline_fit_seed(&mut self) -> Option<Vec<PlotPoint>> {
        match self.fit_run_ui_seed.take() {
            Some(FitRunUiSeed::Spline { initial_curve }) => Some(initial_curve),
            Some(FitRunUiSeed::Parametric { .. }) | None => None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn status_is_fitting(&self) -> bool {
        matches!(
            self.status.as_ref(),
            Some(StatusMessage::FittingInProgress | StatusMessage::FittingStopping)
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn set_fit_stopped_status_if_fitting(&mut self) {
        if self.status_is_fitting() {
            self.status = Some(StatusMessage::FitStopped);
        }
    }

    pub(super) fn clear_fit_preview(&mut self) {
        self.fit_preview_params = None;
        self.fit_preview_iteration = None;
    }

    #[cfg(test)]
    pub(super) fn update_parametric_result_metrics(
        &mut self,
        points: &Points,
        params: &CurveParams,
    ) -> IterationMetricSnapshot {
        let (metrics, result_metrics, residual_plot_points) =
            self.parametric_metrics_and_residuals(points, params);
        self.result_metrics = Some(result_metrics);
        self.residual_plot_points = residual_plot_points;
        metrics
    }

    fn parametric_metrics_and_residuals(
        &self,
        points: &Points,
        params: &CurveParams,
    ) -> (IterationMetricSnapshot, ExtendedMetrics, Vec<PlotPoint>) {
        let metrics = calculate_iteration_metrics_with_quantization(
            points,
            params,
            self.fit_loss_metric,
            self.fit_metric_quantization,
        );
        let result_metrics = ExtendedMetrics {
            mse: metrics.mse,
            rmse: metrics.rmse,
            mae: metrics.mae,
            r2: metrics.r2,
            max_abs_error: metrics.max_abs_error,
        };
        let mut residual_plot_points = Vec::with_capacity(points.len());
        for point in points.as_slice() {
            let residual = params.evaluate(point.x()) - point.y();
            residual_plot_points.push(PlotPoint::new(point.x(), residual));
        }

        (metrics, result_metrics, residual_plot_points)
    }

    fn apply_fit_completion_visual_snapshot(
        &mut self,
        diagnostics: IterationDiagnostics,
        replay_frames: Vec<ReplayFrame>,
    ) {
        self.spline_plot_curve = None;
        self.sampled_curve_cache = None;
        self.iteration_diagnostics = diagnostics;
        self.replay.frames = replay_frames;
        self.replay.selected_index = None;
        self.replay.autoplay = false;
        self.replay.last_step_at = None;
        self.panel.diagnostics_hide_non_loss_by_default_pending = true;
    }
    fn finalize_parametric_fit_completion(
        &mut self,
        result: FitResult,
        trace: Vec<ParametricIterationTraceEntry>,
    ) {
        let fit_points = self.active_fit_points.take();
        let point_count = fit_points.as_ref().map(Points::len).unwrap_or_default();
        let initial_params = self.take_parametric_fit_seed();
        let mut diagnostics = IterationDiagnostics::default();
        let mut replay_frames = Vec::with_capacity(trace.len().saturating_add(2));
        if let (Some(points), Some(initial_params)) = (fit_points.as_ref(), initial_params.as_ref())
        {
            // Для диагностики и replay сохраняем начальное состояние как итерацию 0.
            diagnostics.initialize(
                points,
                initial_params,
                self.fit_loss_metric,
                self.fit_metric_quantization,
            );
            Self::upsert_buffered_parametric_replay_frame(
                &mut replay_frames,
                0,
                initial_params.clone(),
            );
        }
        Self::apply_parametric_trace_to_buffers(trace, &mut diagnostics, &mut replay_frames);

        let mut result_metrics = None;
        let mut residual_plot_points = Vec::new();
        if let Some(points) = fit_points.as_ref() {
            // Финальные метрики пересчитываем по тем точкам и параметрам, которые видит пользователь.
            let (metrics, snapshot_metrics, snapshot_residuals) =
                self.parametric_metrics_and_residuals(points, &result.params);
            diagnostics.append(result.iterations, metrics, &result.params);
            result_metrics = Some(snapshot_metrics);
            residual_plot_points = snapshot_residuals;
        }

        Self::upsert_buffered_parametric_replay_frame(
            &mut replay_frames,
            result.iterations,
            result.params.clone(),
        );

        self.spline_result = None;
        self.result_metrics = result_metrics;
        self.residual_plot_points = residual_plot_points;
        self.apply_fit_completion_visual_snapshot(diagnostics, replay_frames);
        self.finalize_replay_after_fit_completion();
        self.complete_fit_timer_successfully();
        self.store_parametric_fit_export_record(&result, point_count);
        self.fit_result = Some(result);
        self.status = Some(StatusMessage::FitCompleted);
    }

    fn finalize_spline_fit_completion(
        &mut self,
        result: SplineResult,
        metrics: IterationMetricSnapshot,
        trace: Vec<SplineIterationTraceEntry>,
    ) {
        let point_count = result.residuals.len();
        let initial_curve = self.take_spline_fit_seed();
        let mut diagnostics = IterationDiagnostics::default();
        let mut replay_frames = Vec::with_capacity(trace.len().saturating_add(2));
        if let Some(initial_curve) = initial_curve {
            // Начальная кривая нужна, чтобы replay показывал эволюцию сплайна с нулевого шага.
            Self::upsert_buffered_spline_replay_frame(&mut replay_frames, 0, initial_curve);
        }
        Self::apply_spline_trace_to_buffers(trace, &mut diagnostics, &mut replay_frames);
        let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
        let spline_plot_curve = Self::plot_points_from_pairs(result.curve.iter().copied());
        diagnostics.append_spline(result.iterations, metrics, &knot_y);
        Self::upsert_buffered_spline_replay_frame(
            &mut replay_frames,
            result.iterations,
            spline_plot_curve,
        );

        let result_metrics = ExtendedMetrics {
            mse: result.mse,
            rmse: result.rmse,
            mae: result.mae,
            r2: result.r2,
            max_abs_error: result.max_abs_error,
        };
        let residual_plot_points = Self::plot_points_from_pairs(result.residuals.iter().copied());

        self.fit_result = None;
        self.result_metrics = Some(result_metrics);
        self.residual_plot_points = residual_plot_points;
        self.apply_fit_completion_visual_snapshot(diagnostics, replay_frames);
        self.finalize_replay_after_fit_completion();
        self.complete_fit_timer_successfully();
        self.store_spline_fit_export_record(&result, point_count);
        self.spline_result = Some(result);
        self.status = Some(StatusMessage::FitCompleted);
        self.active_fit_points = None;
    }

    #[cfg(target_arch = "wasm32")]
    fn finalize_fit_stopped_runtime(&mut self) {
        self.fit_in_progress = false;
        self.clear_fit_outputs();
        self.status = Some(StatusMessage::FitStopped);
    }

    #[cfg(target_arch = "wasm32")]
    fn finalize_fit_failed_runtime(&mut self, error: String) {
        self.fit_in_progress = false;
        self.clear_fit_outputs();
        self.status = Some(StatusMessage::Error(error));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn cancel_fit_and_discard_updates(&mut self) {
        if !self.fit_in_progress {
            return;
        }

        if let Some(cancel_flag) = &self.fit_cancel_flag {
            cancel_flag.store(true, Ordering::Relaxed);
        }
        self.discard_fit_worker_updates = true;
        if matches!(self.status, Some(StatusMessage::FittingInProgress)) {
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_fit_worker(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.fit_worker_rx.take() else {
            return;
        };

        let mut keep_receiver = true;
        match rx.try_recv() {
            Ok(FitWorkerMessage::Stopped) => {
                self.fit_in_progress = false;
                self.clear_fit_outputs();
                if !self.discard_fit_worker_updates {
                    self.status = Some(StatusMessage::FitStopped);
                } else {
                    self.set_fit_stopped_status_if_fitting();
                }
                keep_receiver = false;
            }
            Ok(FitWorkerMessage::Finished { result, trace }) => {
                self.fit_in_progress = false;
                if !self.discard_fit_worker_updates {
                    self.finalize_parametric_fit_completion(result, trace);
                } else {
                    self.active_fit_points = None;
                    self.reset_fit_timer();
                    self.clear_fit_export_state();
                    self.set_fit_stopped_status_if_fitting();
                }
                keep_receiver = false;
            }
            Ok(FitWorkerMessage::SplineFinished {
                result,
                metrics,
                trace,
            }) => {
                self.fit_in_progress = false;
                if !self.discard_fit_worker_updates {
                    self.finalize_spline_fit_completion(result, metrics, trace);
                } else {
                    self.reset_fit_timer();
                    self.clear_fit_export_state();
                    self.set_fit_stopped_status_if_fitting();
                    self.active_fit_points = None;
                }
                keep_receiver = false;
            }
            Ok(FitWorkerMessage::Failed(error)) => {
                self.fit_in_progress = false;
                self.clear_fit_outputs();
                if !self.discard_fit_worker_updates {
                    self.status = Some(StatusMessage::Error(error));
                } else {
                    self.set_fit_stopped_status_if_fitting();
                }
                keep_receiver = false;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.fit_in_progress = false;
                self.clear_fit_outputs();
                if !self.discard_fit_worker_updates {
                    self.status = Some(StatusMessage::Error(
                        "Fit worker channel disconnected unexpectedly".to_string(),
                    ));
                } else {
                    self.set_fit_stopped_status_if_fitting();
                }
                keep_receiver = false;
            }
        }

        if keep_receiver {
            self.fit_worker_rx = Some(rx);
        } else {
            self.fit_cancel_flag = None;
            self.discard_fit_worker_updates = false;
        }

        if self.fit_in_progress {
            ctx.request_repaint_after(FIT_WORKER_POLL_REPAINT_INTERVAL);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn poll_fit_worker(&mut self, ctx: &egui::Context) {
        let Some(job) = self.wasm_fit_job.take() else {
            return;
        };

        match job {
            WasmFitJob::Deferred(runner) => {
                self.wasm_fit_job = Some(WasmFitJob::Running(runner));
                ctx.request_repaint();
            }
            WasmFitJob::Running(runner) => match runner {
                WasmFitRunner::Parametric {
                    runner,
                    normalization,
                } => self.run_wasm_parametric_fit_continuously(runner, normalization),
                WasmFitRunner::Spline(runner) => self.run_wasm_spline_fit_continuously(runner),
            },
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn cancel_wasm_fit_if_available(&mut self) {
        if !self.fit_in_progress {
            return;
        }
        if let Some(job) = self.wasm_fit_job.as_mut() {
            let runner = match job {
                WasmFitJob::Deferred(runner) | WasmFitJob::Running(runner) => runner,
            };
            match runner {
                WasmFitRunner::Parametric { runner, .. } => runner.cancel(),
                WasmFitRunner::Spline(runner) => runner.cancel(),
            }
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn request_stop_fit(&mut self) {
        if let Some(cancel_flag) = &self.fit_cancel_flag {
            cancel_flag.store(true, Ordering::Relaxed);
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn cancel_fit_and_discard_updates(&mut self) {
        self.cancel_wasm_fit_if_available();
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn request_stop_fit(&mut self) {
        self.cancel_wasm_fit_if_available();
    }

    #[cfg(target_arch = "wasm32")]
    fn run_wasm_parametric_fit_continuously(
        &mut self,
        mut runner: IncrementalFitRunner,
        normalization: Option<ParametricNormalization>,
    ) {
        let mut iteration_trace = Vec::new();
        loop {
            match runner.step() {
                Ok(IncrementalFitStep::Iteration {
                    iteration,
                    mse: _,
                    metrics,
                    params,
                }) => {
                    let params = if let Some(normalization) = normalization {
                        match normalization.denormalize_params(&params) {
                            Ok(params) => params,
                            Err(error) => {
                                self.finalize_fit_failed_runtime(error);
                                break;
                            }
                        }
                    } else {
                        params
                    };
                    let metrics = if let Some(points) = self.active_fit_points.as_ref() {
                        calculate_iteration_metrics_with_quantization(
                            points,
                            &params,
                            self.fit_loss_metric,
                            self.fit_metric_quantization,
                        )
                    } else {
                        metrics
                    };
                    iteration_trace.push(ParametricIterationTraceEntry {
                        iteration,
                        metrics,
                        params,
                    });
                }
                Ok(IncrementalFitStep::Finished(result)) => {
                    self.fit_in_progress = false;
                    let mut result = result;
                    if let Some(normalization) = normalization {
                        result.params = match normalization.denormalize_params(&result.params) {
                            Ok(params) => params,
                            Err(error) => {
                                self.finalize_fit_failed_runtime(error);
                                break;
                            }
                        };
                    }
                    if let Some(points) = self.active_fit_points.as_ref() {
                        let (mse, rmse) = calculate_metrics_with_quantization(
                            points,
                            &result.params,
                            self.fit_metric_quantization,
                        );
                        result.mse = mse;
                        result.rmse = rmse;
                    }
                    self.finalize_parametric_fit_completion(result, iteration_trace);
                    break;
                }
                Ok(IncrementalFitStep::Cancelled) => {
                    self.finalize_fit_stopped_runtime();
                    break;
                }
                Err(error) => {
                    self.finalize_fit_failed_runtime(error.to_string());
                    break;
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn run_wasm_spline_fit_continuously(&mut self, mut runner: IncrementalSplineFitRunner) {
        let mut iteration_trace = Vec::new();
        loop {
            match runner.step() {
                Ok(IncrementalSplineFitStep::Iteration {
                    iteration,
                    mse: _,
                    metrics,
                    knot_y,
                    curve,
                }) => {
                    iteration_trace.push(SplineIterationTraceEntry {
                        iteration,
                        metrics,
                        knot_y,
                        curve,
                    });
                }
                Ok(IncrementalSplineFitStep::Finished { result, metrics }) => {
                    self.fit_in_progress = false;
                    self.finalize_spline_fit_completion(result, metrics, iteration_trace);
                    break;
                }
                Ok(IncrementalSplineFitStep::Cancelled) => {
                    self.finalize_fit_stopped_runtime();
                    break;
                }
                Err(error) => {
                    self.finalize_fit_failed_runtime(error.to_string());
                    break;
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_fit_worker(&mut self, input: ParametricFitWorkerInput) {
        let ParametricFitWorkerInput {
            family,
            optimization_points,
            display_points,
            optimization_initial_params,
            normalization,
            optimizer_config,
            loss_metric,
            metric_quantization,
            cancel_flag,
        } = input;
        let (tx, rx) = mpsc::channel();
        self.fit_worker_rx = Some(rx);
        self.fit_cancel_flag = Some(cancel_flag.clone());
        self.discard_fit_worker_updates = false;
        self.fit_in_progress = true;

        std::thread::spawn(move || {
            let progress_points = display_points;
            let mut iteration_trace = Vec::new();
            let mut runner = match IncrementalFitRunner::new_with_optimizer_config_and_loss_metric_and_metric_quantization(
                &optimization_points,
                family,
                optimization_initial_params,
                &optimizer_config,
                loss_metric,
                metric_quantization,
            ) {
                Ok(runner) => runner,
                Err(error) => {
                    let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                    return;
                }
            };

            loop {
                if cancel_flag.load(Ordering::Relaxed) {
                    runner.cancel();
                }

                match runner.step() {
                    Ok(IncrementalFitStep::Iteration {
                        iteration,
                        mse: _,
                        metrics: _,
                        params,
                    }) => {
                        let params = if let Some(normalization) = normalization {
                            match normalization.denormalize_params(&params) {
                                Ok(params) => params,
                                Err(_) => continue,
                            }
                        } else {
                            params
                        };
                        let metrics = calculate_iteration_metrics_with_quantization(
                            &progress_points,
                            &params,
                            loss_metric,
                            metric_quantization,
                        );
                        iteration_trace.push(ParametricIterationTraceEntry {
                            iteration,
                            metrics,
                            params,
                        });
                    }
                    Ok(IncrementalFitStep::Finished(result)) => {
                        let params = if let Some(normalization) = normalization {
                            match normalization.denormalize_params(&result.params) {
                                Ok(params) => params,
                                Err(error) => {
                                    let _ = tx.send(FitWorkerMessage::Failed(error));
                                    break;
                                }
                            }
                        } else {
                            result.params
                        };
                        let (mse, rmse) = calculate_metrics_with_quantization(
                            &progress_points,
                            &params,
                            metric_quantization,
                        );
                        let _ = tx.send(FitWorkerMessage::Finished {
                            result: FitResult {
                                family: result.family,
                                params,
                                mse,
                                rmse,
                                iterations: result.iterations,
                            },
                            trace: iteration_trace,
                        });
                        break;
                    }
                    Ok(IncrementalFitStep::Cancelled) | Err(FitError::Cancelled) => {
                        let _ = tx.send(FitWorkerMessage::Stopped);
                        break;
                    }
                    Err(error) => {
                        let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                        break;
                    }
                }
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_spline_fit_worker(
        &mut self,
        family: SplineFamilyKind,
        points: Points,
        config: SplineConfig,
        optimizer_config: OptimizerConfig,
        initial_knot_y: Vec<f64>,
        cancel_flag: Arc<AtomicBool>,
    ) {
        let loss_metric = self.fit_loss_metric;
        let metric_quantization = self.fit_metric_quantization;
        let (tx, rx) = mpsc::channel();
        self.fit_worker_rx = Some(rx);
        self.fit_cancel_flag = Some(cancel_flag.clone());
        self.discard_fit_worker_updates = false;
        self.fit_in_progress = true;

        std::thread::spawn(move || {
            let mut iteration_trace = Vec::new();
            let mut runner =
                match IncrementalSplineFitRunner::new_with_initial_knot_y_and_optimizer_config_and_loss_metric(
                    &points,
                    family,
                    config,
                    &optimizer_config,
                    Some(initial_knot_y.as_slice()),
                    loss_metric,
                    metric_quantization,
                ) {
                    Ok(runner) => runner,
                    Err(error) => {
                        let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                        return;
                    }
                };

            loop {
                if cancel_flag.load(Ordering::Relaxed) {
                    runner.cancel();
                }

                match runner.step() {
                    Ok(IncrementalSplineFitStep::Iteration {
                        iteration,
                        mse: _,
                        metrics,
                        knot_y,
                        curve,
                    }) => {
                        iteration_trace.push(SplineIterationTraceEntry {
                            iteration,
                            metrics,
                            knot_y,
                            curve,
                        });
                    }
                    Ok(IncrementalSplineFitStep::Finished { result, metrics }) => {
                        let _ = tx.send(FitWorkerMessage::SplineFinished {
                            result,
                            metrics,
                            trace: iteration_trace,
                        });
                        break;
                    }
                    Ok(IncrementalSplineFitStep::Cancelled) | Err(FitError::Cancelled) => {
                        let _ = tx.send(FitWorkerMessage::Stopped);
                        break;
                    }
                    Err(error) => {
                        let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                        break;
                    }
                }
            }
        });
    }

    pub(super) fn run_fit(&mut self) {
        if self.fit_in_progress {
            return;
        }

        self.fit_optimizer_method = self.optimizer_method;
        self.last_right_panel_fit_snapshot = Some(self.capture_right_panel_fit_snapshot());
        self.auto_refit_pending_rerun = false;

        let points = match self.parse_points_strict() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };
        let optimizer_config = match self.optimizer_config() {
            Ok(config) => config,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };
        let loss_metric = self.optimization_loss_metric;
        self.fit_loss_metric = loss_metric;
        let metric_quantization = match self.selected_metric_quantization() {
            Ok(metric_quantization) => metric_quantization,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };
        self.fit_metric_quantization = metric_quantization;

        let selected_model = self.resolved_model();
        if let Some(spline_family) = selected_model.spline_family() {
            let Some(spline_config) = self.spline_config_for_model(selected_model, points.len())
            else {
                self.status = Some(StatusMessage::Error(
                    "Selected spline model has no spline configuration".to_string(),
                ));
                return;
            };
            self.sync_spline_initial_knot_y_inputs(spline_config.knots);
            let initial_knot_y = match self.parse_spline_initial_knot_y(spline_config.knots) {
                Ok(values) => values,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            let initial_curve = match build_spline_initial_curve_from_knot_y(
                &points,
                spline_family,
                spline_config,
                initial_knot_y.as_slice(),
            ) {
                Ok(curve) => curve,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error.to_string()));
                    return;
                }
            };
            self.reset_fit_runtime_for_new_run();
            self.fit_run_ui_seed = Some(FitRunUiSeed::Spline {
                initial_curve: Self::plot_points_from_pairs(initial_curve),
            });
            self.start_fit_timer();
            self.status = Some(StatusMessage::FittingInProgress);

            #[cfg(not(target_arch = "wasm32"))]
            {
                let cancel_flag = Arc::new(AtomicBool::new(false));
                self.start_spline_fit_worker(
                    spline_family,
                    points,
                    spline_config,
                    optimizer_config,
                    initial_knot_y.into_vec(),
                    cancel_flag,
                );
            }

            #[cfg(target_arch = "wasm32")]
            {
                match IncrementalSplineFitRunner::new_with_initial_knot_y_and_optimizer_config_and_loss_metric(
                    &points,
                    spline_family,
                    spline_config,
                    &optimizer_config,
                    Some(initial_knot_y.as_slice()),
                    loss_metric,
                    metric_quantization,
                ) {
                    Ok(runner) => {
                        self.wasm_fit_job =
                            Some(WasmFitJob::Deferred(WasmFitRunner::Spline(runner)));
                        self.fit_in_progress = true;
                    }
                    Err(error) => {
                        self.reset_fit_timer();
                        self.fit_run_ui_seed = None;
                        self.status = Some(StatusMessage::Error(error.to_string()));
                    }
                }
            }
            return;
        }

        let Some(family) = selected_model.parametric_family() else {
            self.status = Some(StatusMessage::Error(
                "Selected model has no parametric family".to_string(),
            ));
            return;
        };

        let initial_params = match self.parse_initial_params() {
            Ok(params) => params.into_curve_params(),
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        if let Err(error) = family.validate_points(&points) {
            self.status = Some(StatusMessage::Error(error.to_string()));
            return;
        }

        let mut optimization_points = points.clone();
        let mut optimization_initial_params = initial_params.clone();
        let normalization = if self.normalize_parametric_data {
            let normalization = match ParametricNormalization::try_from_points(&points) {
                Ok(normalization) => normalization,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            optimization_points = match normalization.normalize_points(&points) {
                Ok(normalized_points) => normalized_points,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            optimization_initial_params = match normalization.normalize_params(&initial_params) {
                Ok(normalized_params) => normalized_params,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            Some(normalization)
        } else {
            None
        };

        self.reset_fit_runtime_for_new_run();
        self.active_fit_points = Some(points.clone());
        self.fit_run_ui_seed = Some(FitRunUiSeed::Parametric {
            initial_params: initial_params.clone(),
        });
        self.start_fit_timer();
        self.status = Some(StatusMessage::FittingInProgress);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let cancel_flag = Arc::new(AtomicBool::new(false));
            self.start_fit_worker(ParametricFitWorkerInput {
                family,
                optimization_points,
                display_points: points,
                optimization_initial_params,
                normalization,
                optimizer_config,
                loss_metric,
                metric_quantization,
                cancel_flag,
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            match IncrementalFitRunner::new_with_optimizer_config_and_loss_metric_and_metric_quantization(
                &optimization_points,
                family,
                optimization_initial_params,
                &optimizer_config,
                loss_metric,
                metric_quantization,
            ) {
                Ok(runner) => {
                    self.wasm_fit_job = Some(WasmFitJob::Deferred(WasmFitRunner::Parametric {
                        runner,
                        normalization,
                    }));
                    self.fit_in_progress = true;
                }
                Err(error) => {
                    self.reset_fit_timer();
                    self.active_fit_points = None;
                    self.fit_run_ui_seed = None;
                    self.status = Some(StatusMessage::Error(error.to_string()));
                }
            }
        }
    }
}
