use super::*;

#[cfg(not(target_arch = "wasm32"))]
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

    pub(super) fn update_parametric_result_metrics(
        &mut self,
        points: &Points,
        params: &CurveParams,
    ) {
        let metrics = calculate_iteration_metrics_with_quantization(
            points,
            params,
            self.fit_loss_metric,
            self.fit_metric_quantization,
        );
        self.result_metrics = Some(ExtendedMetrics {
            mse: metrics.mse,
            rmse: metrics.rmse,
            mae: metrics.mae,
            r2: metrics.r2,
            max_abs_error: metrics.max_abs_error,
        });

        self.residual_plot_points.clear();
        self.residual_plot_points.reserve(points.len());
        for point in points.as_slice() {
            let residual = params.evaluate(point.x()) - point.y();
            self.residual_plot_points
                .push(PlotPoint::new(point.x(), residual));
        }
    }

    pub(super) fn update_spline_result_metrics(&mut self, result: &SplineResult) {
        self.result_metrics = Some(ExtendedMetrics {
            mse: result.mse,
            rmse: result.rmse,
            mae: result.mae,
            r2: result.r2,
            max_abs_error: result.max_abs_error,
        });
        self.residual_plot_points = Self::plot_points_from_pairs(result.residuals.iter().copied());
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
        loop {
            match rx.try_recv() {
                Ok(FitWorkerMessage::Iteration {
                    iteration,
                    metrics,
                    params,
                }) => {
                    if self.discard_fit_worker_updates {
                        continue;
                    }
                    self.iteration_diagnostics
                        .append(iteration, metrics, &params);
                    self.upsert_parametric_replay_frame(iteration, params);
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(FitWorkerMessage::SplineIteration {
                    iteration,
                    metrics,
                    knot_y,
                    curve,
                }) => {
                    if self.discard_fit_worker_updates {
                        continue;
                    }
                    self.iteration_diagnostics
                        .append_spline(iteration, metrics, &knot_y);
                    self.upsert_spline_replay_frame(iteration, Self::plot_points_from_pairs(curve));
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(FitWorkerMessage::Stopped) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.active_fit_points = None;
                    self.finalize_replay_after_fit_stopped();
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::FitStopped);
                    } else {
                        self.set_fit_stopped_status_if_fitting();
                    }
                    keep_receiver = false;
                    break;
                }
                Ok(FitWorkerMessage::Finished(result)) => {
                    self.fit_in_progress = false;
                    let fit_points = self.active_fit_points.take();
                    if !self.discard_fit_worker_updates {
                        if let Some(points) = fit_points.as_ref() {
                            self.update_parametric_result_metrics(points, &result.params);
                            let metrics = calculate_iteration_metrics_with_quantization(
                                points,
                                &result.params,
                                self.fit_loss_metric,
                                self.fit_metric_quantization,
                            );
                            self.iteration_diagnostics.append(
                                result.iterations,
                                metrics,
                                &result.params,
                            );
                        }
                        self.upsert_parametric_replay_frame(
                            result.iterations,
                            result.params.clone(),
                        );
                        self.finalize_replay_after_fit_completion();
                        self.fit_result = Some(result);
                        self.complete_fit_timer_successfully();
                        self.status = Some(StatusMessage::FitCompleted);
                    } else {
                        self.reset_fit_timer();
                        self.set_fit_stopped_status_if_fitting();
                    }
                    keep_receiver = false;
                    break;
                }
                Ok(FitWorkerMessage::SplineFinished { result, metrics }) => {
                    self.fit_in_progress = false;
                    if !self.discard_fit_worker_updates {
                        let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
                        let spline_plot_curve =
                            Self::plot_points_from_pairs(result.curve.iter().copied());
                        self.update_spline_result_metrics(&result);
                        self.iteration_diagnostics.append_spline(
                            result.iterations,
                            metrics,
                            &knot_y,
                        );
                        self.upsert_spline_replay_frame(result.iterations, spline_plot_curve);
                        self.finalize_replay_after_fit_completion();
                        self.spline_result = Some(result);
                        self.complete_fit_timer_successfully();
                        self.status = Some(StatusMessage::FitCompleted);
                    } else {
                        self.reset_fit_timer();
                        self.set_fit_stopped_status_if_fitting();
                    }
                    self.active_fit_points = None;
                    keep_receiver = false;
                    break;
                }
                Ok(FitWorkerMessage::Failed(error)) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::Error(error));
                    } else {
                        self.set_fit_stopped_status_if_fitting();
                    }
                    keep_receiver = false;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::Error(
                            "Fit worker channel disconnected unexpectedly".to_string(),
                        ));
                    } else {
                        self.set_fit_stopped_status_if_fitting();
                    }
                    keep_receiver = false;
                    break;
                }
            }
        }

        if keep_receiver {
            self.fit_worker_rx = Some(rx);
        } else {
            self.fit_cancel_flag = None;
            self.discard_fit_worker_updates = false;
        }

        if self.fit_in_progress {
            ctx.request_repaint_after(Duration::from_millis(16));
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
                                self.fit_in_progress = false;
                                self.reset_fit_timer();
                                self.status = Some(StatusMessage::Error(error));
                                self.active_fit_points = None;
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
                    self.iteration_diagnostics
                        .append(iteration, metrics, &params);
                    self.upsert_parametric_replay_frame(iteration, params);
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(IncrementalFitStep::Finished(result)) => {
                    self.fit_in_progress = false;
                    let mut result = result;
                    if let Some(normalization) = normalization {
                        result.params = match normalization.denormalize_params(&result.params) {
                            Ok(params) => params,
                            Err(error) => {
                                self.reset_fit_timer();
                                self.status = Some(StatusMessage::Error(error));
                                self.active_fit_points = None;
                                break;
                            }
                        };
                    }
                    let fit_points = self.active_fit_points.take();
                    if let Some(points) = fit_points.as_ref() {
                        let (mse, rmse) = calculate_metrics_with_quantization(
                            points,
                            &result.params,
                            self.fit_metric_quantization,
                        );
                        result.mse = mse;
                        result.rmse = rmse;
                        self.update_parametric_result_metrics(points, &result.params);
                        let metrics = calculate_iteration_metrics_with_quantization(
                            points,
                            &result.params,
                            self.fit_loss_metric,
                            self.fit_metric_quantization,
                        );
                        self.iteration_diagnostics.append(
                            result.iterations,
                            metrics,
                            &result.params,
                        );
                    }
                    self.upsert_parametric_replay_frame(result.iterations, result.params.clone());
                    self.finalize_replay_after_fit_completion();
                    self.fit_result = Some(result);
                    self.complete_fit_timer_successfully();
                    self.status = Some(StatusMessage::FitCompleted);
                    break;
                }
                Ok(IncrementalFitStep::Cancelled) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.finalize_replay_after_fit_stopped();
                    self.status = Some(StatusMessage::FitStopped);
                    self.active_fit_points = None;
                    break;
                }
                Err(error) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.status = Some(StatusMessage::Error(error.to_string()));
                    self.active_fit_points = None;
                    break;
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn run_wasm_spline_fit_continuously(&mut self, mut runner: IncrementalSplineFitRunner) {
        loop {
            match runner.step() {
                Ok(IncrementalSplineFitStep::Iteration {
                    iteration,
                    mse: _,
                    metrics,
                    knot_y,
                    curve,
                }) => {
                    self.iteration_diagnostics
                        .append_spline(iteration, metrics, &knot_y);
                    self.upsert_spline_replay_frame(iteration, Self::plot_points_from_pairs(curve));
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(IncrementalSplineFitStep::Finished { result, metrics }) => {
                    self.fit_in_progress = false;
                    let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
                    let spline_plot_curve =
                        Self::plot_points_from_pairs(result.curve.iter().copied());
                    self.update_spline_result_metrics(&result);
                    self.iteration_diagnostics
                        .append_spline(result.iterations, metrics, &knot_y);
                    self.upsert_spline_replay_frame(result.iterations, spline_plot_curve);
                    self.finalize_replay_after_fit_completion();
                    self.spline_result = Some(result);
                    self.complete_fit_timer_successfully();
                    self.status = Some(StatusMessage::FitCompleted);
                    self.active_fit_points = None;
                    break;
                }
                Ok(IncrementalSplineFitStep::Cancelled) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.finalize_replay_after_fit_stopped();
                    self.status = Some(StatusMessage::FitStopped);
                    self.active_fit_points = None;
                    break;
                }
                Err(error) => {
                    self.fit_in_progress = false;
                    self.reset_fit_timer();
                    self.status = Some(StatusMessage::Error(error.to_string()));
                    self.active_fit_points = None;
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
            let iter_tx = tx.clone();
            let progress_cancel = cancel_flag.clone();
            let progress_points = Arc::new(display_points);
            let callback_points = Arc::clone(&progress_points);
            let result =
                fit_curve_with_progress_and_optimizer_config_and_loss_metric_and_metric_quantization(
                &optimization_points,
                family,
                optimization_initial_params,
                &optimizer_config,
                loss_metric,
                metric_quantization,
                move |iteration, params| {
                    if progress_cancel.load(Ordering::Relaxed) {
                        return false;
                    }
                    if let Some(params) = params {
                        let params = if let Some(normalization) = normalization {
                            match normalization.denormalize_params(&params) {
                                Ok(params) => params,
                                Err(_) => return !progress_cancel.load(Ordering::Relaxed),
                            }
                        } else {
                            params
                        };
                        let metrics = calculate_iteration_metrics_with_quantization(
                            callback_points.as_ref(),
                            &params,
                            loss_metric,
                            metric_quantization,
                        );
                        let _ = iter_tx.send(FitWorkerMessage::Iteration {
                            iteration,
                            metrics,
                            params,
                        });
                    }
                    !progress_cancel.load(Ordering::Relaxed)
                },
            );

            match result {
                Ok(result) => {
                    let params = if let Some(normalization) = normalization {
                        match normalization.denormalize_params(&result.params) {
                            Ok(params) => params,
                            Err(error) => {
                                let _ = tx.send(FitWorkerMessage::Failed(error));
                                return;
                            }
                        }
                    } else {
                        result.params
                    };
                    let (mse, rmse) = calculate_metrics_with_quantization(
                        progress_points.as_ref(),
                        &params,
                        metric_quantization,
                    );
                    let _ = tx.send(FitWorkerMessage::Finished(FitResult {
                        family: result.family,
                        params,
                        mse,
                        rmse,
                        iterations: result.iterations,
                    }));
                }
                Err(FitError::Cancelled) => {
                    let _ = tx.send(FitWorkerMessage::Stopped);
                }
                Err(error) => {
                    let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
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
                        let _ = tx.send(FitWorkerMessage::SplineIteration {
                            iteration,
                            metrics,
                            knot_y,
                            curve,
                        });
                    }
                    Ok(IncrementalSplineFitStep::Finished { result, metrics }) => {
                        let _ = tx.send(FitWorkerMessage::SplineFinished { result, metrics });
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
            let spline_config = self
                .spline_config_for_model(selected_model, points.len())
                .expect("spline model must provide spline config");
            self.sync_spline_initial_knot_y_inputs(spline_config.knots);
            let initial_knot_y = match self.parse_spline_initial_knot_y(spline_config.knots) {
                Ok(values) => values,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            self.clear_fit_outputs();
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
            self.upsert_spline_replay_frame(0, Self::plot_points_from_pairs(initial_curve));
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

        // Очищаем предыдущий успешный результат только когда новый запуск уже валиден.
        self.clear_fit_outputs();
        self.active_fit_points = Some(points.clone());
        self.iteration_diagnostics.initialize(
            &points,
            &initial_params,
            loss_metric,
            metric_quantization,
        );
        self.upsert_parametric_replay_frame(0, initial_params.clone());
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
                    self.status = Some(StatusMessage::Error(error.to_string()));
                }
            }
        }
    }
}
