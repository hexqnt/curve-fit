use super::*;

impl CurveFitApp {
    pub(super) fn clear_fit_preview(&mut self) {
        self.fit_preview_params = None;
        self.fit_preview_iteration = None;
    }

    pub(super) fn update_parametric_result_metrics(
        &mut self,
        points: &Points,
        params: &CurveParams,
    ) {
        let sample_count = points.len() as f64;
        let y_mean = points.as_slice().iter().map(|point| point.y()).sum::<f64>() / sample_count;

        let mut sse = 0.0;
        let mut sae = 0.0;
        let mut max_abs_error = 0.0_f64;
        self.residual_plot_points.clear();
        self.residual_plot_points.reserve(points.len());
        for point in points.as_slice() {
            let residual = params.evaluate(point.x()) - point.y();
            let abs_residual = residual.abs();
            sse += residual * residual;
            sae += abs_residual;
            max_abs_error = max_abs_error.max(abs_residual);
            self.residual_plot_points
                .push(PlotPoint::new(point.x(), residual));
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
        self.result_metrics = Some(ExtendedMetrics {
            mse,
            rmse,
            mae,
            r2,
            max_abs_error,
        });
    }

    pub(super) fn update_spline_result_metrics(&mut self, result: &SplineResult) {
        self.result_metrics = Some(ExtendedMetrics {
            mse: result.mse,
            rmse: result.rmse,
            mae: result.mae,
            r2: result.r2,
            max_abs_error: result.max_abs_error,
        });
        self.residual_plot_points = result
            .residuals
            .iter()
            .map(|point| PlotPoint::new(point[0], point[1]))
            .collect();
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
                    mse,
                    params,
                }) => {
                    if self.discard_fit_worker_updates {
                        continue;
                    }
                    self.fit_preview_iteration = Some(iteration);
                    self.iteration_diagnostics.append(iteration, mse, &params);
                    self.fit_preview_params = Some(params);
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(FitWorkerMessage::SplineIteration {
                    iteration,
                    mse,
                    knot_y,
                    curve,
                }) => {
                    if self.discard_fit_worker_updates {
                        continue;
                    }
                    self.fit_preview_iteration = Some(iteration);
                    self.iteration_diagnostics
                        .append_spline(iteration, mse, &knot_y);
                    self.spline_plot_curve = Some(
                        curve
                            .into_iter()
                            .map(|point| PlotPoint::new(point[0], point[1]))
                            .collect(),
                    );
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(FitWorkerMessage::Stopped) => {
                    self.fit_in_progress = false;
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates
                        || matches!(
                            self.status,
                            Some(StatusMessage::FittingInProgress)
                                | Some(StatusMessage::FittingStopping)
                        )
                    {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    keep_receiver = false;
                }
                Ok(FitWorkerMessage::Finished(result)) => {
                    self.fit_in_progress = false;
                    if !self.discard_fit_worker_updates {
                        if let Some(points) = self.active_fit_points.clone() {
                            self.update_parametric_result_metrics(&points, &result.params);
                        }
                        self.iteration_diagnostics.append(
                            result.iterations,
                            result.mse,
                            &result.params,
                        );
                        self.fit_preview_iteration = Some(result.iterations);
                        self.fit_preview_params = Some(result.params.clone());
                        self.fit_result = Some(result);
                        self.status = Some(StatusMessage::FitCompleted);
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    self.active_fit_points = None;
                    keep_receiver = false;
                }
                Ok(FitWorkerMessage::SplineFinished(result)) => {
                    self.fit_in_progress = false;
                    if !self.discard_fit_worker_updates {
                        let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
                        let spline_plot_curve = result
                            .curve
                            .iter()
                            .map(|point| PlotPoint::new(point[0], point[1]))
                            .collect();
                        self.update_spline_result_metrics(&result);
                        self.iteration_diagnostics.append_spline(
                            result.iterations,
                            result.mse,
                            &knot_y,
                        );
                        self.fit_preview_iteration = Some(result.iterations);
                        self.spline_plot_curve = Some(spline_plot_curve);
                        self.spline_result = Some(result);
                        self.status = Some(StatusMessage::FitCompleted);
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    self.active_fit_points = None;
                    keep_receiver = false;
                }
                Ok(FitWorkerMessage::Failed(error)) => {
                    self.fit_in_progress = false;
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::Error(error));
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    keep_receiver = false;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.fit_in_progress = false;
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::Error(
                            "Fit worker channel disconnected unexpectedly".to_string(),
                        ));
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
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

        if self.fit_in_progress && self.fit_preview_iteration.is_some() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn poll_fit_worker(&mut self, ctx: &egui::Context) {
        enum WasmRunnerStep {
            Parametric(Result<IncrementalFitStep, FitError>),
            Spline(Result<IncrementalSplineFitStep, FitError>),
        }

        let Some(step) = self.wasm_fit_runner.as_mut().map(|runner| match runner {
            WasmFitRunner::Parametric(runner) => WasmRunnerStep::Parametric(runner.step()),
            WasmFitRunner::Spline(runner) => WasmRunnerStep::Spline(runner.step()),
        }) else {
            return;
        };

        match step {
            WasmRunnerStep::Parametric(Ok(IncrementalFitStep::Iteration {
                iteration,
                mse,
                params,
            })) => {
                self.fit_preview_iteration = Some(iteration);
                self.iteration_diagnostics.append(iteration, mse, &params);
                self.fit_preview_params = Some(params);
                self.status = Some(StatusMessage::FittingInProgress);
                if self.iteration_delay_seconds > 0.0 {
                    ctx.request_repaint_after(Duration::from_secs_f64(
                        self.iteration_delay_seconds,
                    ));
                } else {
                    ctx.request_repaint();
                }
            }
            WasmRunnerStep::Parametric(Ok(IncrementalFitStep::Finished(result))) => {
                self.fit_in_progress = false;
                if let Some(points) = self.active_fit_points.clone() {
                    self.update_parametric_result_metrics(&points, &result.params);
                }
                self.iteration_diagnostics
                    .append(result.iterations, result.mse, &result.params);
                self.fit_preview_iteration = Some(result.iterations);
                self.fit_preview_params = Some(result.params.clone());
                self.fit_result = Some(result);
                self.status = Some(StatusMessage::FitCompleted);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Parametric(Ok(IncrementalFitStep::Cancelled)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::FitStopped);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Parametric(Err(error)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::Error(error.to_string()));
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Spline(Ok(IncrementalSplineFitStep::Iteration {
                iteration,
                mse,
                knot_y,
                curve,
            })) => {
                self.fit_preview_iteration = Some(iteration);
                self.iteration_diagnostics
                    .append_spline(iteration, mse, &knot_y);
                self.spline_plot_curve = Some(
                    curve
                        .into_iter()
                        .map(|point| PlotPoint::new(point[0], point[1]))
                        .collect(),
                );
                self.status = Some(StatusMessage::FittingInProgress);
                if self.iteration_delay_seconds > 0.0 {
                    ctx.request_repaint_after(Duration::from_secs_f64(
                        self.iteration_delay_seconds,
                    ));
                } else {
                    ctx.request_repaint();
                }
            }
            WasmRunnerStep::Spline(Ok(IncrementalSplineFitStep::Finished(result))) => {
                self.fit_in_progress = false;
                let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
                let spline_plot_curve = result
                    .curve
                    .iter()
                    .map(|point| PlotPoint::new(point[0], point[1]))
                    .collect();
                self.update_spline_result_metrics(&result);
                self.iteration_diagnostics
                    .append_spline(result.iterations, result.mse, &knot_y);
                self.fit_preview_iteration = Some(result.iterations);
                self.spline_plot_curve = Some(spline_plot_curve);
                self.spline_result = Some(result);
                self.status = Some(StatusMessage::FitCompleted);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Spline(Ok(IncrementalSplineFitStep::Cancelled)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::FitStopped);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Spline(Err(error)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::Error(error.to_string()));
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn cancel_fit_and_discard_updates(&mut self) {
        if !self.fit_in_progress {
            return;
        }
        if let Some(runner) = self.wasm_fit_runner.as_mut() {
            match runner {
                WasmFitRunner::Parametric(runner) => runner.cancel(),
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
    pub(super) fn request_stop_fit(&mut self) {
        if let Some(runner) = self.wasm_fit_runner.as_mut() {
            match runner {
                WasmFitRunner::Parametric(runner) => runner.cancel(),
                WasmFitRunner::Spline(runner) => runner.cancel(),
            }
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_fit_worker(
        &mut self,
        family: CurveFamily,
        points: Points,
        initial_params: CurveParams,
        optimizer_config: OptimizerConfig,
        cancel_flag: Arc<AtomicBool>,
    ) {
        let delay_seconds = self.iteration_delay_seconds;
        let (tx, rx) = mpsc::channel();
        self.fit_worker_rx = Some(rx);
        self.fit_cancel_flag = Some(cancel_flag.clone());
        self.discard_fit_worker_updates = false;
        self.fit_in_progress = true;

        std::thread::spawn(move || {
            let iter_tx = tx.clone();
            let progress_cancel = cancel_flag.clone();
            let progress_points = points.clone();
            let result = fit_curve_with_progress_and_optimizer_config(
                &points,
                family,
                initial_params,
                &optimizer_config,
                move |iteration, params| {
                    if progress_cancel.load(Ordering::Relaxed) {
                        return false;
                    }
                    if let Some(params) = params {
                        let (mse, _) = calculate_metrics(&progress_points, &params);
                        let _ = iter_tx.send(FitWorkerMessage::Iteration {
                            iteration,
                            mse,
                            params,
                        });
                    }
                    if delay_seconds > 0.0 {
                        std::thread::sleep(Duration::from_secs_f64(delay_seconds));
                    }
                    !progress_cancel.load(Ordering::Relaxed)
                },
            );

            match result {
                Ok(result) => {
                    let _ = tx.send(FitWorkerMessage::Finished(result));
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
        let delay_seconds = self.iteration_delay_seconds;
        let (tx, rx) = mpsc::channel();
        self.fit_worker_rx = Some(rx);
        self.fit_cancel_flag = Some(cancel_flag.clone());
        self.discard_fit_worker_updates = false;
        self.fit_in_progress = true;

        std::thread::spawn(move || {
            let mut runner =
                match IncrementalSplineFitRunner::new_with_initial_knot_y_and_optimizer_config(
                    &points,
                    family,
                    config,
                    &optimizer_config,
                    Some(initial_knot_y.as_slice()),
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
                        mse,
                        knot_y,
                        curve,
                    }) => {
                        let _ = tx.send(FitWorkerMessage::SplineIteration {
                            iteration,
                            mse,
                            knot_y,
                            curve,
                        });
                        if delay_seconds > 0.0 {
                            std::thread::sleep(Duration::from_secs_f64(delay_seconds));
                        }
                    }
                    Ok(IncrementalSplineFitStep::Finished(result)) => {
                        let _ = tx.send(FitWorkerMessage::SplineFinished(result));
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

        let selected_model = self.resolved_model();
        if matches!(
            selected_model,
            ResolvedModel::LinearSpline
                | ResolvedModel::MonotoneCubicSpline
                | ResolvedModel::NaturalCubicSpline
                | ResolvedModel::AkimaSpline
        ) {
            let spline_family = match selected_model {
                ResolvedModel::LinearSpline => SplineFamilyKind::Linear,
                ResolvedModel::MonotoneCubicSpline => SplineFamilyKind::MonotoneCubic,
                ResolvedModel::NaturalCubicSpline => SplineFamilyKind::NaturalCubic,
                ResolvedModel::AkimaSpline => SplineFamilyKind::Akima,
                ResolvedModel::Parametric(_) => unreachable!("checked by matches!"),
            };
            let spline_config = self
                .spline_config_for_model(selected_model, points.len())
                .expect("checked by matches!");
            self.sync_spline_initial_knot_y_inputs(spline_config.knots);
            let initial_knot_y = match self.parse_spline_initial_knot_y(spline_config.knots) {
                Ok(values) => values,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            self.clear_fit_outputs();
            self.fit_preview_iteration = Some(0);
            self.status = Some(StatusMessage::FittingInProgress);

            #[cfg(not(target_arch = "wasm32"))]
            {
                let cancel_flag = Arc::new(AtomicBool::new(false));
                self.start_spline_fit_worker(
                    spline_family,
                    points,
                    spline_config,
                    optimizer_config.clone(),
                    initial_knot_y,
                    cancel_flag,
                );
            }

            #[cfg(target_arch = "wasm32")]
            {
                match IncrementalSplineFitRunner::new_with_initial_knot_y_and_optimizer_config(
                    &points,
                    spline_family,
                    spline_config,
                    &optimizer_config,
                    Some(initial_knot_y.as_slice()),
                ) {
                    Ok(runner) => {
                        self.wasm_fit_runner = Some(WasmFitRunner::Spline(runner));
                        self.fit_in_progress = true;
                    }
                    Err(error) => {
                        self.status = Some(StatusMessage::Error(error.to_string()));
                    }
                }
            }
            return;
        }

        let family = match selected_model.parametric_family() {
            Some(family) => family,
            None => {
                self.status = Some(StatusMessage::Error(
                    "Selected model has no parametric family".to_string(),
                ));
                return;
            }
        };

        let initial_params = match self.parse_initial_params() {
            Ok(params) => params,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        if let Err(error) = family.validate_points(&points) {
            self.status = Some(StatusMessage::Error(error.to_string()));
            return;
        }

        // Очищаем предыдущий успешный результат только когда новый запуск уже валиден.
        self.clear_fit_outputs();
        self.active_fit_points = Some(points.clone());
        self.iteration_diagnostics
            .initialize(&points, &initial_params);

        if self.iteration_delay_seconds > 0.0 {
            self.fit_preview_params = Some(initial_params.clone());
            self.fit_preview_iteration = Some(0);
        }
        self.status = Some(StatusMessage::FittingInProgress);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let cancel_flag = Arc::new(AtomicBool::new(false));
            self.start_fit_worker(
                family,
                points,
                initial_params,
                optimizer_config.clone(),
                cancel_flag,
            );
        }

        #[cfg(target_arch = "wasm32")]
        {
            match IncrementalFitRunner::new_with_optimizer_config(
                &points,
                family,
                initial_params,
                &optimizer_config,
            ) {
                Ok(runner) => {
                    self.wasm_fit_runner = Some(WasmFitRunner::Parametric(runner));
                    self.fit_in_progress = true;
                }
                Err(error) => {
                    self.active_fit_points = None;
                    self.status = Some(StatusMessage::Error(error.to_string()));
                }
            }
        }
    }
}
