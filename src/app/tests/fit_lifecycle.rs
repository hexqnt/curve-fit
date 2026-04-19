use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn auto_refit_first_snapshot_only_initializes_baseline() {
    let mut app = make_linear_fit_app();
    app.auto_refit_enabled = true;

    assert!(app.last_right_panel_fit_snapshot.is_none());
    app.track_right_panel_fit_changes_and_maybe_refit();
    assert!(app.last_right_panel_fit_snapshot.is_some());
    assert!(!app.fit_in_progress);
    assert!(!app.auto_refit_pending_rerun);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn auto_refit_runs_fit_when_right_panel_settings_change_in_idle() {
    let mut app = make_linear_fit_app();
    app.auto_refit_enabled = true;
    app.track_right_panel_fit_changes_and_maybe_refit();

    app.optimization_loss_metric = OptimizationLossMetric::Mae;
    app.track_right_panel_fit_changes_and_maybe_refit();

    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn auto_refit_queues_rerun_when_settings_change_during_fit() {
    let mut app = make_linear_fit_app();
    app.auto_refit_enabled = true;
    app.track_right_panel_fit_changes_and_maybe_refit();
    app.run_fit();
    assert!(app.fit_in_progress);

    app.metric_quantization_enabled = true;
    app.metric_quantization_decimal_places = 2;
    app.track_right_panel_fit_changes_and_maybe_refit();

    assert!(app.fit_in_progress);
    assert!(app.auto_refit_pending_rerun);
    assert!(!matches!(app.status, Some(StatusMessage::FittingStopping)));
    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn auto_refit_pending_rerun_starts_after_fit_transitions_to_idle() {
    let mut app = make_linear_fit_app();
    app.auto_refit_enabled = true;
    app.track_right_panel_fit_changes_and_maybe_refit();
    app.run_fit();
    assert!(app.fit_in_progress);

    app.parameter_inputs[0] = "0.1".to_string();
    app.track_right_panel_fit_changes_and_maybe_refit();
    assert!(app.auto_refit_pending_rerun);

    wait_fit_completion(&mut app);
    assert!(!app.fit_in_progress);
    assert!(app.auto_refit_pending_rerun);

    app.maybe_run_pending_auto_refit();
    assert!(app.fit_in_progress);
    assert!(!app.auto_refit_pending_rerun);
    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn disabling_auto_refit_clears_pending_rerun_without_starting_fit() {
    let mut app = make_linear_fit_app();
    app.auto_refit_enabled = true;
    app.auto_refit_pending_rerun = true;

    app.auto_refit_enabled = false;
    app.maybe_run_pending_auto_refit();

    assert!(!app.auto_refit_pending_rerun);
    assert!(!app.fit_in_progress);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn run_fit_clears_pending_auto_refit_rerun_flag() {
    let mut app = make_linear_fit_app();
    app.auto_refit_pending_rerun = true;

    app.run_fit();

    assert!(app.fit_in_progress);
    assert!(!app.auto_refit_pending_rerun);
    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn run_fit_snapshots_metric_quantization_at_start() {
    let mut app = make_linear_fit_app();
    app.metric_quantization_enabled = true;
    app.metric_quantization_decimal_places = 3;

    app.run_fit();
    assert!(app.fit_in_progress);
    assert_eq!(app.fit_metric_quantization, metric_quantization(3));

    app.metric_quantization_enabled = false;
    app.metric_quantization_decimal_places = 0;
    assert_eq!(
        app.fit_metric_quantization,
        metric_quantization(3),
        "active fit must keep its startup snapshot"
    );

    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn successful_fit_starts_replay_from_first_frame() {
    let mut app = make_linear_fit_app();
    app.replay.iteration_delay_seconds = 0.0;

    app.run_fit();
    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);

    assert!(
        matches!(app.status, Some(StatusMessage::FitCompleted)),
        "status after fit: {:?}",
        app.status
    );
    assert!(!app.replay.frames.is_empty());
    assert_eq!(app.replay.selected_index, Some(0));
    assert_eq!(
        app.fit_preview_iteration,
        Some(app.replay.frames[0].iteration)
    );
    if app.replay.frames.len() > 1 {
        assert!(app.replay.autoplay);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn successful_fit_records_duration() {
    let mut app = make_linear_fit_app();

    app.run_fit();
    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);

    assert!(matches!(app.status, Some(StatusMessage::FitCompleted)));
    assert!(app.last_fit_duration.is_some());
    assert!(app.fit_started_at.is_none());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn successful_fit_with_auto_replay_disabled_selects_last_iteration() {
    let mut app = make_linear_fit_app();
    app.replay.autoplay_on_fit = false;
    app.replay.iteration_delay_seconds = 0.0;

    app.run_fit();
    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);

    assert!(
        matches!(app.status, Some(StatusMessage::FitCompleted)),
        "status after fit: {:?}",
        app.status
    );
    assert!(!app.replay.frames.is_empty());
    let last_index = app.replay.frames.len() - 1;
    let last_iteration = app.replay.frames[last_index].iteration;
    assert_eq!(app.replay.selected_index, Some(last_index));
    assert_eq!(app.fit_preview_iteration, Some(last_iteration));
    assert!(!app.replay.autoplay);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn run_fit_with_auto_replay_disabled_does_not_seed_preview_before_completion() {
    let mut app = make_linear_fit_app();
    app.replay.autoplay_on_fit = false;

    app.run_fit();

    assert!(app.fit_in_progress);
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());

    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn run_fit_with_auto_replay_enabled_does_not_seed_preview_before_completion() {
    let mut app = make_linear_fit_app();
    app.replay.autoplay_on_fit = true;

    app.run_fit();

    assert!(app.fit_in_progress);
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());

    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn finished_message_applies_buffered_parametric_trace_in_single_poll() {
    let (tx, rx) = std::sync::mpsc::channel();
    let points = line_points();
    let initial_params = CurveParams::Linear { a: 0.0, b: 0.0 };
    let mut app = CurveFitApp {
        fit_in_progress: true,
        fit_worker_rx: Some(rx),
        active_fit_points: Some(points.clone()),
        fit_run_ui_seed: Some(super::FitRunUiSeed::Parametric {
            initial_params: initial_params.clone(),
        }),
        status: Some(StatusMessage::FittingInProgress),
        replay: super::ReplayState {
            autoplay_on_fit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    app.iteration_diagnostics.initialize(
        &points,
        &initial_params,
        OptimizationLossMetric::Mse,
        MetricQuantization::Disabled,
    );
    app.upsert_parametric_replay_frame(0, initial_params);
    app.start_fit_timer();

    let trace = vec![
        super::ParametricIterationTraceEntry {
            iteration: 1,
            metrics: metrics_snapshot(2.0, 2.0, 1.4, 1.2, 1.1, 0.1, 2.0),
            params: CurveParams::Linear { a: 0.4, b: 0.8 },
        },
        super::ParametricIterationTraceEntry {
            iteration: 2,
            metrics: metrics_snapshot(0.5, 0.5, 0.7, 0.6, 0.55, 0.7, 0.9),
            params: CurveParams::Linear { a: 1.2, b: 0.5 },
        },
    ];
    tx.send(super::FitWorkerMessage::Finished {
        result: FitResult {
            family: CurveFamily::Linear,
            params: CurveParams::Linear { a: 2.0, b: 1.0 },
            mse: 0.0,
            rmse: 0.0,
            iterations: 3,
        },
        trace,
    })
    .expect("worker message must be sent");
    drop(tx);

    assert_eq!(app.replay.frames.len(), 1);
    assert_eq!(app.iteration_diagnostics.loss_points.len(), 1);

    app.poll_fit_worker(&egui::Context::default());

    assert!(!app.fit_in_progress);
    assert!(matches!(app.status, Some(StatusMessage::FitCompleted)));
    assert_eq!(app.replay.frames.len(), 4);
    assert_eq!(app.iteration_diagnostics.loss_points.len(), 4);
    assert_eq!(app.fit_preview_iteration, Some(3));
    assert!(matches!(
        app.fit_result,
        Some(FitResult {
            family: CurveFamily::Linear,
            iterations: 3,
            ..
        })
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn run_fit_keeps_replay_and_diagnostics_seeded_until_finish() {
    let mut app = make_linear_fit_app();
    app.optimizer_method = OptimizerMethod::Sgd;
    app.sgd_inputs.max_iters = 5_000;
    app.sgd_inputs.learning_rate = 1e-3;
    app.replay.autoplay_on_fit = false;

    app.run_fit();
    assert!(app.fit_in_progress);

    let seeded_replay_len = app.replay.frames.len();
    let seeded_loss_len = app.iteration_diagnostics.loss_points.len();
    let ctx = egui::Context::default();
    let mut observed_in_progress_poll = false;

    for _ in 0..200 {
        if !app.fit_in_progress {
            break;
        }
        app.poll_fit_worker(&ctx);
        if app.fit_in_progress {
            observed_in_progress_poll = true;
            assert_eq!(app.replay.frames.len(), seeded_replay_len);
            assert_eq!(app.iteration_diagnostics.loss_points.len(), seeded_loss_len);
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    assert!(
        observed_in_progress_poll,
        "fit completed too quickly to observe in-progress polling"
    );

    wait_fit_completion(&mut app);
    assert!(app.replay.frames.len() > seeded_replay_len);
    assert!(app.iteration_diagnostics.loss_points.len() > seeded_loss_len);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn stopped_fit_clears_ui_outputs_and_sets_stopped_status() {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut app = CurveFitApp {
        fit_in_progress: true,
        replay: super::ReplayState {
            autoplay_on_fit: false,
            frames: vec![
                super::ReplayFrame {
                    iteration: 0,
                    payload: ReplayFramePayload::Parametric {
                        params: CurveParams::Linear { a: 1.0, b: 0.0 },
                    },
                },
                super::ReplayFrame {
                    iteration: 11,
                    payload: ReplayFramePayload::Parametric {
                        params: CurveParams::Linear { a: 3.0, b: -1.0 },
                    },
                },
            ],
            selected_index: Some(0),
            ..Default::default()
        },
        fit_worker_rx: Some(rx),
        status: Some(StatusMessage::FittingInProgress),
        fit_preview_params: Some(CurveParams::Linear { a: 1.0, b: 0.0 }),
        fit_preview_iteration: Some(0),
        ..Default::default()
    };

    tx.send(super::FitWorkerMessage::Stopped)
        .expect("worker message must be sent");
    drop(tx);

    app.poll_fit_worker(&egui::Context::default());

    assert!(!app.fit_in_progress);
    assert!(matches!(app.status, Some(StatusMessage::FitStopped)));
    assert!(app.fit_result.is_none());
    assert!(app.spline_result.is_none());
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());
    assert!(app.replay.frames.is_empty());
    assert!(app.replay.selected_index.is_none());
    assert!(app.iteration_diagnostics.loss_points.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn spline_fit_seeds_iteration_zero_replay_frame_from_initialization() {
    let mut app = CurveFitApp {
        replay: super::ReplayState {
            autoplay_on_fit: false,
            ..Default::default()
        },
        ..make_linear_spline_fit_app()
    };

    app.run_fit();
    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);

    assert!(
        matches!(app.status, Some(StatusMessage::FitCompleted)),
        "status after fit: {:?}",
        app.status
    );
    assert!(!app.replay.frames.is_empty());
    assert_eq!(app.replay.frames[0].iteration, 0);
    match &app.replay.frames[0].payload {
        ReplayFramePayload::Spline { curve } => {
            assert!(
                !curve.is_empty(),
                "iteration zero spline replay frame must contain sampled curve"
            );
        }
        ReplayFramePayload::Parametric { .. } => {
            panic!("iteration zero replay frame for spline fit must be spline payload")
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn replay_step_seconds_does_not_block_fit_completion() {
    for replay_step in [0.0, 0.75, 1.5] {
        let mut app = make_linear_fit_app();
        app.replay.iteration_delay_seconds = replay_step;

        app.run_fit();
        assert!(app.fit_in_progress);
        wait_fit_completion(&mut app);

        assert!(app.fit_result.is_some());
        assert!(
            matches!(app.status, Some(StatusMessage::FitCompleted)),
            "status after fit: {:?}",
            app.status
        );
    }
}

#[test]
fn run_fit_invalid_input_does_not_seed_iteration_diagnostics() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Power,
        ..Default::default()
    };
    app.sync_parameter_inputs();
    app.points.text = "-1 2\n1 3\n".to_string();
    app.invalidate_points_cache();

    app.run_fit();

    assert!(matches!(app.status, Some(StatusMessage::Error(_))));
    assert!(app.iteration_diagnostics.loss_points.is_empty());
    assert!(app.iteration_diagnostics.parameter_series.is_empty());
}

#[test]
fn points_edit_parse_error_status_restores_completed_when_fixed() {
    let mut app = CurveFitApp {
        fit_result: Some(FitResult {
            family: CurveFamily::Linear,
            params: CurveParams::Linear { a: 1.0, b: 0.0 },
            mse: 0.0,
            rmse: 0.0,
            iterations: 1,
        }),
        last_fit_duration: Some(std::time::Duration::from_millis(184)),
        status: Some(StatusMessage::FitCompleted),
        ..Default::default()
    };

    app.points.text = "1 2 3\n".to_string();
    app.invalidate_points_cache();
    app.refresh_status_after_points_edit();
    assert!(matches!(
        app.status.as_ref(),
        Some(StatusMessage::Error(message)) if message.starts_with(super::POINTS_PARSE_ERROR_PREFIX)
    ));

    app.points.text = "1 2\n2 3\n".to_string();
    app.invalidate_points_cache();
    app.refresh_status_after_points_edit();
    assert!(matches!(app.status, Some(StatusMessage::FitCompleted)));
    assert_eq!(
        app.last_fit_duration,
        Some(std::time::Duration::from_millis(184))
    );
}

#[test]
fn clear_fit_outputs_resets_fit_duration_state() {
    let mut app = CurveFitApp::default();
    app.start_fit_timer();
    app.last_fit_duration = Some(std::time::Duration::from_millis(42));

    app.clear_fit_outputs();

    assert!(app.fit_started_at.is_none());
    assert!(app.last_fit_duration.is_none());
}
