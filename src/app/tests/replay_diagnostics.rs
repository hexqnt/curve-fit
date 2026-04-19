use super::*;

#[test]
fn diagnostics_initialize_stores_iteration_zero_state() {
    let points = line_points();
    let params = CurveParams::Linear { a: 2.0, b: 1.0 };
    let mut diagnostics = IterationDiagnostics::default();

    diagnostics.initialize(
        &points,
        &params,
        OptimizationLossMetric::Mse,
        MetricQuantization::Disabled,
    );

    assert_eq!(
        diagnostics.parameter_names,
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(diagnostics.loss_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.mse_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.rmse_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.mae_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.soft_l1_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.r2_abs_points, vec![[0.0, 1.0]]);
    assert_eq!(diagnostics.max_abs_error_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.parameter_series.len(), 2);
    assert_eq!(diagnostics.parameter_series[0], vec![[0.0, 2.0]]);
    assert_eq!(diagnostics.parameter_series[1], vec![[0.0, 1.0]]);
}

#[test]
fn diagnostics_append_replaces_duplicate_iteration() {
    let points = line_points();
    let mut diagnostics = IterationDiagnostics::default();
    diagnostics.initialize(
        &points,
        &CurveParams::Linear { a: 2.0, b: 1.0 },
        OptimizationLossMetric::Mse,
        MetricQuantization::Disabled,
    );

    diagnostics.append(
        2,
        metrics_snapshot(5.0, 5.0, 5.0_f64.sqrt(), 2.0, 1.75, -1.5, 3.0),
        &CurveParams::Linear { a: 1.0, b: 0.0 },
    );
    diagnostics.append(
        2,
        metrics_snapshot(3.0, 3.0, 3.0_f64.sqrt(), 1.5, 1.1, -2.0, 2.0),
        &CurveParams::Linear { a: -1.5, b: 0.5 },
    );

    assert_eq!(diagnostics.loss_points.len(), 2);
    assert_eq!(diagnostics.loss_points[1], [2.0, 3.0]);
    assert_eq!(diagnostics.mse_points[1], [2.0, 3.0]);
    assert_eq!(diagnostics.rmse_points[1], [2.0, 3.0_f64.sqrt()]);
    assert_eq!(diagnostics.mae_points[1], [2.0, 1.5]);
    assert_eq!(diagnostics.soft_l1_points[1], [2.0, 1.1]);
    assert_eq!(diagnostics.r2_abs_points[1], [2.0, 2.0]);
    assert_eq!(diagnostics.max_abs_error_points[1], [2.0, 2.0]);
    assert_eq!(diagnostics.parameter_series[0].len(), 2);
    assert_eq!(diagnostics.parameter_series[0][1], [2.0, -1.5]);
    assert_eq!(diagnostics.parameter_series[1].len(), 2);
    assert_eq!(diagnostics.parameter_series[1][1], [2.0, 0.5]);
}

#[test]
fn diagnostics_append_resets_when_family_changes() {
    let points = line_points();
    let mut diagnostics = IterationDiagnostics::default();
    diagnostics.initialize(
        &points,
        &CurveParams::Linear { a: 2.0, b: 1.0 },
        OptimizationLossMetric::Mse,
        MetricQuantization::Disabled,
    );
    diagnostics.append(
        4,
        metrics_snapshot(1.0, 1.0, 1.0, 1.0, 0.8, 0.2, 1.0),
        &CurveParams::Quadratic {
            a: 1.0,
            b: -2.0,
            c: 3.0,
        },
    );

    assert_eq!(
        diagnostics.parameter_names,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    assert_eq!(diagnostics.loss_points, vec![[4.0, 1.0]]);
    assert_eq!(diagnostics.parameter_series.len(), 3);
    assert_eq!(diagnostics.parameter_series[0], vec![[4.0, 1.0]]);
    assert_eq!(diagnostics.parameter_series[1], vec![[4.0, -2.0]]);
    assert_eq!(diagnostics.parameter_series[2], vec![[4.0, 3.0]]);
}

#[test]
fn diagnostics_append_spline_tracks_knot_parameters() {
    let mut diagnostics = IterationDiagnostics::default();

    diagnostics.append_spline(
        1,
        metrics_snapshot(2.5, 2.5, 2.5_f64.sqrt(), 1.2, 1.6, -0.25, 2.0),
        &[0.5, -1.0],
    );
    diagnostics.append_spline(
        2,
        metrics_snapshot(1.5, 1.5, 1.5_f64.sqrt(), 0.9, 1.0, 0.25, 1.4),
        &[0.75, -0.25],
    );

    assert_eq!(
        diagnostics.parameter_names,
        vec!["knot_y[0]".to_string(), "knot_y[1]".to_string()]
    );
    assert_eq!(diagnostics.loss_points, vec![[1.0, 2.5], [2.0, 1.5]]);
    assert_eq!(diagnostics.soft_l1_points, vec![[1.0, 1.6], [2.0, 1.0]]);
    assert_eq!(diagnostics.r2_abs_points, vec![[1.0, 0.25], [2.0, 0.25]]);
    assert_eq!(
        diagnostics.parameter_series[0],
        vec![[1.0, 0.5], [2.0, 0.75]]
    );
    assert_eq!(
        diagnostics.parameter_series[1],
        vec![[1.0, -1.0], [2.0, -0.25]]
    );
}

#[test]
fn replay_upsert_replaces_duplicate_iteration() {
    let mut app = CurveFitApp::default();
    app.upsert_parametric_replay_frame(3, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(3, CurveParams::Linear { a: 2.0, b: -1.0 });

    assert_eq!(app.replay.frames.len(), 1);
    assert_eq!(app.replay.frames[0].iteration, 3);
    match &app.replay.frames[0].payload {
        ReplayFramePayload::Parametric { params } => {
            assert_eq!(*params, CurveParams::Linear { a: 2.0, b: -1.0 });
        }
        ReplayFramePayload::Spline { .. } => panic!("expected parametric replay frame"),
    }
}

#[test]
fn replay_selected_iteration_matches_selected_replay_frame() {
    let mut app = CurveFitApp::default();
    app.upsert_parametric_replay_frame(2, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(9, CurveParams::Linear { a: 3.0, b: -1.0 });

    app.set_replay_selected_index(1);

    assert_eq!(app.replay_selected_iteration(), Some(9));
}

#[test]
fn replay_selected_iteration_returns_none_without_selection() {
    let mut app = CurveFitApp::default();
    app.upsert_parametric_replay_frame(2, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(9, CurveParams::Linear { a: 3.0, b: -1.0 });

    assert_eq!(app.replay_selected_iteration(), None);
}

#[test]
fn replay_start_from_beginning_selects_first_frame_and_enables_autoplay() {
    let mut app = CurveFitApp::default();
    app.upsert_parametric_replay_frame(1, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(2, CurveParams::Linear { a: 2.0, b: 0.5 });

    app.start_replay_from_beginning();

    assert_eq!(app.replay.selected_index, Some(0));
    assert!(app.replay.autoplay);
    assert_eq!(app.fit_preview_iteration, Some(1));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 1.0, b: 0.0 })
    );
}

#[test]
fn replay_start_from_beginning_respects_auto_replay_toggle() {
    let mut app = CurveFitApp {
        replay: super::ReplayState {
            autoplay_on_fit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    app.upsert_parametric_replay_frame(1, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(2, CurveParams::Linear { a: 2.0, b: 0.5 });

    app.start_replay_from_beginning();

    assert_eq!(app.replay.selected_index, Some(0));
    assert!(!app.replay.autoplay);
    assert_eq!(app.fit_preview_iteration, Some(1));
}

#[test]
fn replay_finalize_after_fit_completion_uses_last_frame_when_auto_replay_is_disabled() {
    let mut app = CurveFitApp {
        replay: super::ReplayState {
            autoplay_on_fit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    app.upsert_parametric_replay_frame(1, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(9, CurveParams::Linear { a: 3.0, b: -1.0 });

    app.finalize_replay_after_fit_completion();

    assert_eq!(app.replay.selected_index, Some(1));
    assert_eq!(app.fit_preview_iteration, Some(9));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 3.0, b: -1.0 })
    );
    assert!(!app.replay.autoplay);
}

#[test]
fn replay_finalize_after_fit_stopped_uses_last_frame_when_auto_replay_is_disabled() {
    let mut app = CurveFitApp {
        replay: super::ReplayState {
            autoplay_on_fit: false,
            autoplay: true,
            ..Default::default()
        },
        ..Default::default()
    };
    app.upsert_parametric_replay_frame(0, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(5, CurveParams::Linear { a: 2.5, b: -0.5 });
    app.set_replay_selected_index(0);

    app.finalize_replay_after_fit_stopped();

    assert_eq!(app.replay.selected_index, Some(1));
    assert_eq!(app.fit_preview_iteration, Some(5));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 2.5, b: -0.5 })
    );
    assert!(!app.replay.autoplay);
}

#[test]
fn replay_select_nearest_iteration_updates_parametric_preview() {
    let mut app = CurveFitApp::default();
    app.upsert_parametric_replay_frame(2, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(10, CurveParams::Linear { a: 4.0, b: -2.0 });

    app.select_nearest_replay_iteration(8);

    assert_eq!(app.fit_preview_iteration, Some(10));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 4.0, b: -2.0 })
    );
    assert!(app.spline_plot_curve.is_none());
}

#[test]
fn replay_select_nearest_iteration_updates_spline_preview() {
    let mut app = CurveFitApp::default();
    app.upsert_spline_replay_frame(1, vec![PlotPoint::new(0.0, 0.0), PlotPoint::new(1.0, 1.0)]);
    app.upsert_spline_replay_frame(5, vec![PlotPoint::new(0.0, 1.0), PlotPoint::new(1.0, 2.0)]);

    app.select_nearest_replay_iteration(4);

    assert_eq!(app.fit_preview_iteration, Some(5));
    assert!(app.fit_preview_params.is_none());
    let spline_curve = app
        .spline_plot_curve
        .as_ref()
        .expect("spline curve preview must be set");
    assert_eq!(spline_curve.len(), 2);
    assert_eq!(spline_curve[0], PlotPoint::new(0.0, 1.0));
    assert_eq!(spline_curve[1], PlotPoint::new(1.0, 2.0));
}

#[test]
fn replay_spline_preview_reuses_arc_storage() {
    let mut app = CurveFitApp::default();
    app.upsert_spline_replay_frame(1, vec![PlotPoint::new(0.0, 0.5), PlotPoint::new(1.0, 1.5)]);

    app.set_replay_selected_index(0);

    let preview_curve = app
        .spline_plot_curve
        .as_ref()
        .expect("spline preview must be available");
    let stored_curve = match &app.replay.frames[0].payload {
        ReplayFramePayload::Spline { curve } => curve,
        ReplayFramePayload::Parametric { .. } => panic!("expected spline replay payload"),
    };
    assert!(std::sync::Arc::ptr_eq(preview_curve, stored_curve));
}
