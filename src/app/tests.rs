use super::{
    CurveFitApp, IterationDiagnostics, ModelChoice, OptimizerMethod, OptimizerPreset,
    ParamInitMethod, ReplayFramePayload, StatusMessage, UiLanguage, data_based_params_for_family,
};
use crate::domain::{CurveFamily, CurveParams, FitResult, OptimizerConfig, Point, Points};
use crate::fit::{IterationMetricSnapshot, OptimizationLossMetric};
use eframe::egui;
use egui_plot::PlotPoint;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

fn line_points() -> Points {
    Points::try_from(vec![
        Point::try_new(0.0, 1.0).expect("x/y must be finite"),
        Point::try_new(1.0, 3.0).expect("x/y must be finite"),
    ])
    .expect("two points are enough for Points")
}

fn points_from_pairs(pairs: &[(f64, f64)]) -> Points {
    let points = pairs
        .iter()
        .copied()
        .map(|(x, y)| Point::try_new(x, y).expect("x/y must be finite"))
        .collect::<Vec<_>>();
    Points::try_from(points).expect("points must satisfy minimum size")
}

fn assert_approx_eq(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

fn metrics_snapshot(
    loss: f64,
    mse: f64,
    rmse: f64,
    mae: f64,
    soft_l1: f64,
    r2: f64,
    max_abs_error: f64,
) -> IterationMetricSnapshot {
    IterationMetricSnapshot {
        loss,
        mse,
        rmse,
        mae,
        soft_l1,
        r2,
        max_abs_error,
    }
}

#[test]
fn ui_language_from_locale_tag_uses_russian_for_ru_tags() {
    assert_eq!(UiLanguage::from_locale_tag("ru"), UiLanguage::Russian);
    assert_eq!(UiLanguage::from_locale_tag("ru-RU"), UiLanguage::Russian);
    assert_eq!(
        UiLanguage::from_locale_tag("ru_RU.UTF-8"),
        UiLanguage::Russian
    );
    assert_eq!(
        UiLanguage::from_locale_tag("ru-RU,en-US;q=0.9"),
        UiLanguage::Russian
    );
}

#[test]
fn ui_language_from_locale_tag_uses_english_for_other_tags() {
    assert_eq!(UiLanguage::from_locale_tag("en-US"), UiLanguage::English);
    assert_eq!(
        UiLanguage::from_locale_tag("de_DE.UTF-8"),
        UiLanguage::English
    );
    assert_eq!(UiLanguage::from_locale_tag(""), UiLanguage::English);
}

#[test]
fn optimizer_config_matches_selected_method() {
    let mut app = CurveFitApp {
        optimizer_method: OptimizerMethod::Lbfgs,
        ..Default::default()
    };
    assert!(matches!(
        app.optimizer_config(),
        Ok(OptimizerConfig::Lbfgs(_))
    ));

    app.optimizer_method = OptimizerMethod::NelderMead;
    assert!(matches!(
        app.optimizer_config(),
        Ok(OptimizerConfig::NelderMead(_))
    ));

    app.optimizer_method = OptimizerMethod::SteepestDescent;
    assert!(matches!(
        app.optimizer_config(),
        Ok(OptimizerConfig::SteepestDescent(_))
    ));
}

#[test]
fn optimization_metric_defaults_to_mse() {
    let app = CurveFitApp::default();
    assert_eq!(app.optimization_loss_metric, OptimizationLossMetric::Mse);
    assert_eq!(app.fit_loss_metric, OptimizationLossMetric::Mse);
    assert!(app.replay_autoplay_on_fit);
    assert!(app.diagnostics_hide_non_loss_by_default_pending);
}

#[test]
fn spray_rate_limiter_emits_initial_batch_without_waiting() {
    let mut app = CurveFitApp {
        spray_points_per_second: 300,
        ..Default::default()
    };

    let emitted = app.next_spray_points_to_add(Instant::now());

    assert_eq!(emitted, 5);
    assert_approx_eq(app.spray_points_budget, 0.0, 1e-12);
}

#[test]
fn spray_rate_limiter_accumulates_fractional_points_between_frames() {
    let start = Instant::now();
    let mut app = CurveFitApp {
        spray_points_per_second: 100,
        spray_last_emit_at: Some(start),
        ..Default::default()
    };

    let first = app.next_spray_points_to_add(start + Duration::from_millis(5));
    let second = app.next_spray_points_to_add(start + Duration::from_millis(10));

    assert_eq!(first, 0);
    assert_eq!(second, 1);
    assert_approx_eq(app.spray_points_budget, 0.0, 1e-12);
}

#[test]
fn spray_rate_limiter_keeps_rate_on_large_frame_gaps() {
    let now = Instant::now();
    let mut app = CurveFitApp {
        spray_points_per_second: 300,
        spray_last_emit_at: Some(now - Duration::from_secs(1)),
        ..Default::default()
    };

    let emitted = app.next_spray_points_to_add(now);

    assert_eq!(emitted, 300);
    assert_approx_eq(app.spray_points_budget, 0.0, 1e-12);
}

#[test]
fn optimizer_presets_are_stored_per_method() {
    let mut app = CurveFitApp {
        optimizer_method: OptimizerMethod::Lbfgs,
        ..Default::default()
    };
    app.apply_selected_optimizer_preset(OptimizerPreset::Fast);

    app.optimizer_method = OptimizerMethod::NelderMead;
    app.apply_selected_optimizer_preset(OptimizerPreset::Precise);

    app.optimizer_method = OptimizerMethod::SteepestDescent;
    app.apply_selected_optimizer_preset(OptimizerPreset::Fast);

    app.optimizer_method = OptimizerMethod::Lbfgs;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Fast);

    app.optimizer_method = OptimizerMethod::NelderMead;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Precise);

    app.optimizer_method = OptimizerMethod::SteepestDescent;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Fast);
}

#[test]
fn optimizer_preset_changes_active_config_values() {
    let mut app = CurveFitApp {
        optimizer_method: OptimizerMethod::NelderMead,
        ..Default::default()
    };
    app.apply_selected_optimizer_preset(OptimizerPreset::Fast);
    let fast_config = app
        .optimizer_config()
        .expect("optimizer config must be valid");

    app.apply_selected_optimizer_preset(OptimizerPreset::Precise);
    let precise_config = app
        .optimizer_config()
        .expect("optimizer config must be valid");

    let (fast_max_iters, precise_max_iters) = match (fast_config, precise_config) {
        (OptimizerConfig::NelderMead(fast), OptimizerConfig::NelderMead(precise)) => {
            (fast.max_iters, precise.max_iters)
        }
        _ => panic!("Nelder-Mead must remain active"),
    };
    assert!(precise_max_iters > fast_max_iters);
}

#[test]
fn diagnostics_initialize_stores_iteration_zero_state() {
    let points = line_points();
    let params = CurveParams::Linear { a: 2.0, b: 1.0 };
    let mut diagnostics = IterationDiagnostics::default();

    diagnostics.initialize(&points, &params, OptimizationLossMetric::Mse);

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

    assert_eq!(app.replay_frames.len(), 1);
    assert_eq!(app.replay_frames[0].iteration, 3);
    match &app.replay_frames[0].payload {
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

    assert_eq!(app.replay_selected_index, Some(0));
    assert!(app.replay_autoplay);
    assert_eq!(app.fit_preview_iteration, Some(1));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 1.0, b: 0.0 })
    );
}

#[test]
fn replay_start_from_beginning_respects_auto_replay_toggle() {
    let mut app = CurveFitApp {
        replay_autoplay_on_fit: false,
        ..Default::default()
    };
    app.upsert_parametric_replay_frame(1, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(2, CurveParams::Linear { a: 2.0, b: 0.5 });

    app.start_replay_from_beginning();

    assert_eq!(app.replay_selected_index, Some(0));
    assert!(!app.replay_autoplay);
    assert_eq!(app.fit_preview_iteration, Some(1));
}

#[test]
fn replay_finalize_after_fit_completion_uses_last_frame_when_auto_replay_is_disabled() {
    let mut app = CurveFitApp {
        replay_autoplay_on_fit: false,
        ..Default::default()
    };
    app.upsert_parametric_replay_frame(1, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(9, CurveParams::Linear { a: 3.0, b: -1.0 });

    app.finalize_replay_after_fit_completion();

    assert_eq!(app.replay_selected_index, Some(1));
    assert_eq!(app.fit_preview_iteration, Some(9));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 3.0, b: -1.0 })
    );
    assert!(!app.replay_autoplay);
}

#[test]
fn replay_finalize_after_fit_stopped_uses_last_frame_when_auto_replay_is_disabled() {
    let mut app = CurveFitApp {
        replay_autoplay_on_fit: false,
        replay_autoplay: true,
        ..Default::default()
    };
    app.upsert_parametric_replay_frame(0, CurveParams::Linear { a: 1.0, b: 0.0 });
    app.upsert_parametric_replay_frame(5, CurveParams::Linear { a: 2.5, b: -0.5 });
    app.set_replay_selected_index(0);

    app.finalize_replay_after_fit_stopped();

    assert_eq!(app.replay_selected_index, Some(1));
    assert_eq!(app.fit_preview_iteration, Some(5));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 2.5, b: -0.5 })
    );
    assert!(!app.replay_autoplay);
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
    let stored_curve = match &app.replay_frames[0].payload {
        ReplayFramePayload::Spline { curve } => curve,
        ReplayFramePayload::Parametric { .. } => panic!("expected spline replay payload"),
    };
    assert!(std::sync::Arc::ptr_eq(preview_curve, stored_curve));
}

#[test]
fn param_init_method_support_matrix_is_correct() {
    assert!(ParamInitMethod::Default.is_supported_for_family(CurveFamily::Arrhenius));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Linear));
    assert!(ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::Power));

    assert!(!ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Arrhenius));
    assert!(!ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::FourPl));
}

#[test]
fn data_based_polynomial_initialization_sets_only_linear_terms() {
    let points = points_from_pairs(&[(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]);
    let params =
        data_based_params_for_family(CurveFamily::Quartic, &points).expect("must initialize");
    let values = params.values();

    assert_eq!(values.len(), 5);
    assert_approx_eq(values[0], 0.0, 1e-12);
    assert_approx_eq(values[1], 0.0, 1e-12);
    assert_approx_eq(values[2], 0.0, 1e-12);
    assert_approx_eq(values[3], 2.0, 1e-12);
    assert_approx_eq(values[4], 1.0, 1e-12);
}

#[test]
fn data_based_power_initialization_rejects_non_positive_y() {
    let points = points_from_pairs(&[(1.0, 0.0), (2.0, 2.0)]);
    let error = data_based_params_for_family(CurveFamily::Power, &points)
        .expect_err("y <= 0 must be rejected for Power data-based init");

    assert!(error.contains("requires y > 0"));
}

#[test]
fn randomized_initialization_stays_within_expected_range() {
    let mut app = CurveFitApp::default();
    let params = app
        .build_randomized_initial_params(CurveFamily::Gaussian)
        .expect("randomized init must succeed");

    let values = params.values();
    assert_eq!(values.len(), CurveFamily::Gaussian.parameter_count());
    for value in values {
        assert!((-1.0..=1.0).contains(&value));
    }
}

#[test]
fn apply_param_init_updates_inputs_and_clears_fit_state() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Polynomial,
        polynomial_degree: 3,
        ..Default::default()
    };
    app.sync_parameter_inputs();
    app.points_text = "0 1\n1 3\n2 5\n3 7\n".to_string();
    app.invalidate_points_cache();

    app.fit_result = Some(FitResult {
        family: CurveFamily::Cubic,
        params: CurveParams::Cubic {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
        },
        mse: 1.0,
        rmse: 1.0,
        iterations: 1,
    });
    app.fit_preview_params = Some(CurveParams::Cubic {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
    });
    app.fit_preview_iteration = Some(1);
    app.iteration_diagnostics.initialize(
        &line_points(),
        &CurveParams::Linear { a: 2.0, b: 1.0 },
        OptimizationLossMetric::Mse,
    );

    app.apply_param_init_method(ParamInitMethod::DataBased);

    let values = app
        .parameter_inputs
        .iter()
        .map(|value| value.parse::<f64>().expect("parameter must parse"))
        .collect::<Vec<_>>();

    assert_eq!(values.len(), 4);
    assert_approx_eq(values[0], 0.0, 1e-12);
    assert_approx_eq(values[1], 0.0, 1e-12);
    assert_approx_eq(values[2], 2.0, 1e-12);
    assert_approx_eq(values[3], 1.0, 1e-12);
    assert!(app.fit_result.is_none());
    assert!(app.spline_result.is_none());
    assert!(app.spline_plot_curve.is_none());
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());
    assert!(app.iteration_diagnostics.loss_points.is_empty());
}

#[test]
fn apply_fitted_param_init_updates_inputs_and_clears_fit_state() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Polynomial,
        polynomial_degree: 3,
        ..Default::default()
    };
    app.sync_parameter_inputs();

    app.fit_result = Some(FitResult {
        family: CurveFamily::Cubic,
        params: CurveParams::Cubic {
            a: 1.5,
            b: -0.25,
            c: 2.0,
            d: 3.0,
        },
        mse: 0.1,
        rmse: 0.31622776601683794,
        iterations: 15,
    });
    app.fit_preview_params = Some(CurveParams::Cubic {
        a: 1.5,
        b: -0.25,
        c: 2.0,
        d: 3.0,
    });
    app.fit_preview_iteration = Some(15);
    app.iteration_diagnostics.initialize(
        &line_points(),
        &CurveParams::Linear { a: 2.0, b: 1.0 },
        OptimizationLossMetric::Mse,
    );

    app.apply_fitted_param_init();

    let values = app
        .parameter_inputs
        .iter()
        .map(|value| value.parse::<f64>().expect("parameter must parse"))
        .collect::<Vec<_>>();

    assert_eq!(values.len(), 4);
    assert_approx_eq(values[0], 1.5, 1e-12);
    assert_approx_eq(values[1], -0.25, 1e-12);
    assert_approx_eq(values[2], 2.0, 1e-12);
    assert_approx_eq(values[3], 3.0, 1e-12);
    assert!(app.fit_result.is_none());
    assert!(app.spline_result.is_none());
    assert!(app.spline_plot_curve.is_none());
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());
    assert!(app.iteration_diagnostics.loss_points.is_empty());
    assert!(matches!(app.status, Some(StatusMessage::Ready)));
}

#[test]
fn apply_fitted_param_init_sets_error_when_fit_is_unavailable() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Polynomial,
        polynomial_degree: 2,
        ..Default::default()
    };
    app.sync_parameter_inputs();

    app.apply_fitted_param_init();

    assert!(matches!(app.status, Some(StatusMessage::Error(_))));
}

#[test]
fn apply_param_init_sets_error_status_on_failure() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Power,
        ..Default::default()
    };
    app.sync_parameter_inputs();
    app.points_text = "1 0\n2 2\n".to_string();
    app.invalidate_points_cache();

    app.apply_param_init_method(ParamInitMethod::DataBased);

    assert!(matches!(app.status, Some(StatusMessage::Error(_))));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn clear_fit_outputs_requests_cancellation_without_dropping_progress_state() {
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut app = CurveFitApp {
        fit_in_progress: true,
        fit_cancel_flag: Some(cancel_flag.clone()),
        status: Some(StatusMessage::FittingInProgress),
        fit_result: Some(FitResult {
            family: CurveFamily::Linear,
            params: CurveParams::Linear { a: 1.0, b: 0.0 },
            mse: 0.0,
            rmse: 0.0,
            iterations: 1,
        }),
        replay_frames: vec![super::ReplayFrame {
            iteration: 0,
            payload: ReplayFramePayload::Parametric {
                params: CurveParams::Linear { a: 1.0, b: 0.0 },
            },
        }],
        replay_selected_index: Some(0),
        replay_autoplay: true,
        ..Default::default()
    };

    app.clear_fit_outputs();

    assert!(cancel_flag.load(Ordering::Relaxed));
    assert!(app.fit_in_progress);
    assert!(app.discard_fit_worker_updates);
    assert!(app.fit_result.is_none());
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());
    assert!(app.replay_frames.is_empty());
    assert!(app.replay_selected_index.is_none());
    assert!(!app.replay_autoplay);
}

#[cfg(not(target_arch = "wasm32"))]
fn make_linear_fit_app() -> CurveFitApp {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Polynomial,
        polynomial_degree: 1,
        ..Default::default()
    };
    app.sync_parameter_inputs();
    app.points_text = "0 1\n1 3\n2 5\n3 7\n".to_string();
    app.invalidate_points_cache();
    app
}

#[cfg(not(target_arch = "wasm32"))]
fn make_linear_spline_fit_app() -> CurveFitApp {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::LinearSpline,
        ..Default::default()
    };
    app.points_text = "0 1\n1 3\n2 5\n3 7\n4 9\n5 11\n6 13\n7 15\n8 17\n9 19\n".to_string();
    app.invalidate_points_cache();
    app
}

#[cfg(not(target_arch = "wasm32"))]
fn wait_fit_completion(app: &mut CurveFitApp) {
    let ctx = egui::Context::default();
    for _ in 0..20_000 {
        app.poll_fit_worker(&ctx);
        if !app.fit_in_progress {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("fit did not complete in time");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn successful_fit_starts_replay_from_first_frame() {
    let mut app = make_linear_fit_app();
    app.iteration_delay_seconds = 0.0;

    app.run_fit();
    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);

    assert!(
        matches!(app.status, Some(StatusMessage::FitCompleted)),
        "status after fit: {:?}",
        app.status
    );
    assert!(!app.replay_frames.is_empty());
    assert_eq!(app.replay_selected_index, Some(0));
    assert_eq!(
        app.fit_preview_iteration,
        Some(app.replay_frames[0].iteration)
    );
    if app.replay_frames.len() > 1 {
        assert!(app.replay_autoplay);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn successful_fit_with_auto_replay_disabled_selects_last_iteration() {
    let mut app = make_linear_fit_app();
    app.replay_autoplay_on_fit = false;
    app.iteration_delay_seconds = 0.0;

    app.run_fit();
    assert!(app.fit_in_progress);
    wait_fit_completion(&mut app);

    assert!(
        matches!(app.status, Some(StatusMessage::FitCompleted)),
        "status after fit: {:?}",
        app.status
    );
    assert!(!app.replay_frames.is_empty());
    let last_index = app.replay_frames.len() - 1;
    let last_iteration = app.replay_frames[last_index].iteration;
    assert_eq!(app.replay_selected_index, Some(last_index));
    assert_eq!(app.fit_preview_iteration, Some(last_iteration));
    assert!(!app.replay_autoplay);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn run_fit_with_auto_replay_disabled_does_not_seed_preview_before_completion() {
    let mut app = make_linear_fit_app();
    app.replay_autoplay_on_fit = false;

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
    app.replay_autoplay_on_fit = true;

    app.run_fit();

    assert!(app.fit_in_progress);
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());

    wait_fit_completion(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn stopped_fit_with_auto_replay_disabled_selects_last_iteration() {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut app = CurveFitApp {
        fit_in_progress: true,
        replay_autoplay_on_fit: false,
        fit_worker_rx: Some(rx),
        status: Some(StatusMessage::FittingInProgress),
        replay_frames: vec![
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
        replay_selected_index: Some(0),
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
    assert_eq!(app.replay_selected_index, Some(1));
    assert_eq!(app.fit_preview_iteration, Some(11));
    assert_eq!(
        app.fit_preview_params,
        Some(CurveParams::Linear { a: 3.0, b: -1.0 })
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn spline_fit_seeds_iteration_zero_replay_frame_from_initialization() {
    let mut app = CurveFitApp {
        replay_autoplay_on_fit: false,
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
    assert!(!app.replay_frames.is_empty());
    assert_eq!(app.replay_frames[0].iteration, 0);
    match &app.replay_frames[0].payload {
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
        app.iteration_delay_seconds = replay_step;

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
    app.points_text = "-1 2\n1 3\n".to_string();
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
        status: Some(StatusMessage::FitCompleted),
        ..Default::default()
    };

    app.points_text = "1 2 3\n".to_string();
    app.invalidate_points_cache();
    app.refresh_status_after_points_edit();
    assert!(matches!(
        app.status.as_ref(),
        Some(StatusMessage::Error(message)) if message.starts_with(super::POINTS_PARSE_ERROR_PREFIX)
    ));

    app.points_text = "1 2\n2 3\n".to_string();
    app.invalidate_points_cache();
    app.refresh_status_after_points_edit();
    assert!(matches!(app.status, Some(StatusMessage::FitCompleted)));
}

#[test]
fn fill_points_with_residuals_replaces_points_text_and_pushes_undo() {
    let mut app = CurveFitApp {
        points_text: "0 1\n1 2\n".to_string(),
        residual_plot_points: vec![PlotPoint::new(0.0, -0.5), PlotPoint::new(1.0, 0.25)],
        ..Default::default()
    };

    app.fill_points_with_residuals();

    assert_eq!(
        app.points_text,
        "0.00000000 -0.50000000\n1.00000000 0.25000000\n"
    );
    assert_eq!(app.points_undo_stack, vec!["0 1\n1 2\n".to_string()]);
    assert!(app.points_redo_stack.is_empty());
}

#[test]
fn fill_points_with_residuals_is_noop_when_residuals_are_absent() {
    let mut app = CurveFitApp {
        points_text: "0 1\n1 2\n".to_string(),
        ..Default::default()
    };

    app.fill_points_with_residuals();

    assert_eq!(app.points_text, "0 1\n1 2\n");
    assert!(app.points_undo_stack.is_empty());
    assert!(app.points_redo_stack.is_empty());
}
