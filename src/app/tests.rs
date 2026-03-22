use super::{
    CurveFitApp, IterationDiagnostics, ModelChoice, OptimizerMethod, OptimizerPreset,
    ParamInitMethod, StatusMessage, UiLanguage, data_based_params_for_family,
};
use crate::domain::{CurveFamily, CurveParams, FitResult, OptimizerConfig, Point, Points};
use egui_plot::PlotPoint;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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

    diagnostics.initialize(&points, &params);

    assert_eq!(
        diagnostics.parameter_names,
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(diagnostics.loss_mse_points, vec![[0.0, 0.0]]);
    assert_eq!(diagnostics.parameter_series.len(), 2);
    assert_eq!(diagnostics.parameter_series[0], vec![[0.0, 2.0]]);
    assert_eq!(diagnostics.parameter_series[1], vec![[0.0, 1.0]]);
}

#[test]
fn diagnostics_append_replaces_duplicate_iteration() {
    let points = line_points();
    let mut diagnostics = IterationDiagnostics::default();
    diagnostics.initialize(&points, &CurveParams::Linear { a: 2.0, b: 1.0 });

    diagnostics.append(2, 5.0, &CurveParams::Linear { a: 1.0, b: 0.0 });
    diagnostics.append(2, 3.0, &CurveParams::Linear { a: -1.5, b: 0.5 });

    assert_eq!(diagnostics.loss_mse_points.len(), 2);
    assert_eq!(diagnostics.loss_mse_points[1], [2.0, 3.0]);
    assert_eq!(diagnostics.parameter_series[0].len(), 2);
    assert_eq!(diagnostics.parameter_series[0][1], [2.0, -1.5]);
    assert_eq!(diagnostics.parameter_series[1].len(), 2);
    assert_eq!(diagnostics.parameter_series[1][1], [2.0, 0.5]);
}

#[test]
fn diagnostics_append_resets_when_family_changes() {
    let points = line_points();
    let mut diagnostics = IterationDiagnostics::default();
    diagnostics.initialize(&points, &CurveParams::Linear { a: 2.0, b: 1.0 });
    diagnostics.append(
        4,
        1.0,
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
    assert_eq!(diagnostics.loss_mse_points, vec![[4.0, 1.0]]);
    assert_eq!(diagnostics.parameter_series.len(), 3);
    assert_eq!(diagnostics.parameter_series[0], vec![[4.0, 1.0]]);
    assert_eq!(diagnostics.parameter_series[1], vec![[4.0, -2.0]]);
    assert_eq!(diagnostics.parameter_series[2], vec![[4.0, 3.0]]);
}

#[test]
fn diagnostics_append_spline_tracks_knot_parameters() {
    let mut diagnostics = IterationDiagnostics::default();

    diagnostics.append_spline(1, 2.5, &[0.5, -1.0]);
    diagnostics.append_spline(2, 1.5, &[0.75, -0.25]);

    assert_eq!(
        diagnostics.parameter_names,
        vec!["knot_y[0]".to_string(), "knot_y[1]".to_string()]
    );
    assert_eq!(diagnostics.loss_mse_points, vec![[1.0, 2.5], [2.0, 1.5]]);
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
    app.iteration_diagnostics
        .initialize(&line_points(), &CurveParams::Linear { a: 2.0, b: 1.0 });

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
    assert!(app.iteration_diagnostics.loss_mse_points.is_empty());
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
        ..Default::default()
    };

    app.clear_fit_outputs();

    assert!(cancel_flag.load(Ordering::Relaxed));
    assert!(app.fit_in_progress);
    assert!(app.discard_fit_worker_updates);
    assert!(app.fit_result.is_none());
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());
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
    assert!(app.iteration_diagnostics.loss_mse_points.is_empty());
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
