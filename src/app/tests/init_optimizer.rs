use super::*;

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

    app.optimizer_method = OptimizerMethod::NewtonCg;
    assert!(matches!(
        app.optimizer_config(),
        Ok(OptimizerConfig::NewtonCg(_))
    ));

    app.optimizer_method = OptimizerMethod::Sgd;
    assert!(matches!(
        app.optimizer_config(),
        Ok(OptimizerConfig::Sgd(_))
    ));

    app.optimizer_method = OptimizerMethod::Adam;
    assert!(matches!(
        app.optimizer_config(),
        Ok(OptimizerConfig::Adam(_))
    ));
}

#[test]
fn optimization_metric_defaults_to_mse() {
    let app = CurveFitApp::default();
    assert_eq!(app.optimization_loss_metric, OptimizationLossMetric::Mse);
    assert_eq!(app.fit_loss_metric, OptimizationLossMetric::Mse);
    assert!(!app.metric_quantization_enabled);
    assert!(!app.auto_refit_enabled);
    assert!(!app.auto_refit_pending_rerun);
    assert!(app.last_right_panel_fit_snapshot.is_none());
    assert_eq!(
        app.metric_quantization_decimal_places,
        super::DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES
    );
    assert_eq!(app.fit_metric_quantization, MetricQuantization::Disabled);
    assert!(app.replay.autoplay_on_fit);
    assert_eq!(app.panel.diagnostics_tab, DiagnosticsTab::Loss);
    assert!(app.panel.diagnostics_hide_non_loss_by_default_pending);
}

#[test]
fn update_parametric_metrics_applies_quantization_and_keeps_raw_residual_plot() {
    let points = points_from_pairs(&[(0.0, 1.225), (1.0, -1.225)]);
    let params = CurveParams::Linear { a: 0.0, b: 0.0 };
    let mut app = CurveFitApp {
        fit_loss_metric: OptimizationLossMetric::Mse,
        fit_metric_quantization: metric_quantization(2),
        ..Default::default()
    };

    app.update_parametric_result_metrics(&points, &params);

    let metrics = app
        .result_metrics
        .expect("result metrics must be computed for parametric fit");
    assert_approx_eq(metrics.mse, 1.5129, 1e-12);
    assert_approx_eq(metrics.rmse, 1.23, 1e-12);
    assert_approx_eq(metrics.mae, 1.23, 1e-12);
    assert_approx_eq(metrics.r2, 0.0, 1e-12);
    assert_approx_eq(metrics.max_abs_error, 1.23, 1e-12);

    assert_eq!(app.residual_plot_points.len(), 2);
    assert_approx_eq(app.residual_plot_points[0].x, 0.0, 1e-12);
    assert_approx_eq(app.residual_plot_points[0].y, -1.225, 1e-12);
    assert_approx_eq(app.residual_plot_points[1].x, 1.0, 1e-12);
    assert_approx_eq(app.residual_plot_points[1].y, 1.225, 1e-12);
}

#[test]
fn fit_export_parametric_record_includes_expected_payload() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Polynomial,
        polynomial_degree: 1,
        fit_loss_metric: OptimizationLossMetric::SoftL1,
        fit_metric_quantization: metric_quantization(3),
        fit_optimizer_method: OptimizerMethod::Adam,
        last_fit_duration: Some(std::time::Duration::from_millis(184)),
        result_metrics: Some(super::ExtendedMetrics {
            mse: 0.01,
            rmse: 0.1,
            mae: 0.08,
            r2: 0.99,
            max_abs_error: 0.2,
        }),
        ..Default::default()
    };
    let result = FitResult {
        family: CurveFamily::Linear,
        params: CurveParams::Linear { a: 2.0, b: 1.0 },
        mse: 0.01,
        rmse: 0.1,
        iterations: 42,
    };

    app.store_parametric_fit_export_record(&result, 2);
    let json = app
        .build_fit_export_json_pretty()
        .expect("fit export JSON must be built");
    let value: serde_json::Value =
        serde_json::from_str(&json).expect("fit export JSON must be valid");

    assert!(
        chrono::DateTime::parse_from_rfc3339(
            value["fitted_at"]
                .as_str()
                .expect("timestamp must be present"),
        )
        .is_ok(),
        "fit timestamp must be valid RFC3339",
    );
    assert_eq!(value["model"]["selected"]["id"], "polynomial");
    assert_eq!(value["model"]["selected"]["name"], "Polynomial");
    assert_eq!(value["model"]["fitted"]["id"], "linear");
    assert_eq!(value["model"]["fitted"]["name"], "Linear");
    assert_eq!(value["point_count"], 2);
    assert_eq!(value["optimizer"]["method"]["id"], "adam");
    assert_eq!(value["optimizer"]["method"]["name"], "Adam");
    assert_eq!(value["optimizer"]["loss_metric"]["id"], "soft_l1");
    assert_eq!(value["optimizer"]["loss_metric"]["name"], "Soft L1");
    assert_eq!(value["optimizer"]["metric_quantization_decimal_places"], 3);
    assert_eq!(value["convergence"]["iterations"], 42);
    assert_eq!(value["convergence"]["duration_ms"], 184);
    assert_eq!(value["metrics"]["mse"], 0.01);
    assert_eq!(value["metrics"]["rmse"], 0.1);
    assert_eq!(value["metrics"]["mae"], 0.08);
    assert_eq!(value["metrics"]["r2"], 0.99);
    assert_eq!(value["metrics"]["max_abs_error"], 0.2);
    assert_eq!(value["result"]["kind"], "parametric");
    assert_eq!(value["result"]["parameter_count"], 2);
    assert!(
        value["result"]
            .as_object()
            .expect("result must be object")
            .get("family")
            .is_none(),
        "resolved model should not be duplicated inside parametric result",
    );
    let params = value["result"]["parameters"]
        .as_array()
        .expect("parameters must be an array");
    assert_eq!(params.len(), 2);
    assert_eq!(params[0]["name"], "a");
    assert_eq!(params[0]["value"], 2.0);
    assert_eq!(params[1]["name"], "b");
    assert_eq!(params[1]["value"], 1.0);
}

#[test]
fn fit_export_rational_record_uses_unified_selected_id_and_degree_specific_fitted_id() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Rational,
        rational_degree: 5,
        fit_loss_metric: OptimizationLossMetric::Mse,
        fit_optimizer_method: OptimizerMethod::Lbfgs,
        ..Default::default()
    };
    let result = FitResult {
        family: CurveFamily::Rational55,
        params: CurveParams::Rational55 {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.2,
            e: 0.8,
            f: 0.1,
            g: 0.04,
            h: 0.01,
            i: 0.0,
            j: 0.0,
            k: 0.0,
        },
        mse: 0.01,
        rmse: 0.1,
        iterations: 12,
    };

    app.store_parametric_fit_export_record(&result, 12);
    let json = app
        .build_fit_export_json_pretty()
        .expect("fit export JSON must be built");
    let value: serde_json::Value =
        serde_json::from_str(&json).expect("fit export JSON must be valid");

    assert_eq!(value["model"]["selected"]["id"], "rational");
    assert_eq!(value["model"]["fitted"]["id"], "rational_55");
}

#[test]
fn fit_export_spline_record_includes_expected_payload() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::LinearSpline,
        fit_loss_metric: OptimizationLossMetric::Mae,
        fit_optimizer_method: OptimizerMethod::Lbfgs,
        ..Default::default()
    };
    let result = SplineResult {
        knots: vec![[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]],
        curve: vec![[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]],
        mse: 0.01,
        rmse: 0.1,
        mae: 0.09,
        r2: 0.96,
        max_abs_error: 0.2,
        residuals: vec![[0.0, 0.1], [1.0, -0.1]],
        iterations: 9,
    };

    app.store_spline_fit_export_record(&result, 2);
    let json = app
        .build_fit_export_json_pretty()
        .expect("fit export JSON must be built");
    let value: serde_json::Value =
        serde_json::from_str(&json).expect("fit export JSON must be valid");

    assert_eq!(value["model"]["selected"]["id"], "linear_spline");
    assert_eq!(value["model"]["selected"]["name"], "Linear Spline");
    assert_eq!(value["model"]["fitted"]["id"], "linear_spline");
    assert_eq!(value["model"]["fitted"]["name"], "Linear Spline");
    assert_eq!(value["point_count"], 2);
    assert_eq!(value["optimizer"]["method"]["id"], "lbfgs");
    assert_eq!(value["optimizer"]["loss_metric"]["id"], "mae");
    assert_eq!(
        value["optimizer"]["loss_metric"]["name"],
        "Mean Absolute Error (L1)"
    );
    assert!(
        value["optimizer"]
            .as_object()
            .expect("optimizer must be object")
            .get("metric_quantization_decimal_places")
            .is_none(),
        "disabled quantization should not be serialized",
    );
    assert_eq!(value["convergence"]["iterations"], 9);
    assert!(
        value["convergence"]
            .as_object()
            .expect("convergence must be object")
            .get("duration_ms")
            .is_none(),
        "missing fit duration should not be serialized",
    );
    assert_eq!(value["result"]["kind"], "spline");
    assert_eq!(value["result"]["knot_count"], 3);
    assert!(
        value["result"]
            .as_object()
            .expect("result must be object")
            .get("spline")
            .is_none(),
        "resolved model should not be duplicated inside spline result",
    );
    let knots = value["result"]["knots"]
        .as_array()
        .expect("knots must be an array");
    assert_eq!(knots.len(), 3);
    assert_eq!(knots[0]["x"], 0.0);
    assert_eq!(knots[0]["y"], 1.0);
    assert_eq!(value["metrics"]["mae"], 0.09);
}

#[test]
fn fit_export_json_errors_when_record_is_absent() {
    let app = CurveFitApp::default();
    let error = app
        .build_fit_export_json_pretty()
        .expect_err("export without result must fail");
    assert!(error.contains("No fit export data is available"));
}

#[test]
fn fit_export_parametric_without_extended_metrics_serializes_only_basic_metrics() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Power,
        fit_loss_metric: OptimizationLossMetric::Mse,
        fit_optimizer_method: OptimizerMethod::Lbfgs,
        result_metrics: None,
        ..Default::default()
    };
    let result = FitResult {
        family: CurveFamily::Power,
        params: CurveParams::Power { a: 2.5, b: 1.2 },
        mse: 0.25,
        rmse: 0.5,
        iterations: 11,
    };

    app.store_parametric_fit_export_record(&result, 2);
    let json = app
        .build_fit_export_json_pretty()
        .expect("fit export JSON must be built");
    let value: serde_json::Value =
        serde_json::from_str(&json).expect("fit export JSON must be valid");
    let metrics = value["metrics"]
        .as_object()
        .expect("metrics must be object");
    assert_eq!(value["metrics"]["mse"], 0.25);
    assert_eq!(value["metrics"]["rmse"], 0.5);
    assert!(
        metrics.get("mae").is_none(),
        "mae should be omitted when unavailable",
    );
    assert!(
        metrics.get("r2").is_none(),
        "r2 should be omitted when unavailable"
    );
    assert!(
        metrics.get("max_abs_error").is_none(),
        "max_abs_error should be omitted when unavailable",
    );
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

    app.optimizer_method = OptimizerMethod::NewtonCg;
    app.apply_selected_optimizer_preset(OptimizerPreset::Precise);

    app.optimizer_method = OptimizerMethod::Sgd;
    app.apply_selected_optimizer_preset(OptimizerPreset::Balanced);

    app.optimizer_method = OptimizerMethod::Adam;
    app.apply_selected_optimizer_preset(OptimizerPreset::Precise);

    app.optimizer_method = OptimizerMethod::Lbfgs;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Fast);

    app.optimizer_method = OptimizerMethod::NelderMead;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Precise);

    app.optimizer_method = OptimizerMethod::SteepestDescent;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Fast);

    app.optimizer_method = OptimizerMethod::NewtonCg;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Precise);

    app.optimizer_method = OptimizerMethod::Sgd;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Balanced);

    app.optimizer_method = OptimizerMethod::Adam;
    assert_eq!(app.selected_optimizer_preset(), OptimizerPreset::Precise);
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
fn sgd_preset_changes_active_config_values() {
    let mut app = CurveFitApp {
        optimizer_method: OptimizerMethod::Sgd,
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

    let (fast_max_iters, precise_max_iters, fast_lr, precise_lr) =
        match (fast_config, precise_config) {
            (OptimizerConfig::Sgd(fast), OptimizerConfig::Sgd(precise)) => (
                fast.max_iters,
                precise.max_iters,
                fast.learning_rate,
                precise.learning_rate,
            ),
            _ => panic!("SGD must remain active"),
        };
    assert!(precise_max_iters > fast_max_iters);
    assert!(precise_lr < fast_lr);
}

#[test]
fn newton_cg_preset_changes_active_config_values() {
    let mut app = CurveFitApp {
        optimizer_method: OptimizerMethod::NewtonCg,
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

    let (fast_max_iters, precise_max_iters, fast_tol, precise_tol) =
        match (fast_config, precise_config) {
            (OptimizerConfig::NewtonCg(fast), OptimizerConfig::NewtonCg(precise)) => {
                (fast.max_iters, precise.max_iters, fast.tol, precise.tol)
            }
            _ => panic!("Newton-CG must remain active"),
        };
    assert!(precise_max_iters > fast_max_iters);
    assert!(precise_tol < fast_tol);
}

#[test]
fn adam_preset_changes_active_config_values() {
    let mut app = CurveFitApp {
        optimizer_method: OptimizerMethod::Adam,
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

    let (fast_max_iters, precise_max_iters, fast_lr, precise_lr) =
        match (fast_config, precise_config) {
            (OptimizerConfig::Adam(fast), OptimizerConfig::Adam(precise)) => (
                fast.max_iters,
                precise.max_iters,
                fast.learning_rate,
                precise.learning_rate,
            ),
            _ => panic!("Adam must remain active"),
        };
    assert!(precise_max_iters > fast_max_iters);
    assert!(precise_lr < fast_lr);
}

#[test]
fn param_init_method_support_matrix_is_correct() {
    assert!(ParamInitMethod::Default.is_supported_for_family(CurveFamily::Arrhenius));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Linear));
    assert!(ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::Power));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::BiExponential));
    assert!(ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::DampedSinusoid));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Rational11));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Rational22));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Rational33));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Rational44));
    assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Rational55));
    assert!(ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::Emg));
    assert!(ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::PseudoVoigt));

    assert!(!ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Arrhenius));
    assert!(!ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::FourPl));
}

#[test]
fn rational_model_degree_controls_resolved_family_and_parameter_count() {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Rational,
        rational_degree: 3,
        ..Default::default()
    };
    app.sync_parameter_inputs();

    assert_eq!(
        app.resolved_model().parametric_family(),
        Some(CurveFamily::Rational33)
    );
    assert_eq!(
        app.parameter_inputs.len(),
        CurveFamily::Rational33.parameter_count()
    );

    app.rational_degree = 5;
    app.sync_parameter_inputs();

    assert_eq!(
        app.resolved_model().parametric_family(),
        Some(CurveFamily::Rational55)
    );
    assert_eq!(
        app.parameter_inputs.len(),
        CurveFamily::Rational55.parameter_count()
    );
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
fn data_based_bi_exponential_initialization_returns_finite_values() {
    let points = points_from_pairs(&[
        (0.0, 2.7),
        (0.4, 2.1),
        (0.9, 1.5),
        (1.6, 1.0),
        (2.3, 0.7),
        (3.2, 0.5),
    ]);
    let params = data_based_params_for_family(CurveFamily::BiExponential, &points)
        .expect("must initialize bi-exponential params");
    let values = params.values();

    assert_eq!(values.len(), CurveFamily::BiExponential.parameter_count());
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values[1] > 0.0, "k1 must be positive");
    assert!(values[3] > 0.0, "k2 must be positive");
}

#[test]
fn data_based_damped_sinusoid_initialization_returns_finite_values() {
    let points = points_from_pairs(&[
        (0.0, 0.6),
        (0.5, 1.2),
        (1.0, 0.3),
        (1.5, -0.8),
        (2.0, -0.5),
        (2.5, 0.4),
        (3.0, 0.7),
        (3.5, 0.1),
        (4.0, -0.4),
    ]);
    let params = data_based_params_for_family(CurveFamily::DampedSinusoid, &points)
        .expect("must initialize damped sinusoid params");
    let values = params.values();

    assert_eq!(values.len(), CurveFamily::DampedSinusoid.parameter_count());
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values[1] > 0.0, "k must be positive");
    assert!(values[2] > 0.0, "omega must be positive");
}

#[test]
fn data_based_rational_and_peak_initialization_returns_finite_values() {
    let points = points_from_pairs(&[
        (-2.0, 0.6),
        (-1.0, 1.1),
        (-0.2, 1.8),
        (0.4, 2.3),
        (1.0, 1.9),
        (1.8, 1.3),
        (2.6, 0.9),
    ]);

    for family in [
        CurveFamily::Rational11,
        CurveFamily::Rational22,
        CurveFamily::Rational33,
        CurveFamily::Rational44,
        CurveFamily::Rational55,
        CurveFamily::Emg,
        CurveFamily::PseudoVoigt,
    ] {
        let params = data_based_params_for_family(family, &points)
            .expect("must initialize params for new family");
        let values = params.values();
        assert_eq!(values.len(), family.parameter_count());
        assert!(
            values.iter().all(|value| value.is_finite()),
            "all initialized params must be finite for {family:?}"
        );
    }
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
    app.points.text = "0 1\n1 3\n2 5\n3 7\n".to_string();
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
        MetricQuantization::Disabled,
    );
    app.panel.diagnostics_tab = DiagnosticsTab::Residuals;

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
    assert_eq!(app.panel.diagnostics_tab, DiagnosticsTab::Residuals);
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
        MetricQuantization::Disabled,
    );
    app.panel.diagnostics_tab = DiagnosticsTab::Residuals;

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
    assert_eq!(app.panel.diagnostics_tab, DiagnosticsTab::Residuals);
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
    app.points.text = "1 0\n2 2\n".to_string();
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
        replay: super::ReplayState {
            frames: vec![super::ReplayFrame {
                iteration: 0,
                payload: ReplayFramePayload::Parametric {
                    params: CurveParams::Linear { a: 1.0, b: 0.0 },
                },
            }],
            selected_index: Some(0),
            autoplay: true,
            ..Default::default()
        },
        ..Default::default()
    };
    app.panel.diagnostics_tab = DiagnosticsTab::Residuals;

    app.clear_fit_outputs();

    assert!(cancel_flag.load(Ordering::Relaxed));
    assert!(app.fit_in_progress);
    assert!(app.discard_fit_worker_updates);
    assert!(app.fit_result.is_none());
    assert!(app.fit_preview_params.is_none());
    assert!(app.fit_preview_iteration.is_none());
    assert!(app.replay.frames.is_empty());
    assert!(app.replay.selected_index.is_none());
    assert!(!app.replay.autoplay);
    assert_eq!(app.panel.diagnostics_tab, DiagnosticsTab::Residuals);
}
