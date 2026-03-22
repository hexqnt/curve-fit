use super::{
    DEFAULT_SPLINE_KNOTS, FitError, IncrementalSplineFitRunner, IncrementalSplineFitStep,
    OptimizationLossMetric, SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation,
    SplineFamilyKind, SplineKnotStrategy, approximate_spline_knots, calculate_iteration_metrics,
    calculate_metrics, evaluate_linear_spline, fit_akima_spline, fit_akima_spline_with_config,
    fit_curve, fit_curve_with_optimizer_config, fit_curve_with_progress,
    fit_curve_with_progress_and_optimizer_config,
    fit_curve_with_progress_and_optimizer_config_and_loss_metric, fit_linear_spline,
    fit_monotone_cubic_spline, fit_natural_cubic_spline, sorted_points_with_duplicate_policy,
};
use crate::domain::{
    CurveFamily, CurveParams, InputError, LbfgsConfig, NelderMeadConfig, OptimizerConfig, Point,
    Points, SteepestDescentConfig,
};

fn build_points<F>(xs: &[f64], f: F) -> Points
where
    F: Fn(f64) -> f64,
{
    let points = xs
        .iter()
        .copied()
        .map(|x| Point::try_new(x, f(x)).unwrap())
        .collect::<Vec<_>>();
    Points::try_from(points).unwrap()
}

#[test]
fn metrics_are_computed_correctly() {
    let points = build_points(&[0.0, 1.0, 2.0], |x| x + 1.0);
    let params = CurveParams::Linear { a: 1.0, b: 0.0 };
    let (mse, rmse) = calculate_metrics(&points, &params);

    assert!((mse - 1.0).abs() < 1e-12);
    assert!((rmse - 1.0).abs() < 1e-12);
}

#[test]
fn iteration_metrics_loss_matches_selected_objective() {
    let points = build_points(&[0.0, 1.0, 2.0], |x| x + 1.0);
    let params = CurveParams::Linear { a: 1.0, b: 0.0 };

    let mse_metrics = calculate_iteration_metrics(&points, &params, OptimizationLossMetric::Mse);
    let mae_metrics = calculate_iteration_metrics(&points, &params, OptimizationLossMetric::Mae);
    let soft_l1_metrics =
        calculate_iteration_metrics(&points, &params, OptimizationLossMetric::SoftL1);

    assert!((mse_metrics.loss - 1.0).abs() < 1e-12);
    assert!((mae_metrics.loss - 1.0).abs() < 1e-12);

    let expected_soft_l1 = 2.0 * (2.0_f64.sqrt() - 1.0);
    assert!((mse_metrics.soft_l1 - expected_soft_l1).abs() < 1e-12);
    assert!((mae_metrics.soft_l1 - expected_soft_l1).abs() < 1e-12);
    assert!((soft_l1_metrics.loss - expected_soft_l1).abs() < 1e-12);
    assert!((soft_l1_metrics.mse - 1.0).abs() < 1e-12);
    assert!((soft_l1_metrics.mae - 1.0).abs() < 1e-12);
    assert!((soft_l1_metrics.soft_l1 - expected_soft_l1).abs() < 1e-12);
}

#[test]
fn parametric_fit_supports_all_objective_metrics() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::Lbfgs(LbfgsConfig::default());
    for loss_metric in OptimizationLossMetric::ALL {
        let result = fit_curve_with_progress_and_optimizer_config_and_loss_metric(
            &points,
            CurveFamily::Linear,
            CurveParams::Linear { a: 0.2, b: 0.1 },
            &optimizer_config,
            loss_metric,
            |_iteration, _params| true,
        )
        .expect("fit with selected objective metric must succeed");
        assert!(result.mse < 1e-8);
    }
}

#[test]
fn incremental_spline_runner_supports_all_objective_metrics() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::Lbfgs(LbfgsConfig::default());
    for loss_metric in OptimizationLossMetric::ALL {
        let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config_and_loss_metric(
            &points,
            SplineFamilyKind::Linear,
            SplineConfig::default(),
            &optimizer_config,
            loss_metric,
        )
        .expect("incremental spline runner must be created");

        let mut finished = false;
        for _ in 0..5_000 {
            match runner.step().expect("runner step must succeed") {
                IncrementalSplineFitStep::Iteration { .. } => {}
                IncrementalSplineFitStep::Finished(result) => {
                    assert!(result.mse < 1e-6);
                    finished = true;
                    break;
                }
                IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
            }
        }
        assert!(finished, "runner must finish in reasonable number of steps");
    }
}

#[test]
fn lbfgs_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &config,
    )
    .expect("linear fit must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn nelder_mead_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NelderMead(NelderMeadConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with Nelder-Mead must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn steepest_descent_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::SteepestDescent(SteepestDescentConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with steepest descent must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn lbfgs_fits_cubic_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| {
        0.4 * x * x * x - 0.8 * x * x + 1.2 * x + 0.5
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Cubic,
        CurveParams::Cubic {
            a: 0.1,
            b: 0.1,
            c: 0.1,
            d: 0.1,
        },
        &config,
    )
    .expect("cubic fit must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn lbfgs_fits_nonic_data() {
    let points = build_points(
        &[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        |x| {
            0.15 * x.powi(9) - 0.05 * x.powi(8) + 0.12 * x.powi(7) - 0.2 * x.powi(6)
                + 0.08 * x.powi(5)
                + 0.1 * x.powi(4)
                - 0.05 * x.powi(3)
                + 0.07 * x.powi(2)
                - 0.03 * x
                + 0.9
        },
    );
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Nonic,
        CurveParams::Nonic {
            a: 0.1,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
            g: 0.0,
            h: 0.0,
            i: 0.0,
            j: 0.0,
        },
        &config,
    )
    .expect("nonic fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_exponential_basic_data() {
    let points = build_points(&[0.0, 0.5, 1.0, 1.5, 2.0], |x| 0.7 + 2.4 * (-0.9 * x).exp());
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::ExponentialBasic,
        CurveParams::ExponentialBasic {
            a: 0.1,
            b: 1.0,
            c: 0.3,
        },
        &config,
    )
    .expect("exponential basic fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_exponential_linear_data() {
    let points = build_points(&[-1.0, -0.5, 0.0, 0.7, 1.4, 2.0], |x| {
        1.6 * (0.45 * x).exp() - 0.8 * x + 0.3
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::ExponentialLinear,
        CurveParams::ExponentialLinear {
            a: 1.0,
            b: 0.2,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("exponential + linear fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_arrhenius_data() {
    let points = build_points(&[0.5, 0.8, 1.0, 1.4, 2.0, 3.0], |x| 1.8 * (0.9 / x).exp());
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Arrhenius,
        CurveParams::Arrhenius { a: 1.0, b: 0.2 },
        &config,
    )
    .expect("arrhenius fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_inverse_data() {
    let points = build_points(&[0.5, 0.75, 1.0, 1.5, 2.0, 3.0], |x| 1.2 + 2.7 / x);
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Inverse,
        CurveParams::Inverse { a: 0.0, b: 1.0 },
        &config,
    )
    .expect("inverse fit must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn lbfgs_fits_logistic_data() {
    let points = build_points(&[-2.0, -1.5, -1.0, -0.2, 0.4, 0.8, 1.2, 1.8], |x| {
        4.0 / (1.0 + (-2.2 * (x - 0.7)).exp())
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Logistic,
        CurveParams::Logistic {
            a: 3.0,
            b: 1.0,
            c: 0.0,
        },
        &config,
    )
    .expect("logistic fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_lorentzian_data() {
    let points = build_points(&[-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0], |x| {
        0.4 + 2.5 / (1.0 + ((x - 0.3) / 0.8).powi(2))
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Lorentzian,
        CurveParams::Lorentzian {
            a: 2.0,
            x0: 0.0,
            gamma: 1.0,
            c: 0.0,
        },
        &config,
    )
    .expect("lorentzian fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_natural_log_data() {
    let points = build_points(&[0.5, 0.8, 1.2, 1.8, 2.5, 3.2], |x| 1.5 * (x / 0.7).ln());
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::NaturalLog,
        CurveParams::NaturalLog { a: 1.0, b: 1.0 },
        &config,
    )
    .expect("natural log fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_michaelis_menten_data() {
    let points = build_points(&[0.5, 1.0, 2.0, 4.0, 8.0], |x| (3.5 * x) / (1.8 + x));
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::MichaelisMenten,
        CurveParams::MichaelisMenten { vmax: 2.0, km: 1.0 },
        &config,
    )
    .expect("michaelis-menten fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_hyperbolic_tangent_data() {
    let points = build_points(&[-2.0, -1.0, -0.4, 0.0, 0.6, 1.1, 1.8], |x| {
        2.2 * (1.3 * (x - 0.35)).tanh() - 0.4
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::HyperbolicTangent,
        CurveParams::HyperbolicTangent {
            a: 1.5,
            b: 0.8,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("hyperbolic tangent fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_arctangent_step_data() {
    let points = build_points(&[-2.0, -1.2, -0.6, 0.0, 0.5, 1.0, 1.8], |x| {
        2.0 * (1.5 * (x - 0.2)).atan() + 0.1
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::ArctangentStep,
        CurveParams::ArctangentStep {
            a: 1.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("arctangent step fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_softplus_data() {
    let points = build_points(&[-2.0, -1.0, -0.2, 0.3, 0.8, 1.4, 2.0], |x| {
        1.8 * super::softplus(2.0 * (x - 0.4)) - 0.35
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Softplus,
        CurveParams::Softplus {
            a: 1.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("softplus fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_power_data() {
    let points = build_points(&[0.5, 1.0, 1.5, 2.0, 3.0], |x| 1.7 * x.powf(1.35));
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Power,
        CurveParams::Power { a: 1.0, b: 1.0 },
        &config,
    )
    .expect("power fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_gaussian_data() {
    let points = build_points(&[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], |x| {
        2.1 * (-(x - 0.4).powi(2) / (2.0 * 0.7 * 0.7)).exp()
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Gaussian,
        CurveParams::Gaussian {
            a: 1.0,
            b: 0.0,
            c: 1.0,
        },
        &config,
    )
    .expect("gaussian fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn fit_curve_validates_positive_x_domain() {
    let points = build_points(&[-1.0, 1.0, 2.0], |x| x);
    let config = LbfgsConfig::default();
    let error = fit_curve(
        &points,
        CurveFamily::Power,
        CurveParams::Power { a: 1.0, b: 1.0 },
        &config,
    )
    .expect_err("power family must reject x <= 0");

    assert!(matches!(
        error,
        super::FitError::InvalidInput(InputError::NonPositiveXForFamily {
            family: CurveFamily::Power,
            ..
        })
    ));
}

#[test]
fn linear_spline_builds_curve() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let result =
        fit_linear_spline(&points, 50, DEFAULT_SPLINE_KNOTS).expect("linear spline must succeed");

    assert!(!result.knots.is_empty());
    assert_eq!(result.curve.len(), 50);
    assert!(result.mse < 1e-12);
    assert!(result.iterations > 0);
}

#[test]
fn monotone_cubic_spline_preserves_monotone_curve() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0, 4.0], |x| x * x + 0.5 * x);
    let result = fit_monotone_cubic_spline(&points, 80, DEFAULT_SPLINE_KNOTS)
        .expect("monotone cubic spline must succeed");

    assert_eq!(result.curve.len(), 80);
    assert!(result.mse < 1e-10);
    assert!(result.iterations > 0);
    for window in result.curve.windows(2) {
        assert!(window[1][1] >= window[0][1] - 1e-10);
    }
}

#[test]
fn natural_cubic_spline_builds_curve() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| x * x * x - x + 1.0);
    let result = fit_natural_cubic_spline(&points, 60, DEFAULT_SPLINE_KNOTS)
        .expect("natural cubic spline must succeed");

    assert_eq!(result.curve.len(), 60);
    assert!(result.mse < 1e-8);
    assert!(result.iterations > 0);
}

#[test]
fn akima_spline_builds_curve() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], |x| {
        x * x * x - 0.5 * x + 1.0
    });
    let result =
        fit_akima_spline(&points, 70, DEFAULT_SPLINE_KNOTS).expect("akima spline must succeed");

    assert_eq!(result.curve.len(), 70);
    assert!(result.mse < 1e-10);
    assert!(result.iterations > 0);
}

#[test]
fn akima_spline_requires_at_least_five_knots() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], |x| {
        x * x * x - 0.5 * x + 1.0
    });
    let error = fit_akima_spline_with_config(
        &points,
        SplineConfig {
            knots: 4,
            samples: 64,
            knot_strategy: SplineKnotStrategy::BinMean,
            extrapolation: SplineExtrapolation::Clamp,
            duplicate_x_policy: SplineDuplicateXPolicy::Error,
        },
    )
    .expect_err("akima should reject knot count below 5");

    assert!(matches!(
        error,
        FitError::InvalidSplineInput(message) if message.contains("at least 5 knots")
    ));
}

#[test]
fn median_knot_strategy_is_robust_to_single_outlier() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], |x| {
        if (x - 1.0).abs() < 1e-12 { 100.0 } else { 0.0 }
    });
    let sorted = sorted_points_with_duplicate_policy(&points, SplineDuplicateXPolicy::Error)
        .expect("x values are unique");
    let mean_knots = approximate_spline_knots(&sorted, 3, SplineKnotStrategy::BinMean);
    let median_knots = approximate_spline_knots(&sorted, 3, SplineKnotStrategy::BinMedian);

    assert!(mean_knots[0][1] > 30.0);
    assert!(median_knots[0][1].abs() < 1e-12);
}

#[test]
fn duplicate_x_policy_mean_y_merges_points() {
    let points = Points::try_from(vec![
        Point::try_new(1.0, 2.0).unwrap(),
        Point::try_new(1.0, 6.0).unwrap(),
        Point::try_new(2.0, 4.0).unwrap(),
    ])
    .unwrap();

    let sorted = sorted_points_with_duplicate_policy(&points, SplineDuplicateXPolicy::MeanY)
        .expect("duplicate x should be merged with mean");

    assert_eq!(sorted.len(), 2);
    assert!((sorted[0][0] - 1.0).abs() < 1e-12);
    assert!((sorted[0][1] - 4.0).abs() < 1e-12);
}

#[test]
fn linear_extrapolation_uses_edge_slope() {
    let knots = [[0.0, 1.0], [2.0, 5.0]];

    let clamped = evaluate_linear_spline(&knots, -1.0, SplineExtrapolation::Clamp);
    let linear = evaluate_linear_spline(&knots, -1.0, SplineExtrapolation::Linear);

    assert!((clamped - 1.0).abs() < 1e-12);
    assert!((linear + 1.0).abs() < 1e-12);
}

#[test]
fn splines_are_approximation_not_exact_interpolation() {
    let points = build_points(&(-20..=20).map(|x| x as f64).collect::<Vec<_>>(), |x| {
        (x * 0.3).sin() + 0.1 * x
    });

    let result =
        fit_natural_cubic_spline(&points, 60, DEFAULT_SPLINE_KNOTS).expect("natural cubic spline");

    assert_eq!(result.curve.len(), 60);
    assert!(
        result.mse > 1e-6,
        "Smoothing should produce non-zero error on dense input"
    );
    assert!(result.iterations > 0);
}

#[test]
fn incremental_spline_runner_reports_iteration_steps() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::Lbfgs(LbfgsConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig {
            knots: DEFAULT_SPLINE_KNOTS,
            samples: 48,
            knot_strategy: SplineKnotStrategy::BinMean,
            extrapolation: SplineExtrapolation::Clamp,
            duplicate_x_policy: SplineDuplicateXPolicy::Error,
        },
        &optimizer_config,
    )
    .expect("incremental linear spline runner must be created");

    let mut saw_iteration = false;
    loop {
        match runner.step().expect("runner step must succeed") {
            IncrementalSplineFitStep::Iteration { .. } => saw_iteration = true,
            IncrementalSplineFitStep::Finished(result) => {
                assert!(saw_iteration);
                assert!(result.iterations > 0);
                break;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
}

#[test]
fn incremental_spline_runner_rejects_wrong_custom_init_length() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::Lbfgs(LbfgsConfig::default());
    let error = IncrementalSplineFitRunner::new_with_initial_knot_y_and_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig {
            knots: DEFAULT_SPLINE_KNOTS,
            samples: 48,
            knot_strategy: SplineKnotStrategy::BinMean,
            extrapolation: SplineExtrapolation::Clamp,
            duplicate_x_policy: SplineDuplicateXPolicy::Error,
        },
        &optimizer_config,
        Some(&[1.0, 2.0, 3.0]),
    );
    let error = match error {
        Ok(_) => panic!("runner must reject mismatched custom initialization length"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        FitError::InvalidSplineInput(message) if message.contains("expects")
    ));
}

#[test]
fn incremental_spline_runner_can_be_cancelled() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| x * x);
    let optimizer_config = OptimizerConfig::Lbfgs(LbfgsConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::NaturalCubic,
        SplineConfig::default(),
        &optimizer_config,
    )
    .expect("incremental spline runner must be created");

    runner.cancel();
    let step = runner.step().expect("cancelled runner step must succeed");
    assert!(matches!(step, IncrementalSplineFitStep::Cancelled));
}

#[test]
fn incremental_spline_runner_supports_nelder_mead() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::NelderMead(NelderMeadConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig::default(),
        &optimizer_config,
    )
    .expect("incremental spline runner with Nelder-Mead must be created");

    for _ in 0..5_000 {
        match runner.step().expect("runner step must succeed") {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished(result) => {
                assert!(result.iterations > 0);
                return;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
    panic!("runner must finish in reasonable number of steps");
}

#[test]
fn incremental_spline_runner_supports_steepest_descent() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::SteepestDescent(SteepestDescentConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig::default(),
        &optimizer_config,
    )
    .expect("incremental spline runner with steepest descent must be created");

    for _ in 0..5_000 {
        match runner.step().expect("runner step must succeed") {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished(result) => {
                assert!(result.iterations > 0);
                return;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
    panic!("runner must finish in reasonable number of steps");
}

#[test]
fn fit_curve_can_be_cancelled_via_progress_callback() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let config = LbfgsConfig::default();
    let result = fit_curve_with_progress(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &config,
        |_iteration, _params| false,
    );

    assert!(matches!(result, Err(FitError::Cancelled)));
}

#[test]
fn fit_curve_with_optimizer_config_can_be_cancelled_via_progress_callback() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NelderMead(NelderMeadConfig::default());
    let result = fit_curve_with_progress_and_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
        |_iteration, _params| false,
    );

    assert!(matches!(result, Err(FitError::Cancelled)));
}
