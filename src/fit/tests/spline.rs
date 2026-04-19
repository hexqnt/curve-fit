use super::*;

#[test]
fn linear_spline_builds_curve() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let result =
        fit_linear_spline(&points, 50, DEFAULT_SPLINE_KNOTS).expect("linear spline must succeed");

    assert!(!result.knots.is_empty());
    assert_eq!(result.curve.len(), 50);
    assert!(result.mse < 1e-12);
}

#[test]
fn linear_spline_curve_extends_beyond_sample_extremes() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let result = fit_linear_spline_with_config(
        &points,
        SplineConfig {
            knots: 2,
            samples: 40,
            knot_strategy: SplineKnotStrategy::BinMean,
            extrapolation: SplineExtrapolation::Linear,
            duplicate_x_policy: SplineDuplicateXPolicy::Error,
        },
    )
    .expect("linear spline with two knots must succeed");

    let first_x = result.curve.first().expect("curve must be sampled")[0];
    let last_x = result.curve.last().expect("curve must be sampled")[0];
    assert!(
        first_x < 0.0,
        "left spline tail must extend beyond minimum sample x; got {first_x}"
    );
    assert!(
        last_x > 3.0,
        "right spline tail must extend beyond maximum sample x; got {last_x}"
    );
}

#[test]
fn spline_initial_curve_preview_extends_beyond_sample_extremes() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let curve = build_spline_initial_curve_from_knot_y(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig {
            knots: 2,
            samples: 40,
            knot_strategy: SplineKnotStrategy::BinMean,
            extrapolation: SplineExtrapolation::Linear,
            duplicate_x_policy: SplineDuplicateXPolicy::Error,
        },
        &[2.0, 6.0],
    )
    .expect("initial spline preview must be built");

    let first_x = curve.first().expect("curve must be sampled")[0];
    let last_x = curve.last().expect("curve must be sampled")[0];
    assert!(
        first_x < 0.0,
        "left spline preview tail must extend beyond minimum sample x; got {first_x}"
    );
    assert!(
        last_x > 3.0,
        "right spline preview tail must extend beyond maximum sample x; got {last_x}"
    );
}

#[test]
fn monotone_cubic_spline_preserves_monotone_curve() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0, 4.0], |x| x * x + 0.5 * x);
    let result = fit_monotone_cubic_spline(&points, 80, DEFAULT_SPLINE_KNOTS)
        .expect("monotone cubic spline must succeed");

    assert_eq!(result.curve.len(), 80);
    assert!(result.mse < 1e-10);
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
            IncrementalSplineFitStep::Finished { result, .. } => {
                assert!(saw_iteration || result.iterations == 0);
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
            IncrementalSplineFitStep::Finished { result, .. } => {
                assert!(result.iterations <= optimizer_config.max_iters());
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
            IncrementalSplineFitStep::Finished { result, .. } => {
                assert!(result.iterations <= optimizer_config.max_iters());
                return;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
    panic!("runner must finish in reasonable number of steps");
}

#[test]
fn incremental_spline_runner_supports_sgd() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::Sgd(SgdConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig::default(),
        &optimizer_config,
    )
    .expect("incremental spline runner with SGD must be created");

    for _ in 0..5_000 {
        match runner.step().expect("runner step must succeed") {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished { result, .. } => {
                assert!(result.iterations <= optimizer_config.max_iters());
                return;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
    panic!("runner must finish in reasonable number of steps");
}

#[test]
fn incremental_spline_runner_supports_adam() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::Adam(AdamConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig::default(),
        &optimizer_config,
    )
    .expect("incremental spline runner with Adam must be created");

    for _ in 0..5_000 {
        match runner.step().expect("runner step must succeed") {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished { result, .. } => {
                assert!(result.iterations <= optimizer_config.max_iters());
                return;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
    panic!("runner must finish in reasonable number of steps");
}

#[test]
fn incremental_spline_runner_supports_newton_cg() {
    let points = build_points(&[0.0, 1.0, 2.0, 3.0], |x| 2.0 * x + 1.0);
    let optimizer_config = OptimizerConfig::NewtonCg(NewtonCgConfig::default());
    let mut runner = IncrementalSplineFitRunner::new_with_optimizer_config(
        &points,
        SplineFamilyKind::Linear,
        SplineConfig::default(),
        &optimizer_config,
    )
    .expect("incremental spline runner with Newton-CG must be created");

    for _ in 0..5_000 {
        match runner.step().expect("runner step must succeed") {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished { result, .. } => {
                assert!(result.iterations <= optimizer_config.max_iters());
                return;
            }
            IncrementalSplineFitStep::Cancelled => panic!("runner must not be cancelled"),
        }
    }
    panic!("runner must finish in reasonable number of steps");
}
