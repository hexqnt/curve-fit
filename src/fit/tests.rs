#[cfg(feature = "portable-simd")]
use super::simd;
use super::{
    CurveProblem, DEFAULT_SPLINE_KNOTS, FitError, HESSIAN_DIAGONAL_JITTER,
    IncrementalSplineFitRunner, IncrementalSplineFitStep, MetricQuantization,
    MetricQuantizationDecimalPlaces, OptimizationLossMetric, SplineConfig, SplineDuplicateXPolicy,
    SplineExtrapolation, SplineFamilyKind, SplineFinalizeContext, SplineKnotStrategy,
    approximate_spline_knots, build_spline_initial_curve_from_knot_y,
    build_spline_result_from_knot_y, calculate_iteration_metrics,
    calculate_iteration_metrics_with_quantization, calculate_metrics, evaluate_linear_spline,
    expanded_spline_curve_x_bounds, fit_akima_spline, fit_akima_spline_with_config, fit_curve,
    fit_curve_with_optimizer_config, fit_curve_with_progress,
    fit_curve_with_progress_and_optimizer_config,
    fit_curve_with_progress_and_optimizer_config_and_loss_metric, fit_linear_spline,
    fit_linear_spline_with_config, fit_monotone_cubic_spline, fit_natural_cubic_spline,
    numerical_hessian_from_gradient, sorted_points_with_duplicate_policy,
};
use crate::domain::{
    AdamConfig, CurveFamily, CurveParams, InputError, LbfgsConfig, NelderMeadConfig,
    NewtonCgConfig, OptimizerConfig, Point, Points, SgdConfig, SteepestDescentConfig,
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

fn assert_near(actual: f64, expected: f64, epsilon: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= epsilon,
        "expected {expected}, got {actual}, delta={delta}, epsilon={epsilon}"
    );
}

fn quantization(decimal_places: u8) -> MetricQuantization {
    MetricQuantization::Enabled(
        MetricQuantizationDecimalPlaces::try_new(decimal_places)
            .expect("test decimal places must be valid"),
    )
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
fn metric_quantization_decimal_places_respect_bounds() {
    assert!(MetricQuantizationDecimalPlaces::try_new(0).is_ok());
    assert!(MetricQuantizationDecimalPlaces::try_new(15).is_ok());
    assert!(MetricQuantizationDecimalPlaces::try_new(16).is_err());
}

#[test]
fn quantized_metrics_round_half_away_from_zero_and_update_r2_inputs() {
    let points = build_points(&[0.0, 1.0], |x| if x < 0.5 { 1.225 } else { -1.225 });
    let params = CurveParams::Linear { a: 0.0, b: 0.0 };
    let metrics = calculate_iteration_metrics_with_quantization(
        &points,
        &params,
        OptimizationLossMetric::Mse,
        quantization(2),
    );

    let expected_abs_residual = 1.23_f64;
    let expected_mse = expected_abs_residual * expected_abs_residual;
    let expected_soft_l1 =
        2.0 * ((1.0 + expected_abs_residual * expected_abs_residual).sqrt() - 1.0);

    assert_near(metrics.loss, expected_mse, 1e-12);
    assert_near(metrics.mse, expected_mse, 1e-12);
    assert_near(metrics.rmse, expected_abs_residual, 1e-12);
    assert_near(metrics.mae, expected_abs_residual, 1e-12);
    assert_near(metrics.soft_l1, expected_soft_l1, 1e-12);
    assert_near(metrics.r2, 0.0, 1e-12);
    assert_near(metrics.max_abs_error, expected_abs_residual, 1e-12);
}

#[test]
fn quantized_objective_differs_from_raw_objective() {
    use argmin::core::CostFunction;

    let points = build_points(&[0.0, 1.0], |x| if x < 0.5 { 1.225 } else { -1.225 });
    let raw_problem = CurveProblem::new_with_metric_quantization(
        CurveFamily::Linear,
        &points,
        OptimizationLossMetric::Mse,
        MetricQuantization::Disabled,
    );
    let quantized_problem = CurveProblem::new_with_metric_quantization(
        CurveFamily::Linear,
        &points,
        OptimizationLossMetric::Mse,
        quantization(2),
    );
    let params = vec![0.0, 0.0];

    let raw_cost = CostFunction::cost(&raw_problem, &params).expect("raw cost must be computed");
    let quantized_cost =
        CostFunction::cost(&quantized_problem, &params).expect("quantized cost must be computed");

    assert_near(raw_cost, 1.500625, 1e-12);
    assert_near(quantized_cost, 1.5129, 1e-12);
    assert!(
        (raw_cost - quantized_cost).abs() > 1e-4,
        "quantized objective must differ for boundary residuals"
    );
}

#[test]
fn spline_final_snapshot_uses_quantized_metrics_while_residuals_stay_raw() {
    let points = build_points(&[0.0, 1.0], |x| x + 0.016);
    let knot_x = [0.0, 1.0];
    let knot_y = [0.004, 1.004];
    let finalize_context = SplineFinalizeContext {
        points: &points,
        family: SplineFamilyKind::Linear,
        config: SplineConfig::default(),
        knot_x: &knot_x,
        curve_x_bounds: expanded_spline_curve_x_bounds(0.0, 1.0),
        loss_metric: OptimizationLossMetric::SoftL1,
        metric_quantization: quantization(2),
    };
    let (result, metrics) = build_spline_result_from_knot_y(&finalize_context, &knot_y, 7)
        .expect("spline result build must succeed");

    assert_eq!(result.iterations, 7);
    assert_eq!(result.residuals.len(), 2);
    assert_near(result.residuals[0][1], -0.012, 1e-12);
    assert_near(result.residuals[1][1], -0.012, 1e-12);

    let raw_mse = result
        .residuals
        .iter()
        .map(|point| point[1] * point[1])
        .sum::<f64>()
        / result.residuals.len() as f64;
    assert_near(raw_mse, 0.000144, 1e-12);

    let expected_abs_residual = 0.02_f64;
    let expected_mse = expected_abs_residual * expected_abs_residual;
    let expected_soft_l1 =
        2.0 * ((1.0 + expected_abs_residual * expected_abs_residual).sqrt() - 1.0);

    assert_near(metrics.mse, expected_mse, 1e-12);
    assert_near(metrics.rmse, expected_abs_residual, 1e-12);
    assert_near(metrics.mae, expected_abs_residual, 1e-12);
    assert_near(metrics.soft_l1, expected_soft_l1, 1e-12);
    assert_near(metrics.loss, expected_soft_l1, 1e-12);
    assert_near(result.mse, metrics.mse, 1e-12);
    assert_near(result.rmse, metrics.rmse, 1e-12);
    assert_near(result.mae, metrics.mae, 1e-12);
    assert!(
        (metrics.mse - raw_mse).abs() > 1e-5,
        "final metrics must not be reconstructed from raw residuals"
    );
}

#[test]
fn analytic_hessian_matches_mse_polynomial_formula() {
    use argmin::core::Hessian;

    let points = build_points(&[-1.0, 0.0, 2.0], |x| 1.5 * x - 0.25);
    let problem = CurveProblem::new(CurveFamily::Linear, &points, OptimizationLossMetric::Mse);
    let hessian = Hessian::hessian(&problem, &vec![0.3, -0.7]).expect("hessian must be computed");

    // Для линейной модели y = a*x + b и MSE:
    // H = (2/n) * Σ [[x^2, x], [x, 1]] + диагональный jitter.
    let n = 3.0;
    let expected_00 = (2.0 / n) * (1.0 + 0.0 + 4.0) + HESSIAN_DIAGONAL_JITTER;
    let expected_01 = (2.0 / n) * (-1.0 + 0.0 + 2.0);
    let expected_11 = (2.0 / n) * 3.0 + HESSIAN_DIAGONAL_JITTER;

    assert_near(hessian[0][0], expected_00, 1e-12);
    assert_near(hessian[0][1], expected_01, 1e-12);
    assert_near(hessian[1][0], expected_01, 1e-12);
    assert_near(hessian[1][1], expected_11, 1e-12);
}

#[test]
fn analytic_hessian_matches_soft_l1_inverse_formula() {
    use argmin::core::Hessian;

    let points = build_points(&[1.0, 2.0, 4.0], |x| 1.0 + 0.5 / x);
    let params = vec![0.9, 0.3];
    let problem = CurveProblem::new(
        CurveFamily::Inverse,
        &points,
        OptimizationLossMetric::SoftL1,
    );
    let hessian = Hessian::hessian(&problem, &params).expect("hessian must be computed");

    let mut expected_00 = 0.0;
    let mut expected_01 = 0.0;
    let mut expected_11 = 0.0;
    for point in points.as_slice() {
        let x = point.x().max(1e-9);
        let inv_x = 1.0 / x;
        let residual = params[0] + params[1] * inv_x - point.y();
        let weight = 2.0 / (1.0 + residual * residual).powf(1.5);
        expected_00 += weight;
        expected_01 += weight * inv_x;
        expected_11 += weight * inv_x * inv_x;
    }
    let sample_scale = 1.0 / points.len() as f64;
    expected_00 = expected_00 * sample_scale + HESSIAN_DIAGONAL_JITTER;
    expected_01 *= sample_scale;
    expected_11 = expected_11 * sample_scale + HESSIAN_DIAGONAL_JITTER;

    assert_near(hessian[0][0], expected_00, 1e-12);
    assert_near(hessian[0][1], expected_01, 1e-12);
    assert_near(hessian[1][0], expected_01, 1e-12);
    assert_near(hessian[1][1], expected_11, 1e-12);
}

#[test]
fn analytic_hessian_exponential_basic_matches_numerical_reference() {
    use argmin::core::Hessian;

    let points = build_points(&[-1.0, -0.2, 0.3, 1.1, 2.0], |x| {
        0.8 + 1.4 * (-0.6 * x).exp()
    });
    let params = vec![0.5, 1.1, 0.3];
    let problem = CurveProblem::new(
        CurveFamily::ExponentialBasic,
        &points,
        OptimizationLossMetric::Mse,
    );

    let analytic = Hessian::hessian(&problem, &params).expect("analytic hessian must be computed");
    let numerical =
        numerical_hessian_from_gradient(&problem, &params).expect("numerical hessian must succeed");

    for row in 0..3 {
        for column in 0..3 {
            assert_near(analytic[row][column], numerical[row][column], 5e-5);
        }
    }
}

#[test]
fn analytic_hessian_exponential_linear_matches_numerical_reference() {
    use argmin::core::Hessian;

    let points = build_points(&[-1.2, -0.5, 0.0, 0.7, 1.4], |x| {
        1.4 * (0.35 * x).exp() - 0.4 * x + 0.2
    });
    let params = vec![1.0, 0.2, -0.2, 0.0];
    let problem = CurveProblem::new(
        CurveFamily::ExponentialLinear,
        &points,
        OptimizationLossMetric::SoftL1,
    );

    let analytic = Hessian::hessian(&problem, &params).expect("analytic hessian must be computed");
    let numerical =
        numerical_hessian_from_gradient(&problem, &params).expect("numerical hessian must succeed");

    for row in 0..4 {
        for column in 0..4 {
            assert_near(analytic[row][column], numerical[row][column], 8e-5);
        }
    }
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
                IncrementalSplineFitStep::Finished { result, .. } => {
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
fn sgd_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::Sgd(SgdConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with SGD must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn adam_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::Adam(
        AdamConfig::try_new(5_000, 2e-2).expect("adam test config must be valid"),
    );
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with Adam must succeed");

    assert!(result.mse < 1e-6, "adam mse={}", result.mse);
}

#[test]
fn newton_cg_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NewtonCg(NewtonCgConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with Newton-CG must succeed");

    assert!(result.mse < 1e-10, "newton-cg mse={}", result.mse);
}

#[test]
fn newton_cg_supports_all_objective_metrics() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NewtonCg(NewtonCgConfig::default());
    for loss_metric in OptimizationLossMetric::ALL {
        let result = fit_curve_with_progress_and_optimizer_config_and_loss_metric(
            &points,
            CurveFamily::Linear,
            CurveParams::Linear { a: 0.2, b: 0.1 },
            &optimizer_config,
            loss_metric,
            |_iteration, _params| true,
        )
        .expect("fit with Newton-CG and selected objective metric must succeed");
        assert!(
            result.mse < 1e-6,
            "loss={loss_metric:?}, mse={}",
            result.mse
        );
    }
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
fn lbfgs_fits_gompertz_data() {
    let points = build_points(&[-2.0, -1.2, -0.5, 0.0, 0.5, 1.0, 1.6, 2.2], |x| {
        4.8 * (-(-1.6 * (x - 0.4)).exp()).exp()
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Gompertz,
        CurveParams::Gompertz {
            a: 3.5,
            b: 0.8,
            c: 0.0,
        },
        &config,
    )
    .expect("gompertz fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_bi_exponential_data() {
    let points = build_points(&[0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.8, 3.6, 4.5], |x| {
        1.8 * (-2.4 * x).exp() + 0.7 * (-0.35 * x).exp() + 0.2
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::BiExponential,
        CurveParams::BiExponential {
            a1: 1.0,
            k1: 1.0,
            a2: 0.4,
            k2: 0.1,
            c: 0.0,
        },
        &config,
    )
    .expect("bi-exponential fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_damped_sinusoid_data() {
    let points = build_points(
        &[
            0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6,
        ],
        |x| 1.9 * (-0.25 * x).exp() * (2.4 * x + 0.35).sin() - 0.2,
    );
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::DampedSinusoid,
        CurveParams::DampedSinusoid {
            a: 1.4,
            k: 0.2,
            omega: 2.0,
            phi: 0.0,
            c: 0.0,
        },
        &config,
    )
    .expect("damped sinusoid fit must succeed");

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

#[cfg(feature = "portable-simd")]
fn assert_close(actual: f64, expected: f64, abs_tolerance: f64, rel_tolerance: f64) {
    let delta = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    assert!(
        delta <= abs_tolerance.max(scale * rel_tolerance),
        "expected {expected}, got {actual}, delta={delta}"
    );
}

#[cfg(feature = "portable-simd")]
fn test_xy_for_polynomial() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let param = vec![0.3, -0.2, 0.08, -0.005, 1.7, -4.0];
    let x_values = (0..1031)
        .map(|index| index as f64 / 9.0 - 60.0)
        .collect::<Vec<_>>();
    let y_values = x_values
        .iter()
        .copied()
        .map(|x| {
            let model = param
                .iter()
                .copied()
                .fold(0.0, |acc, coefficient| acc * x + coefficient);
            model + 0.125 * (x * 0.3).sin()
        })
        .collect::<Vec<_>>();
    (param, x_values, y_values)
}

#[cfg(feature = "portable-simd")]
fn test_xy_for_inverse() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let param = vec![1.25, -0.7];
    let x_values = (0..1019)
        .map(|index| index as f64 / 11.0 - 30.0)
        .collect::<Vec<_>>();
    let y_values = x_values
        .iter()
        .copied()
        .map(|x| {
            let x_safe = x.max(1e-9);
            (param[0] + param[1] / x_safe) + 0.05 * (x * 0.2).cos()
        })
        .collect::<Vec<_>>();
    (param, x_values, y_values)
}

#[cfg(feature = "portable-simd")]
#[test]
fn simd_polynomial_cost_matches_scalar_reference_for_all_loss_metrics() {
    let (param, x_values, y_values) = test_xy_for_polynomial();
    for loss_metric in OptimizationLossMetric::ALL {
        let scalar = simd::polynomial_cost_scalar(&param, &x_values, &y_values, loss_metric);
        let simd = simd::polynomial_cost_simd(&param, &x_values, &y_values, loss_metric);
        assert_close(simd, scalar, 1e-10, 1e-10);
    }
}

#[cfg(feature = "portable-simd")]
#[test]
fn simd_inverse_cost_matches_scalar_reference_for_all_loss_metrics() {
    let (param, x_values, y_values) = test_xy_for_inverse();
    for loss_metric in OptimizationLossMetric::ALL {
        let scalar = simd::inverse_cost_scalar(&param, &x_values, &y_values, loss_metric);
        let simd = simd::inverse_cost_simd(&param, &x_values, &y_values, loss_metric);
        assert_close(simd, scalar, 1e-10, 1e-10);
    }
}

#[cfg(feature = "portable-simd")]
#[test]
fn simd_polynomial_gradient_matches_scalar_reference_for_all_loss_metrics() {
    let (param, x_values, y_values) = test_xy_for_polynomial();
    for loss_metric in OptimizationLossMetric::ALL {
        let mut scalar_gradient = vec![0.0; param.len()];
        simd::accumulate_polynomial_gradient_scalar(
            &x_values,
            &y_values,
            &param,
            loss_metric,
            &mut scalar_gradient,
        );

        let mut simd_gradient = vec![0.0; param.len()];
        simd::accumulate_polynomial_gradient_simd(
            &x_values,
            &y_values,
            &param,
            loss_metric,
            &mut simd_gradient,
        );

        for index in 0..param.len() {
            assert_close(simd_gradient[index], scalar_gradient[index], 1e-7, 1e-7);
        }
    }
}

#[cfg(feature = "portable-simd")]
#[test]
fn simd_inverse_gradient_matches_scalar_reference_for_all_loss_metrics() {
    let (param, x_values, y_values) = test_xy_for_inverse();
    for loss_metric in OptimizationLossMetric::ALL {
        let mut scalar_gradient = vec![0.0; 2];
        simd::accumulate_inverse_gradient_scalar(
            &x_values,
            &y_values,
            &param,
            loss_metric,
            &mut scalar_gradient,
        );

        let mut simd_gradient = vec![0.0; 2];
        simd::accumulate_inverse_gradient_simd(
            &x_values,
            &y_values,
            &param,
            loss_metric,
            &mut simd_gradient,
        );

        for index in 0..2 {
            assert_close(simd_gradient[index], scalar_gradient[index], 1e-8, 1e-8);
        }
    }
}

#[cfg(feature = "portable-simd")]
#[test]
fn polynomial_cost_with_quantization_uses_quantized_scalar_pipeline() {
    use argmin::core::CostFunction;

    let points = build_points(&[0.0, 1.0], |x| if x < 0.5 { 1.225 } else { -1.225 });
    let problem = CurveProblem::new_with_metric_quantization(
        CurveFamily::Linear,
        &points,
        OptimizationLossMetric::Mse,
        quantization(2),
    );

    let cost = CostFunction::cost(&problem, &vec![0.0, 0.0]).expect("cost must be computed");
    assert_near(cost, 1.5129, 1e-12);
}

#[cfg(feature = "portable-simd")]
#[test]
fn simd_cost_returns_large_cost_for_non_finite_inputs() {
    let x_values = vec![0.1, 0.2, 0.3];
    let y_values = vec![1.0, 2.0, 3.0];
    let inf_param = vec![f64::INFINITY, 1.0];
    let loss_metric = OptimizationLossMetric::Mse;

    assert_eq!(
        simd::polynomial_cost_scalar(&inf_param, &x_values, &y_values, loss_metric),
        super::LARGE_COST
    );
    assert_eq!(
        simd::polynomial_cost_simd(&inf_param, &x_values, &y_values, loss_metric),
        super::LARGE_COST
    );

    let nan_y = vec![1.0, f64::NAN, 3.0];
    let inverse_param = vec![1.0, 2.0];
    assert_eq!(
        simd::inverse_cost_scalar(&inverse_param, &x_values, &nan_y, loss_metric),
        super::LARGE_COST
    );
    assert_eq!(
        simd::inverse_cost_simd(&inverse_param, &x_values, &nan_y, loss_metric),
        super::LARGE_COST
    );
}
