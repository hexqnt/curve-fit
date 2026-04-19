use super::*;

#[test]
fn prediction_loss_adapter_respects_metric_and_quantization() {
    let points = build_points(&[0.0, 1.0], |_| 0.0);
    let prediction = 1.225;
    let target = 0.0;
    let raw_residual = prediction - target;
    let quantized_residual = 1.23;

    for metric in OptimizationLossMetric::ALL {
        let raw_problem = CurveProblem::new_with_metric_quantization(
            CurveFamily::Linear,
            &points,
            metric,
            MetricQuantization::Disabled,
        );
        let quantized_problem = CurveProblem::new_with_metric_quantization(
            CurveFamily::Linear,
            &points,
            metric,
            quantization(2),
        );

        let raw_loss = CurveProblemPredictionLoss {
            problem: &raw_problem,
        };
        let quantized_loss = CurveProblemPredictionLoss {
            problem: &quantized_problem,
        };

        assert_near(
            raw_loss.value(prediction, target),
            metric.value_from_residual(raw_residual),
            1e-12,
        );
        assert_near(
            raw_loss.d_prediction(prediction, target),
            metric.residual_derivative(raw_residual),
            1e-12,
        );
        assert_near(
            raw_loss.d2_prediction(prediction, target),
            metric.residual_second_derivative(raw_residual),
            1e-12,
        );

        assert_near(
            quantized_loss.value(prediction, target),
            metric.value_from_residual(quantized_residual),
            1e-12,
        );
        assert_near(
            quantized_loss.d_prediction(prediction, target),
            metric.residual_derivative(quantized_residual),
            1e-12,
        );
        assert_near(
            quantized_loss.d2_prediction(prediction, target),
            metric.residual_second_derivative(quantized_residual),
            1e-12,
        );
    }
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
    let params = ndarray::Array1::from_vec(vec![0.0, 0.0]);

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
