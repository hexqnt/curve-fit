use super::*;

#[test]
fn central_diff_gradient_retries_step_when_primary_step_is_invalid() {
    let param = [1.0_f64];
    let rel_step = 1e-3;
    let min_step = 1e-3;
    let base_step = ((param[0].abs() + 1.0) * rel_step).max(min_step);
    let mut gradient = [0.0];
    models::central_diff_gradient_from_value(
        &param,
        rel_step,
        min_step,
        |probe| {
            let delta = (probe[0] - param[0]).abs();
            if (delta - base_step).abs() <= 1e-14 {
                f64::NAN
            } else {
                probe[0] * probe[0]
            }
        },
        &mut gradient,
    );

    assert_near(gradient[0], 2.0, 1e-10);
}

#[test]
fn central_diff_hessian_retries_step_when_primary_step_is_invalid() {
    let param = [1.0_f64];
    let rel_step = 1e-3;
    let min_step = 1e-3;
    let base_step = ((param[0].abs() + 1.0) * rel_step).max(min_step);
    let hessian = models::central_diff_hessian_from_gradient(
        &param,
        rel_step,
        min_step,
        |probe, gradient_out| {
            let delta = (probe[0] - param[0]).abs();
            gradient_out[0] = if (delta - base_step).abs() <= 1e-14 {
                f64::NAN
            } else {
                2.0 * probe[0]
            };
        },
    );

    assert_near(hessian[[0, 0]], 2.0, 1e-10);
}

#[test]
fn fit_numerical_hessian_retries_step_when_primary_step_is_invalid() {
    let param = Array1::from_vec(vec![1.0_f64]);
    let base_step = ((param[0].abs() + 1.0) * HESSIAN_FD_REL_STEP).max(HESSIAN_FD_MIN_STEP);
    let problem = RetryGradientProblem {
        center: param[0],
        invalid_step: base_step,
    };
    let hessian =
        numerical_hessian_from_gradient(&problem, &param).expect("hessian must be computed");

    assert_near(hessian[[0, 0]], 2.0 + HESSIAN_DIAGONAL_JITTER, 1e-10);
}

#[test]
fn central_diff_gradient_falls_back_to_zero_when_all_retry_steps_invalid() {
    let param = [1.0_f64];
    let mut gradient = [123.0];
    models::central_diff_gradient_from_value(&param, 1e-3, 1e-3, |_probe| f64::NAN, &mut gradient);

    assert_eq!(gradient[0], 0.0);
}

#[test]
fn fit_numerical_hessian_falls_back_to_diagonal_jitter_when_all_retry_steps_invalid() {
    let param = Array1::from_vec(vec![1.0_f64]);
    let hessian = numerical_hessian_from_gradient(&AlwaysInvalidGradientProblem, &param)
        .expect("hessian must be computed even with invalid gradients");

    assert_near(hessian[[0, 0]], HESSIAN_DIAGONAL_JITTER, 1e-15);
}

#[test]
fn curve_objective_arrhenius_is_consistent_across_levels() {
    let x_values = [0.4, 0.8, 1.4, 2.5, 4.0];
    let true_params = [1.5, 0.9];
    let probe_params = [1.2, 0.5];
    let y_values = x_values
        .iter()
        .map(|&x| models::value_at(CurveFamily::Arrhenius, &true_params, x))
        .collect::<Vec<_>>();

    let term = models::DataTerm::new(
        CurveFamily::Arrhenius,
        &x_values,
        &y_values,
        None,
        MsePredictionLoss,
    );
    let objective = models::CurveObjective::new(probe_params.len(), term);

    let value = objective.value(&probe_params);
    let (value_from_grad, gradient) = objective.value_grad(&probe_params);
    let (value_from_raw_hessian, gradient_from_raw_hessian, raw_hessian) =
        objective.value_grad_raw_hessian(&probe_params);
    let (value_from_hessian, gradient_from_hessian, hessian) =
        objective.value_grad_hessian(&probe_params);

    assert_near(value, value_from_grad, 1e-12);
    assert_near(value, value_from_raw_hessian, 1e-12);
    assert_near(value, value_from_hessian, 1e-12);
    for index in 0..probe_params.len() {
        assert_near(gradient[index], gradient_from_raw_hessian[index], 1e-10);
        assert_near(gradient[index], gradient_from_hessian[index], 1e-10);
    }

    let mut numerical_gradient = vec![0.0; probe_params.len()];
    models::central_diff_gradient_from_value(
        &probe_params,
        1e-6,
        1e-7,
        |param| objective.value(param),
        &mut numerical_gradient,
    );
    for index in 0..probe_params.len() {
        assert_near(gradient[index], numerical_gradient[index], 2e-5);
    }

    let numerical_hessian =
        models::central_diff_hessian_from_gradient(&probe_params, 2e-4, 1e-6, |param, output| {
            let (_, gradient_probe) = objective.value_grad(param);
            output.copy_from_slice(&gradient_probe);
        });
    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(raw_hessian[[row, column]], raw_hessian[[column, row]], 3e-4);
            assert_near(hessian[[row, column]], hessian[[column, row]], 3e-4);
            assert_near(
                hessian[[row, column]],
                numerical_hessian[[row, column]],
                3e-4,
            );
        }
    }
}

#[test]
fn curve_objective_emg_matches_numerical_derivatives() {
    let x_values = [-1.5, -0.8, -0.2, 0.3, 1.0, 1.8];
    let true_params = [1.4, 0.2, 0.6, 0.5, 0.1];
    let probe_params = [1.2, 0.1, 0.5, 0.4, 0.0];
    let y_values = x_values
        .iter()
        .map(|&x| models::value_at(CurveFamily::Emg, &true_params, x))
        .collect::<Vec<_>>();

    let term = models::DataTerm::new(
        CurveFamily::Emg,
        &x_values,
        &y_values,
        None,
        MsePredictionLoss,
    );
    let objective = models::CurveObjective::new(probe_params.len(), term);

    let value = objective.value(&probe_params);
    let (value_from_grad, gradient) = objective.value_grad(&probe_params);
    let (value_from_hessian, gradient_from_hessian, hessian) =
        objective.value_grad_hessian(&probe_params);

    assert_near(value, value_from_grad, 1e-12);
    assert_near(value, value_from_hessian, 1e-12);
    for index in 0..probe_params.len() {
        assert_near(gradient[index], gradient_from_hessian[index], 1e-9);
    }

    let mut numerical_gradient = vec![0.0; probe_params.len()];
    models::central_diff_gradient_from_value(
        &probe_params,
        1e-6,
        1e-7,
        |param| objective.value(param),
        &mut numerical_gradient,
    );
    for index in 0..probe_params.len() {
        assert_near(gradient[index], numerical_gradient[index], 5e-5);
    }

    let numerical_hessian =
        models::central_diff_hessian_from_gradient(&probe_params, 2e-4, 1e-6, |param, output| {
            let (_, gradient_probe) = objective.value_grad(param);
            output.copy_from_slice(&gradient_probe);
        });
    for row in 0..probe_params.len() {
        for column in 0..probe_params.len() {
            assert_near(
                hessian[[row, column]],
                numerical_hessian[[row, column]],
                2e-3,
            );
        }
    }
}
fn assert_close(actual: f64, expected: f64, abs_tolerance: f64, rel_tolerance: f64) {
    let delta = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    assert!(
        delta <= abs_tolerance.max(scale * rel_tolerance),
        "expected {expected}, got {actual}, delta={delta}"
    );
}

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

#[test]
fn simd_polynomial_cost_matches_scalar_reference_for_all_loss_metrics() {
    let (param, x_values, y_values) = test_xy_for_polynomial();
    for loss_metric in OptimizationLossMetric::ALL {
        let scalar = simd::polynomial_cost_scalar(&param, &x_values, &y_values, loss_metric);
        let simd = simd::polynomial_cost_simd(&param, &x_values, &y_values, loss_metric);
        assert_close(simd, scalar, 1e-10, 1e-10);
    }
}

#[test]
fn simd_inverse_cost_matches_scalar_reference_for_all_loss_metrics() {
    let (param, x_values, y_values) = test_xy_for_inverse();
    for loss_metric in OptimizationLossMetric::ALL {
        let scalar = simd::inverse_cost_scalar(&param, &x_values, &y_values, loss_metric);
        let simd = simd::inverse_cost_simd(&param, &x_values, &y_values, loss_metric);
        assert_close(simd, scalar, 1e-10, 1e-10);
    }
}

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

#[test]
fn polynomial_cost_with_quantization_uses_quantized_scalar_pipeline() {
    use argmin::core::CostFunction;

    let points = build_points(&[0.0, 1.0], |x| if x < 0.5 { 1.225 } else { -1.225 });
    let problem = CurveProblem::new_with_metric_quantization(
        CurveFamily::Linear,
        &points,
        None,
        OptimizationLossMetric::Mse,
        quantization(2),
    );

    let cost = CostFunction::cost(&problem, &ndarray::Array1::from_vec(vec![0.0, 0.0]))
        .expect("cost must be computed");
    assert_near(cost, 1.5129, 1e-12);
}

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
