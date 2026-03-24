#![cfg(feature = "portable-simd")]

use std::hint::black_box;
use std::time::{Duration, Instant};

use curve_fit::fit::{OptimizationLossMetric, simd_bench};

fn speedup_ratio(scalar_duration: Duration, simd_duration: Duration) -> f64 {
    scalar_duration.as_secs_f64() / simd_duration.as_secs_f64()
}

fn measure_cost<F>(iterations: usize, mut run: F) -> Duration
where
    F: FnMut() -> f64,
{
    let start = Instant::now();
    let mut sink = 0.0;
    for _ in 0..iterations {
        sink += black_box(run());
    }
    black_box(sink);
    start.elapsed()
}

fn measure_gradient<F>(iterations: usize, mut run: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    start.elapsed()
}

fn polynomial_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let param = vec![0.2, -0.08, 0.04, -0.015, 0.007, -0.003, 0.8, -0.4, 1.2, 0.3];
    let x_values = (0..160_003)
        .map(|index| index as f64 / 20_000.0 - 4.0)
        .collect::<Vec<_>>();
    let y_values = x_values
        .iter()
        .copied()
        .map(|x| {
            let model = param
                .iter()
                .copied()
                .fold(0.0, |acc, coefficient| acc * x + coefficient);
            model + 0.01 * (x * 3.0).sin()
        })
        .collect::<Vec<_>>();
    (param, x_values, y_values)
}

fn inverse_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let param = vec![1.4, -0.9];
    let x_values = (0..160_003)
        .map(|index| 0.05 + index as f64 / 8_000.0)
        .collect::<Vec<_>>();
    let y_values = x_values
        .iter()
        .copied()
        .map(|x| (param[0] + param[1] / x) + 0.005 * (x * 0.8).cos())
        .collect::<Vec<_>>();
    (param, x_values, y_values)
}

#[test]
#[ignore]
fn simd_perf_gate() {
    let loss_metric = OptimizationLossMetric::Mse;

    let (poly_param, poly_x, poly_y) = polynomial_dataset();
    let poly_cost_scalar = measure_cost(160, || {
        simd_bench::polynomial_cost_scalar(&poly_param, &poly_x, &poly_y, loss_metric)
    });
    let poly_cost_simd = measure_cost(160, || {
        simd_bench::polynomial_cost_simd(&poly_param, &poly_x, &poly_y, loss_metric)
    });
    let poly_cost_ratio = speedup_ratio(poly_cost_scalar, poly_cost_simd);
    println!(
        "Polynomial cost speedup: {:.3}x (scalar={:?}, simd={:?})",
        poly_cost_ratio, poly_cost_scalar, poly_cost_simd
    );
    assert!(
        poly_cost_ratio >= 2.0,
        "Polynomial cost speedup is below threshold: {:.3}x",
        poly_cost_ratio
    );

    let mut poly_gradient = vec![0.0; poly_param.len()];
    let poly_grad_scalar = measure_gradient(120, || {
        poly_gradient.fill(0.0);
        simd_bench::polynomial_gradient_scalar(
            &poly_x,
            &poly_y,
            &poly_param,
            loss_metric,
            &mut poly_gradient,
        );
        black_box(&poly_gradient);
    });
    let poly_grad_simd = measure_gradient(120, || {
        poly_gradient.fill(0.0);
        simd_bench::polynomial_gradient_simd(
            &poly_x,
            &poly_y,
            &poly_param,
            loss_metric,
            &mut poly_gradient,
        );
        black_box(&poly_gradient);
    });
    let poly_grad_ratio = speedup_ratio(poly_grad_scalar, poly_grad_simd);
    println!(
        "Polynomial gradient speedup: {:.3}x (scalar={:?}, simd={:?})",
        poly_grad_ratio, poly_grad_scalar, poly_grad_simd
    );
    assert!(
        poly_grad_ratio >= 1.7,
        "Polynomial gradient speedup is below threshold: {:.3}x",
        poly_grad_ratio
    );

    let (inverse_param, inverse_x, inverse_y) = inverse_dataset();
    let inverse_cost_scalar = measure_cost(200, || {
        simd_bench::inverse_cost_scalar(&inverse_param, &inverse_x, &inverse_y, loss_metric)
    });
    let inverse_cost_simd = measure_cost(200, || {
        simd_bench::inverse_cost_simd(&inverse_param, &inverse_x, &inverse_y, loss_metric)
    });
    let inverse_cost_ratio = speedup_ratio(inverse_cost_scalar, inverse_cost_simd);
    println!(
        "Inverse cost speedup: {:.3}x (scalar={:?}, simd={:?})",
        inverse_cost_ratio, inverse_cost_scalar, inverse_cost_simd
    );
    assert!(
        inverse_cost_ratio >= 1.4,
        "Inverse cost speedup is below threshold: {:.3}x",
        inverse_cost_ratio
    );

    let mut inverse_gradient = vec![0.0; 2];
    let inverse_grad_scalar = measure_gradient(200, || {
        inverse_gradient.fill(0.0);
        simd_bench::inverse_gradient_scalar(
            &inverse_x,
            &inverse_y,
            &inverse_param,
            loss_metric,
            &mut inverse_gradient,
        );
        black_box(&inverse_gradient);
    });
    let inverse_grad_simd = measure_gradient(200, || {
        inverse_gradient.fill(0.0);
        simd_bench::inverse_gradient_simd(
            &inverse_x,
            &inverse_y,
            &inverse_param,
            loss_metric,
            &mut inverse_gradient,
        );
        black_box(&inverse_gradient);
    });
    let inverse_grad_ratio = speedup_ratio(inverse_grad_scalar, inverse_grad_simd);
    println!(
        "Inverse gradient speedup: {:.3}x (scalar={:?}, simd={:?})",
        inverse_grad_ratio, inverse_grad_scalar, inverse_grad_simd
    );
    assert!(
        inverse_grad_ratio >= 1.25,
        "Inverse gradient speedup is below threshold: {:.3}x",
        inverse_grad_ratio
    );
}
