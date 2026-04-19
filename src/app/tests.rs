//! Общие тестовые фикстуры и вспомогательные функции для модулей `app/tests/*`.

#[cfg(not(target_arch = "wasm32"))]
use super::FitWorkerMessage;
use super::{
    CLIPBOARD_IMPORT_ERROR_PREFIX, CurveFitApp, DiagnosticsTab, ExtendedMetrics, FitRunUiSeed,
    IterationDiagnostics, ModelChoice, OptimizerMethod, OptimizerPreset, POINTS_PARSE_ERROR_PREFIX,
    POINTS_POSITIVE_AXIS_EPS, ParamInitMethod, ParametricIterationTraceEntry, PointsEditorState,
    ReplayFrame, ReplayFramePayload, ReplayState, StatusMessage, UiLanguage,
    data_based_params_for_family, dialog_directory_from_path,
};
use crate::domain::{CurveFamily, CurveParams, FitResult, OptimizerConfig, Point, Points};
use crate::fit::{
    DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES, IterationMetricSnapshot, MetricQuantization,
    OptimizationLossMetric, SplineResult,
};
use eframe::egui;
use egui_plot::PlotPoint;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

// Все import-ы специально поднимаются в этот модуль,
// чтобы подмодули `app/tests/*` брали их через `use super::*;`.
mod fit_lifecycle;
mod import_export;
mod init_optimizer;
mod points_editing;
mod replay_diagnostics;

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

fn parsed_point_pairs(app: &mut CurveFitApp) -> Vec<(f64, f64)> {
    let cache = app.points_cache_with_policy(true);
    let points = cache
        .parsed_points
        .as_ref()
        .expect("points must parse successfully");
    points
        .iter()
        .map(|point| (point.x(), point.y()))
        .collect::<Vec<_>>()
}

#[cfg(not(target_arch = "wasm32"))]
fn write_temp_points_csv(contents: &[u8]) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};

    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must be after UNIX_EPOCH")
        .as_nanos();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "curve-fit-dialog-memory-{}-{suffix}.csv",
        std::process::id()
    ));
    std::fs::write(&path, contents).expect("temporary CSV test file must be writable");
    path
}

#[cfg(not(target_arch = "wasm32"))]
fn cleanup_temp_file(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

fn assert_approx_eq(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

fn metric_quantization(decimal_places: u8) -> MetricQuantization {
    MetricQuantization::from_ui_state(true, decimal_places)
        .expect("test decimal places must be valid")
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

#[cfg(not(target_arch = "wasm32"))]
fn make_linear_fit_app() -> CurveFitApp {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::Polynomial,
        polynomial_degree: 1,
        ..Default::default()
    };
    app.sync_parameter_inputs();
    app.points.text = "0 1\n1 3\n2 5\n3 7\n".to_string();
    app.invalidate_points_cache();
    app
}

#[cfg(not(target_arch = "wasm32"))]
fn make_linear_spline_fit_app() -> CurveFitApp {
    let mut app = CurveFitApp {
        selected_model: ModelChoice::LinearSpline,
        ..Default::default()
    };
    app.points.text = "0 1\n1 3\n2 5\n3 7\n4 9\n5 11\n6 13\n7 15\n8 17\n9 19\n".to_string();
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
