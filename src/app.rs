//! Главный модуль UI-приложения: состояние, панели, импорт данных и запуск фитинга.

use std::f64::consts::TAU;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use egui_file_dialog::FileDialog;
use egui_plot::{
    Legend, Line, LineStyle, Plot, PlotBounds, PlotPoint, PlotPoints, PlotResponse,
    Points as PlotPointsItem, VLine,
};

mod bootstrap;
mod layout;
mod model_catalog;
mod panel_state;
mod point_layers;
mod state;
mod status;
mod style;
mod types;

mod clipboard_import;
mod diagnostics;
mod file_import;
mod fit_worker;
mod formula;
mod i18n;
mod input_parse;
mod normalization;
mod optimizer;
mod param_init;
mod plot_utils;
mod points_state;
mod points_text;
mod replay;
mod result_export;
mod ui;

use self::diagnostics::{IterationDiagnostics, diagnostics_plot_y_axis_width};
use self::formula::model_formula_info;
use self::formula::{formula_svg_bytes, formula_svg_uri};
#[cfg(not(target_arch = "wasm32"))]
use self::i18n::file_import_icon_image;
use self::i18n::{
    actions_icon_image, center_origin_icon_image, clear_icon_image, clipboard_import_icon_image,
    family_label, fit_icon_image, fit_to_content_icon_image, github_mark_image,
    language_flag_image, layer_delete_icon_image, layer_duplicate_icon_image,
    layer_hidden_icon_image, layer_new_icon_image, layer_visible_icon_image, model_choice_label,
    open_formula_icon_image, optimization_loss_metric_label, origin_bottom_left_icon_image,
    panels_icon_image, param_init_method_disabled_label, param_init_method_label,
    param_init_method_name_en, redo_icon_image, replay_pause_icon_image, replay_play_icon_image,
    reset_icon_image, spline_extrapolation_label, spline_knot_strategy_label, spray_brush_label,
    stop_icon_image, tool_icon_image, tr, undo_icon_image, view_icon_image,
};
use self::model_catalog::{
    ModelChoice, ModelGroup, ResolvedModel, model_group, model_group_label,
    spline_duplicate_policy_label,
};
use self::normalization::ParametricNormalization;
use self::optimizer::{
    AdamInputState, LbfgsInputState, NelderMeadInputState, NewtonCgInputState, OptimizerPreset,
    OptimizerUiMode, SgdInputState, SteepestDescentInputState, adam_config_from_preset,
    infer_adam_preset, infer_lbfgs_preset, infer_nelder_mead_preset, infer_newton_cg_preset,
    infer_sgd_preset, infer_steepest_descent_preset, lbfgs_config_from_preset,
    nelder_mead_config_from_preset, newton_cg_config_from_preset, optimizer_method_label,
    optimizer_preset_label, sgd_config_from_preset, steepest_descent_config_from_preset,
};
use self::panel_state::{DiagnosticsTab, PanelState};
use self::param_init::{
    data_based_params_for_family, is_advanced_param_init_supported, polynomial_family,
    rational_family,
};
use self::plot_utils::{fit_bounds_for_content, plot_domain};
use self::point_layers::{PointLayer, PointLayerId, PointLayersState};
use self::points_state::{
    ParsedPointsCache, PointsEditorState, points_editor_cache_with_policy,
    set_points_editor_cache_from_valid_points,
};
use self::points_text::{
    parse_f64, parse_points_from_clipboard_text, parse_points_text_cache, points_to_text,
};
use self::replay::ReplayState;
#[cfg(test)]
use self::replay::{ReplayFrame, ReplayFramePayload};
use self::result_export::FitExportRecord;
pub use self::state::CurveFitApp;
#[cfg(not(target_arch = "wasm32"))]
use self::state::FitWorkerMessage;
use self::state::{FitRunUiSeed, ParametricIterationTraceEntry, SplineIterationTraceEntry};
#[cfg(target_arch = "wasm32")]
use self::state::{WasmFitJob, WasmFitRunner};
use self::status::StatusMessage;
#[cfg(not(target_arch = "wasm32"))]
use self::types::dialog_directory_from_path;
use self::types::{
    ExtendedMetrics, FormulaReferenceSection, FormulaSvgCache, ModelFormulaInfo, ParamInitMethod,
    PlotTool, SampledCurveCache, SprayBrush, UiLanguage, params_to_input_strings,
    tau_grid_to_input_strings,
};
use crate::domain::{
    AdamConfig, CurveFamily, CurveParams, DEFAULT_SATURATING_TREND_TAUS_YEARS, FitResult,
    LbfgsConfig, MAX_RATIONAL_DEGREE, MAX_SATURATING_TREND_TAU_COUNT, MIN_RATIONAL_DEGREE,
    MIN_SATURATING_TREND_TAU_COUNT, NelderMeadConfig, NewtonCgConfig, OptimizerConfig,
    OptimizerMethod, Point, Points, SaturatingTrendTauGrid, SgdConfig, SteepestDescentConfig,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::fit::FitError;
use crate::fit::IterationMetricSnapshot;
use crate::fit::OptimizationLossMetric;
use crate::fit::{
    DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES, MetricQuantization,
    MetricQuantizationDecimalPlaces, SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation,
    SplineFamilyKind, SplineKnotStrategy, SplineResult, build_spline_initial_curve_from_knot_y,
    calculate_iteration_metrics_with_quantization, calculate_metrics_with_quantization,
    default_spline_initial_knot_y, sample_curve,
};
use crate::fit::{
    IncrementalFitRunner, IncrementalFitStep, IncrementalSplineFitRunner, IncrementalSplineFitStep,
};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, TryRecvError};
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

const PARAMETRIC_PLOT_SAMPLES: usize = 200;
const C1_MIN: f64 = 1e-8;
const C2_MAX: f64 = 0.9999;
const STEP_MIN_MIN: f64 = 1e-12;
const STEP_MAX_MAX: f64 = 1e3;
const SPRAY_GAUSSIAN_SIGMA: f64 = 1.0 / 3.0;
const SPRAY_REFERENCE_FPS: f64 = 60.0;
const PARAM_INIT_RANDOM_MIN: f64 = -1.0;
const PARAM_INIT_RANDOM_MAX: f64 = 1.0;
const SPLINE_AUTO_SAMPLES_MIN: usize = 80;
const SPLINE_AUTO_SAMPLES_MAX: usize = 2_000;
const SPLINE_AUTO_SAMPLES_PER_KNOT: usize = 30;
const SPLINE_AUTO_SAMPLES_PER_POINT: usize = 3;
const DIAGNOSTICS_PANEL_DEFAULT_HEIGHT: f32 = 230.0;
const DIAGNOSTICS_PANEL_MIN_HEIGHT: f32 = 120.0;
const LEFT_PANEL_DEFAULT_WIDTH: f32 = 320.0;
const LEFT_PANEL_MIN_WIDTH: f32 = 320.0;
const LEFT_PANEL_MAX_WIDTH: f32 = 460.0;
const RIGHT_PANEL_DEFAULT_WIDTH: f32 = 280.0;
const RIGHT_PANEL_MIN_WIDTH: f32 = 280.0;
const POINTS_PARSE_DEBOUNCE_MS: u64 = 180;
const POINTS_HISTORY_LIMIT: usize = 256;
const POINTS_PARSE_ERROR_PREFIX: &str = "Points parse error: ";
const CLIPBOARD_IMPORT_ERROR_PREFIX: &str = "Clipboard import error: ";
const FILE_IMPORT_ERROR_PREFIX: &str = "File import error: ";
#[cfg(target_arch = "wasm32")]
const CLIPBOARD_COPY_ERROR_PREFIX: &str = "Clipboard copy error: ";
#[cfg(not(target_arch = "wasm32"))]
const CLIPBOARD_IMPORT_PASTE_TIMEOUT_MS: u64 = 1_500;
const POINTS_POSITIVE_AXIS_EPS: f64 = 1e-6;
const UI_CORNER_RADIUS: u8 = 6;
const PANEL_INNER_MARGIN_X: i8 = 10;
const PANEL_INNER_MARGIN_Y: i8 = 8;
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const APP_REPOSITORY_URL: &str = env!("CARGO_PKG_REPOSITORY");
const REPLAY_FAST_REPAINT_INTERVAL_MS: u64 = 16;

#[cfg(test)]
mod tests;
