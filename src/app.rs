use std::f64::consts::TAU;
use std::sync::Arc;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use eframe::egui;
use egui_plot::{
    Legend, Line, LineStyle, Plot, PlotBounds, PlotPoint, PlotPoints, PlotResponse,
    Points as PlotPointsItem, VLine,
};

mod diagnostics;
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
mod ui;

use self::diagnostics::{IterationDiagnostics, diagnostics_plot_y_axis_width};
use self::formula::formula_plain_text;
use self::formula::model_formula_info;
#[cfg(not(target_arch = "wasm32"))]
use self::formula::{formula_svg_bytes, formula_svg_uri};
use self::i18n::{
    center_origin_icon_image, clear_icon_image, family_label, fit_icon_image,
    fit_to_content_icon_image, github_mark_image, language_flag_image, model_choice_label,
    open_formula_icon_image, optimization_loss_metric_label, origin_bottom_left_icon_image,
    panels_icon_image, param_init_method_disabled_label, param_init_method_label,
    param_init_method_name_en, redo_icon_image, replay_pause_icon_image, replay_play_icon_image,
    reset_icon_image, spline_extrapolation_label, spline_knot_strategy_label, spray_brush_label,
    stop_icon_image, tool_icon_image, tool_label, tr, undo_icon_image, view_icon_image,
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
use self::param_init::{
    data_based_params_for_family, is_advanced_param_init_supported, polynomial_family,
};
use self::plot_utils::{fit_bounds_for_content, plot_domain};
use self::points_state::{ParsedPointsCache, PointsEditorState};
use self::points_text::{parse_f64, parse_points_text_cache, points_to_text};
use self::replay::ReplayState;
#[cfg(test)]
use self::replay::{ReplayFrame, ReplayFramePayload};
use crate::domain::{
    AdamConfig, CurveFamily, CurveParams, FitResult, LbfgsConfig, NelderMeadConfig, NewtonCgConfig,
    OptimizerConfig, OptimizerMethod, Point, Points, SgdConfig, SteepestDescentConfig,
};
use crate::fit::IterationMetricSnapshot;
use crate::fit::OptimizationLossMetric;
use crate::fit::{
    DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES, MetricQuantization,
    MetricQuantizationDecimalPlaces, SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation,
    SplineFamilyKind, SplineKnotStrategy, SplineResult, build_spline_initial_curve_from_knot_y,
    calculate_iteration_metrics_with_quantization, calculate_metrics_with_quantization,
    default_spline_initial_knot_y, sample_curve,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::fit::{
    FitError, fit_curve_with_progress_and_optimizer_config_and_loss_metric_and_metric_quantization,
};
#[cfg(target_arch = "wasm32")]
use crate::fit::{
    IncrementalFitRunner, IncrementalFitStep, IncrementalSplineFitRunner, IncrementalSplineFitStep,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::fit::{IncrementalSplineFitRunner, IncrementalSplineFitStep};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, TryRecvError};

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
const RIGHT_PANEL_DEFAULT_WIDTH: f32 = 280.0;
const RIGHT_PANEL_MIN_WIDTH: f32 = 280.0;
const POINTS_PARSE_DEBOUNCE_MS: u64 = 180;
const POINTS_HISTORY_LIMIT: usize = 256;
const POINTS_PARSE_ERROR_PREFIX: &str = "Points parse error: ";
const UI_CORNER_RADIUS: u8 = 6;
const PANEL_INNER_MARGIN_X: i8 = 10;
const PANEL_INNER_MARGIN_Y: i8 = 8;
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const APP_REPOSITORY_URL: &str = env!("CARGO_PKG_REPOSITORY");
const REPLAY_FAST_REPAINT_INTERVAL_MS: u64 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum UiLanguage {
    #[default]
    English,
    Russian,
}

impl UiLanguage {
    const ALL: [Self; 2] = [Self::English, Self::Russian];

    fn native_name(self) -> &'static str {
        match self {
            Self::English => "English",
            Self::Russian => "Русский",
        }
    }

    fn from_locale_tag(locale: &str) -> Self {
        let language = locale
            .trim()
            .split(['-', '_', '.', '@', ':', ','])
            .next()
            .unwrap_or_default();

        if language.eq_ignore_ascii_case("ru") {
            Self::Russian
        } else {
            Self::English
        }
    }

    fn from_system_locale() -> Self {
        system_locale_tag()
            .as_deref()
            .map(Self::from_locale_tag)
            .unwrap_or(Self::English)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn system_locale_tag() -> Option<String> {
    sys_locale::get_locale()
}

#[cfg(target_arch = "wasm32")]
fn system_locale_tag() -> Option<String> {
    web_sys::window().and_then(|window| window.navigator().language())
}

fn params_to_input_strings(params: &CurveParams) -> Vec<String> {
    params
        .values()
        .into_iter()
        .map(|value| value.to_string())
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum PlotTool {
    None,
    #[default]
    SinglePoint,
    Spray,
    Eraser,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SprayBrush {
    #[default]
    Uniform,
    Gaussian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamInitMethod {
    Default,
    DataBased,
    Randomized,
}

impl ParamInitMethod {
    const ALL: [Self; 3] = [Self::Default, Self::DataBased, Self::Randomized];

    fn is_supported_for_family(self, family: CurveFamily) -> bool {
        match self {
            Self::Default => true,
            Self::DataBased | Self::Randomized => is_advanced_param_init_supported(family),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ModelChoice {
    #[default]
    Polynomial,
    Arrhenius,
    Inverse,
    Logistic,
    Gompertz,
    BiExponential,
    DampedSinusoid,
    Lorentzian,
    NaturalLog,
    FourPl,
    FivePl,
    MichaelisMenten,
    ExponentialBasic,
    ExponentialLinear,
    ExponentialHalfLife,
    FallingExponential,
    HyperbolicTangent,
    ArctangentStep,
    Softplus,
    Power,
    Gaussian,
    LinearSpline,
    MonotoneCubicSpline,
    NaturalCubicSpline,
    AkimaSpline,
}

impl ModelChoice {
    const ALL: [Self; 25] = [
        Self::Polynomial,
        Self::Arrhenius,
        Self::Inverse,
        Self::Logistic,
        Self::Gompertz,
        Self::BiExponential,
        Self::DampedSinusoid,
        Self::Lorentzian,
        Self::NaturalLog,
        Self::FourPl,
        Self::FivePl,
        Self::MichaelisMenten,
        Self::ExponentialBasic,
        Self::ExponentialLinear,
        Self::ExponentialHalfLife,
        Self::FallingExponential,
        Self::HyperbolicTangent,
        Self::ArctangentStep,
        Self::Softplus,
        Self::Power,
        Self::Gaussian,
        Self::LinearSpline,
        Self::MonotoneCubicSpline,
        Self::NaturalCubicSpline,
        Self::AkimaSpline,
    ];

    fn is_polynomial(self) -> bool {
        matches!(self, Self::Polynomial)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedModel {
    Parametric(CurveFamily),
    LinearSpline,
    MonotoneCubicSpline,
    NaturalCubicSpline,
    AkimaSpline,
}

impl ResolvedModel {
    fn from_choice(choice: ModelChoice, polynomial_degree: usize) -> Self {
        match choice {
            ModelChoice::Polynomial => Self::Parametric(polynomial_family(polynomial_degree)),
            ModelChoice::Arrhenius => Self::Parametric(CurveFamily::Arrhenius),
            ModelChoice::Inverse => Self::Parametric(CurveFamily::Inverse),
            ModelChoice::Logistic => Self::Parametric(CurveFamily::Logistic),
            ModelChoice::Gompertz => Self::Parametric(CurveFamily::Gompertz),
            ModelChoice::BiExponential => Self::Parametric(CurveFamily::BiExponential),
            ModelChoice::DampedSinusoid => Self::Parametric(CurveFamily::DampedSinusoid),
            ModelChoice::Lorentzian => Self::Parametric(CurveFamily::Lorentzian),
            ModelChoice::NaturalLog => Self::Parametric(CurveFamily::NaturalLog),
            ModelChoice::FourPl => Self::Parametric(CurveFamily::FourPl),
            ModelChoice::FivePl => Self::Parametric(CurveFamily::FivePl),
            ModelChoice::MichaelisMenten => Self::Parametric(CurveFamily::MichaelisMenten),
            ModelChoice::ExponentialBasic => Self::Parametric(CurveFamily::ExponentialBasic),
            ModelChoice::ExponentialLinear => Self::Parametric(CurveFamily::ExponentialLinear),
            ModelChoice::ExponentialHalfLife => Self::Parametric(CurveFamily::ExponentialHalfLife),
            ModelChoice::FallingExponential => Self::Parametric(CurveFamily::FallingExponential),
            ModelChoice::HyperbolicTangent => Self::Parametric(CurveFamily::HyperbolicTangent),
            ModelChoice::ArctangentStep => Self::Parametric(CurveFamily::ArctangentStep),
            ModelChoice::Softplus => Self::Parametric(CurveFamily::Softplus),
            ModelChoice::Power => Self::Parametric(CurveFamily::Power),
            ModelChoice::Gaussian => Self::Parametric(CurveFamily::Gaussian),
            ModelChoice::LinearSpline => Self::LinearSpline,
            ModelChoice::MonotoneCubicSpline => Self::MonotoneCubicSpline,
            ModelChoice::NaturalCubicSpline => Self::NaturalCubicSpline,
            ModelChoice::AkimaSpline => Self::AkimaSpline,
        }
    }

    fn parametric_family(self) -> Option<CurveFamily> {
        match self {
            Self::Parametric(family) => Some(family),
            Self::LinearSpline
            | Self::MonotoneCubicSpline
            | Self::NaturalCubicSpline
            | Self::AkimaSpline => None,
        }
    }

    fn spline_family(self) -> Option<SplineFamilyKind> {
        match self {
            Self::LinearSpline => Some(SplineFamilyKind::Linear),
            Self::MonotoneCubicSpline => Some(SplineFamilyKind::MonotoneCubic),
            Self::NaturalCubicSpline => Some(SplineFamilyKind::NaturalCubic),
            Self::AkimaSpline => Some(SplineFamilyKind::Akima),
            Self::Parametric(_) => None,
        }
    }

    fn spline_min_knots(self) -> Option<usize> {
        match self {
            Self::Parametric(_) => None,
            Self::LinearSpline | Self::MonotoneCubicSpline => Some(2),
            Self::NaturalCubicSpline => Some(3),
            Self::AkimaSpline => Some(5),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelGroup {
    Polynomial,
    ParametricGeneral,
    ParametricSigmoid,
    ParametricPeak,
    Spline,
}

impl ModelGroup {
    const ALL: [Self; 5] = [
        Self::Polynomial,
        Self::ParametricGeneral,
        Self::ParametricSigmoid,
        Self::ParametricPeak,
        Self::Spline,
    ];
}

fn model_group(model: ModelChoice) -> ModelGroup {
    match model {
        ModelChoice::Polynomial => ModelGroup::Polynomial,
        ModelChoice::Logistic
        | ModelChoice::Gompertz
        | ModelChoice::FourPl
        | ModelChoice::FivePl
        | ModelChoice::HyperbolicTangent
        | ModelChoice::ArctangentStep
        | ModelChoice::Softplus => ModelGroup::ParametricSigmoid,
        ModelChoice::Lorentzian | ModelChoice::Gaussian => ModelGroup::ParametricPeak,
        ModelChoice::LinearSpline
        | ModelChoice::MonotoneCubicSpline
        | ModelChoice::NaturalCubicSpline
        | ModelChoice::AkimaSpline => ModelGroup::Spline,
        ModelChoice::Arrhenius
        | ModelChoice::Inverse
        | ModelChoice::BiExponential
        | ModelChoice::DampedSinusoid
        | ModelChoice::NaturalLog
        | ModelChoice::MichaelisMenten
        | ModelChoice::ExponentialBasic
        | ModelChoice::ExponentialLinear
        | ModelChoice::ExponentialHalfLife
        | ModelChoice::FallingExponential
        | ModelChoice::Power => ModelGroup::ParametricGeneral,
    }
}

fn model_group_label(language: UiLanguage, group: ModelGroup) -> &'static str {
    match (language, group) {
        (UiLanguage::English, ModelGroup::Polynomial) => "Polynomial",
        (UiLanguage::English, ModelGroup::ParametricGeneral) => "Parametric (General)",
        (UiLanguage::English, ModelGroup::ParametricSigmoid) => "Parametric (Sigmoid/Step)",
        (UiLanguage::English, ModelGroup::ParametricPeak) => "Parametric (Peak)",
        (UiLanguage::English, ModelGroup::Spline) => "Spline",
        (UiLanguage::Russian, ModelGroup::Polynomial) => "Полиномы",
        (UiLanguage::Russian, ModelGroup::ParametricGeneral) => "Параметрические (общие)",
        (UiLanguage::Russian, ModelGroup::ParametricSigmoid) => {
            "Параметрические (сигмоиды/переходы)"
        }
        (UiLanguage::Russian, ModelGroup::ParametricPeak) => "Параметрические (пики)",
        (UiLanguage::Russian, ModelGroup::Spline) => "Сплайны",
    }
}

fn spline_duplicate_policy_label(
    language: UiLanguage,
    policy: SplineDuplicateXPolicy,
) -> &'static str {
    match (language, policy) {
        (UiLanguage::English, SplineDuplicateXPolicy::Error) => "Error on duplicates",
        (UiLanguage::English, SplineDuplicateXPolicy::MeanY) => "Merge by mean(y)",
        (UiLanguage::English, SplineDuplicateXPolicy::MedianY) => "Merge by median(y)",
        (UiLanguage::English, SplineDuplicateXPolicy::FirstY) => "Keep first y",
        (UiLanguage::Russian, SplineDuplicateXPolicy::Error) => "Ошибка при дублях",
        (UiLanguage::Russian, SplineDuplicateXPolicy::MeanY) => "Слить по mean(y)",
        (UiLanguage::Russian, SplineDuplicateXPolicy::MedianY) => "Слить по median(y)",
        (UiLanguage::Russian, SplineDuplicateXPolicy::FirstY) => "Оставить первый y",
    }
}

#[derive(Debug, Clone)]
struct ModelFormulaInfo {
    full_formula: String,
    notes: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
struct FormulaSvgCache {
    formula: String,
    dark_mode: bool,
    uri: String,
    bytes: Arc<[u8]>,
}

#[derive(Debug, Clone)]
struct SampledCurveCache {
    params: CurveParams,
    x_min_bits: u64,
    x_max_bits: u64,
    samples: usize,
    curve: Arc<[PlotPoint]>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ExtendedMetrics {
    mse: f64,
    rmse: f64,
    mae: f64,
    r2: f64,
    max_abs_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DiagnosticsTab {
    #[default]
    Loss,
    Residuals,
}

#[derive(Debug, Clone)]
struct PanelState {
    show_left: bool,
    show_right: bool,
    show_formula_window: bool,
    show_diagnostics: bool,
    diagnostics_tab: DiagnosticsTab,
    diagnostics_hide_non_loss_by_default_pending: bool,
    diagnostics_shared_axis_width: f32,
}

impl Default for PanelState {
    fn default() -> Self {
        Self {
            show_left: true,
            show_right: true,
            show_formula_window: false,
            show_diagnostics: true,
            diagnostics_tab: DiagnosticsTab::Loss,
            diagnostics_hide_non_loss_by_default_pending: true,
            diagnostics_shared_axis_width: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
enum StatusMessage {
    Ready,
    Cleared,
    FittingInProgress,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    FittingStopping,
    FitStopped,
    FitCompleted,
    Error(String),
}

impl StatusMessage {
    fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    fn text(&self, language: UiLanguage) -> &str {
        match self {
            Self::Ready => tr(language, "Ready", "Готово"),
            Self::Cleared => tr(language, "Input cleared", "Поле ввода очищено"),
            Self::FittingInProgress => tr(language, "Fitting in progress", "Подгонка в процессе"),
            Self::FittingStopping => tr(language, "Stopping fit...", "Останавливаем подгонку..."),
            Self::FitStopped => tr(language, "Fit stopped", "Подгонка остановлена"),
            Self::FitCompleted => tr(language, "Fit completed", "Фитинг завершен"),
            Self::Error(message) => message.as_str(),
        }
    }
}

enum ActiveOptimizerView<'a> {
    Lbfgs {
        inputs: &'a LbfgsInputState,
        preset: OptimizerPreset,
    },
    NelderMead {
        inputs: &'a NelderMeadInputState,
        preset: OptimizerPreset,
    },
    SteepestDescent {
        inputs: &'a SteepestDescentInputState,
        preset: OptimizerPreset,
    },
    NewtonCg {
        inputs: &'a NewtonCgInputState,
        preset: OptimizerPreset,
    },
    Sgd {
        inputs: &'a SgdInputState,
        preset: OptimizerPreset,
    },
    Adam {
        inputs: &'a AdamInputState,
        preset: OptimizerPreset,
    },
}

impl ActiveOptimizerView<'_> {
    fn preset(self) -> OptimizerPreset {
        match self {
            Self::Lbfgs { preset, .. }
            | Self::NelderMead { preset, .. }
            | Self::SteepestDescent { preset, .. }
            | Self::NewtonCg { preset, .. }
            | Self::Sgd { preset, .. }
            | Self::Adam { preset, .. } => preset,
        }
    }

    fn config(self) -> Result<OptimizerConfig, String> {
        match self {
            Self::Lbfgs { inputs, .. } => inputs.to_config().map(OptimizerConfig::Lbfgs),
            Self::NelderMead { inputs, .. } => inputs.to_config().map(OptimizerConfig::NelderMead),
            Self::SteepestDescent { inputs, .. } => {
                inputs.to_config().map(OptimizerConfig::SteepestDescent)
            }
            Self::NewtonCg { inputs, .. } => inputs.to_config().map(OptimizerConfig::NewtonCg),
            Self::Sgd { inputs, .. } => inputs.to_config().map(OptimizerConfig::Sgd),
            Self::Adam { inputs, .. } => inputs.to_config().map(OptimizerConfig::Adam),
        }
    }
}

enum ActiveOptimizerViewMut<'a> {
    Lbfgs {
        inputs: &'a mut LbfgsInputState,
        preset: &'a mut OptimizerPreset,
    },
    NelderMead {
        inputs: &'a mut NelderMeadInputState,
        preset: &'a mut OptimizerPreset,
    },
    SteepestDescent {
        inputs: &'a mut SteepestDescentInputState,
        preset: &'a mut OptimizerPreset,
    },
    NewtonCg {
        inputs: &'a mut NewtonCgInputState,
        preset: &'a mut OptimizerPreset,
    },
    Sgd {
        inputs: &'a mut SgdInputState,
        preset: &'a mut OptimizerPreset,
    },
    Adam {
        inputs: &'a mut AdamInputState,
        preset: &'a mut OptimizerPreset,
    },
}

impl ActiveOptimizerViewMut<'_> {
    fn set_preset(self, value: OptimizerPreset) {
        match self {
            Self::Lbfgs { preset, .. }
            | Self::NelderMead { preset, .. }
            | Self::SteepestDescent { preset, .. }
            | Self::NewtonCg { preset, .. }
            | Self::Sgd { preset, .. }
            | Self::Adam { preset, .. } => *preset = value,
        }
    }

    fn apply_preset(self, value: OptimizerPreset) {
        match self {
            Self::Lbfgs { inputs, preset } => {
                *inputs = LbfgsInputState::from_config(&lbfgs_config_from_preset(value));
                *preset = value;
            }
            Self::NelderMead { inputs, preset } => {
                *inputs = NelderMeadInputState::from_config(&nelder_mead_config_from_preset(value));
                *preset = value;
            }
            Self::SteepestDescent { inputs, preset } => {
                *inputs = SteepestDescentInputState::from_config(
                    &steepest_descent_config_from_preset(value),
                );
                *preset = value;
            }
            Self::NewtonCg { inputs, preset } => {
                *inputs = NewtonCgInputState::from_config(&newton_cg_config_from_preset(value));
                *preset = value;
            }
            Self::Sgd { inputs, preset } => {
                *inputs = SgdInputState::from_config(&sgd_config_from_preset(value));
                *preset = value;
            }
            Self::Adam { inputs, preset } => {
                *inputs = AdamInputState::from_config(&adam_config_from_preset(value));
                *preset = value;
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
enum FitWorkerMessage {
    Iteration {
        iteration: u64,
        metrics: IterationMetricSnapshot,
        params: CurveParams,
    },
    SplineIteration {
        iteration: u64,
        metrics: IterationMetricSnapshot,
        knot_y: Vec<f64>,
        curve: Vec<[f64; 2]>,
    },
    Stopped,
    Finished(FitResult),
    SplineFinished {
        result: SplineResult,
        metrics: IterationMetricSnapshot,
    },
    Failed(String),
}

#[cfg(target_arch = "wasm32")]
enum WasmFitRunner {
    Parametric {
        runner: IncrementalFitRunner,
        normalization: Option<ParametricNormalization>,
    },
    Spline(IncrementalSplineFitRunner),
}

#[cfg(target_arch = "wasm32")]
enum WasmFitJob {
    Deferred(WasmFitRunner),
    Running(WasmFitRunner),
}

/// Состояние и UI-логика интерактивного приложения для подгонки кривых.
pub struct CurveFitApp {
    points: PointsEditorState,
    selected_model: ModelChoice,
    polynomial_degree: usize,
    parameter_inputs: Vec<String>,
    optimizer_method: OptimizerMethod,
    optimizer_mode: OptimizerUiMode,
    optimization_loss_metric: OptimizationLossMetric,
    metric_quantization_enabled: bool,
    metric_quantization_decimal_places: u8,
    normalize_parametric_data: bool,
    lbfgs_inputs: LbfgsInputState,
    lbfgs_preset: OptimizerPreset,
    nelder_mead_inputs: NelderMeadInputState,
    nelder_mead_preset: OptimizerPreset,
    steepest_descent_inputs: SteepestDescentInputState,
    steepest_descent_preset: OptimizerPreset,
    newton_cg_inputs: NewtonCgInputState,
    newton_cg_preset: OptimizerPreset,
    sgd_inputs: SgdInputState,
    sgd_preset: OptimizerPreset,
    adam_inputs: AdamInputState,
    adam_preset: OptimizerPreset,
    ui_language: UiLanguage,
    plot_tool: PlotTool,
    spray_points_per_second: usize,
    spray_radius_rel: f64,
    spray_brush: SprayBrush,
    eraser_radius_rel: f64,
    spray_seed: u64,
    spray_points_budget: f64,
    spray_last_emit_at: Option<Instant>,
    fit_to_content_requested: bool,
    center_origin_requested: bool,
    origin_bottom_left_requested: bool,
    last_plot_bounds: Option<PlotBounds>,
    active_tool_bounds: Option<PlotBounds>,
    panel: PanelState,
    replay: ReplayState,
    spline_knots: usize,
    spline_knot_strategy: SplineKnotStrategy,
    spline_extrapolation: SplineExtrapolation,
    spline_duplicate_x_policy: SplineDuplicateXPolicy,
    spline_initial_knot_y_inputs: Vec<String>,
    fit_in_progress: bool,
    fit_loss_metric: OptimizationLossMetric,
    fit_metric_quantization: MetricQuantization,
    fit_preview_params: Option<CurveParams>,
    fit_preview_iteration: Option<u64>,
    fit_result: Option<FitResult>,
    spline_result: Option<SplineResult>,
    active_fit_points: Option<Points>,
    result_metrics: Option<ExtendedMetrics>,
    residual_plot_points: Vec<PlotPoint>,
    spline_plot_curve: Option<Arc<[PlotPoint]>>,
    #[cfg(not(target_arch = "wasm32"))]
    formula_svg_cache: Option<FormulaSvgCache>,
    sampled_curve_cache: Option<SampledCurveCache>,
    iteration_diagnostics: IterationDiagnostics,
    status: Option<StatusMessage>,
    #[cfg(not(target_arch = "wasm32"))]
    fit_worker_rx: Option<Receiver<FitWorkerMessage>>,
    #[cfg(not(target_arch = "wasm32"))]
    fit_cancel_flag: Option<Arc<AtomicBool>>,
    #[cfg(not(target_arch = "wasm32"))]
    discard_fit_worker_updates: bool,
    #[cfg(target_arch = "wasm32")]
    wasm_fit_job: Option<WasmFitJob>,
}

impl CurveFitApp {
    /// Создает приложение и настраивает загрузчики изображений для иконок/формул.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        Self {
            ui_language: UiLanguage::from_system_locale(),
            ..Self::default()
        }
    }

    fn resolved_model(&self) -> ResolvedModel {
        ResolvedModel::from_choice(self.selected_model, self.polynomial_degree)
    }

    fn active_optimizer_view(&self) -> ActiveOptimizerView<'_> {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => ActiveOptimizerView::Lbfgs {
                inputs: &self.lbfgs_inputs,
                preset: self.lbfgs_preset,
            },
            OptimizerMethod::NelderMead => ActiveOptimizerView::NelderMead {
                inputs: &self.nelder_mead_inputs,
                preset: self.nelder_mead_preset,
            },
            OptimizerMethod::SteepestDescent => ActiveOptimizerView::SteepestDescent {
                inputs: &self.steepest_descent_inputs,
                preset: self.steepest_descent_preset,
            },
            OptimizerMethod::NewtonCg => ActiveOptimizerView::NewtonCg {
                inputs: &self.newton_cg_inputs,
                preset: self.newton_cg_preset,
            },
            OptimizerMethod::Sgd => ActiveOptimizerView::Sgd {
                inputs: &self.sgd_inputs,
                preset: self.sgd_preset,
            },
            OptimizerMethod::Adam => ActiveOptimizerView::Adam {
                inputs: &self.adam_inputs,
                preset: self.adam_preset,
            },
        }
    }

    fn active_optimizer_view_mut(&mut self) -> ActiveOptimizerViewMut<'_> {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => ActiveOptimizerViewMut::Lbfgs {
                inputs: &mut self.lbfgs_inputs,
                preset: &mut self.lbfgs_preset,
            },
            OptimizerMethod::NelderMead => ActiveOptimizerViewMut::NelderMead {
                inputs: &mut self.nelder_mead_inputs,
                preset: &mut self.nelder_mead_preset,
            },
            OptimizerMethod::SteepestDescent => ActiveOptimizerViewMut::SteepestDescent {
                inputs: &mut self.steepest_descent_inputs,
                preset: &mut self.steepest_descent_preset,
            },
            OptimizerMethod::NewtonCg => ActiveOptimizerViewMut::NewtonCg {
                inputs: &mut self.newton_cg_inputs,
                preset: &mut self.newton_cg_preset,
            },
            OptimizerMethod::Sgd => ActiveOptimizerViewMut::Sgd {
                inputs: &mut self.sgd_inputs,
                preset: &mut self.sgd_preset,
            },
            OptimizerMethod::Adam => ActiveOptimizerViewMut::Adam {
                inputs: &mut self.adam_inputs,
                preset: &mut self.adam_preset,
            },
        }
    }

    fn selected_optimizer_preset(&self) -> OptimizerPreset {
        self.active_optimizer_view().preset()
    }

    fn set_selected_optimizer_preset(&mut self, preset: OptimizerPreset) {
        self.active_optimizer_view_mut().set_preset(preset);
    }

    fn apply_selected_optimizer_preset(&mut self, preset: OptimizerPreset) {
        self.active_optimizer_view_mut().apply_preset(preset);
    }

    fn optimizer_config(&self) -> Result<OptimizerConfig, String> {
        self.active_optimizer_view().config()
    }

    fn auto_spline_samples(points_len: usize, knots: usize) -> usize {
        // На больших датасетах используем более плотную дискретизацию,
        // но ограничиваем верхний порог ради отзывчивости UI.
        let by_knots = knots.saturating_mul(SPLINE_AUTO_SAMPLES_PER_KNOT);
        let by_points = points_len.saturating_mul(SPLINE_AUTO_SAMPLES_PER_POINT);
        by_knots
            .max(by_points)
            .clamp(SPLINE_AUTO_SAMPLES_MIN, SPLINE_AUTO_SAMPLES_MAX)
    }

    fn spline_config_for_model(
        &self,
        model: ResolvedModel,
        points_len: usize,
    ) -> Option<SplineConfig> {
        let min_knots = model.spline_min_knots()?;
        let knots = self.spline_knots.max(min_knots);
        Some(
            SplineConfig {
                knots,
                samples: Self::auto_spline_samples(points_len, knots),
                knot_strategy: self.spline_knot_strategy,
                extrapolation: self.spline_extrapolation,
                duplicate_x_policy: self.spline_duplicate_x_policy,
            }
            .normalized(),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn cached_formula_svg(&mut self, formula: &str, dark_mode: bool) -> (String, Arc<[u8]>) {
        if let Some(cache) = &self.formula_svg_cache
            && cache.formula == formula
            && cache.dark_mode == dark_mode
        {
            return (cache.uri.clone(), Arc::clone(&cache.bytes));
        }

        let uri = formula_svg_uri(formula, dark_mode);
        let bytes: Arc<[u8]> = formula_svg_bytes(formula, dark_mode).into();
        self.formula_svg_cache = Some(FormulaSvgCache {
            formula: formula.to_string(),
            dark_mode,
            uri: uri.clone(),
            bytes: Arc::clone(&bytes),
        });
        (uri, bytes)
    }

    fn cached_sampled_curve(
        &mut self,
        params: &CurveParams,
        x_min: f64,
        x_max: f64,
        samples: usize,
    ) -> Arc<[PlotPoint]> {
        let x_min_bits = x_min.to_bits();
        let x_max_bits = x_max.to_bits();
        if let Some(cache) = &self.sampled_curve_cache
            && cache.params == *params
            && cache.x_min_bits == x_min_bits
            && cache.x_max_bits == x_max_bits
            && cache.samples == samples
        {
            return Arc::clone(&cache.curve);
        }

        let curve: Arc<[PlotPoint]> = sample_curve(params, x_min, x_max, samples)
            .into_iter()
            .map(|[x, y]| PlotPoint::new(x, y))
            .collect::<Vec<_>>()
            .into();
        self.sampled_curve_cache = Some(SampledCurveCache {
            params: params.clone(),
            x_min_bits,
            x_max_bits,
            samples,
            curve: Arc::clone(&curve),
        });
        curve
    }

    fn sync_parameter_inputs(&mut self) {
        if let Some(family) = self.resolved_model().parametric_family() {
            let default_params = family.default_params();
            self.set_parameter_inputs_from_params(&default_params);
        } else {
            self.parameter_inputs.clear();
        }
    }

    fn set_parameter_inputs_from_params(&mut self, params: &CurveParams) {
        self.parameter_inputs = params_to_input_strings(params);
    }

    fn sync_spline_initial_knot_y_inputs(&mut self, knot_count: usize) {
        if self.spline_initial_knot_y_inputs.len() < knot_count {
            self.spline_initial_knot_y_inputs
                .resize_with(knot_count, || "0.0".to_string());
        } else {
            self.spline_initial_knot_y_inputs.truncate(knot_count);
        }
    }

    fn set_spline_initial_knot_y_inputs(&mut self, values: &[f64]) {
        self.spline_initial_knot_y_inputs = values.iter().map(|value| value.to_string()).collect();
    }

    fn selected_metric_quantization(&self) -> Result<MetricQuantization, String> {
        MetricQuantization::from_ui_state(
            self.metric_quantization_enabled,
            self.metric_quantization_decimal_places,
        )
    }

    fn clear_fit_outputs(&mut self) {
        self.cancel_fit_and_discard_updates();
        self.fit_result = None;
        self.spline_result = None;
        self.active_fit_points = None;
        self.result_metrics = None;
        self.residual_plot_points.clear();
        self.spline_plot_curve = None;
        self.sampled_curve_cache = None;
        self.iteration_diagnostics.clear();
        self.panel.diagnostics_tab = DiagnosticsTab::Loss;
        self.panel.diagnostics_hide_non_loss_by_default_pending = true;
        self.clear_fit_preview();
        self.clear_replay_state();
    }

    fn spline_family_and_init_config(&self) -> Option<(SplineFamilyKind, SplineConfig)> {
        let model = self.resolved_model();
        let family = model.spline_family()?;
        let config = self.spline_config_for_model(model, 2)?;
        Some((family, config))
    }

    fn build_randomized_spline_initial_knot_y(&mut self, knot_count: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(knot_count);
        for _ in 0..knot_count {
            let random = self.next_unit_random();
            let value =
                PARAM_INIT_RANDOM_MIN + (PARAM_INIT_RANDOM_MAX - PARAM_INIT_RANDOM_MIN) * random;
            values.push(value);
        }
        values
    }

    fn build_data_based_initial_params(
        &mut self,
        family: CurveFamily,
    ) -> Result<CurveParams, String> {
        if !ParamInitMethod::DataBased.is_supported_for_family(family) {
            return Err(format!(
                "Data-based initialization is not supported for family {family}"
            ));
        }

        let points = self.parse_points_strict()?;
        family
            .validate_points(&points)
            .map_err(|error| error.to_string())?;
        data_based_params_for_family(family, &points)
    }

    fn build_randomized_initial_params(
        &mut self,
        family: CurveFamily,
    ) -> Result<CurveParams, String> {
        if !ParamInitMethod::Randomized.is_supported_for_family(family) {
            return Err(format!(
                "Randomized initialization is not supported for family {family}"
            ));
        }

        let mut values = Vec::with_capacity(family.parameter_count());
        for _ in 0..family.parameter_count() {
            let random = self.next_unit_random();
            let value =
                PARAM_INIT_RANDOM_MIN + (PARAM_INIT_RANDOM_MAX - PARAM_INIT_RANDOM_MIN) * random;
            values.push(value);
        }

        CurveParams::try_from_values(family, values).map_err(|error| error.to_string())
    }

    fn has_fitted_params_for_family(&self, family: CurveFamily) -> bool {
        self.fit_result
            .as_ref()
            .is_some_and(|result| result.family == family)
    }

    fn build_fitted_initial_params(&self, family: CurveFamily) -> Result<CurveParams, String> {
        let Some(result) = &self.fit_result else {
            return Err("No fitted model parameters are available for initialization".to_string());
        };

        if result.family != family {
            return Err(format!(
                "Fitted model family mismatch: expected {family}, got {}",
                result.family
            ));
        }

        Ok(result.params.clone())
    }

    fn apply_fitted_param_init(&mut self) {
        let Some(family) = self.resolved_model().parametric_family() else {
            self.status = Some(StatusMessage::Error(
                "Current model is non-parametric and has no initial parameters".to_string(),
            ));
            return;
        };

        match self.build_fitted_initial_params(family) {
            Ok(params) => {
                self.set_parameter_inputs_from_params(&params);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Ready);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
            }
        }
    }

    fn apply_param_init_method(&mut self, method: ParamInitMethod) {
        let Some(family) = self.resolved_model().parametric_family() else {
            self.status = Some(StatusMessage::Error(
                "Current model is non-parametric and has no initial parameters".to_string(),
            ));
            return;
        };

        if !method.is_supported_for_family(family) {
            self.status = Some(StatusMessage::Error(format!(
                "Initialization method '{}' is not supported for family {family}",
                param_init_method_name_en(method),
            )));
            return;
        }

        let params_result = match method {
            ParamInitMethod::Default => Ok(family.default_params()),
            ParamInitMethod::DataBased => self.build_data_based_initial_params(family),
            ParamInitMethod::Randomized => self.build_randomized_initial_params(family),
        };

        match params_result {
            Ok(params) => {
                self.set_parameter_inputs_from_params(&params);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Ready);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
            }
        }
    }

    fn apply_spline_param_init_method(&mut self, method: ParamInitMethod) {
        let Some((family, config)) = self.spline_family_and_init_config() else {
            self.status = Some(StatusMessage::Error(
                "Current model is parametric and has no spline parameters".to_string(),
            ));
            return;
        };

        self.sync_spline_initial_knot_y_inputs(config.knots);
        let values_result = match method {
            ParamInitMethod::Default => Ok(vec![0.0; config.knots]),
            ParamInitMethod::DataBased => {
                let points = match self.parse_points_strict() {
                    Ok(points) => points,
                    Err(error) => {
                        self.status = Some(StatusMessage::Error(error));
                        return;
                    }
                };
                default_spline_initial_knot_y(&points, family, config)
                    .map_err(|error| error.to_string())
            }
            ParamInitMethod::Randomized => {
                Ok(self.build_randomized_spline_initial_knot_y(config.knots))
            }
        };

        match values_result {
            Ok(values) => {
                self.set_spline_initial_knot_y_inputs(&values);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Ready);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
            }
        }
    }

    fn apply_visual_style(ctx: &egui::Context) {
        ctx.global_style_mut(|style| {
            style.spacing.item_spacing = egui::vec2(10.0, 8.0);
            style.spacing.button_padding = egui::vec2(8.0, 5.0);
            style.spacing.interact_size = egui::vec2(44.0, 26.0);
            style.spacing.slider_width = 170.0;
            style.spacing.combo_width = 180.0;
            style.spacing.indent = 14.0;

            style.text_styles.insert(
                egui::TextStyle::Heading,
                egui::FontId::new(21.0, egui::FontFamily::Proportional),
            );
            style.text_styles.insert(
                egui::TextStyle::Body,
                egui::FontId::new(14.0, egui::FontFamily::Proportional),
            );
            style.text_styles.insert(
                egui::TextStyle::Button,
                egui::FontId::new(14.0, egui::FontFamily::Proportional),
            );
            style.text_styles.insert(
                egui::TextStyle::Monospace,
                egui::FontId::new(13.0, egui::FontFamily::Monospace),
            );
            style.text_styles.insert(
                egui::TextStyle::Small,
                egui::FontId::new(12.0, egui::FontFamily::Proportional),
            );

            let visuals = &mut style.visuals;
            visuals.widgets.noninteractive.corner_radius =
                egui::CornerRadius::same(UI_CORNER_RADIUS);
            visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(UI_CORNER_RADIUS);
            visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(UI_CORNER_RADIUS);
            visuals.widgets.active.corner_radius = egui::CornerRadius::same(UI_CORNER_RADIUS);
            visuals.widgets.open.corner_radius = egui::CornerRadius::same(UI_CORNER_RADIUS);

            if visuals.dark_mode {
                visuals.panel_fill = egui::Color32::from_rgb(14, 17, 22);
                visuals.window_fill = egui::Color32::from_rgb(17, 20, 26);
                visuals.faint_bg_color = egui::Color32::from_rgb(24, 30, 38);
                visuals.extreme_bg_color = egui::Color32::from_rgb(8, 11, 16);
                visuals.code_bg_color = egui::Color32::from_rgb(10, 20, 28);
                visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(52, 70, 85));
                visuals.selection.bg_fill = egui::Color32::from_rgb(22, 88, 120);
                visuals.selection.stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(152, 226, 255));
                visuals.hyperlink_color = egui::Color32::from_rgb(94, 204, 255);
                visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(28, 35, 44);
                visuals.widgets.inactive.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(52, 70, 85));
                visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(34, 49, 61);
                visuals.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 113, 138));
                visuals.widgets.active.weak_bg_fill = egui::Color32::from_rgb(27, 84, 108);
                visuals.widgets.active.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(86, 171, 211));
                visuals.widgets.open.weak_bg_fill = egui::Color32::from_rgb(33, 57, 73);
                visuals.widgets.open.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(72, 122, 150));
            } else {
                visuals.panel_fill = egui::Color32::from_rgb(239, 245, 249);
                visuals.window_fill = egui::Color32::from_rgb(246, 250, 252);
                visuals.faint_bg_color = egui::Color32::from_rgb(225, 236, 242);
                visuals.extreme_bg_color = egui::Color32::from_rgb(251, 253, 255);
                visuals.code_bg_color = egui::Color32::from_rgb(235, 245, 250);
                visuals.window_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(165, 188, 201));
                visuals.selection.bg_fill = egui::Color32::from_rgb(150, 214, 235);
                visuals.selection.stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(20, 76, 96));
                visuals.hyperlink_color = egui::Color32::from_rgb(0, 118, 163);
                visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(220, 234, 241);
                visuals.widgets.inactive.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(163, 189, 203));
                visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(208, 227, 237);
                visuals.widgets.hovered.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(128, 170, 192));
                visuals.widgets.active.weak_bg_fill = egui::Color32::from_rgb(183, 220, 236);
                visuals.widgets.active.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(87, 151, 182));
                visuals.widgets.open.weak_bg_fill = egui::Color32::from_rgb(198, 224, 236);
                visuals.widgets.open.bg_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(103, 160, 188));
            }
        });
    }

    fn side_panel_frame(style: &egui::Style) -> egui::Frame {
        egui::Frame::side_top_panel(style)
            .inner_margin(egui::Margin::symmetric(
                PANEL_INNER_MARGIN_X,
                PANEL_INNER_MARGIN_Y,
            ))
            .fill(style.visuals.panel_fill)
            .stroke(egui::Stroke::new(
                1.0,
                style.visuals.widgets.noninteractive.bg_stroke.color,
            ))
    }

    fn top_bottom_panel_frame(style: &egui::Style) -> egui::Frame {
        egui::Frame::side_top_panel(style)
            .inner_margin(egui::Margin::symmetric(PANEL_INNER_MARGIN_X, 6))
            .fill(style.visuals.panel_fill)
            .stroke(egui::Stroke::new(
                1.0,
                style.visuals.widgets.noninteractive.bg_stroke.color,
            ))
    }
}

impl Default for CurveFitApp {
    fn default() -> Self {
        let selected_model = ModelChoice::Polynomial;
        let polynomial_degree = 1;
        let selected_family = polynomial_family(polynomial_degree);
        let default_lbfgs = LbfgsConfig::default();
        let default_nelder_mead = NelderMeadConfig::default();
        let default_steepest_descent = SteepestDescentConfig::default();
        let default_newton_cg = NewtonCgConfig::default();
        let default_sgd = SgdConfig::default();
        let default_adam = AdamConfig::default();

        Self {
            points: PointsEditorState::default(),
            selected_model,
            polynomial_degree,
            parameter_inputs: params_to_input_strings(&selected_family.default_params()),
            optimizer_method: OptimizerMethod::Lbfgs,
            optimizer_mode: OptimizerUiMode::Basic,
            optimization_loss_metric: OptimizationLossMetric::default(),
            metric_quantization_enabled: false,
            metric_quantization_decimal_places: DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES,
            normalize_parametric_data: false,
            lbfgs_inputs: LbfgsInputState::from_config(&default_lbfgs),
            lbfgs_preset: infer_lbfgs_preset(&default_lbfgs),
            nelder_mead_inputs: NelderMeadInputState::from_config(&default_nelder_mead),
            nelder_mead_preset: infer_nelder_mead_preset(&default_nelder_mead),
            steepest_descent_inputs: SteepestDescentInputState::from_config(
                &default_steepest_descent,
            ),
            steepest_descent_preset: infer_steepest_descent_preset(&default_steepest_descent),
            newton_cg_inputs: NewtonCgInputState::from_config(&default_newton_cg),
            newton_cg_preset: infer_newton_cg_preset(&default_newton_cg),
            sgd_inputs: SgdInputState::from_config(&default_sgd),
            sgd_preset: infer_sgd_preset(&default_sgd),
            adam_inputs: AdamInputState::from_config(&default_adam),
            adam_preset: infer_adam_preset(&default_adam),
            ui_language: UiLanguage::English,
            plot_tool: PlotTool::SinglePoint,
            spray_points_per_second: 140,
            spray_radius_rel: 0.02,
            spray_brush: SprayBrush::Uniform,
            eraser_radius_rel: 0.03,
            spray_seed: 0xDEADBEEFCAFEBABE,
            spray_points_budget: 0.0,
            spray_last_emit_at: None,
            fit_to_content_requested: false,
            center_origin_requested: false,
            origin_bottom_left_requested: true,
            last_plot_bounds: None,
            active_tool_bounds: None,
            panel: PanelState::default(),
            spline_knots: crate::fit::DEFAULT_SPLINE_KNOTS,
            spline_knot_strategy: SplineKnotStrategy::default(),
            spline_extrapolation: SplineExtrapolation::default(),
            spline_duplicate_x_policy: SplineDuplicateXPolicy::default(),
            spline_initial_knot_y_inputs: Vec::new(),
            replay: ReplayState::default(),
            fit_in_progress: false,
            fit_loss_metric: OptimizationLossMetric::default(),
            fit_metric_quantization: MetricQuantization::Disabled,
            fit_preview_params: None,
            fit_preview_iteration: None,
            fit_result: None,
            spline_result: None,
            active_fit_points: None,
            result_metrics: None,
            residual_plot_points: Vec::new(),
            spline_plot_curve: None,
            #[cfg(not(target_arch = "wasm32"))]
            formula_svg_cache: None,
            sampled_curve_cache: None,
            iteration_diagnostics: IterationDiagnostics::default(),
            status: Some(StatusMessage::Ready),
            #[cfg(not(target_arch = "wasm32"))]
            fit_worker_rx: None,
            #[cfg(not(target_arch = "wasm32"))]
            fit_cancel_flag: None,
            #[cfg(not(target_arch = "wasm32"))]
            discard_fit_worker_updates: false,
            #[cfg(target_arch = "wasm32")]
            wasm_fit_job: None,
        }
    }
}

impl eframe::App for CurveFitApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        Self::apply_visual_style(ctx);
        self.poll_fit_worker(ctx);
        self.tick_replay(ctx);
        self.maybe_refresh_points_cache_after_debounce();

        if !self.fit_in_progress {
            let undo_requested = ctx.input(|input| {
                input.modifiers.command && !input.modifiers.shift && input.key_pressed(egui::Key::Z)
            });
            let redo_requested = ctx.input(|input| {
                input.modifiers.command
                    && (input.key_pressed(egui::Key::Y)
                        || (input.modifiers.shift && input.key_pressed(egui::Key::Z)))
            });
            if undo_requested {
                self.undo_points_edit();
            } else if redo_requested {
                self.redo_points_edit();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let panel_style = ctx.global_style();
        let panel_style = panel_style.as_ref();

        egui::Panel::top("header_panel")
            .frame(Self::top_bottom_panel_frame(panel_style))
            .show_inside(ui, |ui| {
                self.ui_header(ui);
            });

        egui::Panel::bottom("status_bar")
            .frame(Self::top_bottom_panel_frame(panel_style))
            .show_inside(ui, |ui| {
                self.ui_status_bar(ui);
            });

        if self.panel.show_left {
            egui::Panel::left("points_panel")
                .default_size(LEFT_PANEL_DEFAULT_WIDTH)
                .min_size(LEFT_PANEL_MIN_WIDTH)
                .resizable(true)
                .frame(Self::side_panel_frame(panel_style))
                .show_inside(ui, |ui| {
                    let language = self.ui_language;
                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 8.0);
                    ui.set_width(ui.available_width());
                    Self::panel_card_collapsible(
                        ui,
                        "left_section_tools",
                        tr(language, "Tools", "Инструменты"),
                        |ui| {
                            self.ui_tools(ui);
                        },
                    );
                    Self::panel_card_collapsible(
                        ui,
                        "left_section_points",
                        tr(language, "Input Points", "Точки"),
                        |ui| {
                            self.ui_points_editor(ui);
                        },
                    );
                });
        }

        if self.panel.show_right {
            egui::Panel::right("settings_panel")
                .default_size(RIGHT_PANEL_DEFAULT_WIDTH)
                .min_size(RIGHT_PANEL_MIN_WIDTH)
                .resizable(true)
                .frame(Self::side_panel_frame(panel_style))
                .show_inside(ui, |ui| {
                    let language = self.ui_language;
                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 8.0);
                    ui.set_width(ui.available_width());
                    Self::panel_card_collapsible(
                        ui,
                        "right_section_model",
                        tr(language, "Model", "Модель"),
                        |ui| {
                            self.ui_family_and_params(ui);
                        },
                    );
                    Self::panel_card_collapsible(
                        ui,
                        "right_section_metric",
                        tr(language, "Optimization metric", "Метрика оптимизации"),
                        |ui| {
                            self.ui_optimization_metric(ui);
                        },
                    );
                    Self::panel_card_collapsible(
                        ui,
                        "right_section_optimizer",
                        tr(language, "Optimizer", "Оптимизатор"),
                        |ui| {
                            self.ui_optimizer(ui);
                        },
                    );
                    Self::panel_card_collapsible(
                        ui,
                        "right_section_result",
                        tr(language, "Result", "Результат"),
                        |ui| {
                            self.ui_result(ui);
                        },
                    );
                });
        }

        self.ui_formula_window(&ctx);

        if self.panel.show_diagnostics {
            egui::Panel::bottom("diagnostics_panel")
                .resizable(true)
                .default_size(DIAGNOSTICS_PANEL_DEFAULT_HEIGHT)
                .min_size(DIAGNOSTICS_PANEL_MIN_HEIGHT)
                .frame(Self::top_bottom_panel_frame(panel_style))
                .show_inside(ui, |ui| {
                    let available_height = ui.available_height();
                    ui.set_height(available_height);
                    self.ui_iteration_diagnostics(ui);
                });
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.ui_plot(ui, ui.available_height().max(2.0));
        });
    }
}

#[cfg(test)]
mod tests;
