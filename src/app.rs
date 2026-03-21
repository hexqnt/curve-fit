use std::f64::consts::TAU;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use egui_plot::{
    Legend, Line, Plot, PlotBounds, PlotPoint, PlotPoints, PlotResponse, Points as PlotPointsItem,
};

mod formula;
mod i18n;
mod param_init;
mod plot_utils;
mod points_text;

use self::formula::{formula_svg_bytes, formula_svg_uri, model_formula_info};
use self::i18n::{
    center_origin_icon_image, clear_icon_image, family_label, fit_icon_image,
    fit_to_content_icon_image, github_mark_image, language_flag_image, model_choice_label,
    param_init_method_disabled_label, param_init_method_label, param_init_method_name_en,
    redo_icon_image, reset_icon_image, spline_extrapolation_label, spline_knot_strategy_label,
    spray_brush_label, stop_icon_image, tool_icon_image, tool_label, tr, undo_icon_image,
};
use self::param_init::{
    data_based_params_for_family, is_advanced_param_init_supported, polynomial_family,
};
use self::plot_utils::{fit_bounds_for_content, plot_domain};
use self::points_text::{parse_f64, parse_points_text_cache, points_to_text};
use crate::domain::{CurveFamily, CurveParams, FitResult, LbfgsConfig, Point, Points};
use crate::fit::{
    FitError, SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation, SplineFamilyKind,
    SplineKnotStrategy, SplineResult, calculate_metrics, default_spline_initial_knot_y,
    sample_curve,
};
#[cfg(target_arch = "wasm32")]
use crate::fit::{
    IncrementalFitRunner, IncrementalFitStep, IncrementalSplineFitRunner, IncrementalSplineFitStep,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::fit::{IncrementalSplineFitRunner, IncrementalSplineFitStep, fit_curve_with_progress};

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
const PARAM_INIT_RANDOM_MIN: f64 = -1.0;
const PARAM_INIT_RANDOM_MAX: f64 = 1.0;
const SPLINE_AUTO_SAMPLES_MIN: usize = 80;
const SPLINE_AUTO_SAMPLES_MAX: usize = 2_000;
const SPLINE_AUTO_SAMPLES_PER_KNOT: usize = 30;
const SPLINE_AUTO_SAMPLES_PER_POINT: usize = 3;
const DIAGNOSTICS_PANEL_DEFAULT_HEIGHT: f32 = 230.0;
const DIAGNOSTICS_PANEL_MIN_HEIGHT: f32 = 120.0;
const POINTS_PARSE_DEBOUNCE_MS: u64 = 180;
const POINTS_HISTORY_LIMIT: usize = 256;
const POINTS_PARSE_ERROR_PREFIX: &str = "Points parse error: ";
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const APP_REPOSITORY_URL: &str = env!("CARGO_PKG_REPOSITORY");

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
    const ALL: [Self; 22] = [
        Self::Polynomial,
        Self::Arrhenius,
        Self::Inverse,
        Self::Logistic,
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

fn lbfgs_preset_label(language: UiLanguage, preset: LbfgsPreset) -> &'static str {
    match (language, preset) {
        (UiLanguage::English, LbfgsPreset::Fast) => "Fast",
        (UiLanguage::English, LbfgsPreset::Balanced) => "Balanced",
        (UiLanguage::English, LbfgsPreset::Precise) => "Precise",
        (UiLanguage::English, LbfgsPreset::Robust) => "Robust",
        (UiLanguage::English, LbfgsPreset::Custom) => "Custom",
        (UiLanguage::Russian, LbfgsPreset::Fast) => "Быстрый",
        (UiLanguage::Russian, LbfgsPreset::Balanced) => "Сбалансированный",
        (UiLanguage::Russian, LbfgsPreset::Precise) => "Точный",
        (UiLanguage::Russian, LbfgsPreset::Robust) => "Устойчивый",
        (UiLanguage::Russian, LbfgsPreset::Custom) => "Произвольный",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LbfgsUiMode {
    #[default]
    Basic,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LbfgsPreset {
    Fast,
    #[default]
    Balanced,
    Precise,
    Robust,
    Custom,
}

impl LbfgsPreset {
    const ALL: [Self; 4] = [Self::Fast, Self::Balanced, Self::Precise, Self::Robust];

    fn to_config(self) -> LbfgsConfig {
        match self {
            Self::Fast => LbfgsConfig::try_new(5, 80, 1e-6, 1e-9, 1e-4, 0.9, 1e-10, 1.0, 1e-8)
                .expect("fast preset must be valid"),
            Self::Balanced => LbfgsConfig::default(),
            Self::Precise => {
                LbfgsConfig::try_new(10, 500, 1e-10, 1e-14, 1e-4, 0.95, 1e-12, 10.0, 1e-12)
                    .expect("precise preset must be valid")
            }
            Self::Robust => {
                LbfgsConfig::try_new(15, 1_000, 1e-9, 1e-12, 1e-4, 0.8, 1e-12, 1.0, 1e-10)
                    .expect("robust preset must be valid")
            }
            Self::Custom => LbfgsConfig::default(),
        }
    }

    fn infer_from_config(config: &LbfgsConfig) -> Self {
        for preset in Self::ALL {
            if &preset.to_config() == config {
                return preset;
            }
        }
        Self::Custom
    }
}

#[derive(Debug, Clone)]
struct ParsedPointsCache {
    parsed_points: Result<Vec<Point>, String>,
    parse_error_line: Option<usize>,
    plot_points: Vec<PlotPoint>,
}

#[derive(Debug, Clone)]
struct ModelFormulaInfo {
    full_formula: String,
    notes: String,
}

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

#[derive(Debug, Clone)]
enum StatusMessage {
    Ready,
    Cleared,
    FittingInProgress,
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
enum FitWorkerMessage {
    Iteration {
        iteration: u64,
        mse: f64,
        params: CurveParams,
    },
    SplineIteration {
        iteration: u64,
        mse: f64,
        knot_y: Vec<f64>,
        curve: Vec<[f64; 2]>,
    },
    Stopped,
    Finished(FitResult),
    SplineFinished(SplineResult),
    Failed(String),
}

#[cfg(target_arch = "wasm32")]
enum WasmFitRunner {
    Parametric(IncrementalFitRunner),
    Spline(IncrementalSplineFitRunner),
}

#[derive(Debug, Clone, Default)]
struct IterationDiagnostics {
    family: Option<CurveFamily>,
    spline_parameter_count: Option<usize>,
    loss_mse_points: Vec<[f64; 2]>,
    parameter_names: Vec<String>,
    parameter_series: Vec<Vec<[f64; 2]>>,
}

impl IterationDiagnostics {
    fn clear(&mut self) {
        self.family = None;
        self.spline_parameter_count = None;
        self.loss_mse_points.clear();
        self.parameter_names.clear();
        self.parameter_series.clear();
    }

    fn initialize(&mut self, points: &Points, params: &CurveParams) {
        let family = params.family();
        self.reset_for_family(family);
        let (mse, _) = calculate_metrics(points, params);
        self.append(0, mse, params);
    }

    fn append(&mut self, iteration: u64, mse: f64, params: &CurveParams) {
        let family = params.family();
        if self.family != Some(family) || self.spline_parameter_count.is_some() {
            self.reset_for_family(family);
        }

        let values = params.values();
        if values.len() != self.parameter_series.len() {
            self.reset_for_family(family);
        }

        let iteration = iteration as f64;
        upsert_iteration_point(&mut self.loss_mse_points, iteration, mse);
        for (series, value) in self.parameter_series.iter_mut().zip(values) {
            upsert_iteration_point(series, iteration, value);
        }
    }

    fn append_spline(&mut self, iteration: u64, mse: f64, knot_y: &[f64]) {
        let parameter_count = knot_y.len();
        if self.family.is_some() || self.spline_parameter_count != Some(parameter_count) {
            self.reset_for_spline(parameter_count);
        }

        let iteration = iteration as f64;
        upsert_iteration_point(&mut self.loss_mse_points, iteration, mse);
        for (series, value) in self.parameter_series.iter_mut().zip(knot_y.iter().copied()) {
            upsert_iteration_point(series, iteration, value);
        }
    }

    fn reset_for_family(&mut self, family: CurveFamily) {
        self.family = Some(family);
        self.spline_parameter_count = None;
        self.loss_mse_points.clear();
        self.parameter_names = family
            .parameter_names()
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        self.parameter_series = (0..self.parameter_names.len())
            .map(|_| Vec::new())
            .collect();
    }

    fn reset_for_spline(&mut self, parameter_count: usize) {
        self.family = None;
        self.spline_parameter_count = Some(parameter_count);
        self.loss_mse_points.clear();
        self.parameter_names = (0..parameter_count)
            .map(|index| format!("knot_y[{index}]"))
            .collect();
        self.parameter_series = (0..self.parameter_names.len())
            .map(|_| Vec::new())
            .collect();
    }
}

fn upsert_iteration_point(series: &mut Vec<[f64; 2]>, iteration: f64, value: f64) {
    if let Some(last) = series.last_mut()
        && (last[0] - iteration).abs() <= f64::EPSILON
    {
        last[1] = value;
    } else {
        series.push([iteration, value]);
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LbfgsInputState {
    history_size: usize,
    max_iters: u64,
    tol_grad: f64,
    tol_cost: f64,
    c1: f64,
    c2: f64,
    step_min: f64,
    step_max: f64,
    width_tolerance: f64,
}

impl LbfgsInputState {
    fn from_config(config: &LbfgsConfig) -> Self {
        Self {
            history_size: config.history_size,
            max_iters: config.max_iters,
            tol_grad: config.tol_grad,
            tol_cost: config.tol_cost,
            c1: config.c1,
            c2: config.c2,
            step_min: config.step_min,
            step_max: config.step_max,
            width_tolerance: config.width_tolerance,
        }
    }

    fn normalize_after_ui(&mut self) {
        self.c1 = self.c1.clamp(C1_MIN, C2_MAX - 1e-4);
        self.c2 = self.c2.clamp(self.c1 + 1e-4, C2_MAX);

        self.step_min = self.step_min.clamp(STEP_MIN_MIN, STEP_MAX_MAX - 1e-6);
        self.step_max = self.step_max.clamp(self.step_min + 1e-6, STEP_MAX_MAX);
    }

    fn to_config(&self) -> Result<LbfgsConfig, String> {
        LbfgsConfig::try_new(
            self.history_size,
            self.max_iters,
            self.tol_grad,
            self.tol_cost,
            self.c1,
            self.c2,
            self.step_min,
            self.step_max,
            self.width_tolerance,
        )
        .map_err(|error| error.to_string())
    }
}

/// Состояние и UI-логика интерактивного приложения для подгонки кривых.
pub struct CurveFitApp {
    points_text: String,
    points_cache: Option<ParsedPointsCache>,
    points_cache_dirty: bool,
    points_parse_debounce_deadline: Option<Instant>,
    points_undo_stack: Vec<String>,
    points_redo_stack: Vec<String>,
    selected_model: ModelChoice,
    polynomial_degree: usize,
    parameter_inputs: Vec<String>,
    lbfgs_inputs: LbfgsInputState,
    lbfgs_mode: LbfgsUiMode,
    lbfgs_preset: LbfgsPreset,
    ui_language: UiLanguage,
    plot_tool: PlotTool,
    spray_density: usize,
    spray_radius_rel: f64,
    spray_brush: SprayBrush,
    eraser_radius_rel: f64,
    spray_seed: u64,
    fit_to_content_requested: bool,
    center_origin_requested: bool,
    origin_bottom_left_requested: bool,
    last_plot_span: Option<[f64; 2]>,
    active_tool_bounds: Option<PlotBounds>,
    show_left_panel: bool,
    show_right_panel: bool,
    show_diagnostics_panel: bool,
    diagnostics_loss_axis_width: f32,
    diagnostics_residual_axis_width: f32,
    diagnostics_params_axis_width: f32,
    iteration_delay_seconds: f64,
    spline_knots: usize,
    spline_knot_strategy: SplineKnotStrategy,
    spline_extrapolation: SplineExtrapolation,
    spline_duplicate_x_policy: SplineDuplicateXPolicy,
    spline_initial_knot_y_inputs: Vec<String>,
    fit_in_progress: bool,
    fit_preview_params: Option<CurveParams>,
    fit_preview_iteration: Option<u64>,
    fit_result: Option<FitResult>,
    spline_result: Option<SplineResult>,
    active_fit_points: Option<Points>,
    result_metrics: Option<ExtendedMetrics>,
    residual_plot_points: Vec<PlotPoint>,
    spline_plot_curve: Option<Vec<PlotPoint>>,
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
    wasm_fit_runner: Option<WasmFitRunner>,
}

impl CurveFitApp {
    /// Создает приложение и настраивает загрузчики изображений для иконок/формул.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        Self::default()
    }

    fn resolved_model(&self) -> ResolvedModel {
        ResolvedModel::from_choice(self.selected_model, self.polynomial_degree)
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

    fn invalidate_points_cache(&mut self) {
        self.points_cache_dirty = true;
        // Небольшой debounce уменьшает число парсингов во время быстрого ввода текста.
        self.points_parse_debounce_deadline =
            Some(Instant::now() + Duration::from_millis(POINTS_PARSE_DEBOUNCE_MS));
    }

    fn points_cache_with_policy(&mut self, force: bool) -> &ParsedPointsCache {
        // Политика пересчета кэша:
        // - сразу, если кэша нет;
        // - по force;
        // - или после окончания debounce.
        let should_parse = if self.points_cache.is_none() {
            true
        } else if !self.points_cache_dirty {
            false
        } else if force {
            true
        } else {
            self.points_parse_debounce_deadline
                .map(|deadline| Instant::now() >= deadline)
                .unwrap_or(true)
        };

        if should_parse {
            self.points_cache = Some(parse_points_text_cache(&self.points_text));
            self.points_cache_dirty = false;
            self.points_parse_debounce_deadline = None;
        }
        self.points_cache
            .as_ref()
            .expect("points cache must be initialized")
    }

    fn points_cache(&mut self) -> &ParsedPointsCache {
        self.points_cache_with_policy(false)
    }

    fn maybe_refresh_points_cache_after_debounce(&mut self) {
        if self.points_cache_dirty
            && self
                .points_parse_debounce_deadline
                .map(|deadline| Instant::now() >= deadline)
                .unwrap_or(true)
        {
            self.points_cache_with_policy(true);
            self.refresh_status_after_points_edit();
        }
    }

    fn idle_status_after_points_edit(&self) -> StatusMessage {
        if self.fit_result.is_some() || self.spline_result.is_some() {
            StatusMessage::FitCompleted
        } else {
            StatusMessage::Ready
        }
    }

    fn refresh_status_after_points_edit(&mut self) {
        let parse_error = match &self.points_cache_with_policy(true).parsed_points {
            Ok(_) => None,
            Err(error) => Some(error.clone()),
        };

        if let Some(error) = parse_error {
            self.status = Some(StatusMessage::Error(format!(
                "{POINTS_PARSE_ERROR_PREFIX}{error}"
            )));
            return;
        }

        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
    }

    fn push_points_undo_snapshot(&mut self, snapshot: String) {
        if self
            .points_undo_stack
            .last()
            .is_some_and(|last| *last == snapshot)
        {
            return;
        }
        self.points_undo_stack.push(snapshot);
        // Ограничиваем историю фиксированным размером, чтобы не раздувать память.
        if self.points_undo_stack.len() > POINTS_HISTORY_LIMIT {
            let overflow = self.points_undo_stack.len() - POINTS_HISTORY_LIMIT;
            self.points_undo_stack.drain(0..overflow);
        }
    }

    fn apply_points_text_change(&mut self, new_text: String, keep_redo: bool) {
        if self.points_text == new_text {
            return;
        }
        self.points_text = new_text;
        self.invalidate_points_cache();
        if !keep_redo {
            self.points_redo_stack.clear();
        }
    }

    fn undo_points_edit(&mut self) {
        if self.fit_in_progress {
            return;
        }
        let Some(previous) = self.points_undo_stack.pop() else {
            return;
        };
        self.points_redo_stack.push(self.points_text.clone());
        self.apply_points_text_change(previous, true);
        self.refresh_status_after_points_edit();
    }

    fn redo_points_edit(&mut self) {
        if self.fit_in_progress {
            return;
        }
        let Some(next) = self.points_redo_stack.pop() else {
            return;
        };
        self.push_points_undo_snapshot(self.points_text.clone());
        self.apply_points_text_change(next, true);
        self.refresh_status_after_points_edit();
    }

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
            self.parameter_inputs = family
                .default_params()
                .values()
                .into_iter()
                .map(|value| value.to_string())
                .collect();
        } else {
            self.parameter_inputs.clear();
        }
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
        self.clear_fit_preview();
    }

    fn parse_points_for_edit(&mut self) -> Result<Vec<Point>, String> {
        match &self.points_cache_with_policy(true).parsed_points {
            Ok(points) => Ok(points.clone()),
            Err(error) => Err(error.clone()),
        }
    }

    fn parse_points_strict(&mut self) -> Result<Points, String> {
        Points::try_from(self.parse_points_for_edit()?).map_err(|error| error.to_string())
    }

    fn parse_initial_params(&self) -> Result<CurveParams, String> {
        let family = self.resolved_model().parametric_family().ok_or_else(|| {
            "Current model is non-parametric and has no initial parameters".to_string()
        })?;
        let mut values = Vec::with_capacity(self.parameter_inputs.len());
        for (index, raw_value) in self.parameter_inputs.iter().enumerate() {
            let field = format!("parameter[{index}]");
            values.push(parse_f64(&field, raw_value)?);
        }

        CurveParams::try_from_values(family, values).map_err(|error| error.to_string())
    }

    fn parse_spline_initial_knot_y(&self, expected_count: usize) -> Result<Vec<f64>, String> {
        if self.spline_initial_knot_y_inputs.len() != expected_count {
            return Err(format!(
                "Spline initialization expects {expected_count} values, got {}",
                self.spline_initial_knot_y_inputs.len()
            ));
        }

        let mut values = Vec::with_capacity(expected_count);
        for (index, raw_value) in self.spline_initial_knot_y_inputs.iter().enumerate() {
            let field = format!("spline_knot_y[{index}]");
            values.push(parse_f64(&field, raw_value)?);
        }
        Ok(values)
    }

    fn spline_family_and_init_config(&self) -> Option<(SplineFamilyKind, SplineConfig)> {
        let model = self.resolved_model();
        let family = match model {
            ResolvedModel::LinearSpline => SplineFamilyKind::Linear,
            ResolvedModel::MonotoneCubicSpline => SplineFamilyKind::MonotoneCubic,
            ResolvedModel::NaturalCubicSpline => SplineFamilyKind::NaturalCubic,
            ResolvedModel::AkimaSpline => SplineFamilyKind::Akima,
            ResolvedModel::Parametric(_) => return None,
        };
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
                self.parameter_inputs = params
                    .values()
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect();
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

    fn write_points_text(&mut self, points: &[Point]) {
        self.push_points_undo_snapshot(self.points_text.clone());
        self.apply_points_text_change(points_to_text(points), false);
        self.refresh_status_after_points_edit();
    }

    fn clear_fit_preview(&mut self) {
        self.fit_preview_params = None;
        self.fit_preview_iteration = None;
    }

    fn update_parametric_result_metrics(&mut self, points: &Points, params: &CurveParams) {
        let sample_count = points.len() as f64;
        let y_mean = points.as_slice().iter().map(|point| point.y()).sum::<f64>() / sample_count;

        let mut sse = 0.0;
        let mut sae = 0.0;
        let mut max_abs_error = 0.0_f64;
        self.residual_plot_points.clear();
        self.residual_plot_points.reserve(points.len());
        for point in points.as_slice() {
            let residual = params.evaluate(point.x()) - point.y();
            let abs_residual = residual.abs();
            sse += residual * residual;
            sae += abs_residual;
            max_abs_error = max_abs_error.max(abs_residual);
            self.residual_plot_points
                .push(PlotPoint::new(point.x(), residual));
        }

        let sst = points
            .as_slice()
            .iter()
            .map(|point| {
                let centered = point.y() - y_mean;
                centered * centered
            })
            .sum::<f64>();
        let mse = sse / sample_count;
        let rmse = mse.sqrt();
        let mae = sae / sample_count;
        let r2 = if sst <= 1e-15 {
            if sse <= 1e-15 { 1.0 } else { 0.0 }
        } else {
            1.0 - sse / sst
        };
        self.result_metrics = Some(ExtendedMetrics {
            mse,
            rmse,
            mae,
            r2,
            max_abs_error,
        });
    }

    fn update_spline_result_metrics(&mut self, result: &SplineResult) {
        self.result_metrics = Some(ExtendedMetrics {
            mse: result.mse,
            rmse: result.rmse,
            mae: result.mae,
            r2: result.r2,
            max_abs_error: result.max_abs_error,
        });
        self.residual_plot_points = result
            .residuals
            .iter()
            .map(|point| PlotPoint::new(point[0], point[1]))
            .collect();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn cancel_fit_and_discard_updates(&mut self) {
        if !self.fit_in_progress {
            return;
        }

        if let Some(cancel_flag) = &self.fit_cancel_flag {
            cancel_flag.store(true, Ordering::Relaxed);
        }
        self.discard_fit_worker_updates = true;
        if matches!(self.status, Some(StatusMessage::FittingInProgress)) {
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_fit_worker(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.fit_worker_rx.take() else {
            return;
        };

        let mut keep_receiver = true;
        loop {
            match rx.try_recv() {
                Ok(FitWorkerMessage::Iteration {
                    iteration,
                    mse,
                    params,
                }) => {
                    if self.discard_fit_worker_updates {
                        continue;
                    }
                    self.fit_preview_iteration = Some(iteration);
                    self.iteration_diagnostics.append(iteration, mse, &params);
                    self.fit_preview_params = Some(params);
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(FitWorkerMessage::SplineIteration {
                    iteration,
                    mse,
                    knot_y,
                    curve,
                }) => {
                    if self.discard_fit_worker_updates {
                        continue;
                    }
                    self.fit_preview_iteration = Some(iteration);
                    self.iteration_diagnostics
                        .append_spline(iteration, mse, &knot_y);
                    self.spline_plot_curve = Some(
                        curve
                            .into_iter()
                            .map(|point| PlotPoint::new(point[0], point[1]))
                            .collect(),
                    );
                    self.status = Some(StatusMessage::FittingInProgress);
                }
                Ok(FitWorkerMessage::Stopped) => {
                    self.fit_in_progress = false;
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates
                        || matches!(
                            self.status,
                            Some(StatusMessage::FittingInProgress)
                                | Some(StatusMessage::FittingStopping)
                        )
                    {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    keep_receiver = false;
                }
                Ok(FitWorkerMessage::Finished(result)) => {
                    self.fit_in_progress = false;
                    if !self.discard_fit_worker_updates {
                        if let Some(points) = self.active_fit_points.clone() {
                            self.update_parametric_result_metrics(&points, &result.params);
                        }
                        self.iteration_diagnostics.append(
                            result.iterations,
                            result.mse,
                            &result.params,
                        );
                        self.fit_preview_iteration = Some(result.iterations);
                        self.fit_preview_params = Some(result.params.clone());
                        self.fit_result = Some(result);
                        self.status = Some(StatusMessage::FitCompleted);
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    self.active_fit_points = None;
                    keep_receiver = false;
                }
                Ok(FitWorkerMessage::SplineFinished(result)) => {
                    self.fit_in_progress = false;
                    if !self.discard_fit_worker_updates {
                        let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
                        let spline_plot_curve = result
                            .curve
                            .iter()
                            .map(|point| PlotPoint::new(point[0], point[1]))
                            .collect();
                        self.update_spline_result_metrics(&result);
                        self.iteration_diagnostics.append_spline(
                            result.iterations,
                            result.mse,
                            &knot_y,
                        );
                        self.fit_preview_iteration = Some(result.iterations);
                        self.spline_plot_curve = Some(spline_plot_curve);
                        self.spline_result = Some(result);
                        self.status = Some(StatusMessage::FitCompleted);
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    self.active_fit_points = None;
                    keep_receiver = false;
                }
                Ok(FitWorkerMessage::Failed(error)) => {
                    self.fit_in_progress = false;
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::Error(error));
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    keep_receiver = false;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.fit_in_progress = false;
                    self.active_fit_points = None;
                    if !self.discard_fit_worker_updates {
                        self.status = Some(StatusMessage::Error(
                            "Fit worker channel disconnected unexpectedly".to_string(),
                        ));
                    } else if matches!(
                        self.status,
                        Some(StatusMessage::FittingInProgress)
                            | Some(StatusMessage::FittingStopping)
                    ) {
                        self.status = Some(StatusMessage::FitStopped);
                    }
                    keep_receiver = false;
                    break;
                }
            }
        }

        if keep_receiver {
            self.fit_worker_rx = Some(rx);
        } else {
            self.fit_cancel_flag = None;
            self.discard_fit_worker_updates = false;
        }

        if self.fit_in_progress && self.fit_preview_iteration.is_some() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_fit_worker(&mut self, ctx: &egui::Context) {
        enum WasmRunnerStep {
            Parametric(Result<IncrementalFitStep, FitError>),
            Spline(Result<IncrementalSplineFitStep, FitError>),
        }

        let Some(step) = self.wasm_fit_runner.as_mut().map(|runner| match runner {
            WasmFitRunner::Parametric(runner) => WasmRunnerStep::Parametric(runner.step()),
            WasmFitRunner::Spline(runner) => WasmRunnerStep::Spline(runner.step()),
        }) else {
            return;
        };

        match step {
            WasmRunnerStep::Parametric(Ok(IncrementalFitStep::Iteration {
                iteration,
                mse,
                params,
            })) => {
                self.fit_preview_iteration = Some(iteration);
                self.iteration_diagnostics.append(iteration, mse, &params);
                self.fit_preview_params = Some(params);
                self.status = Some(StatusMessage::FittingInProgress);
                if self.iteration_delay_seconds > 0.0 {
                    ctx.request_repaint_after(Duration::from_secs_f64(
                        self.iteration_delay_seconds,
                    ));
                } else {
                    ctx.request_repaint();
                }
            }
            WasmRunnerStep::Parametric(Ok(IncrementalFitStep::Finished(result))) => {
                self.fit_in_progress = false;
                if let Some(points) = self.active_fit_points.clone() {
                    self.update_parametric_result_metrics(&points, &result.params);
                }
                self.iteration_diagnostics
                    .append(result.iterations, result.mse, &result.params);
                self.fit_preview_iteration = Some(result.iterations);
                self.fit_preview_params = Some(result.params.clone());
                self.fit_result = Some(result);
                self.status = Some(StatusMessage::FitCompleted);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Parametric(Ok(IncrementalFitStep::Cancelled)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::FitStopped);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Parametric(Err(error)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::Error(error.to_string()));
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Spline(Ok(IncrementalSplineFitStep::Iteration {
                iteration,
                mse,
                knot_y,
                curve,
            })) => {
                self.fit_preview_iteration = Some(iteration);
                self.iteration_diagnostics
                    .append_spline(iteration, mse, &knot_y);
                self.spline_plot_curve = Some(
                    curve
                        .into_iter()
                        .map(|point| PlotPoint::new(point[0], point[1]))
                        .collect(),
                );
                self.status = Some(StatusMessage::FittingInProgress);
                if self.iteration_delay_seconds > 0.0 {
                    ctx.request_repaint_after(Duration::from_secs_f64(
                        self.iteration_delay_seconds,
                    ));
                } else {
                    ctx.request_repaint();
                }
            }
            WasmRunnerStep::Spline(Ok(IncrementalSplineFitStep::Finished(result))) => {
                self.fit_in_progress = false;
                let knot_y = result.knots.iter().map(|knot| knot[1]).collect::<Vec<_>>();
                let spline_plot_curve = result
                    .curve
                    .iter()
                    .map(|point| PlotPoint::new(point[0], point[1]))
                    .collect();
                self.update_spline_result_metrics(&result);
                self.iteration_diagnostics
                    .append_spline(result.iterations, result.mse, &knot_y);
                self.fit_preview_iteration = Some(result.iterations);
                self.spline_plot_curve = Some(spline_plot_curve);
                self.spline_result = Some(result);
                self.status = Some(StatusMessage::FitCompleted);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Spline(Ok(IncrementalSplineFitStep::Cancelled)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::FitStopped);
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
            WasmRunnerStep::Spline(Err(error)) => {
                self.fit_in_progress = false;
                self.status = Some(StatusMessage::Error(error.to_string()));
                self.active_fit_points = None;
                self.wasm_fit_runner = None;
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn cancel_fit_and_discard_updates(&mut self) {
        if !self.fit_in_progress {
            return;
        }
        if let Some(runner) = self.wasm_fit_runner.as_mut() {
            match runner {
                WasmFitRunner::Parametric(runner) => runner.cancel(),
                WasmFitRunner::Spline(runner) => runner.cancel(),
            }
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn request_stop_fit(&mut self) {
        if let Some(cancel_flag) = &self.fit_cancel_flag {
            cancel_flag.store(true, Ordering::Relaxed);
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn request_stop_fit(&mut self) {
        if let Some(runner) = self.wasm_fit_runner.as_mut() {
            match runner {
                WasmFitRunner::Parametric(runner) => runner.cancel(),
                WasmFitRunner::Spline(runner) => runner.cancel(),
            }
            self.status = Some(StatusMessage::FittingStopping);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_fit_worker(
        &mut self,
        family: CurveFamily,
        points: Points,
        initial_params: CurveParams,
        config: LbfgsConfig,
        cancel_flag: Arc<AtomicBool>,
    ) {
        let delay_seconds = self.iteration_delay_seconds;
        let (tx, rx) = mpsc::channel();
        self.fit_worker_rx = Some(rx);
        self.fit_cancel_flag = Some(cancel_flag.clone());
        self.discard_fit_worker_updates = false;
        self.fit_in_progress = true;

        std::thread::spawn(move || {
            let iter_tx = tx.clone();
            let progress_cancel = cancel_flag.clone();
            let progress_points = points.clone();
            let result = fit_curve_with_progress(
                &points,
                family,
                initial_params,
                &config,
                move |iteration, params| {
                    if progress_cancel.load(Ordering::Relaxed) {
                        return false;
                    }
                    if let Some(params) = params {
                        let (mse, _) = calculate_metrics(&progress_points, &params);
                        let _ = iter_tx.send(FitWorkerMessage::Iteration {
                            iteration,
                            mse,
                            params,
                        });
                    }
                    if delay_seconds > 0.0 {
                        std::thread::sleep(Duration::from_secs_f64(delay_seconds));
                    }
                    !progress_cancel.load(Ordering::Relaxed)
                },
            );

            match result {
                Ok(result) => {
                    let _ = tx.send(FitWorkerMessage::Finished(result));
                }
                Err(FitError::Cancelled) => {
                    let _ = tx.send(FitWorkerMessage::Stopped);
                }
                Err(error) => {
                    let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                }
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_spline_fit_worker(
        &mut self,
        family: SplineFamilyKind,
        points: Points,
        config: SplineConfig,
        initial_knot_y: Vec<f64>,
        cancel_flag: Arc<AtomicBool>,
    ) {
        let delay_seconds = self.iteration_delay_seconds;
        let (tx, rx) = mpsc::channel();
        self.fit_worker_rx = Some(rx);
        self.fit_cancel_flag = Some(cancel_flag.clone());
        self.discard_fit_worker_updates = false;
        self.fit_in_progress = true;

        std::thread::spawn(move || {
            let mut runner = match IncrementalSplineFitRunner::new_with_initial_knot_y(
                &points,
                family,
                config,
                Some(initial_knot_y.as_slice()),
            ) {
                Ok(runner) => runner,
                Err(error) => {
                    let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                    return;
                }
            };

            loop {
                if cancel_flag.load(Ordering::Relaxed) {
                    runner.cancel();
                }

                match runner.step() {
                    Ok(IncrementalSplineFitStep::Iteration {
                        iteration,
                        mse,
                        knot_y,
                        curve,
                    }) => {
                        let _ = tx.send(FitWorkerMessage::SplineIteration {
                            iteration,
                            mse,
                            knot_y,
                            curve,
                        });
                        if delay_seconds > 0.0 {
                            std::thread::sleep(Duration::from_secs_f64(delay_seconds));
                        }
                    }
                    Ok(IncrementalSplineFitStep::Finished(result)) => {
                        let _ = tx.send(FitWorkerMessage::SplineFinished(result));
                        break;
                    }
                    Ok(IncrementalSplineFitStep::Cancelled) | Err(FitError::Cancelled) => {
                        let _ = tx.send(FitWorkerMessage::Stopped);
                        break;
                    }
                    Err(error) => {
                        let _ = tx.send(FitWorkerMessage::Failed(error.to_string()));
                        break;
                    }
                }
            }
        });
    }

    fn run_fit(&mut self) {
        if self.fit_in_progress {
            return;
        }

        let points = match self.parse_points_strict() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        let selected_model = self.resolved_model();
        if matches!(
            selected_model,
            ResolvedModel::LinearSpline
                | ResolvedModel::MonotoneCubicSpline
                | ResolvedModel::NaturalCubicSpline
                | ResolvedModel::AkimaSpline
        ) {
            let spline_family = match selected_model {
                ResolvedModel::LinearSpline => SplineFamilyKind::Linear,
                ResolvedModel::MonotoneCubicSpline => SplineFamilyKind::MonotoneCubic,
                ResolvedModel::NaturalCubicSpline => SplineFamilyKind::NaturalCubic,
                ResolvedModel::AkimaSpline => SplineFamilyKind::Akima,
                ResolvedModel::Parametric(_) => unreachable!("checked by matches!"),
            };
            let spline_config = self
                .spline_config_for_model(selected_model, points.len())
                .expect("checked by matches!");
            self.sync_spline_initial_knot_y_inputs(spline_config.knots);
            let initial_knot_y = match self.parse_spline_initial_knot_y(spline_config.knots) {
                Ok(values) => values,
                Err(error) => {
                    self.status = Some(StatusMessage::Error(error));
                    return;
                }
            };
            self.clear_fit_outputs();
            self.fit_preview_iteration = Some(0);
            self.status = Some(StatusMessage::FittingInProgress);

            #[cfg(not(target_arch = "wasm32"))]
            {
                let cancel_flag = Arc::new(AtomicBool::new(false));
                self.start_spline_fit_worker(
                    spline_family,
                    points,
                    spline_config,
                    initial_knot_y,
                    cancel_flag,
                );
            }

            #[cfg(target_arch = "wasm32")]
            {
                match IncrementalSplineFitRunner::new_with_initial_knot_y(
                    &points,
                    spline_family,
                    spline_config,
                    Some(initial_knot_y.as_slice()),
                ) {
                    Ok(runner) => {
                        self.wasm_fit_runner = Some(WasmFitRunner::Spline(runner));
                        self.fit_in_progress = true;
                    }
                    Err(error) => {
                        self.status = Some(StatusMessage::Error(error.to_string()));
                    }
                }
            }
            return;
        }

        let family = match selected_model.parametric_family() {
            Some(family) => family,
            None => {
                self.status = Some(StatusMessage::Error(
                    "Selected model has no parametric family".to_string(),
                ));
                return;
            }
        };

        let initial_params = match self.parse_initial_params() {
            Ok(params) => params,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        let config = match self.lbfgs_inputs.to_config() {
            Ok(config) => config,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        if let Err(error) = family.validate_points(&points) {
            self.status = Some(StatusMessage::Error(error.to_string()));
            return;
        }

        // Очищаем предыдущий успешный результат только когда новый запуск уже валиден.
        self.clear_fit_outputs();
        self.active_fit_points = Some(points.clone());
        self.iteration_diagnostics
            .initialize(&points, &initial_params);

        if self.iteration_delay_seconds > 0.0 {
            self.fit_preview_params = Some(initial_params.clone());
            self.fit_preview_iteration = Some(0);
        }
        self.status = Some(StatusMessage::FittingInProgress);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let cancel_flag = Arc::new(AtomicBool::new(false));
            self.start_fit_worker(family, points, initial_params, config, cancel_flag);
        }

        #[cfg(target_arch = "wasm32")]
        {
            match IncrementalFitRunner::new(&points, family, initial_params, &config) {
                Ok(runner) => {
                    self.wasm_fit_runner = Some(WasmFitRunner::Parametric(runner));
                    self.fit_in_progress = true;
                }
                Err(error) => {
                    self.active_fit_points = None;
                    self.status = Some(StatusMessage::Error(error.to_string()));
                }
            }
        }
    }

    fn next_unit_random(&mut self) -> f64 {
        self.spray_seed = self
            .spray_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let value = self.spray_seed >> 11;
        value as f64 / ((1_u64 << 53) as f64)
    }

    fn next_uniform_unit_disk_offset(&mut self) -> [f64; 2] {
        let radial = self.next_unit_random().sqrt();
        let angle = TAU * self.next_unit_random();
        [radial * angle.cos(), radial * angle.sin()]
    }

    fn next_gaussian_unit_disk_offset(&mut self) -> [f64; 2] {
        loop {
            let u = self.next_unit_random();
            let radial = SPRAY_GAUSSIAN_SIGMA * (-2.0 * (1.0 - u).ln()).sqrt();
            if radial <= 1.0 {
                let angle = TAU * self.next_unit_random();
                return [radial * angle.cos(), radial * angle.sin()];
            }
        }
    }

    fn next_spray_unit_disk_offset(&mut self) -> [f64; 2] {
        match self.spray_brush {
            SprayBrush::Uniform => self.next_uniform_unit_disk_offset(),
            SprayBrush::Gaussian => self.next_gaussian_unit_disk_offset(),
        }
    }

    fn add_point_from_plot(&mut self, x: f64, y: f64) {
        let mut points = match self.parse_points_for_edit() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        match Point::try_new(x, y) {
            Ok(point) => {
                points.push(point);
                self.write_points_text(&points);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error.to_string()));
            }
        }
    }

    fn spray_points_from_plot(
        &mut self,
        center_x: f64,
        center_y: f64,
        radius_x: f64,
        radius_y: f64,
    ) {
        let mut points = match self.parse_points_for_edit() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        for _ in 0..self.spray_density {
            let [offset_x, offset_y] = self.next_spray_unit_disk_offset();
            let x = center_x + offset_x * radius_x;
            let y = center_y + offset_y * radius_y;
            if let Ok(point) = Point::try_new(x, y) {
                points.push(point);
            }
        }

        self.write_points_text(&points);
    }

    fn erase_points_from_plot(
        &mut self,
        center_x: f64,
        center_y: f64,
        radius_x: f64,
        radius_y: f64,
    ) {
        let mut points = match self.parse_points_for_edit() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        if radius_x <= 0.0 || radius_y <= 0.0 {
            return;
        }

        points.retain(|point| {
            let dx = (point.x() - center_x) / radius_x;
            let dy = (point.y() - center_y) / radius_y;
            dx * dx + dy * dy > 1.0
        });

        self.write_points_text(&points);
    }

    fn plot_position_from_screen(
        plot_response: &PlotResponse<()>,
        screen_pos: egui::Pos2,
    ) -> Option<PlotPoint> {
        // Инструменты рисования должны работать только внутри области данных графика.
        // Иначе курсор над осями/легендой даёт координаты вне ожидаемого диапазона.
        if !plot_response.transform.frame().contains(screen_pos) {
            return None;
        }
        Some(plot_response.transform.value_from_position(screen_pos))
    }

    fn handle_plot_tools(&mut self, plot_response: &PlotResponse<()>) {
        if self.fit_in_progress {
            return;
        }

        let response = &plot_response.response;
        let primary_down_on_plot = response.is_pointer_button_down_on();
        if matches!(self.plot_tool, PlotTool::Spray | PlotTool::Eraser) && primary_down_on_plot {
            self.active_tool_bounds
                .get_or_insert(*plot_response.transform.bounds());
        } else {
            self.active_tool_bounds = None;
        }
        let bounds = plot_response.transform.bounds();
        let radius_x_scale = bounds.width().abs().max(1e-6);
        let radius_y_scale = bounds.height().abs().max(1e-6);

        match self.plot_tool {
            PlotTool::None => {}
            PlotTool::SinglePoint => {
                if response.clicked_by(egui::PointerButton::Primary)
                    && let Some(screen_pos) = response.interact_pointer_pos()
                    && let Some(plot_pos) =
                        Self::plot_position_from_screen(plot_response, screen_pos)
                {
                    self.add_point_from_plot(plot_pos.x, plot_pos.y);
                }
            }
            PlotTool::Spray => {
                if primary_down_on_plot
                    && let Some(screen_pos) = response.interact_pointer_pos()
                    && let Some(plot_pos) =
                        Self::plot_position_from_screen(plot_response, screen_pos)
                {
                    let radius_x = self.spray_radius_rel * radius_x_scale;
                    let radius_y = self.spray_radius_rel * radius_y_scale;
                    self.spray_points_from_plot(plot_pos.x, plot_pos.y, radius_x, radius_y);
                }
            }
            PlotTool::Eraser => {
                if primary_down_on_plot
                    && let Some(screen_pos) = response.interact_pointer_pos()
                    && let Some(plot_pos) =
                        Self::plot_position_from_screen(plot_response, screen_pos)
                {
                    let radius_x = self.eraser_radius_rel * radius_x_scale;
                    let radius_y = self.eraser_radius_rel * radius_y_scale;
                    self.erase_points_from_plot(plot_pos.x, plot_pos.y, radius_x, radius_y);
                }
            }
        }
    }

    fn ui_header(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();

        egui::ScrollArea::horizontal()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let delay_supported = !self.fit_in_progress;
                    let delay_slider =
                        egui::Slider::new(&mut self.iteration_delay_seconds, 0.0..=3.0)
                            .step_by(0.01)
                            .text(tr(
                                language,
                                "Iteration delay, sec",
                                "Задержка итерации, сек",
                            ));
                    let response = ui.add_enabled(delay_supported, delay_slider);
                    if !delay_supported {
                        let hint = tr(
                            language,
                            "Delay changes are locked while fitting is running.",
                            "Изменение задержки недоступно во время подгонки.",
                        );
                        response.on_disabled_hover_text(hint);
                    }

                    ui.separator();
                    ui.menu_button(tr(language, "View", "Вид"), |ui| {
                        if ui
                            .add(egui::Button::image_and_text(
                                fit_to_content_icon_image(icon_tint),
                                tr(language, "Fit to content", "Подогнать по содержимому"),
                            ))
                            .clicked()
                        {
                            self.fit_to_content_requested = true;
                            ui.close();
                        }
                        if ui
                            .add(egui::Button::image_and_text(
                                center_origin_icon_image(icon_tint),
                                tr(language, "Center to 0,0", "Центр к 0,0"),
                            ))
                            .clicked()
                        {
                            self.center_origin_requested = true;
                            self.origin_bottom_left_requested = false;
                            ui.close();
                        }
                        if ui
                            .add(egui::Button::new(tr(
                                language,
                                "Set 0,0 to bottom-left",
                                "0,0 в левый нижний угол",
                            )))
                            .clicked()
                        {
                            self.origin_bottom_left_requested = true;
                            self.center_origin_requested = false;
                            ui.close();
                        }
                    });

                    ui.separator();
                    ui.menu_button(tr(language, "Panels", "Панели"), |ui| {
                        ui.toggle_value(
                            &mut self.show_left_panel,
                            tr(language, "Left panel", "Левая панель"),
                        );
                        ui.toggle_value(
                            &mut self.show_right_panel,
                            tr(language, "Right panel", "Правая панель"),
                        );
                        ui.toggle_value(
                            &mut self.show_diagnostics_panel,
                            tr(language, "Diagnostics", "Диагностика"),
                        );
                    });
                });
            });
    }

    fn ui_status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            self.ui_status(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(APP_VERSION_LABEL);
                ui.separator();
                let github_response =
                    ui.add(egui::Button::image_and_text(github_mark_image(), "GitHub"));
                if github_response.clicked() {
                    ui.ctx()
                        .open_url(egui::OpenUrl::new_tab(APP_REPOSITORY_URL));
                }

                ui.separator();
                egui::widgets::global_theme_preference_buttons(ui);

                ui.separator();
                ui.menu_image_text_button(
                    language_flag_image(self.ui_language),
                    self.ui_language.native_name(),
                    |ui| {
                        for candidate in UiLanguage::ALL {
                            let selected = self.ui_language == candidate;
                            if ui
                                .add(
                                    egui::Button::image_and_text(
                                        language_flag_image(candidate),
                                        candidate.native_name(),
                                    )
                                    .selected(selected),
                                )
                                .clicked()
                            {
                                self.ui_language = candidate;
                                ui.close();
                            }
                        }
                    },
                );
            });
        });
    }

    fn ui_tools(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();
        ui.heading(tr(language, "Tools", "Инструменты"));

        ui.horizontal_wrapped(|ui| {
            for tool in [
                PlotTool::None,
                PlotTool::SinglePoint,
                PlotTool::Spray,
                PlotTool::Eraser,
            ] {
                let selected = self.plot_tool == tool;
                let button = egui::Button::image_and_text(
                    tool_icon_image(tool, icon_tint),
                    tool_label(language, tool),
                )
                .selected(selected);
                if ui.add(button).clicked() {
                    self.plot_tool = tool;
                }
            }
        });

        match self.plot_tool {
            PlotTool::None => {
                ui.label(tr(
                    language,
                    "Navigation mode: drag, zoom, and scroll the plot.",
                    "Режим навигации: перемещение, зум и прокрутка графика.",
                ));
            }
            PlotTool::SinglePoint => {}
            PlotTool::Spray => {
                ui.add(egui::Slider::new(&mut self.spray_density, 1..=30).text(tr(
                    language,
                    "Density",
                    "Плотность",
                )));
                ui.add(
                    egui::Slider::new(&mut self.spray_radius_rel, 0.002..=0.2)
                        .logarithmic(true)
                        .text(tr(language, "Radius", "Радиус")),
                );
                ui.horizontal_wrapped(|ui| {
                    ui.label(tr(language, "Brush", "Кисть"));
                    ui.selectable_value(
                        &mut self.spray_brush,
                        SprayBrush::Uniform,
                        spray_brush_label(language, SprayBrush::Uniform),
                    );
                    ui.selectable_value(
                        &mut self.spray_brush,
                        SprayBrush::Gaussian,
                        spray_brush_label(language, SprayBrush::Gaussian),
                    );
                });
            }
            PlotTool::Eraser => {
                ui.add(
                    egui::Slider::new(&mut self.eraser_radius_rel, 0.002..=0.2)
                        .logarithmic(true)
                        .text(tr(language, "Radius", "Радиус")),
                );
            }
        }
    }

    fn ui_points_editor(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();
        let can_edit_points = !self.fit_in_progress;
        ui.heading(tr(language, "Input Points", "Точки"));
        ui.label(tr(
            language,
            "One point per line: x and y separated by space, tab, or ';'",
            "Одна точка на строку: x и y через пробел, табуляцию или ';'",
        ));

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    can_edit_points && !self.points_undo_stack.is_empty(),
                    egui::Button::image_and_text(
                        undo_icon_image(icon_tint),
                        tr(language, "Undo", "Отменить"),
                    ),
                )
                .clicked()
            {
                self.undo_points_edit();
            }
            if ui
                .add_enabled(
                    can_edit_points && !self.points_redo_stack.is_empty(),
                    egui::Button::image_and_text(
                        redo_icon_image(icon_tint),
                        tr(language, "Redo", "Повторить"),
                    ),
                )
                .clicked()
            {
                self.redo_points_edit();
            }
            if ui
                .add_enabled(
                    can_edit_points,
                    egui::Button::image_and_text(
                        clear_icon_image(icon_tint),
                        tr(language, "Clear", "Очистить"),
                    ),
                )
                .clicked()
            {
                self.push_points_undo_snapshot(self.points_text.clone());
                self.apply_points_text_change(String::new(), false);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Cleared);
            }
        });

        let hint = tr(
            language,
            "Example:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
            "Пример:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
        );
        let text_height = ui.available_height();
        let row_height = ui.text_style_height(&egui::TextStyle::Monospace).max(1.0);
        let desired_rows = (text_height / row_height).floor().max(1.0) as usize;
        egui::ScrollArea::vertical()
            .id_salt("points_text_scroll")
            .max_height(text_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let text_width = ui.available_width();
                let before_edit = self.points_text.clone();
                let parse_error_line = self.points_cache().parse_error_line;
                let mut layouter = move |ui: &egui::Ui,
                                         text: &dyn egui::TextBuffer,
                                         wrap_width: f32|
                      -> std::sync::Arc<egui::Galley> {
                    let mut job = egui::text::LayoutJob::default();
                    job.wrap.max_width = wrap_width;
                    let text_color = ui.visuals().text_color();
                    let font_id = egui::TextStyle::Monospace.resolve(ui.style());
                    let error_bg = if ui.visuals().dark_mode {
                        egui::Color32::from_rgb(70, 26, 26)
                    } else {
                        egui::Color32::from_rgb(255, 230, 230)
                    };
                    for (index, line) in text.as_str().split_inclusive('\n').enumerate() {
                        let mut format = egui::TextFormat {
                            font_id: font_id.clone(),
                            color: text_color,
                            ..Default::default()
                        };
                        if parse_error_line == Some(index + 1) {
                            format.background = error_bg;
                        }
                        job.append(line, 0.0, format);
                    }
                    if text.as_str().is_empty() {
                        job.append(
                            "",
                            0.0,
                            egui::TextFormat {
                                font_id,
                                color: text_color,
                                ..Default::default()
                            },
                        );
                    }
                    ui.fonts_mut(|fonts| fonts.layout_job(job))
                };
                let response = ui.add(
                    egui::TextEdit::multiline(&mut self.points_text)
                        .desired_width(text_width)
                        .desired_rows(desired_rows)
                        .font(egui::TextStyle::Monospace)
                        .hint_text(hint)
                        .layouter(&mut layouter)
                        .interactive(can_edit_points),
                );
                if response.changed() {
                    self.push_points_undo_snapshot(before_edit);
                    self.points_redo_stack.clear();
                    self.invalidate_points_cache();
                }
            });

        if let Err(error) = &self.points_cache().parsed_points {
            ui.colored_label(
                egui::Color32::from_rgb(200, 64, 64),
                format!("{POINTS_PARSE_ERROR_PREFIX}{error}"),
            );
        }

        if self.fit_in_progress {
            ui.label(tr(
                language,
                "Point editing is disabled while fitting is running.",
                "Редактирование точек отключено во время подгонки.",
            ));
        }
    }

    fn ui_family_and_params(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let can_edit_params = !self.fit_in_progress;
        ui.heading(tr(language, "Model", "Модель"));

        let previous_model = self.selected_model;
        ui.add_enabled_ui(can_edit_params, |ui| {
            egui::ComboBox::from_label(tr(language, "Model type", "Тип модели"))
                .selected_text(model_choice_label(language, self.selected_model))
                .show_ui(ui, |ui| {
                    ui.set_min_width(280.0);
                    let mut is_first_group = true;
                    for group in ModelGroup::ALL {
                        if !is_first_group {
                            ui.separator();
                        }
                        is_first_group = false;
                        ui.label(egui::RichText::new(model_group_label(language, group)).strong());
                        for model in ModelChoice::ALL {
                            if model_group(model) != group {
                                continue;
                            }
                            let model_label = model_choice_label(language, model);
                            let response =
                                ui.selectable_label(self.selected_model == model, model_label);
                            if response.clicked() {
                                self.selected_model = model;
                            }
                        }
                    }
                });
        });

        let mut params_need_sync = false;
        if previous_model != self.selected_model {
            params_need_sync = true;
        }

        if self.selected_model.is_polynomial() {
            let previous_degree = self.polynomial_degree;
            ui.add_enabled(
                can_edit_params,
                egui::Slider::new(&mut self.polynomial_degree, 1..=9).text(tr(
                    language,
                    "Polynomial degree",
                    "Степень полинома",
                )),
            );
            if previous_degree != self.polynomial_degree {
                params_need_sync = true;
            }
        }

        if params_need_sync {
            self.sync_parameter_inputs();
            self.clear_fit_outputs();
        }

        let formula_info =
            model_formula_info(language, self.selected_model, self.polynomial_degree);
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new(tr(language, "Model Formula", "Формула модели")).strong());
            let dark_mode = ui.visuals().dark_mode;
            let (svg_uri, svg_bytes) =
                self.cached_formula_svg(&formula_info.full_formula, dark_mode);
            ui.add(
                egui::Image::from_bytes(svg_uri, svg_bytes)
                    .max_width(ui.available_width())
                    .fit_to_original_size(1.0),
            );
            ui.label(egui::RichText::new(formula_info.notes).small());
        });

        if let Some(family) = self.resolved_model().parametric_family() {
            let mut method_to_apply = None;
            ui.horizontal_wrapped(|ui| {
                ui.label(tr(language, "Initial parameters", "Начальные параметры"));
                ui.add_enabled_ui(can_edit_params, |ui| {
                    ui.menu_button(
                        tr(language, "+ Initialize", "+ Инициализация"),
                        |ui| {
                            for method in ParamInitMethod::ALL {
                                if method.is_supported_for_family(family) {
                                    if ui
                                        .button(param_init_method_label(language, method))
                                        .clicked()
                                    {
                                        method_to_apply = Some(method);
                                        ui.close();
                                    }
                                } else {
                                    ui.add_enabled(
                                        false,
                                        egui::Button::new(param_init_method_disabled_label(
                                            language, method,
                                        )),
                                    );
                                }
                            }
                        },
                    );
                });
            });

            if let Some(method) = method_to_apply {
                self.apply_param_init_method(method);
            }

            for (index, parameter_name) in family.parameter_names().iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(*parameter_name);
                    ui.add_enabled(
                        can_edit_params,
                        egui::TextEdit::singleline(&mut self.parameter_inputs[index])
                            .desired_width(120.0),
                    );
                });
            }
        } else {
            ui.label(tr(
                language,
                "Spline models are non-parametric, but they optimize knot y-values as parameters.",
                "Сплайны непараметрические, но оптимизируют knot y как параметры.",
            ));
            ui.add_space(4.0);
            let min_knots = self
                .resolved_model()
                .spline_min_knots()
                .expect("non-parametric branch guarantees spline model");
            self.spline_knots = self.spline_knots.max(min_knots);
            self.sync_spline_initial_knot_y_inputs(self.spline_knots);
            let mut spline_method_to_apply = None;

            ui.horizontal_wrapped(|ui| {
                ui.label(tr(language, "Initial parameters", "Начальные параметры"));
                ui.add_enabled_ui(can_edit_params, |ui| {
                    ui.menu_button(
                        tr(language, "+ Initialize", "+ Инициализация"),
                        |ui| {
                            for method in ParamInitMethod::ALL {
                                if ui
                                    .button(param_init_method_label(language, method))
                                    .clicked()
                                {
                                    spline_method_to_apply = Some(method);
                                    ui.close();
                                }
                            }
                        },
                    );
                });
            });

            if let Some(method) = spline_method_to_apply {
                self.apply_spline_param_init_method(method);
            }

            ui.add_enabled_ui(can_edit_params, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.spline_knots, min_knots..=40).text(tr(
                        language,
                        "Spline knot count",
                        "Число узлов сплайна",
                    )),
                );
                egui::ComboBox::from_label(tr(language, "Knot reduction", "Редукция узлов"))
                    .selected_text(spline_knot_strategy_label(
                        language,
                        self.spline_knot_strategy,
                    ))
                    .show_ui(ui, |ui| {
                        for strategy in SplineKnotStrategy::ALL {
                            ui.selectable_value(
                                &mut self.spline_knot_strategy,
                                strategy,
                                spline_knot_strategy_label(language, strategy),
                            );
                        }
                    });
                egui::ComboBox::from_label(tr(language, "Extrapolation", "Экстраполяция"))
                    .selected_text(spline_extrapolation_label(
                        language,
                        self.spline_extrapolation,
                    ))
                    .show_ui(ui, |ui| {
                        for extrapolation in SplineExtrapolation::ALL {
                            ui.selectable_value(
                                &mut self.spline_extrapolation,
                                extrapolation,
                                spline_extrapolation_label(language, extrapolation),
                            );
                        }
                    });
                egui::ComboBox::from_label(tr(language, "Duplicate x", "Дубли x"))
                    .selected_text(spline_duplicate_policy_label(
                        language,
                        self.spline_duplicate_x_policy,
                    ))
                    .show_ui(ui, |ui| {
                        for policy in SplineDuplicateXPolicy::ALL {
                            ui.selectable_value(
                                &mut self.spline_duplicate_x_policy,
                                policy,
                                spline_duplicate_policy_label(language, policy),
                            );
                        }
                    });
            });
            self.sync_spline_initial_knot_y_inputs(self.spline_knots);
            ui.label(format!(
                "{}: {}",
                tr(
                    language,
                    "Target spline parameter count",
                    "Целевое число параметров сплайна"
                ),
                self.spline_knots
            ));
            ui.label(tr(
                language,
                "Initial knot y values",
                "Начальные значения knot y",
            ));
            for (index, value) in self.spline_initial_knot_y_inputs.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("knot_y[{index}]"));
                    ui.add_enabled(
                        can_edit_params,
                        egui::TextEdit::singleline(value).desired_width(120.0),
                    );
                });
            }
            ui.label(egui::RichText::new(tr(
                language,
                "More knots means better fit, less smoothing; fewer knots means stronger smoothing.",
                "Больше узлов — более точная подгонка, меньше сглаживания; меньше узлов — более сильное сглаживание.",
            )).small());
            ui.label(egui::RichText::new(tr(
                language,
                "When x-values contain duplicates you can merge them automatically instead of failing.",
                "При повторяющихся x можно автоматически объединять точки вместо ошибки.",
            )).small());
            ui.label(egui::RichText::new(tr(
                language,
                "Sample density is selected automatically from knot count and data size.",
                "Плотность сэмплирования выбирается автоматически по числу узлов и размеру данных.",
            )).small());
        }
    }

    fn ui_lbfgs(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();
        ui.separator();
        ui.heading("LBFGS");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.lbfgs_mode,
                LbfgsUiMode::Basic,
                tr(language, "Basic", "Базовый"),
            );
            ui.selectable_value(
                &mut self.lbfgs_mode,
                LbfgsUiMode::Advanced,
                tr(language, "Advanced", "Продвинутый"),
            );
        });

        if self.lbfgs_mode == LbfgsUiMode::Basic {
            ui.label(tr(
                language,
                "Use presets to quickly control convergence speed and stability.",
                "Используйте пресеты для быстрого выбора баланса скорости и устойчивости.",
            ));

            let previous_preset = self.lbfgs_preset;
            egui::ComboBox::from_label(tr(language, "Preset", "Пресет"))
                .selected_text(lbfgs_preset_label(language, self.lbfgs_preset))
                .show_ui(ui, |ui| {
                    for preset in LbfgsPreset::ALL {
                        ui.selectable_value(
                            &mut self.lbfgs_preset,
                            preset,
                            lbfgs_preset_label(language, preset),
                        );
                    }
                    if self.lbfgs_preset == LbfgsPreset::Custom {
                        ui.add_enabled(
                            false,
                            egui::Button::new(lbfgs_preset_label(language, LbfgsPreset::Custom)),
                        );
                    }
                });
            if self.lbfgs_preset != previous_preset && self.lbfgs_preset != LbfgsPreset::Custom {
                self.lbfgs_inputs = LbfgsInputState::from_config(&self.lbfgs_preset.to_config());
            }

            ui.label(format!("history_size = {}", self.lbfgs_inputs.history_size));
            ui.label(format!("max_iters = {}", self.lbfgs_inputs.max_iters));
            ui.label(format!("tol_grad = {:.2e}", self.lbfgs_inputs.tol_grad));
            ui.label(format!("tol_cost = {:.2e}", self.lbfgs_inputs.tol_cost));
        } else {
            ui.label(tr(
                language,
                "Use sliders to tune optimizer parameters.",
                "Используйте бегунки для настройки оптимизатора.",
            ));

            let before = self.lbfgs_inputs.clone();
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.history_size, 1..=50).text("history_size"),
            );
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.max_iters, 10..=10_000).text("max_iters"),
            );
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.tol_grad, 1e-12..=1e-2)
                    .logarithmic(true)
                    .text("tol_grad"),
            );
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.tol_cost, 1e-14..=1e-4)
                    .logarithmic(true)
                    .text("tol_cost"),
            );
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.c1, C1_MIN..=0.2)
                    .logarithmic(true)
                    .text("c1"),
            );
            ui.add(egui::Slider::new(&mut self.lbfgs_inputs.c2, 0.1..=C2_MAX).text("c2"));
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.step_min, STEP_MIN_MIN..=1.0)
                    .logarithmic(true)
                    .text("step_min"),
            );
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.step_max, 1e-6..=STEP_MAX_MAX)
                    .logarithmic(true)
                    .text("step_max"),
            );
            ui.add(
                egui::Slider::new(&mut self.lbfgs_inputs.width_tolerance, 1e-14..=1e-3)
                    .logarithmic(true)
                    .text("width_tolerance"),
            );

            self.lbfgs_inputs.normalize_after_ui();
            if self.lbfgs_inputs != before {
                self.lbfgs_preset = LbfgsPreset::Custom;
            }
        }

        if ui
            .add(egui::Button::image_and_text(
                reset_icon_image(icon_tint),
                tr(language, "Reset Defaults", "Сбросить по умолчанию"),
            ))
            .clicked()
        {
            self.lbfgs_inputs = LbfgsInputState::from_config(&LbfgsConfig::default());
            self.lbfgs_preset = LbfgsPreset::Balanced;
        }
    }

    fn ui_status(&self, ui: &mut egui::Ui) {
        if let Some(status) = &self.status {
            let color = if status.is_error() {
                egui::Color32::from_rgb(200, 64, 64)
            } else {
                egui::Color32::from_rgb(64, 160, 96)
            };
            ui.colored_label(color, status.text(self.ui_language));
        }
    }

    fn ui_plot(&mut self, ui: &mut egui::Ui, height: f32) {
        let language = self.ui_language;
        let points = self.points_cache().plot_points.clone();
        let points_slice = points.as_slice();
        let (x_min, x_max) = plot_domain(points_slice);
        let navigation_mode = matches!(self.plot_tool, PlotTool::None);
        let spline_curve = self.spline_plot_curve.clone();
        let spline_curve_slice = spline_curve.as_deref();
        let sampled_curve = if spline_curve_slice.is_none() {
            let active_params = self
                .fit_preview_params
                .clone()
                .or_else(|| self.fit_result.as_ref().map(|result| result.params.clone()));
            active_params.map(|params| {
                self.cached_sampled_curve(&params, x_min, x_max, PARAMETRIC_PLOT_SAMPLES)
            })
        } else {
            None
        };
        let fitted_curve_points = spline_curve_slice.or(sampled_curve.as_deref());
        let fitted_line_name = if spline_curve_slice.is_some() {
            model_choice_label(language, self.selected_model).to_string()
        } else if self.fit_in_progress {
            if let Some(iteration) = self.fit_preview_iteration {
                format!(
                    "{} ({})",
                    tr(language, "Fitted", "Фитинг"),
                    format_args!("{} {iteration}", tr(language, "iter", "итер."))
                )
            } else {
                tr(language, "Fitted", "Фитинг").to_string()
            }
        } else {
            tr(language, "Fitted", "Фитинг").to_string()
        };
        let content_bounds = fit_bounds_for_content(points_slice, fitted_curve_points);
        let fit_bounds = if self.fit_to_content_requested {
            content_bounds
        } else {
            None
        };
        let center_bounds = if self.center_origin_requested {
            let [span_x, span_y] = self
                .last_plot_span
                .or_else(|| {
                    content_bounds.map(|bounds| [bounds.width().abs(), bounds.height().abs()])
                })
                .unwrap_or([2.0, 2.0]);
            let half_x = span_x.max(1e-6) * 0.5;
            let half_y = span_y.max(1e-6) * 0.5;
            Some(PlotBounds::from_min_max(
                [-half_x, -half_y],
                [half_x, half_y],
            ))
        } else {
            None
        };
        let origin_bottom_left_bounds = if self.origin_bottom_left_requested {
            let [span_x, span_y] = self
                .last_plot_span
                .or_else(|| {
                    content_bounds.map(|bounds| [bounds.width().abs(), bounds.height().abs()])
                })
                .unwrap_or([2.0, 2.0]);
            Some(PlotBounds::from_min_max(
                [0.0, 0.0],
                [span_x.max(1e-6), span_y.max(1e-6)],
            ))
        } else {
            None
        };
        let locked_tool_bounds = self.active_tool_bounds;

        let plot_response = Plot::new("fit_plot")
            .height(height)
            .legend(Legend::default())
            .allow_drag(navigation_mode)
            .allow_zoom(navigation_mode)
            .allow_scroll(navigation_mode)
            .allow_double_click_reset(navigation_mode)
            .allow_boxed_zoom(navigation_mode)
            .show(ui, |plot_ui| {
                if let Some(bounds) = locked_tool_bounds {
                    plot_ui.set_plot_bounds(bounds);
                }
                if let Some(bounds) = fit_bounds {
                    plot_ui.set_plot_bounds(bounds);
                }
                if let Some(bounds) = center_bounds {
                    plot_ui.set_plot_bounds(bounds);
                }
                if let Some(bounds) = origin_bottom_left_bounds {
                    plot_ui.set_plot_bounds(bounds);
                }
                if !points_slice.is_empty() {
                    plot_ui.points(
                        PlotPointsItem::new(tr(language, "Samples", "Точки"), points_slice)
                            .radius(3.0),
                    );
                }
                if let Some(fitted) = spline_curve_slice {
                    plot_ui.line(Line::new(fitted_line_name.clone(), fitted));
                } else if let Some(fitted) = sampled_curve.as_deref() {
                    plot_ui.line(Line::new(fitted_line_name.clone(), fitted));
                }
            });

        let bounds = plot_response.transform.bounds();
        self.last_plot_span = Some([
            bounds.width().abs().max(1e-6),
            bounds.height().abs().max(1e-6),
        ]);

        if self.fit_to_content_requested {
            self.fit_to_content_requested = false;
        }
        if self.center_origin_requested {
            self.center_origin_requested = false;
        }
        if self.origin_bottom_left_requested {
            self.origin_bottom_left_requested = false;
        }

        self.handle_plot_tools(&plot_response);
    }

    fn ui_iteration_diagnostics(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        ui.heading(tr(
            language,
            "Iteration diagnostics",
            "Диагностика итераций",
        ));

        if self.iteration_diagnostics.loss_mse_points.is_empty() {
            ui.label(tr(
                language,
                "Run Fit to collect iteration history.",
                "Запустите фитинг, чтобы получить историю итераций.",
            ));
            self.diagnostics_loss_axis_width = 0.0;
            self.diagnostics_residual_axis_width = 0.0;
            self.diagnostics_params_axis_width = 0.0;
            return;
        }

        let has_residual_plot = !self.residual_plot_points.is_empty();
        let available_height = ui.available_height().max(2.0);
        let spacing = ui.spacing().item_spacing.y;
        let plot_count = if has_residual_plot { 3.0 } else { 2.0 };
        let total_spacing = spacing * (plot_count - 1.0);
        let plot_height = ((available_height - total_spacing).max(2.0)) / plot_count;
        let shared_axis_width = self
            .diagnostics_loss_axis_width
            .max(self.diagnostics_residual_axis_width)
            .max(self.diagnostics_params_axis_width);
        let loss_extra_padding = (shared_axis_width - self.diagnostics_loss_axis_width).max(0.0);
        let residual_extra_padding =
            (shared_axis_width - self.diagnostics_residual_axis_width).max(0.0);
        let params_extra_padding =
            (shared_axis_width - self.diagnostics_params_axis_width).max(0.0);
        let mut measured_loss_axis_width = 0.0;
        let mut measured_residual_axis_width = 0.0;
        let mut measured_params_axis_width = 0.0;

        {
            let loss_points = &self.iteration_diagnostics.loss_mse_points;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height),
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    if loss_extra_padding > 0.0 {
                        ui.add_space(loss_extra_padding);
                    }
                    let plot_response = Plot::new("loss_mse_plot")
                        .height(plot_height)
                        .legend(Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .allow_boxed_zoom(false)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(
                                tr(language, "Loss (MSE)", "Лосс (MSE)"),
                                PlotPoints::from_iter(loss_points.iter().copied()),
                            ));
                        });
                    measured_loss_axis_width = diagnostics_plot_y_axis_width(&plot_response);
                },
            );
        }

        if has_residual_plot {
            let residual_points = &self.residual_plot_points;
            let x_min = residual_points
                .iter()
                .map(|point| point.x)
                .fold(f64::INFINITY, f64::min);
            let x_max = residual_points
                .iter()
                .map(|point| point.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let zero_line = [[x_min, 0.0], [x_max, 0.0]];

            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height),
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    if residual_extra_padding > 0.0 {
                        ui.add_space(residual_extra_padding);
                    }
                    let plot_response = Plot::new("residuals_diagnostics_plot")
                        .height(plot_height)
                        .legend(Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .allow_boxed_zoom(false)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(
                                tr(language, "Zero", "Ноль"),
                                PlotPoints::from_iter(zero_line),
                            ));
                            plot_ui.points(
                                PlotPointsItem::new(
                                    tr(language, "Residuals", "Остатки"),
                                    residual_points.as_slice(),
                                )
                                .radius(2.5),
                            );
                        });
                    measured_residual_axis_width = diagnostics_plot_y_axis_width(&plot_response);
                },
            );
        }

        {
            let parameter_names = &self.iteration_diagnostics.parameter_names;
            let parameter_series = &self.iteration_diagnostics.parameter_series;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height),
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    if params_extra_padding > 0.0 {
                        ui.add_space(params_extra_padding);
                    }
                    let plot_response = Plot::new("parameter_iteration_plot")
                        .height(plot_height)
                        .legend(Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .allow_boxed_zoom(false)
                        .show(ui, |plot_ui| {
                            for (name, series) in
                                parameter_names.iter().zip(parameter_series.iter())
                            {
                                plot_ui.line(Line::new(
                                    name.clone(),
                                    PlotPoints::from_iter(series.iter().copied()),
                                ));
                            }
                        });
                    measured_params_axis_width = diagnostics_plot_y_axis_width(&plot_response);
                },
            );
        }

        self.diagnostics_loss_axis_width = measured_loss_axis_width;
        self.diagnostics_residual_axis_width = measured_residual_axis_width;
        self.diagnostics_params_axis_width = measured_params_axis_width;
    }

    fn ui_result(&self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        ui.heading(tr(language, "Result", "Результат"));
        if self.fit_in_progress {
            ui.label(tr(
                language,
                "Fitting in progress. Curve updates after each iteration.",
                "Подгонка в процессе. Кривая обновляется после каждой итерации.",
            ));
            if let Some(iteration) = self.fit_preview_iteration {
                ui.label(format!(
                    "{}: {iteration}",
                    tr(language, "Iteration", "Итерация")
                ));
            }
            if let Some(params) = &self.fit_preview_params {
                ui.separator();
                ui.label(tr(language, "Current parameters", "Текущие параметры"));
                for (name, value) in params
                    .family()
                    .parameter_names()
                    .iter()
                    .zip(params.values())
                {
                    ui.label(format!("{name} = {value:.8}"));
                }
            }
            return;
        }

        let metrics = self.result_metrics.unwrap_or_else(|| {
            if let Some(result) = &self.fit_result {
                ExtendedMetrics {
                    mse: result.mse,
                    rmse: result.rmse,
                    ..ExtendedMetrics::default()
                }
            } else if let Some(result) = &self.spline_result {
                ExtendedMetrics {
                    mse: result.mse,
                    rmse: result.rmse,
                    mae: result.mae,
                    r2: result.r2,
                    max_abs_error: result.max_abs_error,
                }
            } else {
                ExtendedMetrics::default()
            }
        });

        if let Some(result) = &self.fit_result {
            ui.label(format!(
                "{}: {}",
                tr(language, "Family", "Семейство"),
                family_label(language, result.family)
            ));
            ui.label(format!("MSE: {:.8}", metrics.mse));
            ui.label(format!("RMSE: {:.8}", metrics.rmse));
            ui.label(format!("MAE: {:.8}", metrics.mae));
            ui.label(format!("R²: {:.8}", metrics.r2));
            ui.label(format!(
                "{}: {:.8}",
                tr(language, "Max |error|", "Макс |ошибка|"),
                metrics.max_abs_error
            ));
            ui.label(format!(
                "{}: {}",
                tr(language, "Iterations", "Итерации"),
                result.iterations
            ));
            ui.separator();
            ui.label(tr(language, "Parameters", "Параметры"));
            for (name, value) in result
                .family
                .parameter_names()
                .iter()
                .zip(result.params.values())
            {
                ui.label(format!("{name} = {value:.8}"));
            }
        } else if let Some(result) = &self.spline_result {
            ui.label(format!(
                "{}: {}",
                tr(language, "Family", "Семейство"),
                model_choice_label(language, self.selected_model)
            ));
            ui.label(format!("MSE: {:.8}", metrics.mse));
            ui.label(format!("RMSE: {:.8}", metrics.rmse));
            ui.label(format!("MAE: {:.8}", metrics.mae));
            ui.label(format!("R²: {:.8}", metrics.r2));
            ui.label(format!(
                "{}: {:.8}",
                tr(language, "Max |error|", "Макс |ошибка|"),
                metrics.max_abs_error
            ));
            ui.label(format!(
                "{}: {}",
                tr(language, "Iterations", "Итерации"),
                result.iterations
            ));
            ui.separator();
            ui.label(format!(
                "{}: {}",
                tr(language, "Parameters", "Параметры"),
                result.knots.len()
            ));
            for (index, knot) in result.knots.iter().enumerate() {
                ui.label(format!(
                    "knot_y[{index}] @ x={:.8}: {:.8}",
                    knot[0], knot[1]
                ));
            }
        } else {
            ui.label(tr(
                language,
                "Run Fit to see optimization results.",
                "Нажмите Fit, чтобы увидеть результат оптимизации.",
            ));
        }
    }
}

impl Default for CurveFitApp {
    fn default() -> Self {
        let selected_model = ModelChoice::Polynomial;
        let polynomial_degree = 1;
        let selected_family = polynomial_family(polynomial_degree);
        let default_lbfgs = LbfgsConfig::default();

        Self {
            points_text: String::new(),
            points_cache: None,
            points_cache_dirty: true,
            points_parse_debounce_deadline: None,
            points_undo_stack: Vec::new(),
            points_redo_stack: Vec::new(),
            selected_model,
            polynomial_degree,
            parameter_inputs: selected_family
                .default_params()
                .values()
                .into_iter()
                .map(|value| value.to_string())
                .collect(),
            lbfgs_inputs: LbfgsInputState::from_config(&default_lbfgs),
            lbfgs_mode: LbfgsUiMode::Basic,
            lbfgs_preset: LbfgsPreset::infer_from_config(&default_lbfgs),
            ui_language: UiLanguage::English,
            plot_tool: PlotTool::SinglePoint,
            spray_density: 8,
            spray_radius_rel: 0.02,
            spray_brush: SprayBrush::Uniform,
            eraser_radius_rel: 0.03,
            spray_seed: 0xDEADBEEFCAFEBABE,
            fit_to_content_requested: false,
            center_origin_requested: false,
            origin_bottom_left_requested: true,
            last_plot_span: None,
            active_tool_bounds: None,
            show_left_panel: true,
            spline_knots: crate::fit::DEFAULT_SPLINE_KNOTS,
            spline_knot_strategy: SplineKnotStrategy::default(),
            spline_extrapolation: SplineExtrapolation::default(),
            spline_duplicate_x_policy: SplineDuplicateXPolicy::default(),
            spline_initial_knot_y_inputs: Vec::new(),
            show_right_panel: true,
            show_diagnostics_panel: true,
            diagnostics_loss_axis_width: 0.0,
            diagnostics_residual_axis_width: 0.0,
            diagnostics_params_axis_width: 0.0,
            iteration_delay_seconds: 0.25,
            fit_in_progress: false,
            fit_preview_params: None,
            fit_preview_iteration: None,
            fit_result: None,
            spline_result: None,
            active_fit_points: None,
            result_metrics: None,
            residual_plot_points: Vec::new(),
            spline_plot_curve: None,
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
            wasm_fit_runner: None,
        }
    }
}

impl eframe::App for CurveFitApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_fit_worker(ctx);
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

        egui::TopBottomPanel::top("header_panel").show(ctx, |ui| {
            self.ui_header(ui);
        });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            self.ui_status_bar(ui);
        });

        if self.show_left_panel {
            egui::SidePanel::left("points_panel")
                .default_width(340.0)
                .resizable(true)
                .show(ctx, |ui| {
                    self.ui_tools(ui);
                    ui.separator();
                    self.ui_points_editor(ui);
                });
        }

        if self.show_right_panel {
            egui::SidePanel::right("settings_panel")
                .default_width(320.0)
                .resizable(true)
                .show(ctx, |ui| {
                    let icon_tint = ui.visuals().text_color();
                    self.ui_family_and_params(ui);
                    if self.resolved_model().parametric_family().is_some() {
                        self.ui_lbfgs(ui);
                    }

                    ui.separator();
                    let action_button = if self.fit_in_progress {
                        egui::Button::image_and_text(
                            stop_icon_image(icon_tint),
                            tr(self.ui_language, "Stop", "Стоп"),
                        )
                    } else {
                        egui::Button::image_and_text(
                            fit_icon_image(icon_tint),
                            tr(self.ui_language, "Fit", "Фитинг"),
                        )
                    };
                    if ui.add(action_button).clicked() {
                        if self.fit_in_progress {
                            self.request_stop_fit();
                        } else {
                            self.run_fit();
                        }
                    }
                    if self.fit_in_progress
                        && let Some(iteration) = self.fit_preview_iteration
                    {
                        ui.label(format!(
                            "{}: {iteration}",
                            tr(self.ui_language, "Iteration", "Итерация")
                        ));
                    }

                    ui.separator();
                    self.ui_result(ui);
                });
        }

        if self.show_diagnostics_panel {
            egui::TopBottomPanel::bottom("diagnostics_panel")
                .resizable(true)
                .default_height(DIAGNOSTICS_PANEL_DEFAULT_HEIGHT)
                .min_height(DIAGNOSTICS_PANEL_MIN_HEIGHT)
                .show(ctx, |ui| {
                    let available_height = ui.available_height();
                    ui.set_height(available_height);
                    self.ui_iteration_diagnostics(ui);
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_plot(ui, ui.available_height().max(2.0));
        });
    }
}

fn diagnostics_plot_y_axis_width(plot_response: &PlotResponse<()>) -> f32 {
    (plot_response.transform.frame().left() - plot_response.response.rect.left()).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::{
        CurveFitApp, IterationDiagnostics, ModelChoice, ParamInitMethod, StatusMessage,
        data_based_params_for_family,
    };
    use crate::domain::{CurveFamily, CurveParams, FitResult, Point, Points};
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

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

    fn assert_approx_eq(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }

    #[test]
    fn diagnostics_initialize_stores_iteration_zero_state() {
        let points = line_points();
        let params = CurveParams::Linear { a: 2.0, b: 1.0 };
        let mut diagnostics = IterationDiagnostics::default();

        diagnostics.initialize(&points, &params);

        assert_eq!(
            diagnostics.parameter_names,
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(diagnostics.loss_mse_points, vec![[0.0, 0.0]]);
        assert_eq!(diagnostics.parameter_series.len(), 2);
        assert_eq!(diagnostics.parameter_series[0], vec![[0.0, 2.0]]);
        assert_eq!(diagnostics.parameter_series[1], vec![[0.0, 1.0]]);
    }

    #[test]
    fn diagnostics_append_replaces_duplicate_iteration() {
        let points = line_points();
        let mut diagnostics = IterationDiagnostics::default();
        diagnostics.initialize(&points, &CurveParams::Linear { a: 2.0, b: 1.0 });

        diagnostics.append(2, 5.0, &CurveParams::Linear { a: 1.0, b: 0.0 });
        diagnostics.append(2, 3.0, &CurveParams::Linear { a: -1.5, b: 0.5 });

        assert_eq!(diagnostics.loss_mse_points.len(), 2);
        assert_eq!(diagnostics.loss_mse_points[1], [2.0, 3.0]);
        assert_eq!(diagnostics.parameter_series[0].len(), 2);
        assert_eq!(diagnostics.parameter_series[0][1], [2.0, -1.5]);
        assert_eq!(diagnostics.parameter_series[1].len(), 2);
        assert_eq!(diagnostics.parameter_series[1][1], [2.0, 0.5]);
    }

    #[test]
    fn diagnostics_append_resets_when_family_changes() {
        let points = line_points();
        let mut diagnostics = IterationDiagnostics::default();
        diagnostics.initialize(&points, &CurveParams::Linear { a: 2.0, b: 1.0 });
        diagnostics.append(
            4,
            1.0,
            &CurveParams::Quadratic {
                a: 1.0,
                b: -2.0,
                c: 3.0,
            },
        );

        assert_eq!(
            diagnostics.parameter_names,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(diagnostics.loss_mse_points, vec![[4.0, 1.0]]);
        assert_eq!(diagnostics.parameter_series.len(), 3);
        assert_eq!(diagnostics.parameter_series[0], vec![[4.0, 1.0]]);
        assert_eq!(diagnostics.parameter_series[1], vec![[4.0, -2.0]]);
        assert_eq!(diagnostics.parameter_series[2], vec![[4.0, 3.0]]);
    }

    #[test]
    fn diagnostics_append_spline_tracks_knot_parameters() {
        let mut diagnostics = IterationDiagnostics::default();

        diagnostics.append_spline(1, 2.5, &[0.5, -1.0]);
        diagnostics.append_spline(2, 1.5, &[0.75, -0.25]);

        assert_eq!(
            diagnostics.parameter_names,
            vec!["knot_y[0]".to_string(), "knot_y[1]".to_string()]
        );
        assert_eq!(diagnostics.loss_mse_points, vec![[1.0, 2.5], [2.0, 1.5]]);
        assert_eq!(
            diagnostics.parameter_series[0],
            vec![[1.0, 0.5], [2.0, 0.75]]
        );
        assert_eq!(
            diagnostics.parameter_series[1],
            vec![[1.0, -1.0], [2.0, -0.25]]
        );
    }

    #[test]
    fn param_init_method_support_matrix_is_correct() {
        assert!(ParamInitMethod::Default.is_supported_for_family(CurveFamily::Arrhenius));
        assert!(ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Linear));
        assert!(ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::Power));

        assert!(!ParamInitMethod::DataBased.is_supported_for_family(CurveFamily::Arrhenius));
        assert!(!ParamInitMethod::Randomized.is_supported_for_family(CurveFamily::FourPl));
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
        app.points_text = "0 1\n1 3\n2 5\n3 7\n".to_string();
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
        app.iteration_diagnostics
            .initialize(&line_points(), &CurveParams::Linear { a: 2.0, b: 1.0 });

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
        assert!(app.iteration_diagnostics.loss_mse_points.is_empty());
    }

    #[test]
    fn apply_param_init_sets_error_status_on_failure() {
        let mut app = CurveFitApp {
            selected_model: ModelChoice::Power,
            ..Default::default()
        };
        app.sync_parameter_inputs();
        app.points_text = "1 0\n2 2\n".to_string();
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
            ..Default::default()
        };

        app.clear_fit_outputs();

        assert!(cancel_flag.load(Ordering::Relaxed));
        assert!(app.fit_in_progress);
        assert!(app.discard_fit_worker_updates);
        assert!(app.fit_result.is_none());
        assert!(app.fit_preview_params.is_none());
        assert!(app.fit_preview_iteration.is_none());
    }

    #[test]
    fn run_fit_invalid_input_does_not_seed_iteration_diagnostics() {
        let mut app = CurveFitApp {
            selected_model: ModelChoice::Power,
            ..Default::default()
        };
        app.sync_parameter_inputs();
        app.points_text = "-1 2\n1 3\n".to_string();
        app.invalidate_points_cache();

        app.run_fit();

        assert!(matches!(app.status, Some(StatusMessage::Error(_))));
        assert!(app.iteration_diagnostics.loss_mse_points.is_empty());
        assert!(app.iteration_diagnostics.parameter_series.is_empty());
    }

    #[test]
    fn points_edit_parse_error_status_restores_completed_when_fixed() {
        let mut app = CurveFitApp {
            fit_result: Some(FitResult {
                family: CurveFamily::Linear,
                params: CurveParams::Linear { a: 1.0, b: 0.0 },
                mse: 0.0,
                rmse: 0.0,
                iterations: 1,
            }),
            status: Some(StatusMessage::FitCompleted),
            ..Default::default()
        };

        app.points_text = "1 2 3\n".to_string();
        app.invalidate_points_cache();
        app.refresh_status_after_points_edit();
        assert!(matches!(
            app.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(super::POINTS_PARSE_ERROR_PREFIX)
        ));

        app.points_text = "1 2\n2 3\n".to_string();
        app.invalidate_points_cache();
        app.refresh_status_after_points_edit();
        assert!(matches!(app.status, Some(StatusMessage::FitCompleted)));
    }
}
