use std::f64::consts::TAU;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use egui_plot::{
    Legend, Line, Plot, PlotBounds, PlotPoint, PlotPoints, PlotResponse, Points as PlotPointsItem,
};

mod diagnostics;
mod fit_worker;
mod formula;
mod i18n;
mod optimizer;
mod param_init;
mod plot_utils;
mod points_text;
mod ui;

use self::diagnostics::{IterationDiagnostics, diagnostics_plot_y_axis_width};
#[cfg(target_arch = "wasm32")]
use self::formula::formula_plain_text;
use self::formula::model_formula_info;
#[cfg(not(target_arch = "wasm32"))]
use self::formula::{formula_svg_bytes, formula_svg_uri};
use self::i18n::{
    center_origin_icon_image, clear_icon_image, family_label, fit_icon_image,
    fit_to_content_icon_image, github_mark_image, language_flag_image, model_choice_label,
    optimization_loss_metric_label, param_init_method_disabled_label, param_init_method_label,
    param_init_method_name_en, redo_icon_image, replay_pause_icon_image, replay_play_icon_image,
    reset_icon_image, spline_extrapolation_label, spline_knot_strategy_label, spray_brush_label,
    stop_icon_image, tool_icon_image, tool_label, tr, undo_icon_image,
};
use self::optimizer::{
    LbfgsInputState, NelderMeadInputState, OptimizerPreset, OptimizerUiMode,
    SteepestDescentInputState, infer_lbfgs_preset, infer_nelder_mead_preset,
    infer_steepest_descent_preset, lbfgs_config_from_preset, nelder_mead_config_from_preset,
    optimizer_method_label, optimizer_preset_label, steepest_descent_config_from_preset,
};
use self::param_init::{
    data_based_params_for_family, is_advanced_param_init_supported, polynomial_family,
};
use self::plot_utils::{fit_bounds_for_content, plot_domain};
use self::points_text::{parse_f64, parse_points_text_cache, points_to_text};
use crate::domain::{
    CurveFamily, CurveParams, FitResult, LbfgsConfig, NelderMeadConfig, OptimizerConfig,
    OptimizerMethod, Point, Points, SteepestDescentConfig,
};
use crate::fit::{
    FitError, SplineConfig, SplineDuplicateXPolicy, SplineExtrapolation, SplineFamilyKind,
    SplineKnotStrategy, SplineResult, calculate_iteration_metrics, default_spline_initial_knot_y,
    fit_curve_with_progress_and_optimizer_config_and_loss_metric, sample_curve,
};
#[cfg(target_arch = "wasm32")]
use crate::fit::{
    IncrementalFitRunner, IncrementalFitStep, IncrementalSplineFitRunner, IncrementalSplineFitStep,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::fit::{IncrementalSplineFitRunner, IncrementalSplineFitStep};
use crate::fit::{IterationMetricSnapshot, OptimizationLossMetric};

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
const LEFT_PANEL_DEFAULT_WIDTH: f32 = 350.0;
const LEFT_PANEL_MIN_WIDTH: f32 = 350.0;
const RIGHT_PANEL_DEFAULT_WIDTH: f32 = 300.0;
const RIGHT_PANEL_MIN_WIDTH: f32 = 300.0;
const POINTS_PARSE_DEBOUNCE_MS: u64 = 180;
const POINTS_HISTORY_LIMIT: usize = 256;
const POINTS_PARSE_ERROR_PREFIX: &str = "Points parse error: ";
const UI_CORNER_RADIUS: u8 = 6;
const PANEL_INNER_MARGIN_X: i8 = 10;
const PANEL_INNER_MARGIN_Y: i8 = 8;
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const APP_REPOSITORY_URL: &str = env!("CARGO_PKG_REPOSITORY");
#[cfg(target_arch = "wasm32")]
const WASM_FIT_BURST_STEPS: usize = 1;
#[cfg(target_arch = "wasm32")]
const WASM_FIT_BURST_TIME_BUDGET_MS: u64 = 6;
#[cfg(target_arch = "wasm32")]
const WASM_FIT_REPAINT_INTERVAL_MS: u64 = 16;

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

#[derive(Debug, Clone, PartialEq)]
enum ReplayFramePayload {
    Parametric { params: CurveParams },
    Spline { curve: Vec<PlotPoint> },
}

#[derive(Debug, Clone, PartialEq)]
struct ReplayFrame {
    iteration: u64,
    payload: ReplayFramePayload,
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
    SplineFinished(SplineResult),
    Failed(String),
}

#[cfg(target_arch = "wasm32")]
enum WasmFitRunner {
    Parametric(IncrementalFitRunner),
    Spline(IncrementalSplineFitRunner),
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
    optimizer_method: OptimizerMethod,
    optimizer_mode: OptimizerUiMode,
    optimization_loss_metric: OptimizationLossMetric,
    lbfgs_inputs: LbfgsInputState,
    lbfgs_preset: OptimizerPreset,
    nelder_mead_inputs: NelderMeadInputState,
    nelder_mead_preset: OptimizerPreset,
    steepest_descent_inputs: SteepestDescentInputState,
    steepest_descent_preset: OptimizerPreset,
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
    last_plot_bounds: Option<PlotBounds>,
    active_tool_bounds: Option<PlotBounds>,
    show_left_panel: bool,
    show_right_panel: bool,
    show_diagnostics_panel: bool,
    diagnostics_hide_non_loss_by_default_pending: bool,
    diagnostics_shared_axis_width: f32,
    iteration_delay_seconds: f64,
    replay_frames: Vec<ReplayFrame>,
    replay_selected_index: Option<usize>,
    replay_autoplay_on_fit: bool,
    replay_autoplay: bool,
    replay_last_step_at: Option<Instant>,
    spline_knots: usize,
    spline_knot_strategy: SplineKnotStrategy,
    spline_extrapolation: SplineExtrapolation,
    spline_duplicate_x_policy: SplineDuplicateXPolicy,
    spline_initial_knot_y_inputs: Vec<String>,
    fit_in_progress: bool,
    fit_loss_metric: OptimizationLossMetric,
    fit_preview_params: Option<CurveParams>,
    fit_preview_iteration: Option<u64>,
    fit_result: Option<FitResult>,
    spline_result: Option<SplineResult>,
    active_fit_points: Option<Points>,
    result_metrics: Option<ExtendedMetrics>,
    residual_plot_points: Vec<PlotPoint>,
    spline_plot_curve: Option<Vec<PlotPoint>>,
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
    wasm_fit_runner: Option<WasmFitRunner>,
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

    fn selected_optimizer_preset(&self) -> OptimizerPreset {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => self.lbfgs_preset,
            OptimizerMethod::NelderMead => self.nelder_mead_preset,
            OptimizerMethod::SteepestDescent => self.steepest_descent_preset,
        }
    }

    fn set_selected_optimizer_preset(&mut self, preset: OptimizerPreset) {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => self.lbfgs_preset = preset,
            OptimizerMethod::NelderMead => self.nelder_mead_preset = preset,
            OptimizerMethod::SteepestDescent => self.steepest_descent_preset = preset,
        }
    }

    fn apply_selected_optimizer_preset(&mut self, preset: OptimizerPreset) {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => {
                self.lbfgs_inputs = LbfgsInputState::from_config(&lbfgs_config_from_preset(preset));
                self.lbfgs_preset = preset;
            }
            OptimizerMethod::NelderMead => {
                self.nelder_mead_inputs =
                    NelderMeadInputState::from_config(&nelder_mead_config_from_preset(preset));
                self.nelder_mead_preset = preset;
            }
            OptimizerMethod::SteepestDescent => {
                self.steepest_descent_inputs = SteepestDescentInputState::from_config(
                    &steepest_descent_config_from_preset(preset),
                );
                self.steepest_descent_preset = preset;
            }
        }
    }

    fn optimizer_config(&self) -> Result<OptimizerConfig, String> {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => self.lbfgs_inputs.to_config().map(OptimizerConfig::Lbfgs),
            OptimizerMethod::NelderMead => self
                .nelder_mead_inputs
                .to_config()
                .map(OptimizerConfig::NelderMead),
            OptimizerMethod::SteepestDescent => self
                .steepest_descent_inputs
                .to_config()
                .map(OptimizerConfig::SteepestDescent),
        }
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
        self.diagnostics_hide_non_loss_by_default_pending = true;
        self.clear_fit_preview();
        self.clear_replay_state();
    }

    fn clear_replay_state(&mut self) {
        self.replay_frames.clear();
        self.replay_selected_index = None;
        self.replay_autoplay = false;
        self.replay_last_step_at = None;
    }

    fn upsert_parametric_replay_frame(&mut self, iteration: u64, params: CurveParams) {
        self.upsert_replay_frame(ReplayFrame {
            iteration,
            payload: ReplayFramePayload::Parametric { params },
        });
    }

    fn upsert_spline_replay_frame(&mut self, iteration: u64, curve: Vec<PlotPoint>) {
        self.upsert_replay_frame(ReplayFrame {
            iteration,
            payload: ReplayFramePayload::Spline { curve },
        });
    }

    fn upsert_replay_frame(&mut self, frame: ReplayFrame) {
        if let Some(last) = self.replay_frames.last_mut()
            && last.iteration == frame.iteration
        {
            *last = frame;
            return;
        }

        self.replay_frames.push(frame);
    }

    fn replay_iteration_bounds(&self) -> Option<(u64, u64)> {
        let first = self.replay_frames.first()?;
        let last = self.replay_frames.last()?;
        Some((first.iteration, last.iteration))
    }

    fn replay_selected_iteration(&self) -> Option<u64> {
        let index = self.replay_selected_index?;
        self.replay_frames.get(index).map(|frame| frame.iteration)
    }

    fn set_replay_selected_index(&mut self, index: usize) {
        let Some(frame) = self.replay_frames.get(index).cloned() else {
            return;
        };

        self.replay_selected_index = Some(index);
        self.fit_preview_iteration = Some(frame.iteration);

        match frame.payload {
            ReplayFramePayload::Parametric { params } => {
                self.fit_preview_params = Some(params);
                self.spline_plot_curve = None;
            }
            ReplayFramePayload::Spline { curve } => {
                self.fit_preview_params = None;
                self.spline_plot_curve = Some(curve);
            }
        }
    }

    fn select_nearest_replay_iteration(&mut self, iteration: u64) {
        let Some(index) = self.nearest_replay_frame_index(iteration) else {
            return;
        };
        self.set_replay_selected_index(index);
    }

    fn nearest_replay_frame_index(&self, iteration: u64) -> Option<usize> {
        let frames = self.replay_frames.as_slice();
        if frames.is_empty() {
            return None;
        }

        match frames.binary_search_by_key(&iteration, |frame| frame.iteration) {
            Ok(index) => Some(index),
            Err(insert) => {
                if insert == 0 {
                    Some(0)
                } else if insert >= frames.len() {
                    Some(frames.len() - 1)
                } else {
                    let prev = insert - 1;
                    let prev_distance = iteration.saturating_sub(frames[prev].iteration);
                    let next_distance = frames[insert].iteration.saturating_sub(iteration);
                    if next_distance < prev_distance {
                        Some(insert)
                    } else {
                        Some(prev)
                    }
                }
            }
        }
    }

    fn start_replay_from_beginning(&mut self) {
        if self.replay_frames.is_empty() {
            self.replay_autoplay = false;
            self.replay_last_step_at = None;
            return;
        }

        self.set_replay_selected_index(0);
        self.replay_autoplay = self.replay_autoplay_on_fit && self.replay_frames.len() > 1;
        self.replay_last_step_at = None;
    }

    fn toggle_replay_autoplay(&mut self) {
        if self.replay_autoplay {
            self.replay_autoplay = false;
            self.replay_last_step_at = None;
            return;
        }

        if self.replay_frames.len() < 2 {
            return;
        }

        let at_end = self
            .replay_selected_index
            .map_or(true, |index| index + 1 >= self.replay_frames.len());
        if at_end {
            self.set_replay_selected_index(0);
        } else if self.replay_selected_index.is_none() {
            self.set_replay_selected_index(0);
        }

        self.replay_autoplay = true;
        self.replay_last_step_at = None;
    }

    fn pause_replay(&mut self) {
        self.replay_autoplay = false;
        self.replay_last_step_at = None;
    }

    fn tick_replay(&mut self, ctx: &egui::Context) {
        if self.fit_in_progress || !self.replay_autoplay {
            return;
        }

        let Some(current_index) = self.replay_selected_index else {
            self.pause_replay();
            return;
        };
        if current_index + 1 >= self.replay_frames.len() {
            self.pause_replay();
            return;
        }

        if self.replay_last_step_at.is_none() {
            self.replay_last_step_at = Some(Instant::now());
            if self.iteration_delay_seconds > 0.0 {
                ctx.request_repaint_after(Duration::from_secs_f64(self.iteration_delay_seconds));
            } else {
                ctx.request_repaint();
            }
            return;
        }

        if self.iteration_delay_seconds <= 0.0 {
            self.set_replay_selected_index(current_index + 1);
            if current_index + 2 < self.replay_frames.len() {
                ctx.request_repaint();
            } else {
                self.pause_replay();
            }
            return;
        }

        let now = Instant::now();
        if let Some(last_step_at) = self.replay_last_step_at {
            let elapsed = now.saturating_duration_since(last_step_at).as_secs_f64();
            if elapsed < self.iteration_delay_seconds {
                ctx.request_repaint_after(Duration::from_secs_f64(
                    self.iteration_delay_seconds - elapsed,
                ));
                return;
            }
        }

        self.set_replay_selected_index(current_index + 1);
        self.replay_last_step_at = Some(now);

        if current_index + 2 < self.replay_frames.len() {
            ctx.request_repaint_after(Duration::from_secs_f64(self.iteration_delay_seconds));
        } else {
            self.pause_replay();
        }
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

    fn set_points_cache_from_valid_points(&mut self, points: &[Point]) {
        let parsed_points = points.to_vec();
        let plot_points = parsed_points
            .iter()
            .map(|point| PlotPoint::new(point.x(), point.y()))
            .collect();
        self.points_cache = Some(ParsedPointsCache {
            parsed_points: Ok(parsed_points),
            parse_error_line: None,
            plot_points,
        });
        self.points_cache_dirty = false;
        self.points_parse_debounce_deadline = None;
    }

    fn clear_points_text(&mut self, record_undo: bool) {
        if self.points_text.is_empty() {
            return;
        }
        let previous = std::mem::take(&mut self.points_text);
        if record_undo {
            self.push_points_undo_snapshot(previous);
        }
        self.points_redo_stack.clear();
        self.set_points_cache_from_valid_points(&[]);
        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
    }

    fn write_points_text(&mut self, points: &[Point], record_undo: bool) {
        let new_text = points_to_text(points);
        if self.points_text == new_text {
            return;
        }
        if record_undo {
            self.push_points_undo_snapshot(self.points_text.clone());
        }
        self.points_text = new_text;
        self.points_redo_stack.clear();
        self.set_points_cache_from_valid_points(points);
        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
    }

    fn fill_points_with_residuals(&mut self) {
        if self.residual_plot_points.is_empty() {
            return;
        }

        let points = match self
            .residual_plot_points
            .iter()
            .map(|point| Point::try_new(point.x, point.y))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(format!(
                    "Failed to convert residual into point: {error}"
                )));
                return;
            }
        };

        self.write_points_text(&points, true);
    }

    fn apply_visual_style(ctx: &egui::Context) {
        ctx.style_mut(|style| {
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
            optimizer_method: OptimizerMethod::Lbfgs,
            optimizer_mode: OptimizerUiMode::Basic,
            optimization_loss_metric: OptimizationLossMetric::default(),
            lbfgs_inputs: LbfgsInputState::from_config(&default_lbfgs),
            lbfgs_preset: infer_lbfgs_preset(&default_lbfgs),
            nelder_mead_inputs: NelderMeadInputState::from_config(&default_nelder_mead),
            nelder_mead_preset: infer_nelder_mead_preset(&default_nelder_mead),
            steepest_descent_inputs: SteepestDescentInputState::from_config(
                &default_steepest_descent,
            ),
            steepest_descent_preset: infer_steepest_descent_preset(&default_steepest_descent),
            ui_language: UiLanguage::English,
            plot_tool: PlotTool::SinglePoint,
            spray_density: 5,
            spray_radius_rel: 0.02,
            spray_brush: SprayBrush::Uniform,
            eraser_radius_rel: 0.03,
            spray_seed: 0xDEADBEEFCAFEBABE,
            fit_to_content_requested: false,
            center_origin_requested: false,
            origin_bottom_left_requested: true,
            last_plot_bounds: None,
            active_tool_bounds: None,
            show_left_panel: true,
            spline_knots: crate::fit::DEFAULT_SPLINE_KNOTS,
            spline_knot_strategy: SplineKnotStrategy::default(),
            spline_extrapolation: SplineExtrapolation::default(),
            spline_duplicate_x_policy: SplineDuplicateXPolicy::default(),
            spline_initial_knot_y_inputs: Vec::new(),
            show_right_panel: true,
            show_diagnostics_panel: true,
            diagnostics_hide_non_loss_by_default_pending: true,
            diagnostics_shared_axis_width: 0.0,
            iteration_delay_seconds: 0.0,
            replay_frames: Vec::new(),
            replay_selected_index: None,
            replay_autoplay_on_fit: true,
            replay_autoplay: false,
            replay_last_step_at: None,
            fit_in_progress: false,
            fit_loss_metric: OptimizationLossMetric::default(),
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
            wasm_fit_runner: None,
        }
    }
}

impl eframe::App for CurveFitApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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

        let panel_style = ctx.style();
        let panel_style = panel_style.as_ref();

        egui::TopBottomPanel::top("header_panel")
            .frame(Self::top_bottom_panel_frame(panel_style))
            .show(ctx, |ui| {
                self.ui_header(ui);
            });

        egui::TopBottomPanel::bottom("status_bar")
            .frame(Self::top_bottom_panel_frame(panel_style))
            .show(ctx, |ui| {
                self.ui_status_bar(ui);
            });

        if self.show_left_panel {
            egui::SidePanel::left("points_panel")
                .default_width(LEFT_PANEL_DEFAULT_WIDTH)
                .min_width(LEFT_PANEL_MIN_WIDTH)
                .resizable(true)
                .frame(Self::side_panel_frame(panel_style))
                .show(ctx, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 8.0);
                    ui.set_width(ui.available_width());
                    Self::panel_card_frame(ui).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        self.ui_tools(ui);
                    });
                    // ui.set_width(ui.available_width());
                    Self::panel_card_frame(ui).show(ui, |ui| {
                        // ui.set_min_width(ui.available_width());
                        self.ui_points_editor(ui);
                    });
                });
        }

        if self.show_right_panel {
            egui::SidePanel::right("settings_panel")
                .default_width(RIGHT_PANEL_DEFAULT_WIDTH)
                .min_width(RIGHT_PANEL_MIN_WIDTH)
                .resizable(true)
                .frame(Self::side_panel_frame(panel_style))
                .show(ctx, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 8.0);
                    ui.set_width(ui.available_width());
                    Self::panel_card_frame(ui).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        // ui.set_width(ui.available_width());
                        self.ui_family_and_params(ui);
                    });
                    Self::panel_card_frame(ui).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        // ui.set_width(ui.available_width());
                        self.ui_optimization_metric(ui);
                    });
                    Self::panel_card_frame(ui).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        // ui.set_width(ui.available_width());
                        self.ui_optimizer(ui);
                    });
                    Self::panel_card_frame(ui).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        // ui.set_width(ui.available_width());
                        self.ui_result(ui);
                    });
                });
        }

        if self.show_diagnostics_panel {
            egui::TopBottomPanel::bottom("diagnostics_panel")
                .resizable(true)
                .default_height(DIAGNOSTICS_PANEL_DEFAULT_HEIGHT)
                .min_height(DIAGNOSTICS_PANEL_MIN_HEIGHT)
                .frame(Self::top_bottom_panel_frame(panel_style))
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

#[cfg(test)]
mod tests;
