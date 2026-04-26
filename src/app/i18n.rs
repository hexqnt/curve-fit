use eframe::egui;

use crate::domain::CurveFamily;
use crate::fit::{OptimizationLossMetric, SplineExtrapolation, SplineKnotStrategy};

use super::{ModelChoice, ParamInitMethod, PlotTool, SprayBrush, UiLanguage};

const TABLER_ICON_SIZE: f32 = 16.0;

macro_rules! tabler_icon {
    ($path:literal, $tint:expr) => {
        egui::Image::new(egui::include_image!($path))
            .fit_to_exact_size(egui::vec2(TABLER_ICON_SIZE, TABLER_ICON_SIZE))
            .tint($tint)
    };
}

/// Небольшая утилита для выбора строки по текущему языку интерфейса.
pub(super) fn tr(language: UiLanguage, en: &'static str, ru: &'static str) -> &'static str {
    match language {
        UiLanguage::English => en,
        UiLanguage::Russian => ru,
    }
}

pub(super) fn language_flag_image(language: UiLanguage) -> egui::Image<'static> {
    let source = match language {
        UiLanguage::English => egui::include_image!("../../assets/flags/us.svg"),
        UiLanguage::Russian => egui::include_image!("../../assets/flags/ru.svg"),
    };
    egui::Image::new(source).fit_to_exact_size(egui::vec2(18.0, 13.5))
}

pub(super) fn github_mark_image(dark_mode: bool) -> egui::Image<'static> {
    let source = if dark_mode {
        egui::include_image!("../../assets/tm/Octicons-mark-github-white.svg")
    } else {
        egui::include_image!("../../assets/tm/Octicons-mark-github.svg")
    };
    egui::Image::new(source).fit_to_exact_size(egui::vec2(16.0, 16.0))
}

pub(super) fn fit_to_content_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/arrows-maximize.svg", tint)
}

pub(super) fn center_origin_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/focus-2.svg", tint)
}

pub(super) fn origin_bottom_left_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/align-box-bottom-left.svg", tint)
}

pub(super) fn open_formula_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/math-function.svg", tint)
}

pub(super) fn view_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/zoom-scan.svg", tint)
}

pub(super) fn panels_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/layout.svg", tint)
}

pub(super) fn undo_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/arrow-back-up.svg", tint)
}

pub(super) fn redo_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/arrow-forward-up.svg", tint)
}

pub(super) fn clear_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/clear-all.svg", tint)
}

pub(super) fn layer_new_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/plus.svg", tint)
}

pub(super) fn layer_duplicate_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/copy.svg", tint)
}

pub(super) fn layer_delete_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/trash.svg", tint)
}

pub(super) fn layer_visible_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/eye.svg", tint)
}

pub(super) fn layer_hidden_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/eye-x.svg", tint)
}

pub(super) fn actions_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/dots.svg", tint)
}

pub(super) fn clipboard_import_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/clipboard-text.svg", tint)
}

pub(super) fn file_import_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/file-import.svg", tint)
}

pub(super) fn reset_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/restore.svg", tint)
}

pub(super) fn fit_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/player-play.svg", tint)
}

pub(super) fn replay_play_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/player-play.svg", tint)
}

pub(super) fn replay_pause_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/player-pause.svg", tint)
}

pub(super) fn stop_icon_image(tint: egui::Color32) -> egui::Image<'static> {
    tabler_icon!("../../assets/icons/tabler/player-stop.svg", tint)
}

pub(super) fn tool_icon_image(tool: PlotTool, tint: egui::Color32) -> egui::Image<'static> {
    match tool {
        PlotTool::None => tabler_icon!("../../assets/icons/tabler/pointer.svg", tint),
        PlotTool::SinglePoint => tabler_icon!("../../assets/icons/tabler/point.svg", tint),
        PlotTool::Dotted => tabler_icon!("../../assets/icons/tabler/line-dotted.svg", tint),
        PlotTool::Spray => tabler_icon!("../../assets/icons/tabler/spray.svg", tint),
        PlotTool::Eraser => tabler_icon!("../../assets/icons/tabler/eraser.svg", tint),
    }
}

pub(super) fn spray_brush_label(language: UiLanguage, brush: SprayBrush) -> &'static str {
    match (language, brush) {
        (UiLanguage::English, SprayBrush::Uniform) => "Uniform",
        (UiLanguage::English, SprayBrush::Gaussian) => "Gaussian",
        (UiLanguage::Russian, SprayBrush::Uniform) => "Равномерная",
        (UiLanguage::Russian, SprayBrush::Gaussian) => "Гауссова",
    }
}

pub(super) fn spline_knot_strategy_label(
    language: UiLanguage,
    strategy: SplineKnotStrategy,
) -> &'static str {
    match (language, strategy) {
        (UiLanguage::English, SplineKnotStrategy::BinMean) => "Bin mean",
        (UiLanguage::English, SplineKnotStrategy::BinMedian) => "Bin median",
        (UiLanguage::Russian, SplineKnotStrategy::BinMean) => "Среднее по окнам",
        (UiLanguage::Russian, SplineKnotStrategy::BinMedian) => "Медиана по окнам",
    }
}

pub(super) fn spline_extrapolation_label(
    language: UiLanguage,
    extrapolation: SplineExtrapolation,
) -> &'static str {
    match (language, extrapolation) {
        (UiLanguage::English, SplineExtrapolation::Clamp) => "Clamp to edge",
        (UiLanguage::English, SplineExtrapolation::Linear) => "Linear",
        (UiLanguage::Russian, SplineExtrapolation::Clamp) => "Фиксация на краю",
        (UiLanguage::Russian, SplineExtrapolation::Linear) => "Линейная",
    }
}

pub(super) fn optimization_loss_metric_label(
    _language: UiLanguage,
    metric: OptimizationLossMetric,
) -> &'static str {
    match metric {
        OptimizationLossMetric::Mse => "MSE (L2)",
        OptimizationLossMetric::Mae => "MAE (L1)",
        OptimizationLossMetric::SoftL1 => "soft_l1",
        OptimizationLossMetric::Chebyshev => "Chebyshev",
        OptimizationLossMetric::Msle => "MSLE",
    }
}

pub(super) fn param_init_method_name_en(method: ParamInitMethod) -> &'static str {
    match method {
        ParamInitMethod::Default => "Default",
        ParamInitMethod::DataBased => "Data-based",
        ParamInitMethod::Randomized => "Randomized",
    }
}

pub(super) fn param_init_method_label(
    language: UiLanguage,
    method: ParamInitMethod,
) -> &'static str {
    match (language, method) {
        (UiLanguage::English, ParamInitMethod::Default) => "Default",
        (UiLanguage::English, ParamInitMethod::DataBased) => "Data-based",
        (UiLanguage::English, ParamInitMethod::Randomized) => "Randomized",
        (UiLanguage::Russian, ParamInitMethod::Default) => "По умолчанию",
        (UiLanguage::Russian, ParamInitMethod::DataBased) => "По данным",
        (UiLanguage::Russian, ParamInitMethod::Randomized) => "Случайно",
    }
}

pub(super) fn param_init_method_disabled_label(
    language: UiLanguage,
    method: ParamInitMethod,
) -> &'static str {
    match (language, method) {
        (UiLanguage::English, ParamInitMethod::DataBased) => {
            "Data-based (Polynomial/Logistic/Gompertz/Bi-Exponential/Damped Sinusoid/Gaussian/Exponential (Basic)/Power/Rational/EMG/Pseudo-Voigt only)"
        }
        (UiLanguage::English, ParamInitMethod::Randomized) => {
            "Randomized (Polynomial/Logistic/Gompertz/Bi-Exponential/Damped Sinusoid/Gaussian/Exponential (Basic)/Power/Rational/EMG/Pseudo-Voigt only)"
        }
        (UiLanguage::English, ParamInitMethod::Default) => "Default",
        (UiLanguage::Russian, ParamInitMethod::DataBased) => {
            "По данным (только Polynomial/Logistic/Gompertz/Bi-Exponential/Damped Sinusoid/Gaussian/Exponential (Basic)/Power/Rational/EMG/Pseudo-Voigt)"
        }
        (UiLanguage::Russian, ParamInitMethod::Randomized) => {
            "Случайно (только Polynomial/Logistic/Gompertz/Bi-Exponential/Damped Sinusoid/Gaussian/Exponential (Basic)/Power/Rational/EMG/Pseudo-Voigt)"
        }
        (UiLanguage::Russian, ParamInitMethod::Default) => "По умолчанию",
    }
}

pub(super) fn model_choice_label(language: UiLanguage, model: ModelChoice) -> &'static str {
    match (language, model) {
        (UiLanguage::English, ModelChoice::Polynomial) => "Polynomial",
        (UiLanguage::English, ModelChoice::Arrhenius) => "Arrhenius",
        (UiLanguage::English, ModelChoice::Inverse) => "Inverse",
        (UiLanguage::English, ModelChoice::Logistic) => "Logistic",
        (UiLanguage::English, ModelChoice::Gompertz) => "Gompertz",
        (UiLanguage::English, ModelChoice::BiExponential) => "Bi-Exponential",
        (UiLanguage::English, ModelChoice::DampedSinusoid) => "Damped Sinusoid",
        (UiLanguage::English, ModelChoice::Lorentzian) => "Lorentzian",
        (UiLanguage::English, ModelChoice::NaturalLog) => "Natural Log",
        (UiLanguage::English, ModelChoice::FourPl) => "4PL",
        (UiLanguage::English, ModelChoice::FivePl) => "5PL",
        (UiLanguage::English, ModelChoice::MichaelisMenten) => "Michaelis-Menten",
        (UiLanguage::English, ModelChoice::ExponentialBasic) => "Exponential (Basic)",
        (UiLanguage::English, ModelChoice::ExponentialLinear) => "Exponential + Linear",
        (UiLanguage::English, ModelChoice::ExponentialHalfLife) => "Exponential (Half-life)",
        (UiLanguage::English, ModelChoice::FallingExponential) => "Falling Exponential",
        (UiLanguage::English, ModelChoice::HyperbolicTangent) => "Hyperbolic Tangent",
        (UiLanguage::English, ModelChoice::ArctangentStep) => "Arctangent Step",
        (UiLanguage::English, ModelChoice::Softplus) => "Softplus",
        (UiLanguage::English, ModelChoice::Power) => "Power",
        (UiLanguage::English, ModelChoice::Gaussian) => "Gaussian",
        (UiLanguage::English, ModelChoice::Rational) => "Rational",
        (UiLanguage::English, ModelChoice::Emg) => "EMG",
        (UiLanguage::English, ModelChoice::PseudoVoigt) => "Pseudo-Voigt",
        (UiLanguage::English, ModelChoice::SaturatingTrendBasis) => "Saturating Trend Basis",
        (UiLanguage::English, ModelChoice::LinearSpline) => "Linear Spline",
        (UiLanguage::English, ModelChoice::MonotoneCubicSpline) => "Monotone Cubic (PCHIP)",
        (UiLanguage::English, ModelChoice::NaturalCubicSpline) => "Natural Cubic Spline",
        (UiLanguage::English, ModelChoice::AkimaSpline) => "Akima Cubic Spline",
        (UiLanguage::Russian, ModelChoice::Polynomial) => "Полином",
        (UiLanguage::Russian, ModelChoice::Arrhenius) => "Аррениус",
        (UiLanguage::Russian, ModelChoice::Inverse) => "Обратная",
        (UiLanguage::Russian, ModelChoice::Logistic) => "Логистическая",
        (UiLanguage::Russian, ModelChoice::Gompertz) => "Гомпертц",
        (UiLanguage::Russian, ModelChoice::BiExponential) => "Би-экспонента",
        (UiLanguage::Russian, ModelChoice::DampedSinusoid) => "Затухающая синусоида",
        (UiLanguage::Russian, ModelChoice::Lorentzian) => "Лоренциан",
        (UiLanguage::Russian, ModelChoice::NaturalLog) => "Натуральный логарифм",
        (UiLanguage::Russian, ModelChoice::FourPl) => "4PL",
        (UiLanguage::Russian, ModelChoice::FivePl) => "5PL",
        (UiLanguage::Russian, ModelChoice::MichaelisMenten) => "Михаэлис-Ментен",
        (UiLanguage::Russian, ModelChoice::ExponentialBasic) => "Экспонента (базовая)",
        (UiLanguage::Russian, ModelChoice::ExponentialLinear) => "Экспонента + линейный тренд",
        (UiLanguage::Russian, ModelChoice::ExponentialHalfLife) => "Экспонента (полураспад)",
        (UiLanguage::Russian, ModelChoice::FallingExponential) => "Падающая экспонента",
        (UiLanguage::Russian, ModelChoice::HyperbolicTangent) => "Гиперболический тангенс",
        (UiLanguage::Russian, ModelChoice::ArctangentStep) => "Арктангенс переход",
        (UiLanguage::Russian, ModelChoice::Softplus) => "Softplus",
        (UiLanguage::Russian, ModelChoice::Power) => "Степенная",
        (UiLanguage::Russian, ModelChoice::Gaussian) => "Гаусс",
        (UiLanguage::Russian, ModelChoice::Rational) => "Рациональная",
        (UiLanguage::Russian, ModelChoice::Emg) => "EMG",
        (UiLanguage::Russian, ModelChoice::PseudoVoigt) => "Псевдо-Фойгт",
        (UiLanguage::Russian, ModelChoice::SaturatingTrendBasis) => "Базис насыщаемых трендов",
        (UiLanguage::Russian, ModelChoice::LinearSpline) => "Линейный сплайн",
        (UiLanguage::Russian, ModelChoice::MonotoneCubicSpline) => "Монотонный кубический (PCHIP)",
        (UiLanguage::Russian, ModelChoice::NaturalCubicSpline) => "Натуральный кубический сплайн",
        (UiLanguage::Russian, ModelChoice::AkimaSpline) => "Кубический сплайн Акимы",
    }
}

pub(super) fn family_label(language: UiLanguage, family: CurveFamily) -> &'static str {
    match (language, family) {
        (UiLanguage::English, CurveFamily::Linear) => "Linear",
        (UiLanguage::English, CurveFamily::Quadratic) => "Quadratic",
        (UiLanguage::English, CurveFamily::Cubic) => "Cubic",
        (UiLanguage::English, CurveFamily::Quartic) => "Quartic",
        (UiLanguage::English, CurveFamily::Quintic) => "Quintic",
        (UiLanguage::English, CurveFamily::Sextic) => "Sextic",
        (UiLanguage::English, CurveFamily::Septic) => "Septic",
        (UiLanguage::English, CurveFamily::Octic) => "Octic",
        (UiLanguage::English, CurveFamily::Nonic) => "Nonic",
        (UiLanguage::English, CurveFamily::Arrhenius) => "Arrhenius",
        (UiLanguage::English, CurveFamily::Inverse) => "Inverse",
        (UiLanguage::English, CurveFamily::Logistic) => "Logistic",
        (UiLanguage::English, CurveFamily::Gompertz) => "Gompertz",
        (UiLanguage::English, CurveFamily::BiExponential) => "Bi-Exponential",
        (UiLanguage::English, CurveFamily::DampedSinusoid) => "Damped Sinusoid",
        (UiLanguage::English, CurveFamily::Lorentzian) => "Lorentzian",
        (UiLanguage::English, CurveFamily::NaturalLog) => "Natural Log",
        (UiLanguage::English, CurveFamily::FourPl) => "4PL",
        (UiLanguage::English, CurveFamily::FivePl) => "5PL",
        (UiLanguage::English, CurveFamily::MichaelisMenten) => "Michaelis-Menten",
        (UiLanguage::English, CurveFamily::ExponentialBasic) => "Exponential (Basic)",
        (UiLanguage::English, CurveFamily::ExponentialLinear) => "Exponential + Linear",
        (UiLanguage::English, CurveFamily::ExponentialHalfLife) => "Exponential (Half-life)",
        (UiLanguage::English, CurveFamily::FallingExponential) => "Falling Exponential",
        (UiLanguage::English, CurveFamily::HyperbolicTangent) => "Hyperbolic Tangent",
        (UiLanguage::English, CurveFamily::ArctangentStep) => "Arctangent Step",
        (UiLanguage::English, CurveFamily::Softplus) => "Softplus",
        (UiLanguage::English, CurveFamily::Power) => "Power",
        (UiLanguage::English, CurveFamily::Gaussian) => "Gaussian",
        (UiLanguage::English, CurveFamily::Rational11) => "Rational (1/1)",
        (UiLanguage::English, CurveFamily::Rational22) => "Rational (2/2)",
        (UiLanguage::English, CurveFamily::Rational33) => "Rational (3/3)",
        (UiLanguage::English, CurveFamily::Rational44) => "Rational (4/4)",
        (UiLanguage::English, CurveFamily::Rational55) => "Rational (5/5)",
        (UiLanguage::English, CurveFamily::Emg) => "EMG",
        (UiLanguage::English, CurveFamily::PseudoVoigt) => "Pseudo-Voigt",
        (UiLanguage::English, CurveFamily::SaturatingTrendBasis1) => {
            "Saturating Trend Basis (1 tau)"
        }
        (UiLanguage::English, CurveFamily::SaturatingTrendBasis2) => {
            "Saturating Trend Basis (2 tau)"
        }
        (UiLanguage::English, CurveFamily::SaturatingTrendBasis3) => {
            "Saturating Trend Basis (3 tau)"
        }
        (UiLanguage::English, CurveFamily::SaturatingTrendBasis4) => {
            "Saturating Trend Basis (4 tau)"
        }
        (UiLanguage::English, CurveFamily::SaturatingTrendBasis5) => {
            "Saturating Trend Basis (5 tau)"
        }
        (UiLanguage::English, CurveFamily::SaturatingTrendBasis6) => {
            "Saturating Trend Basis (6 tau)"
        }
        (UiLanguage::Russian, CurveFamily::Linear) => "Линейная",
        (UiLanguage::Russian, CurveFamily::Quadratic) => "Квадратичная",
        (UiLanguage::Russian, CurveFamily::Cubic) => "Кубическая",
        (UiLanguage::Russian, CurveFamily::Quartic) => "4-й степени",
        (UiLanguage::Russian, CurveFamily::Quintic) => "5-й степени",
        (UiLanguage::Russian, CurveFamily::Sextic) => "6-й степени",
        (UiLanguage::Russian, CurveFamily::Septic) => "7-й степени",
        (UiLanguage::Russian, CurveFamily::Octic) => "8-й степени",
        (UiLanguage::Russian, CurveFamily::Nonic) => "9-й степени",
        (UiLanguage::Russian, CurveFamily::Arrhenius) => "Аррениус",
        (UiLanguage::Russian, CurveFamily::Inverse) => "Обратная",
        (UiLanguage::Russian, CurveFamily::Logistic) => "Логистическая",
        (UiLanguage::Russian, CurveFamily::Gompertz) => "Гомпертц",
        (UiLanguage::Russian, CurveFamily::BiExponential) => "Би-экспонента",
        (UiLanguage::Russian, CurveFamily::DampedSinusoid) => "Затухающая синусоида",
        (UiLanguage::Russian, CurveFamily::Lorentzian) => "Лоренциан",
        (UiLanguage::Russian, CurveFamily::NaturalLog) => "Натуральный логарифм",
        (UiLanguage::Russian, CurveFamily::FourPl) => "4PL",
        (UiLanguage::Russian, CurveFamily::FivePl) => "5PL",
        (UiLanguage::Russian, CurveFamily::MichaelisMenten) => "Михаэлис-Ментен",
        (UiLanguage::Russian, CurveFamily::ExponentialBasic) => "Экспонента (базовая)",
        (UiLanguage::Russian, CurveFamily::ExponentialLinear) => "Экспонента + линейный тренд",
        (UiLanguage::Russian, CurveFamily::ExponentialHalfLife) => "Экспонента (полураспад)",
        (UiLanguage::Russian, CurveFamily::FallingExponential) => "Падающая экспонента",
        (UiLanguage::Russian, CurveFamily::HyperbolicTangent) => "Гиперболический тангенс",
        (UiLanguage::Russian, CurveFamily::ArctangentStep) => "Арктангенс переход",
        (UiLanguage::Russian, CurveFamily::Softplus) => "Softplus",
        (UiLanguage::Russian, CurveFamily::Power) => "Степенная",
        (UiLanguage::Russian, CurveFamily::Gaussian) => "Гаусс",
        (UiLanguage::Russian, CurveFamily::Rational11) => "Рациональная (1/1)",
        (UiLanguage::Russian, CurveFamily::Rational22) => "Рациональная (2/2)",
        (UiLanguage::Russian, CurveFamily::Rational33) => "Рациональная (3/3)",
        (UiLanguage::Russian, CurveFamily::Rational44) => "Рациональная (4/4)",
        (UiLanguage::Russian, CurveFamily::Rational55) => "Рациональная (5/5)",
        (UiLanguage::Russian, CurveFamily::Emg) => "EMG",
        (UiLanguage::Russian, CurveFamily::PseudoVoigt) => "Псевдо-Фойгт",
        (UiLanguage::Russian, CurveFamily::SaturatingTrendBasis1) => {
            "Базис насыщаемых трендов (1 tau)"
        }
        (UiLanguage::Russian, CurveFamily::SaturatingTrendBasis2) => {
            "Базис насыщаемых трендов (2 tau)"
        }
        (UiLanguage::Russian, CurveFamily::SaturatingTrendBasis3) => {
            "Базис насыщаемых трендов (3 tau)"
        }
        (UiLanguage::Russian, CurveFamily::SaturatingTrendBasis4) => {
            "Базис насыщаемых трендов (4 tau)"
        }
        (UiLanguage::Russian, CurveFamily::SaturatingTrendBasis5) => {
            "Базис насыщаемых трендов (5 tau)"
        }
        (UiLanguage::Russian, CurveFamily::SaturatingTrendBasis6) => {
            "Базис насыщаемых трендов (6 tau)"
        }
    }
}
