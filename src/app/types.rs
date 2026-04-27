//! Небольшие UI-типы и кэши, которые живут рядом с состоянием приложения.

use super::*;

/// Язык пользовательского интерфейса.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum UiLanguage {
    #[default]
    English,
    Russian,
}

impl UiLanguage {
    pub(super) const ALL: [Self; 2] = [Self::English, Self::Russian];

    pub(super) fn native_name(self) -> &'static str {
        match self {
            Self::English => "English",
            Self::Russian => "Русский",
        }
    }

    pub(super) fn from_locale_tag(locale: &str) -> Self {
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

    pub(super) fn from_system_locale() -> Self {
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

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn dialog_directory_from_path(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    path.parent().map(Path::to_path_buf)
}

pub(super) fn params_to_input_strings(params: &CurveParams) -> Vec<String> {
    params.with_values(|values| values.iter().map(|value| value.to_string()).collect())
}

pub(super) fn tau_grid_to_input_strings(values: &[f64]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

/// Инструмент редактирования точек непосредственно на графике.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum PlotTool {
    None,
    #[default]
    SinglePoint,
    Dotted,
    Spray,
    Eraser,
}

impl PlotTool {
    pub(super) fn is_navigation(self) -> bool {
        matches!(self, Self::None)
    }

    pub(super) fn is_continuous_point_editing(self) -> bool {
        matches!(self, Self::Dotted | Self::Spray | Self::Eraser)
    }
}

/// Распределение точек для spray-кисти.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum SprayBrush {
    #[default]
    Uniform,
    Gaussian,
}

/// Способ инициализации начальных параметров перед запуском оптимизатора.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParamInitMethod {
    Default,
    DataBased,
    Randomized,
}

impl ParamInitMethod {
    pub(super) const ALL: [Self; 3] = [Self::Default, Self::DataBased, Self::Randomized];

    pub(super) fn is_supported_for_family(self, family: CurveFamily) -> bool {
        match self {
            Self::Default => true,
            Self::DataBased | Self::Randomized => is_advanced_param_init_supported(family),
        }
    }
}

/// Отдельный справочный блок в окне формул: заголовок, LaTeX и пояснение.
#[derive(Debug, Clone)]
pub(super) struct FormulaReferenceSection {
    pub(super) title: String,
    pub(super) render_latex: String,
    pub(super) plain_text: String,
    pub(super) description: String,
}

/// Набор справочных блоков для окна `Model Formula`.
#[derive(Debug, Clone)]
pub(super) struct ModelFormulaInfo {
    pub(super) model_plain_text: String,
    pub(super) reference_plain_text: String,
    pub(super) sections: Vec<FormulaReferenceSection>,
}

/// Кэш уже отрендеренной SVG-формулы, зависящий от темы и текста формулы.
#[derive(Debug, Clone)]
pub(super) struct FormulaSvgCache {
    pub(super) formula: String,
    pub(super) dark_mode: bool,
    pub(super) uri: String,
    pub(super) render_result: Result<Arc<[u8]>, String>,
}

/// Кэш уже сэмплированной параметрической кривой для текущего диапазона `x`.
#[derive(Debug, Clone)]
pub(super) struct SampledCurveCache {
    pub(super) params: CurveParams,
    pub(super) x_min_bits: u64,
    pub(super) x_max_bits: u64,
    pub(super) samples: usize,
    pub(super) curve: Arc<[PlotPoint]>,
}

/// Набор метрик, показываемых в панели результата.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ExtendedMetrics {
    pub(super) mse: f64,
    pub(super) rmse: f64,
    pub(super) mae: f64,
    pub(super) r2: f64,
    pub(super) max_abs_error: f64,
}
