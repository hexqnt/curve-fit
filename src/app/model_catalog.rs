//! Связка UI-выбора моделей с доменными семействами параметрических кривых и сплайнов.

use super::*;

/// Модели, которые пользователь может выбрать в интерфейсе.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ModelChoice {
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
    Rational11,
    Rational22,
    Emg,
    PseudoVoigt,
    LinearSpline,
    MonotoneCubicSpline,
    NaturalCubicSpline,
    AkimaSpline,
}

impl ModelChoice {
    pub(super) const ALL: [Self; 29] = [
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
        Self::Rational11,
        Self::Rational22,
        Self::Emg,
        Self::PseudoVoigt,
        Self::LinearSpline,
        Self::MonotoneCubicSpline,
        Self::NaturalCubicSpline,
        Self::AkimaSpline,
    ];

    pub(super) fn is_polynomial(self) -> bool {
        matches!(self, Self::Polynomial)
    }
}

/// Нормализованный выбор модели после учета степени полинома.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResolvedModel {
    Parametric(CurveFamily),
    LinearSpline,
    MonotoneCubicSpline,
    NaturalCubicSpline,
    AkimaSpline,
}

impl ResolvedModel {
    pub(super) fn from_choice(choice: ModelChoice, polynomial_degree: usize) -> Self {
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
            ModelChoice::Rational11 => Self::Parametric(CurveFamily::Rational11),
            ModelChoice::Rational22 => Self::Parametric(CurveFamily::Rational22),
            ModelChoice::Emg => Self::Parametric(CurveFamily::Emg),
            ModelChoice::PseudoVoigt => Self::Parametric(CurveFamily::PseudoVoigt),
            ModelChoice::LinearSpline => Self::LinearSpline,
            ModelChoice::MonotoneCubicSpline => Self::MonotoneCubicSpline,
            ModelChoice::NaturalCubicSpline => Self::NaturalCubicSpline,
            ModelChoice::AkimaSpline => Self::AkimaSpline,
        }
    }

    pub(super) fn parametric_family(self) -> Option<CurveFamily> {
        match self {
            Self::Parametric(family) => Some(family),
            Self::LinearSpline
            | Self::MonotoneCubicSpline
            | Self::NaturalCubicSpline
            | Self::AkimaSpline => None,
        }
    }

    pub(super) fn spline_family(self) -> Option<SplineFamilyKind> {
        match self {
            Self::LinearSpline => Some(SplineFamilyKind::Linear),
            Self::MonotoneCubicSpline => Some(SplineFamilyKind::MonotoneCubic),
            Self::NaturalCubicSpline => Some(SplineFamilyKind::NaturalCubic),
            Self::AkimaSpline => Some(SplineFamilyKind::Akima),
            Self::Parametric(_) => None,
        }
    }

    pub(super) fn spline_min_knots(self) -> Option<usize> {
        match self {
            Self::Parametric(_) => None,
            Self::LinearSpline | Self::MonotoneCubicSpline => Some(2),
            Self::NaturalCubicSpline => Some(3),
            Self::AkimaSpline => Some(5),
        }
    }
}

/// Группы моделей для компактного меню выбора в UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelGroup {
    Polynomial,
    ParametricGeneral,
    ParametricSigmoid,
    ParametricPeak,
    Spline,
}

impl ModelGroup {
    pub(super) const ALL: [Self; 5] = [
        Self::Polynomial,
        Self::ParametricGeneral,
        Self::ParametricSigmoid,
        Self::ParametricPeak,
        Self::Spline,
    ];
}

pub(super) fn model_group(model: ModelChoice) -> ModelGroup {
    match model {
        ModelChoice::Polynomial => ModelGroup::Polynomial,
        ModelChoice::Logistic
        | ModelChoice::Gompertz
        | ModelChoice::FourPl
        | ModelChoice::FivePl
        | ModelChoice::HyperbolicTangent
        | ModelChoice::ArctangentStep
        | ModelChoice::Softplus => ModelGroup::ParametricSigmoid,
        ModelChoice::Lorentzian
        | ModelChoice::Gaussian
        | ModelChoice::Emg
        | ModelChoice::PseudoVoigt => ModelGroup::ParametricPeak,
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
        | ModelChoice::Power
        | ModelChoice::Rational11
        | ModelChoice::Rational22 => ModelGroup::ParametricGeneral,
    }
}

pub(super) fn model_group_label(language: UiLanguage, group: ModelGroup) -> &'static str {
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

pub(super) fn spline_duplicate_policy_label(
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
