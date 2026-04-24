//! Перечень поддерживаемых семейств кривых и их статические метаданные.

use std::fmt;

use super::{
    CurveParams, DEFAULT_SATURATING_TREND_TAUS_YEARS, InputError, MAX_SATURATING_TREND_TAU_COUNT,
    MIN_SATURATING_TREND_TAU_COUNT, Points, SaturatingTrendTauGrid,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Поддерживаемые семейства аналитических кривых.
pub enum CurveFamily {
    Linear,
    Quadratic,
    Cubic,
    Quartic,
    Quintic,
    Sextic,
    Septic,
    Octic,
    Nonic,
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
    Rational33,
    Rational44,
    Rational55,
    Emg,
    PseudoVoigt,
    SaturatingTrendBasis1,
    SaturatingTrendBasis2,
    SaturatingTrendBasis3,
    SaturatingTrendBasis4,
    SaturatingTrendBasis5,
    SaturatingTrendBasis6,
}

pub(crate) const CURVE_FAMILY_COUNT: usize = CurveFamily::SaturatingTrendBasis6 as usize + 1;
/// Минимально поддерживаемая степень рациональной модели `n/n`.
pub const MIN_RATIONAL_DEGREE: usize = 1;
/// Максимально поддерживаемая степень рациональной модели `n/n`.
pub const MAX_RATIONAL_DEGREE: usize = 5;

#[derive(Debug, Clone, Copy)]
struct CurveFamilyMetadata {
    label: &'static str,
    parameter_names: &'static [&'static str],
    min_points: usize,
    requires_positive_x: bool,
}

const CURVE_FAMILY_METADATA: [CurveFamilyMetadata; CURVE_FAMILY_COUNT] = [
    CurveFamilyMetadata {
        label: "Linear",
        parameter_names: &["a", "b"],
        min_points: 2,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Quadratic",
        parameter_names: &["a", "b", "c"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Cubic",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Quartic",
        parameter_names: &["a", "b", "c", "d", "e"],
        min_points: 5,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Quintic",
        parameter_names: &["a", "b", "c", "d", "e", "f"],
        min_points: 6,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Sextic",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g"],
        min_points: 7,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Septic",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g", "h"],
        min_points: 8,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Octic",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        min_points: 9,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Nonic",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        min_points: 10,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Arrhenius",
        parameter_names: &["A", "B"],
        min_points: 2,
        requires_positive_x: true,
    },
    CurveFamilyMetadata {
        label: "Inverse",
        parameter_names: &["A", "B"],
        min_points: 2,
        requires_positive_x: true,
    },
    CurveFamilyMetadata {
        label: "Logistic",
        parameter_names: &["A", "B", "C"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Gompertz",
        parameter_names: &["A", "B", "C"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Bi-Exponential",
        parameter_names: &["a1", "k1", "a2", "k2", "c"],
        min_points: 5,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Damped Sinusoid",
        parameter_names: &["a", "k", "omega", "phi", "c"],
        min_points: 5,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Lorentzian",
        parameter_names: &["A", "x0", "gamma", "C"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Natural Log",
        parameter_names: &["A", "B"],
        min_points: 2,
        requires_positive_x: true,
    },
    CurveFamilyMetadata {
        label: "4PL",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: true,
    },
    CurveFamilyMetadata {
        label: "5PL",
        parameter_names: &["a", "b", "c", "d", "m"],
        min_points: 5,
        requires_positive_x: true,
    },
    CurveFamilyMetadata {
        label: "Michaelis-Menten",
        parameter_names: &["Vmax", "Km"],
        min_points: 2,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Exponential (Basic)",
        parameter_names: &["a", "b", "c"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Exponential + Linear",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Exponential (Half-life)",
        parameter_names: &["a", "b", "c"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Falling Exponential",
        parameter_names: &["Y0", "V0", "K"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Hyperbolic Tangent",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Arctangent Step",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Softplus",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Power",
        parameter_names: &["a", "b"],
        min_points: 2,
        requires_positive_x: true,
    },
    CurveFamilyMetadata {
        label: "Gaussian",
        parameter_names: &["a", "b", "c"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Rational (1/1)",
        parameter_names: &["a", "b", "c", "d"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Rational (2/2)",
        parameter_names: &["a", "b", "c", "d", "e"],
        min_points: 5,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Rational (3/3)",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g"],
        min_points: 7,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Rational (4/4)",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        min_points: 9,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Rational (5/5)",
        parameter_names: &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        min_points: 11,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "EMG",
        parameter_names: &["a", "mu", "sigma", "tau", "c"],
        min_points: 5,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Pseudo-Voigt",
        parameter_names: &["a", "x0", "sigma", "gamma", "eta", "c"],
        min_points: 6,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Saturating Trend Basis (1 tau)",
        parameter_names: &["c", "w1"],
        min_points: 2,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Saturating Trend Basis (2 tau)",
        parameter_names: &["c", "w1", "w2"],
        min_points: 3,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Saturating Trend Basis (3 tau)",
        parameter_names: &["c", "w1", "w2", "w3"],
        min_points: 4,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Saturating Trend Basis (4 tau)",
        parameter_names: &["c", "w1", "w2", "w3", "w4"],
        min_points: 5,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Saturating Trend Basis (5 tau)",
        parameter_names: &["c", "w1", "w2", "w3", "w4", "w5"],
        min_points: 6,
        requires_positive_x: false,
    },
    CurveFamilyMetadata {
        label: "Saturating Trend Basis (6 tau)",
        parameter_names: &["c", "w1", "w2", "w3", "w4", "w5", "w6"],
        min_points: 7,
        requires_positive_x: false,
    },
];

impl CurveFamily {
    /// Полный список семейств в стабильном порядке для UI и переборов.
    pub const ALL: [Self; CURVE_FAMILY_COUNT] = [
        Self::Linear,
        Self::Quadratic,
        Self::Cubic,
        Self::Quartic,
        Self::Quintic,
        Self::Sextic,
        Self::Septic,
        Self::Octic,
        Self::Nonic,
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
        Self::Rational33,
        Self::Rational44,
        Self::Rational55,
        Self::Emg,
        Self::PseudoVoigt,
        Self::SaturatingTrendBasis1,
        Self::SaturatingTrendBasis2,
        Self::SaturatingTrendBasis3,
        Self::SaturatingTrendBasis4,
        Self::SaturatingTrendBasis5,
        Self::SaturatingTrendBasis6,
    ];

    #[inline]
    fn metadata(self) -> &'static CurveFamilyMetadata {
        debug_assert_eq!(Self::ALL.len(), CURVE_FAMILY_METADATA.len());
        &CURVE_FAMILY_METADATA[self as usize]
    }

    /// Короткое человекочитаемое имя семейства.
    pub fn label(self) -> &'static str {
        self.metadata().label
    }

    /// Имена параметров в порядке внутреннего вектора значений.
    pub fn parameter_names(self) -> &'static [&'static str] {
        self.metadata().parameter_names
    }

    /// Возвращает `true`, если семейство является полиномом степени `1..=9`.
    pub fn is_polynomial(self) -> bool {
        matches!(
            self,
            Self::Linear
                | Self::Quadratic
                | Self::Cubic
                | Self::Quartic
                | Self::Quintic
                | Self::Sextic
                | Self::Septic
                | Self::Octic
                | Self::Nonic
        )
    }

    /// Возвращает `true`, если семейство является рациональной моделью `n/n`.
    pub fn is_rational(self) -> bool {
        matches!(
            self,
            Self::Rational11
                | Self::Rational22
                | Self::Rational33
                | Self::Rational44
                | Self::Rational55
        )
    }

    /// Возвращает `true`, если семейство — saturating basis по префиксу фиксированной сетки `τ`.
    pub fn is_saturating_trend_basis(self) -> bool {
        matches!(
            self,
            Self::SaturatingTrendBasis1
                | Self::SaturatingTrendBasis2
                | Self::SaturatingTrendBasis3
                | Self::SaturatingTrendBasis4
                | Self::SaturatingTrendBasis5
                | Self::SaturatingTrendBasis6
        )
    }

    /// Возвращает рациональное семейство `n/n` для заданной степени.
    ///
    /// Значение автоматически ограничивается поддерживаемым диапазоном.
    pub fn from_rational_degree(degree: usize) -> Self {
        match degree.clamp(MIN_RATIONAL_DEGREE, MAX_RATIONAL_DEGREE) {
            1 => Self::Rational11,
            2 => Self::Rational22,
            3 => Self::Rational33,
            4 => Self::Rational44,
            _ => Self::Rational55,
        }
    }

    /// Возвращает степень рациональной модели `n/n`, если семейство рациональное.
    pub fn rational_degree(self) -> Option<usize> {
        match self {
            Self::Rational11 => Some(1),
            Self::Rational22 => Some(2),
            Self::Rational33 => Some(3),
            Self::Rational44 => Some(4),
            Self::Rational55 => Some(5),
            _ => None,
        }
    }

    /// Возвращает saturating-basis семейство по числу активных `τ`.
    pub fn from_saturating_trend_tau_count(count: usize) -> Self {
        match count.clamp(
            MIN_SATURATING_TREND_TAU_COUNT,
            MAX_SATURATING_TREND_TAU_COUNT,
        ) {
            1 => Self::SaturatingTrendBasis1,
            2 => Self::SaturatingTrendBasis2,
            3 => Self::SaturatingTrendBasis3,
            4 => Self::SaturatingTrendBasis4,
            5 => Self::SaturatingTrendBasis5,
            _ => Self::SaturatingTrendBasis6,
        }
    }

    /// Возвращает число активных `τ`, если семейство saturating-basis.
    pub fn saturating_trend_tau_count(self) -> Option<usize> {
        match self {
            Self::SaturatingTrendBasis1 => Some(1),
            Self::SaturatingTrendBasis2 => Some(2),
            Self::SaturatingTrendBasis3 => Some(3),
            Self::SaturatingTrendBasis4 => Some(4),
            Self::SaturatingTrendBasis5 => Some(5),
            Self::SaturatingTrendBasis6 => Some(6),
            _ => None,
        }
    }

    /// Количество параметров модели.
    pub fn parameter_count(self) -> usize {
        self.parameter_names().len()
    }

    /// Минимальное число точек, необходимое для устойчивой подгонки.
    pub fn min_points(self) -> usize {
        self.metadata().min_points
    }

    /// Возвращает `true`, если семейство определено только при `x > 0`.
    pub fn requires_positive_x(self) -> bool {
        self.metadata().requires_positive_x
    }

    /// Возвращает `true`, если для семейства допустима масштабная нормализация по `x` и `y`.
    ///
    /// Для фиксированного saturating-базиса набор `τ` задан в единицах исходного `x`,
    /// поэтому x-нормализация меняет смысл самих базисных функций.
    pub fn supports_parametric_normalization(self) -> bool {
        !self.is_saturating_trend_basis()
    }

    /// Проверяет набор точек на совместимость с выбранным семейством.
    pub fn validate_points(self, points: &Points) -> Result<(), InputError> {
        let min_required = self.min_points();
        let len = points.len();
        if len < min_required {
            return Err(InputError::TooFewPointsForFamily {
                family: self,
                len,
                min_required,
            });
        }

        if self.requires_positive_x()
            && let Some((index, point)) = points
                .iter()
                .copied()
                .enumerate()
                .find(|(_, point)| point.x() <= 0.0)
        {
            return Err(InputError::NonPositiveXForFamily {
                family: self,
                index,
                value: point.x(),
            });
        }

        Ok(())
    }

    /// Базовые стартовые параметры для оптимизации.
    pub fn default_params(self) -> CurveParams {
        match self {
            Self::Linear => CurveParams::Linear { a: 1.0, b: 0.0 },
            Self::Quadratic => CurveParams::Quadratic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
            },
            Self::Cubic => CurveParams::Cubic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
            },
            Self::Quartic => CurveParams::Quartic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
            },
            Self::Quintic => CurveParams::Quintic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
            },
            Self::Sextic => CurveParams::Sextic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
            },
            Self::Septic => CurveParams::Septic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
                h: 0.0,
            },
            Self::Octic => CurveParams::Octic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
                h: 0.0,
                i: 0.0,
            },
            Self::Nonic => CurveParams::Nonic {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
                h: 0.0,
                i: 0.0,
                j: 0.0,
            },
            Self::Arrhenius => CurveParams::Arrhenius { a: 1.0, b: 1.0 },
            Self::Inverse => CurveParams::Inverse { a: 0.0, b: 1.0 },
            Self::Logistic => CurveParams::Logistic {
                a: 1.0,
                b: 1.0,
                c: 0.0,
            },
            Self::Gompertz => CurveParams::Gompertz {
                a: 1.0,
                b: 1.0,
                c: 0.0,
            },
            Self::BiExponential => CurveParams::BiExponential {
                a1: 1.0,
                k1: 1.0,
                a2: 0.5,
                k2: 0.2,
                c: 0.0,
            },
            Self::DampedSinusoid => CurveParams::DampedSinusoid {
                a: 1.0,
                k: 0.2,
                omega: 2.0,
                phi: 0.0,
                c: 0.0,
            },
            Self::Lorentzian => CurveParams::Lorentzian {
                a: 1.0,
                x0: 0.0,
                gamma: 1.0,
                c: 0.0,
            },
            Self::NaturalLog => CurveParams::NaturalLog { a: 1.0, b: 1.0 },
            Self::FourPl => CurveParams::FourPl {
                a: 0.0,
                b: 1.0,
                c: 1.0,
                d: 1.0,
            },
            Self::FivePl => CurveParams::FivePl {
                a: 0.0,
                b: 1.0,
                c: 1.0,
                d: 1.0,
                m: 1.0,
            },
            Self::MichaelisMenten => CurveParams::MichaelisMenten { vmax: 1.0, km: 1.0 },
            Self::ExponentialBasic => CurveParams::ExponentialBasic {
                a: 0.0,
                b: 1.0,
                c: 0.5,
            },
            Self::ExponentialLinear => CurveParams::ExponentialLinear {
                a: 1.0,
                b: -0.5,
                c: 0.0,
                d: 0.0,
            },
            Self::ExponentialHalfLife => CurveParams::ExponentialHalfLife {
                a: 0.0,
                b: 1.0,
                c: 1.0,
            },
            Self::FallingExponential => CurveParams::FallingExponential {
                y0: 1.0,
                v0: 1.0,
                k: 0.5,
            },
            Self::HyperbolicTangent => CurveParams::HyperbolicTangent {
                a: 1.0,
                b: 1.0,
                c: 0.0,
                d: 0.0,
            },
            Self::ArctangentStep => CurveParams::ArctangentStep {
                a: 1.0,
                b: 1.0,
                c: 0.0,
                d: 0.0,
            },
            Self::Softplus => CurveParams::Softplus {
                a: 1.0,
                b: 1.0,
                c: 0.0,
                d: 0.0,
            },
            Self::Power => CurveParams::Power { a: 1.0, b: 1.0 },
            Self::Gaussian => CurveParams::Gaussian {
                a: 1.0,
                b: 0.0,
                c: 1.0,
            },
            Self::Rational11 => CurveParams::Rational11 {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
            },
            Self::Rational22 => CurveParams::Rational22 {
                a: 0.0,
                b: 1.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
            },
            Self::Rational33 => CurveParams::Rational33 {
                a: 0.0,
                b: 0.0,
                c: 1.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
            },
            Self::Rational44 => CurveParams::Rational44 {
                a: 0.0,
                b: 0.0,
                c: 0.0,
                d: 1.0,
                e: 0.0,
                f: 0.0,
                g: 0.0,
                h: 0.0,
                i: 0.0,
            },
            Self::Rational55 => CurveParams::Rational55 {
                a: 0.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
                g: 0.0,
                h: 0.0,
                i: 0.0,
                j: 0.0,
                k: 0.0,
            },
            Self::Emg => CurveParams::Emg {
                a: 1.0,
                mu: 0.0,
                sigma: 1.0,
                tau: 0.5,
                c: 0.0,
            },
            Self::PseudoVoigt => CurveParams::PseudoVoigt {
                a: 1.0,
                x0: 0.0,
                sigma: 1.0,
                gamma: 1.0,
                eta: 0.0,
                c: 0.0,
            },
            Self::SaturatingTrendBasis1 => CurveParams::SaturatingTrendBasis1 {
                c: 0.0,
                w1: 0.0,
                taus: SaturatingTrendTauGrid::from_values(
                    &DEFAULT_SATURATING_TREND_TAUS_YEARS[..1],
                )
                .expect("default tau grid must be valid"),
            },
            Self::SaturatingTrendBasis2 => CurveParams::SaturatingTrendBasis2 {
                c: 0.0,
                w1: 0.0,
                w2: 0.0,
                taus: SaturatingTrendTauGrid::from_values(
                    &DEFAULT_SATURATING_TREND_TAUS_YEARS[..2],
                )
                .expect("default tau grid must be valid"),
            },
            Self::SaturatingTrendBasis3 => CurveParams::SaturatingTrendBasis3 {
                c: 0.0,
                w1: 0.0,
                w2: 0.0,
                w3: 0.0,
                taus: SaturatingTrendTauGrid::from_values(
                    &DEFAULT_SATURATING_TREND_TAUS_YEARS[..3],
                )
                .expect("default tau grid must be valid"),
            },
            Self::SaturatingTrendBasis4 => CurveParams::SaturatingTrendBasis4 {
                c: 0.0,
                w1: 0.0,
                w2: 0.0,
                w3: 0.0,
                w4: 0.0,
                taus: SaturatingTrendTauGrid::from_values(
                    &DEFAULT_SATURATING_TREND_TAUS_YEARS[..4],
                )
                .expect("default tau grid must be valid"),
            },
            Self::SaturatingTrendBasis5 => CurveParams::SaturatingTrendBasis5 {
                c: 0.0,
                w1: 0.0,
                w2: 0.0,
                w3: 0.0,
                w4: 0.0,
                w5: 0.0,
                taus: SaturatingTrendTauGrid::from_values(
                    &DEFAULT_SATURATING_TREND_TAUS_YEARS[..5],
                )
                .expect("default tau grid must be valid"),
            },
            Self::SaturatingTrendBasis6 => CurveParams::SaturatingTrendBasis6 {
                c: 0.0,
                w1: 0.0,
                w2: 0.0,
                w3: 0.0,
                w4: 0.0,
                w5: 0.0,
                w6: 0.0,
                taus: SaturatingTrendTauGrid::default_for_count(MAX_SATURATING_TREND_TAU_COUNT),
            },
        }
    }
}

impl fmt::Display for CurveFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}
