//! Доменная модель для точек, семейств кривых и параметров оптимизации.
//! Здесь сосредоточены типы с инвариантами, которые проверяются на границах системы.

use std::fmt;

use crate::models;

const MIN_POINTS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Точка наблюдения `(x, y)` с гарантией конечных координат.
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    /// Создает точку и проверяет, что обе координаты конечны.
    pub fn try_new(x: f64, y: f64) -> Result<Self, InputError> {
        if !x.is_finite() {
            return Err(InputError::NonFinitePoint {
                field: "x",
                value: x,
            });
        }
        if !y.is_finite() {
            return Err(InputError::NonFinitePoint {
                field: "y",
                value: y,
            });
        }
        Ok(Self { x, y })
    }

    /// Возвращает абсциссу точки.
    pub fn x(self) -> f64 {
        self.x
    }

    /// Возвращает ординату точки.
    pub fn y(self) -> f64 {
        self.y
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Набор точек с инвариантом минимального размера (`>= 2`).
pub struct Points {
    points: Box<[Point]>,
}

impl Points {
    /// Возвращает неизменяемый срез точек без дополнительных аллокаций.
    pub fn as_slice(&self) -> &[Point] {
        &self.points
    }

    /// Число точек в наборе.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Проверяет, пустой ли набор.
    ///
    /// В текущей модели всегда `false`, но метод полезен для обобщенного кода.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Возвращает `(min_x, max_x)` по всем точкам.
    ///
    /// Предполагается, что инвариант минимального размера уже соблюден.
    pub fn x_bounds(&self) -> (f64, f64) {
        let first = self.points[0].x();
        self.points
            .iter()
            .skip(1)
            .fold((first, first), |(min_x, max_x), point| {
                (min_x.min(point.x()), max_x.max(point.x()))
            })
    }
}

impl TryFrom<Vec<Point>> for Points {
    type Error = InputError;

    fn try_from(points: Vec<Point>) -> Result<Self, Self::Error> {
        if points.len() < MIN_POINTS {
            return Err(InputError::TooFewPoints {
                len: points.len(),
                min_required: MIN_POINTS,
            });
        }
        Ok(Self {
            points: points.into_boxed_slice(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    Emg,
    PseudoVoigt,
}

impl CurveFamily {
    /// Полный список семейств в стабильном порядке для UI и переборов.
    pub const ALL: [Self; 33] = [
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
        Self::Emg,
        Self::PseudoVoigt,
    ];

    /// Короткое человекочитаемое имя семейства.
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Quadratic => "Quadratic",
            Self::Cubic => "Cubic",
            Self::Quartic => "Quartic",
            Self::Quintic => "Quintic",
            Self::Sextic => "Sextic",
            Self::Septic => "Septic",
            Self::Octic => "Octic",
            Self::Nonic => "Nonic",
            Self::Arrhenius => "Arrhenius",
            Self::Inverse => "Inverse",
            Self::Logistic => "Logistic",
            Self::Gompertz => "Gompertz",
            Self::BiExponential => "Bi-Exponential",
            Self::DampedSinusoid => "Damped Sinusoid",
            Self::Lorentzian => "Lorentzian",
            Self::NaturalLog => "Natural Log",
            Self::FourPl => "4PL",
            Self::FivePl => "5PL",
            Self::MichaelisMenten => "Michaelis-Menten",
            Self::ExponentialBasic => "Exponential (Basic)",
            Self::ExponentialLinear => "Exponential + Linear",
            Self::ExponentialHalfLife => "Exponential (Half-life)",
            Self::FallingExponential => "Falling Exponential",
            Self::HyperbolicTangent => "Hyperbolic Tangent",
            Self::ArctangentStep => "Arctangent Step",
            Self::Softplus => "Softplus",
            Self::Power => "Power",
            Self::Gaussian => "Gaussian",
            Self::Rational11 => "Rational (1/1)",
            Self::Rational22 => "Rational (2/2)",
            Self::Emg => "EMG",
            Self::PseudoVoigt => "Pseudo-Voigt",
        }
    }

    /// Имена параметров в порядке внутреннего вектора значений.
    pub fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::Linear => &["a", "b"],
            Self::Quadratic => &["a", "b", "c"],
            Self::Cubic => &["a", "b", "c", "d"],
            Self::Quartic => &["a", "b", "c", "d", "e"],
            Self::Quintic => &["a", "b", "c", "d", "e", "f"],
            Self::Sextic => &["a", "b", "c", "d", "e", "f", "g"],
            Self::Septic => &["a", "b", "c", "d", "e", "f", "g", "h"],
            Self::Octic => &["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            Self::Nonic => &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            Self::Arrhenius => &["A", "B"],
            Self::Inverse => &["A", "B"],
            Self::Logistic => &["A", "B", "C"],
            Self::Gompertz => &["A", "B", "C"],
            Self::BiExponential => &["a1", "k1", "a2", "k2", "c"],
            Self::DampedSinusoid => &["a", "k", "omega", "phi", "c"],
            Self::Lorentzian => &["A", "x0", "gamma", "C"],
            Self::NaturalLog => &["A", "B"],
            Self::FourPl => &["a", "b", "c", "d"],
            Self::FivePl => &["a", "b", "c", "d", "m"],
            Self::MichaelisMenten => &["Vmax", "Km"],
            Self::ExponentialBasic => &["a", "b", "c"],
            Self::ExponentialLinear => &["a", "b", "c", "d"],
            Self::ExponentialHalfLife => &["a", "b", "c"],
            Self::FallingExponential => &["Y0", "V0", "K"],
            Self::HyperbolicTangent => &["a", "b", "c", "d"],
            Self::ArctangentStep => &["a", "b", "c", "d"],
            Self::Softplus => &["a", "b", "c", "d"],
            Self::Power => &["a", "b"],
            Self::Gaussian => &["a", "b", "c"],
            Self::Rational11 => &["a", "b", "c", "d"],
            Self::Rational22 => &["a", "b", "c", "d", "e"],
            Self::Emg => &["a", "mu", "sigma", "tau", "c"],
            Self::PseudoVoigt => &["a", "x0", "sigma", "gamma", "eta", "c"],
        }
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

    /// Количество параметров модели.
    pub fn parameter_count(self) -> usize {
        self.parameter_names().len()
    }

    /// Минимальное число точек, необходимое для устойчивой подгонки.
    pub fn min_points(self) -> usize {
        match self {
            Self::Linear | Self::MichaelisMenten | Self::Power => 2,
            Self::Quadratic
            | Self::ExponentialBasic
            | Self::ExponentialHalfLife
            | Self::FallingExponential
            | Self::Gaussian
            | Self::Logistic
            | Self::Gompertz => 3,
            Self::FourPl
            | Self::Cubic
            | Self::ExponentialLinear
            | Self::HyperbolicTangent
            | Self::ArctangentStep
            | Self::Softplus
            | Self::Rational11 => 4,
            Self::BiExponential
            | Self::DampedSinusoid
            | Self::FivePl
            | Self::Quartic
            | Self::Rational22
            | Self::Emg => 5,
            Self::PseudoVoigt => 6,
            Self::Quintic => 6,
            Self::Sextic => 7,
            Self::Septic => 8,
            Self::Octic => 9,
            Self::Nonic => 10,
            Self::Arrhenius | Self::Inverse | Self::NaturalLog => 2,
            Self::Lorentzian => 4,
        }
    }

    /// Возвращает `true`, если семейство определено только при `x > 0`.
    pub fn requires_positive_x(self) -> bool {
        matches!(
            self,
            Self::FourPl
                | Self::FivePl
                | Self::Power
                | Self::Arrhenius
                | Self::Inverse
                | Self::NaturalLog
        )
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
                .as_slice()
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
        }
    }

    pub(crate) fn evaluate_raw(self, params: &[f64], x: f64) -> f64 {
        models::evaluate_raw(self, params, x)
    }
}

impl fmt::Display for CurveFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Типобезопасное представление параметров всех поддерживаемых семейств.
pub enum CurveParams {
    Linear {
        a: f64,
        b: f64,
    },
    Quadratic {
        a: f64,
        b: f64,
        c: f64,
    },
    Cubic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    Quartic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
    },
    Quintic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
    },
    Sextic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
    },
    Septic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
        h: f64,
    },
    Octic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
        h: f64,
        i: f64,
    },
    Nonic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
        h: f64,
        i: f64,
        j: f64,
    },
    Arrhenius {
        a: f64,
        b: f64,
    },
    Inverse {
        a: f64,
        b: f64,
    },
    Logistic {
        a: f64,
        b: f64,
        c: f64,
    },
    Gompertz {
        a: f64,
        b: f64,
        c: f64,
    },
    BiExponential {
        a1: f64,
        k1: f64,
        a2: f64,
        k2: f64,
        c: f64,
    },
    DampedSinusoid {
        a: f64,
        k: f64,
        omega: f64,
        phi: f64,
        c: f64,
    },
    Lorentzian {
        a: f64,
        x0: f64,
        gamma: f64,
        c: f64,
    },
    NaturalLog {
        a: f64,
        b: f64,
    },
    FourPl {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    FivePl {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        m: f64,
    },
    MichaelisMenten {
        vmax: f64,
        km: f64,
    },
    ExponentialBasic {
        a: f64,
        b: f64,
        c: f64,
    },
    ExponentialLinear {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    ExponentialHalfLife {
        a: f64,
        b: f64,
        c: f64,
    },
    FallingExponential {
        y0: f64,
        v0: f64,
        k: f64,
    },
    HyperbolicTangent {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    ArctangentStep {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    Softplus {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    Power {
        a: f64,
        b: f64,
    },
    Gaussian {
        a: f64,
        b: f64,
        c: f64,
    },
    Rational11 {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    Rational22 {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
    },
    Emg {
        a: f64,
        mu: f64,
        sigma: f64,
        tau: f64,
        c: f64,
    },
    PseudoVoigt {
        a: f64,
        x0: f64,
        sigma: f64,
        gamma: f64,
        eta: f64,
        c: f64,
    },
}

impl CurveParams {
    /// Определяет семейство по варианту параметров.
    pub fn family(&self) -> CurveFamily {
        match self {
            Self::Linear { .. } => CurveFamily::Linear,
            Self::Quadratic { .. } => CurveFamily::Quadratic,
            Self::Cubic { .. } => CurveFamily::Cubic,
            Self::Quartic { .. } => CurveFamily::Quartic,
            Self::Quintic { .. } => CurveFamily::Quintic,
            Self::Sextic { .. } => CurveFamily::Sextic,
            Self::Septic { .. } => CurveFamily::Septic,
            Self::Octic { .. } => CurveFamily::Octic,
            Self::Nonic { .. } => CurveFamily::Nonic,
            Self::Arrhenius { .. } => CurveFamily::Arrhenius,
            Self::Inverse { .. } => CurveFamily::Inverse,
            Self::Logistic { .. } => CurveFamily::Logistic,
            Self::Gompertz { .. } => CurveFamily::Gompertz,
            Self::BiExponential { .. } => CurveFamily::BiExponential,
            Self::DampedSinusoid { .. } => CurveFamily::DampedSinusoid,
            Self::Lorentzian { .. } => CurveFamily::Lorentzian,
            Self::NaturalLog { .. } => CurveFamily::NaturalLog,
            Self::FourPl { .. } => CurveFamily::FourPl,
            Self::FivePl { .. } => CurveFamily::FivePl,
            Self::MichaelisMenten { .. } => CurveFamily::MichaelisMenten,
            Self::ExponentialBasic { .. } => CurveFamily::ExponentialBasic,
            Self::ExponentialLinear { .. } => CurveFamily::ExponentialLinear,
            Self::ExponentialHalfLife { .. } => CurveFamily::ExponentialHalfLife,
            Self::FallingExponential { .. } => CurveFamily::FallingExponential,
            Self::HyperbolicTangent { .. } => CurveFamily::HyperbolicTangent,
            Self::ArctangentStep { .. } => CurveFamily::ArctangentStep,
            Self::Softplus { .. } => CurveFamily::Softplus,
            Self::Power { .. } => CurveFamily::Power,
            Self::Gaussian { .. } => CurveFamily::Gaussian,
            Self::Rational11 { .. } => CurveFamily::Rational11,
            Self::Rational22 { .. } => CurveFamily::Rational22,
            Self::Emg { .. } => CurveFamily::Emg,
            Self::PseudoVoigt { .. } => CurveFamily::PseudoVoigt,
        }
    }

    /// Возвращает параметры в виде вектора в каноническом порядке.
    pub fn values(&self) -> Vec<f64> {
        match self {
            Self::Linear { a, b } => vec![*a, *b],
            Self::Quadratic { a, b, c } => vec![*a, *b, *c],
            Self::Cubic { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::Quartic { a, b, c, d, e } => vec![*a, *b, *c, *d, *e],
            Self::Quintic { a, b, c, d, e, f } => vec![*a, *b, *c, *d, *e, *f],
            Self::Sextic {
                a,
                b,
                c,
                d,
                e,
                f,
                g,
            } => vec![*a, *b, *c, *d, *e, *f, *g],
            Self::Septic {
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
            } => vec![*a, *b, *c, *d, *e, *f, *g, *h],
            Self::Octic {
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
                i,
            } => vec![*a, *b, *c, *d, *e, *f, *g, *h, *i],
            Self::Nonic {
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
                i,
                j,
            } => vec![*a, *b, *c, *d, *e, *f, *g, *h, *i, *j],
            Self::Arrhenius { a, b } => vec![*a, *b],
            Self::Inverse { a, b } => vec![*a, *b],
            Self::Logistic { a, b, c } => vec![*a, *b, *c],
            Self::Gompertz { a, b, c } => vec![*a, *b, *c],
            Self::BiExponential { a1, k1, a2, k2, c } => vec![*a1, *k1, *a2, *k2, *c],
            Self::DampedSinusoid {
                a,
                k,
                omega,
                phi,
                c,
            } => vec![*a, *k, *omega, *phi, *c],
            Self::Lorentzian { a, x0, gamma, c } => vec![*a, *x0, *gamma, *c],
            Self::NaturalLog { a, b } => vec![*a, *b],
            Self::FourPl { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::FivePl { a, b, c, d, m } => vec![*a, *b, *c, *d, *m],
            Self::MichaelisMenten { vmax, km } => vec![*vmax, *km],
            Self::ExponentialBasic { a, b, c } => vec![*a, *b, *c],
            Self::ExponentialLinear { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::ExponentialHalfLife { a, b, c } => vec![*a, *b, *c],
            Self::FallingExponential { y0, v0, k } => vec![*y0, *v0, *k],
            Self::HyperbolicTangent { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::ArctangentStep { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::Softplus { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::Power { a, b } => vec![*a, *b],
            Self::Gaussian { a, b, c } => vec![*a, *b, *c],
            Self::Rational11 { a, b, c, d } => vec![*a, *b, *c, *d],
            Self::Rational22 { a, b, c, d, e } => vec![*a, *b, *c, *d, *e],
            Self::Emg {
                a,
                mu,
                sigma,
                tau,
                c,
            } => vec![*a, *mu, *sigma, *tau, *c],
            Self::PseudoVoigt {
                a,
                x0,
                sigma,
                gamma,
                eta,
                c,
            } => vec![*a, *x0, *sigma, *gamma, *eta, *c],
        }
    }

    /// Вычисляет значение модели для заданного `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        models::evaluate_curve_params(self, x)
    }

    /// Конструирует параметры семейства из среза значений.
    ///
    /// Проверяет длину и конечность значений на границе ввода.
    pub fn try_from_slice(family: CurveFamily, values: &[f64]) -> Result<Self, InputError> {
        let expected = family.parameter_count();
        if values.len() != expected {
            return Err(InputError::WrongParameterCount {
                family,
                expected,
                got: values.len(),
            });
        }

        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(InputError::NonFiniteParameter {
                family,
                index,
                value,
            });
        }

        Ok(match family {
            CurveFamily::Linear => Self::Linear {
                a: values[0],
                b: values[1],
            },
            CurveFamily::Quadratic => Self::Quadratic {
                a: values[0],
                b: values[1],
                c: values[2],
            },
            CurveFamily::Cubic => Self::Cubic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::Quartic => Self::Quartic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
            },
            CurveFamily::Quintic => Self::Quintic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
                f: values[5],
            },
            CurveFamily::Sextic => Self::Sextic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
                f: values[5],
                g: values[6],
            },
            CurveFamily::Septic => Self::Septic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
                f: values[5],
                g: values[6],
                h: values[7],
            },
            CurveFamily::Octic => Self::Octic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
                f: values[5],
                g: values[6],
                h: values[7],
                i: values[8],
            },
            CurveFamily::Nonic => Self::Nonic {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
                f: values[5],
                g: values[6],
                h: values[7],
                i: values[8],
                j: values[9],
            },
            CurveFamily::Arrhenius => Self::Arrhenius {
                a: values[0],
                b: values[1],
            },
            CurveFamily::Inverse => Self::Inverse {
                a: values[0],
                b: values[1],
            },
            CurveFamily::Logistic => Self::Logistic {
                a: values[0],
                b: values[1],
                c: values[2],
            },
            CurveFamily::Gompertz => Self::Gompertz {
                a: values[0],
                b: values[1],
                c: values[2],
            },
            CurveFamily::BiExponential => Self::BiExponential {
                a1: values[0],
                k1: values[1],
                a2: values[2],
                k2: values[3],
                c: values[4],
            },
            CurveFamily::DampedSinusoid => Self::DampedSinusoid {
                a: values[0],
                k: values[1],
                omega: values[2],
                phi: values[3],
                c: values[4],
            },
            CurveFamily::Lorentzian => Self::Lorentzian {
                a: values[0],
                x0: values[1],
                gamma: values[2],
                c: values[3],
            },
            CurveFamily::NaturalLog => Self::NaturalLog {
                a: values[0],
                b: values[1],
            },
            CurveFamily::FourPl => Self::FourPl {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::FivePl => Self::FivePl {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                m: values[4],
            },
            CurveFamily::MichaelisMenten => Self::MichaelisMenten {
                vmax: values[0],
                km: values[1],
            },
            CurveFamily::ExponentialBasic => Self::ExponentialBasic {
                a: values[0],
                b: values[1],
                c: values[2],
            },
            CurveFamily::ExponentialLinear => Self::ExponentialLinear {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::ExponentialHalfLife => Self::ExponentialHalfLife {
                a: values[0],
                b: values[1],
                c: values[2],
            },
            CurveFamily::FallingExponential => Self::FallingExponential {
                y0: values[0],
                v0: values[1],
                k: values[2],
            },
            CurveFamily::HyperbolicTangent => Self::HyperbolicTangent {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::ArctangentStep => Self::ArctangentStep {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::Softplus => Self::Softplus {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::Power => Self::Power {
                a: values[0],
                b: values[1],
            },
            CurveFamily::Gaussian => Self::Gaussian {
                a: values[0],
                b: values[1],
                c: values[2],
            },
            CurveFamily::Rational11 => Self::Rational11 {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            },
            CurveFamily::Rational22 => Self::Rational22 {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
                e: values[4],
            },
            CurveFamily::Emg => Self::Emg {
                a: values[0],
                mu: values[1],
                sigma: values[2],
                tau: values[3],
                c: values[4],
            },
            CurveFamily::PseudoVoigt => Self::PseudoVoigt {
                a: values[0],
                x0: values[1],
                sigma: values[2],
                gamma: values[3],
                eta: values[4],
                c: values[5],
            },
        })
    }

    /// Удобная обертка над [`Self::try_from_slice`] для владения `Vec<f64>`.
    pub fn try_from_values(family: CurveFamily, values: Vec<f64>) -> Result<Self, InputError> {
        Self::try_from_slice(family, &values)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры L-BFGS и line-search с проверяемыми инвариантами.
pub struct LbfgsConfig {
    pub history_size: usize,
    pub max_iters: u64,
    pub tol_grad: f64,
    pub tol_cost: f64,
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    pub width_tolerance: f64,
}

impl LbfgsConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        history_size: usize,
        max_iters: u64,
        tol_grad: f64,
        tol_cost: f64,
        c1: f64,
        c2: f64,
        step_min: f64,
        step_max: f64,
        width_tolerance: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            history_size,
            max_iters,
            tol_grad,
            tol_cost,
            c1,
            c2,
            step_min,
            step_max,
            width_tolerance,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.history_size == 0 {
            return Err(InputError::InvalidLbfgsConfig(
                "history_size must be greater than 0",
            ));
        }
        if self.max_iters == 0 {
            return Err(InputError::InvalidLbfgsConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.tol_grad.is_finite() || self.tol_grad < 0.0 {
            return Err(InputError::InvalidLbfgsConfig(
                "tol_grad must be finite and >= 0",
            ));
        }
        if !self.tol_cost.is_finite() || self.tol_cost < 0.0 {
            return Err(InputError::InvalidLbfgsConfig(
                "tol_cost must be finite and >= 0",
            ));
        }
        if !self.c1.is_finite()
            || !self.c2.is_finite()
            || self.c1 <= 0.0
            || self.c1 >= self.c2
            || self.c2 >= 1.0
        {
            return Err(InputError::InvalidLbfgsConfig(
                "c1 and c2 must satisfy 0 < c1 < c2 < 1",
            ));
        }
        if !self.step_min.is_finite()
            || !self.step_max.is_finite()
            || self.step_min < 0.0
            || self.step_max <= self.step_min
        {
            return Err(InputError::InvalidLbfgsConfig(
                "step bounds must satisfy 0 <= step_min < step_max",
            ));
        }
        if !self.width_tolerance.is_finite() || self.width_tolerance < 0.0 {
            return Err(InputError::InvalidLbfgsConfig(
                "width_tolerance must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            history_size: 7,
            max_iters: 200,
            tol_grad: 1e-8,
            tol_cost: 1e-12,
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-12,
            step_max: 10.0,
            width_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры метода Nelder-Mead с проверяемыми инвариантами.
pub struct NelderMeadConfig {
    pub max_iters: u64,
    pub simplex_scale: f64,
    pub sd_tolerance: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub rho: f64,
    pub sigma: f64,
}

impl NelderMeadConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        max_iters: u64,
        simplex_scale: f64,
        sd_tolerance: f64,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            simplex_scale,
            sd_tolerance,
            alpha,
            gamma,
            rho,
            sigma,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.simplex_scale.is_finite() || self.simplex_scale <= 0.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "simplex_scale must be finite and > 0",
            ));
        }
        if !self.sd_tolerance.is_finite() || self.sd_tolerance < 0.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "sd_tolerance must be finite and >= 0",
            ));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "alpha must be finite and > 0",
            ));
        }
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "gamma must be finite and > 1",
            ));
        }
        if !self.rho.is_finite() || self.rho <= 0.0 || self.rho > 0.5 {
            return Err(InputError::InvalidNelderMeadConfig(
                "rho must be finite and in (0, 0.5]",
            ));
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 || self.sigma > 1.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "sigma must be finite and in (0, 1]",
            ));
        }
        Ok(())
    }
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iters: 400,
            simplex_scale: 0.05,
            sd_tolerance: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры steepest descent с line-search и проверяемыми инвариантами.
pub struct SteepestDescentConfig {
    pub max_iters: u64,
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    pub width_tolerance: f64,
}

impl SteepestDescentConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        max_iters: u64,
        c1: f64,
        c2: f64,
        step_min: f64,
        step_max: f64,
        width_tolerance: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            c1,
            c2,
            step_min,
            step_max,
            width_tolerance,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidSteepestDescentConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.c1.is_finite()
            || !self.c2.is_finite()
            || self.c1 <= 0.0
            || self.c1 >= self.c2
            || self.c2 >= 1.0
        {
            return Err(InputError::InvalidSteepestDescentConfig(
                "c1 and c2 must satisfy 0 < c1 < c2 < 1",
            ));
        }
        if !self.step_min.is_finite()
            || !self.step_max.is_finite()
            || self.step_min < 0.0
            || self.step_max <= self.step_min
        {
            return Err(InputError::InvalidSteepestDescentConfig(
                "step bounds must satisfy 0 <= step_min < step_max",
            ));
        }
        if !self.width_tolerance.is_finite() || self.width_tolerance < 0.0 {
            return Err(InputError::InvalidSteepestDescentConfig(
                "width_tolerance must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl Default for SteepestDescentConfig {
    fn default() -> Self {
        Self {
            max_iters: 300,
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-12,
            step_max: 10.0,
            width_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры Newton-CG с line-search и проверяемыми инвариантами.
pub struct NewtonCgConfig {
    pub max_iters: u64,
    pub tol: f64,
    pub curvature_threshold: f64,
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    pub width_tolerance: f64,
}

impl NewtonCgConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        max_iters: u64,
        tol: f64,
        curvature_threshold: f64,
        c1: f64,
        c2: f64,
        step_min: f64,
        step_max: f64,
        width_tolerance: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            tol,
            curvature_threshold,
            c1,
            c2,
            step_min,
            step_max,
            width_tolerance,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "tol must be finite and > 0",
            ));
        }
        if !self.curvature_threshold.is_finite() || self.curvature_threshold < 0.0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "curvature_threshold must be finite and >= 0",
            ));
        }
        if !self.c1.is_finite()
            || !self.c2.is_finite()
            || self.c1 <= 0.0
            || self.c1 >= self.c2
            || self.c2 >= 1.0
        {
            return Err(InputError::InvalidNewtonCgConfig(
                "c1 and c2 must satisfy 0 < c1 < c2 < 1",
            ));
        }
        if !self.step_min.is_finite()
            || !self.step_max.is_finite()
            || self.step_min < 0.0
            || self.step_max <= self.step_min
        {
            return Err(InputError::InvalidNewtonCgConfig(
                "step bounds must satisfy 0 <= step_min < step_max",
            ));
        }
        if !self.width_tolerance.is_finite() || self.width_tolerance < 0.0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "width_tolerance must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl Default for NewtonCgConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol: 1e-10,
            curvature_threshold: 0.0,
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-12,
            step_max: 10.0,
            width_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры стохастического градиентного спуска (SGD).
pub struct SgdConfig {
    pub max_iters: u64,
    pub learning_rate: f64,
}

impl SgdConfig {
    /// Создает конфигурацию и валидирует ограничения аргументов.
    pub fn try_new(max_iters: u64, learning_rate: f64) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            learning_rate,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidSgdConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(InputError::InvalidSgdConfig(
                "learning_rate must be finite and > 0",
            ));
        }
        Ok(())
    }
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            max_iters: 1_000,
            learning_rate: 1e-2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры оптимизатора Adam.
pub struct AdamConfig {
    pub max_iters: u64,
    pub learning_rate: f64,
}

impl AdamConfig {
    /// Создает конфигурацию и валидирует ограничения аргументов.
    pub fn try_new(max_iters: u64, learning_rate: f64) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            learning_rate,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidAdamConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(InputError::InvalidAdamConfig(
                "learning_rate must be finite and > 0",
            ));
        }
        Ok(())
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            max_iters: 800,
            learning_rate: 5e-3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Метод оптимизации для подгонки параметрических моделей и сплайнов.
pub enum OptimizerMethod {
    #[default]
    Lbfgs,
    NelderMead,
    SteepestDescent,
    NewtonCg,
    Sgd,
    Adam,
}

impl OptimizerMethod {
    /// Полный список методов для UI и переборов.
    pub const ALL: [Self; 6] = [
        Self::Lbfgs,
        Self::NelderMead,
        Self::SteepestDescent,
        Self::NewtonCg,
        Self::Sgd,
        Self::Adam,
    ];
}

#[derive(Debug, Clone, PartialEq)]
/// Объединенная конфигурация оптимизатора.
pub enum OptimizerConfig {
    Lbfgs(LbfgsConfig),
    NelderMead(NelderMeadConfig),
    SteepestDescent(SteepestDescentConfig),
    NewtonCg(NewtonCgConfig),
    Sgd(SgdConfig),
    Adam(AdamConfig),
}

impl OptimizerConfig {
    /// Возвращает выбранный метод оптимизации.
    pub fn method(&self) -> OptimizerMethod {
        match self {
            Self::Lbfgs(_) => OptimizerMethod::Lbfgs,
            Self::NelderMead(_) => OptimizerMethod::NelderMead,
            Self::SteepestDescent(_) => OptimizerMethod::SteepestDescent,
            Self::NewtonCg(_) => OptimizerMethod::NewtonCg,
            Self::Sgd(_) => OptimizerMethod::Sgd,
            Self::Adam(_) => OptimizerMethod::Adam,
        }
    }

    /// Возвращает ограничение на число итераций для выбранного метода.
    pub fn max_iters(&self) -> u64 {
        match self {
            Self::Lbfgs(config) => config.max_iters,
            Self::NelderMead(config) => config.max_iters,
            Self::SteepestDescent(config) => config.max_iters,
            Self::NewtonCg(config) => config.max_iters,
            Self::Sgd(config) => config.max_iters,
            Self::Adam(config) => config.max_iters,
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::Lbfgs(LbfgsConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Результат подгонки параметрической модели.
pub struct FitResult {
    pub family: CurveFamily,
    pub params: CurveParams,
    pub mse: f64,
    pub rmse: f64,
    pub iterations: u64,
}

#[derive(Debug, Clone, PartialEq)]
/// Ошибки входных данных и конфигурации на границах доменной модели.
pub enum InputError {
    NonFinitePoint {
        field: &'static str,
        value: f64,
    },
    TooFewPoints {
        len: usize,
        min_required: usize,
    },
    TooFewPointsForFamily {
        family: CurveFamily,
        len: usize,
        min_required: usize,
    },
    NonPositiveXForFamily {
        family: CurveFamily,
        index: usize,
        value: f64,
    },
    WrongParameterCount {
        family: CurveFamily,
        expected: usize,
        got: usize,
    },
    NonFiniteParameter {
        family: CurveFamily,
        index: usize,
        value: f64,
    },
    FamilyMismatch {
        expected: CurveFamily,
        got: CurveFamily,
    },
    InvalidLbfgsConfig(&'static str),
    InvalidNelderMeadConfig(&'static str),
    InvalidSteepestDescentConfig(&'static str),
    InvalidNewtonCgConfig(&'static str),
    InvalidSgdConfig(&'static str),
    InvalidAdamConfig(&'static str),
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinitePoint { field, value } => {
                write!(f, "Point field '{field}' must be finite, got {value}")
            }
            Self::TooFewPoints { len, min_required } => {
                write!(f, "Need at least {min_required} points, got {len}")
            }
            Self::TooFewPointsForFamily {
                family,
                len,
                min_required,
            } => write!(
                f,
                "Family {family} requires at least {min_required} points, got {len}"
            ),
            Self::NonPositiveXForFamily {
                family,
                index,
                value,
            } => write!(
                f,
                "Family {family} requires x > 0, but point #{index} has x={value}"
            ),
            Self::WrongParameterCount {
                family,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Family {family} expects {expected} parameters, got {got}"
                )
            }
            Self::NonFiniteParameter {
                family,
                index,
                value,
            } => {
                write!(
                    f,
                    "Parameter {index} for family {family} must be finite, got {value}"
                )
            }
            Self::FamilyMismatch { expected, got } => {
                write!(
                    f,
                    "Initial parameters belong to {got}, but selected family is {expected}"
                )
            }
            Self::InvalidLbfgsConfig(message) => f.write_str(message),
            Self::InvalidNelderMeadConfig(message) => f.write_str(message),
            Self::InvalidSteepestDescentConfig(message) => f.write_str(message),
            Self::InvalidNewtonCgConfig(message) => f.write_str(message),
            Self::InvalidSgdConfig(message) => f.write_str(message),
            Self::InvalidAdamConfig(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for InputError {}

#[cfg(test)]
mod tests {
    use super::{
        AdamConfig, CurveFamily, CurveParams, InputError, LbfgsConfig, NelderMeadConfig,
        NewtonCgConfig, Point, Points, SgdConfig, SteepestDescentConfig,
    };

    #[test]
    fn point_rejects_non_finite_values() {
        let error = Point::try_new(f64::NAN, 1.0).expect_err("NaN x must be rejected");
        assert!(matches!(
            error,
            InputError::NonFinitePoint {
                field: "x",
                value
            } if value.is_nan()
        ));

        let error = Point::try_new(1.0, f64::INFINITY).expect_err("Inf y must be rejected");
        assert_eq!(
            error,
            InputError::NonFinitePoint {
                field: "y",
                value: f64::INFINITY,
            }
        );
    }

    #[test]
    fn points_require_at_least_two_values() {
        let points = vec![Point::try_new(0.0, 0.0).unwrap()];
        let error = Points::try_from(points).expect_err("must reject short vectors");

        assert_eq!(
            error,
            InputError::TooFewPoints {
                len: 1,
                min_required: 2,
            }
        );
    }

    #[test]
    fn family_validation_checks_min_points_and_domain() {
        let points = Points::try_from(vec![
            Point::try_new(-1.0, 1.0).unwrap(),
            Point::try_new(1.0, 2.0).unwrap(),
        ])
        .unwrap();

        let error = CurveFamily::Power
            .validate_points(&points)
            .expect_err("power family requires x > 0");
        assert!(matches!(
            error,
            InputError::NonPositiveXForFamily {
                family: CurveFamily::Power,
                index: 0,
                value: -1.0
            }
        ));

        let error = CurveFamily::NaturalLog
            .validate_points(&points)
            .expect_err("natural log requires x > 0");
        assert!(matches!(
            error,
            InputError::NonPositiveXForFamily {
                family: CurveFamily::NaturalLog,
                index: 0,
                value: -1.0
            }
        ));

        let short_points = Points::try_from(vec![
            Point::try_new(1.0, 1.0).unwrap(),
            Point::try_new(2.0, 2.0).unwrap(),
        ])
        .unwrap();
        let error = CurveFamily::Quadratic
            .validate_points(&short_points)
            .expect_err("quadratic requires at least 3 points");
        assert!(matches!(
            error,
            InputError::TooFewPointsForFamily {
                family: CurveFamily::Quadratic,
                len: 2,
                min_required: 3
            }
        ));

        let error = CurveFamily::PseudoVoigt
            .validate_points(&short_points)
            .expect_err("pseudo-voigt requires at least 6 points");
        assert!(matches!(
            error,
            InputError::TooFewPointsForFamily {
                family: CurveFamily::PseudoVoigt,
                len: 2,
                min_required: 6
            }
        ));
    }

    #[test]
    fn lbfgs_config_validates_constraints() {
        let result = LbfgsConfig::try_new(0, 100, 1e-6, 1e-8, 1e-4, 0.9, 1e-6, 10.0, 1e-10);
        assert!(result.is_err());

        let result = LbfgsConfig::try_new(5, 100, 1e-6, 1e-8, 0.95, 0.9, 1e-6, 10.0, 1e-10);
        assert!(result.is_err());

        let result = LbfgsConfig::try_new(5, 100, 1e-6, 1e-8, 1e-4, 0.9, 10.0, 1.0, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn nelder_mead_config_validates_constraints() {
        let result = NelderMeadConfig::try_new(0, 0.1, 1e-8, 1.0, 2.0, 0.5, 0.5);
        assert!(result.is_err());

        let result = NelderMeadConfig::try_new(200, 0.0, 1e-8, 1.0, 2.0, 0.5, 0.5);
        assert!(result.is_err());

        let result = NelderMeadConfig::try_new(200, 0.1, 1e-8, 1.0, 1.0, 0.5, 0.5);
        assert!(result.is_err());

        let result = NelderMeadConfig::try_new(200, 0.1, 1e-8, 1.0, 2.0, 0.0, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn steepest_descent_config_validates_constraints() {
        let result = SteepestDescentConfig::try_new(0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
        assert!(result.is_err());

        let result = SteepestDescentConfig::try_new(100, 0.9, 0.9, 1e-12, 10.0, 1e-10);
        assert!(result.is_err());

        let result = SteepestDescentConfig::try_new(100, 1e-4, 0.9, 10.0, 1.0, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn newton_cg_config_validates_constraints() {
        let result = NewtonCgConfig::try_new(0, 1e-8, 0.0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
        assert!(result.is_err());

        let result = NewtonCgConfig::try_new(100, 0.0, 0.0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
        assert!(result.is_err());

        let result = NewtonCgConfig::try_new(100, 1e-8, -1.0, 1e-4, 0.9, 1e-12, 10.0, 1e-10);
        assert!(result.is_err());

        let result = NewtonCgConfig::try_new(100, 1e-8, 0.0, 0.9, 0.9, 1e-12, 10.0, 1e-10);
        assert!(result.is_err());

        let result = NewtonCgConfig::try_new(100, 1e-8, 0.0, 1e-4, 0.9, 10.0, 1.0, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn sgd_config_validates_constraints() {
        let result = SgdConfig::try_new(0, 1e-2);
        assert!(result.is_err());

        let result = SgdConfig::try_new(100, 0.0);
        assert!(result.is_err());

        let result = SgdConfig::try_new(100, f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn adam_config_validates_constraints() {
        let result = AdamConfig::try_new(0, 1e-3);
        assert!(result.is_err());

        let result = AdamConfig::try_new(100, -1e-3);
        assert!(result.is_err());

        let result = AdamConfig::try_new(100, f64::INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn curve_params_reject_non_finite_values() {
        let values = vec![1.0, f64::NEG_INFINITY];
        let error = CurveParams::try_from_values(CurveFamily::Linear, values)
            .expect_err("non-finite parameters must be rejected");

        assert_eq!(
            error,
            InputError::NonFiniteParameter {
                family: CurveFamily::Linear,
                index: 1,
                value: f64::NEG_INFINITY,
            }
        );
    }
}
