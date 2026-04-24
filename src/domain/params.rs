//! Типизированное представление параметров всех поддерживаемых семейств кривых.

use crate::models;

use super::{CurveFamily, InputError, SaturatingTrendTauGrid};
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
    Rational33 {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
    },
    Rational44 {
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
    Rational55 {
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
        k: f64,
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
    SaturatingTrendBasis1 {
        c: f64,
        w1: f64,
        taus: SaturatingTrendTauGrid,
    },
    SaturatingTrendBasis2 {
        c: f64,
        w1: f64,
        w2: f64,
        taus: SaturatingTrendTauGrid,
    },
    SaturatingTrendBasis3 {
        c: f64,
        w1: f64,
        w2: f64,
        w3: f64,
        taus: SaturatingTrendTauGrid,
    },
    SaturatingTrendBasis4 {
        c: f64,
        w1: f64,
        w2: f64,
        w3: f64,
        w4: f64,
        taus: SaturatingTrendTauGrid,
    },
    SaturatingTrendBasis5 {
        c: f64,
        w1: f64,
        w2: f64,
        w3: f64,
        w4: f64,
        w5: f64,
        taus: SaturatingTrendTauGrid,
    },
    SaturatingTrendBasis6 {
        c: f64,
        w1: f64,
        w2: f64,
        w3: f64,
        w4: f64,
        w5: f64,
        w6: f64,
        taus: SaturatingTrendTauGrid,
    },
}

// Таблица вариантов `CurveParams` как единый источник правды для family/values/try_from_slice.
macro_rules! curve_params_variants {
    ($macro:ident, $($args:tt)*) => {
        $macro! {
            $($args)*
            (Linear, Linear, [a, b]),
            (Quadratic, Quadratic, [a, b, c]),
            (Cubic, Cubic, [a, b, c, d]),
            (Quartic, Quartic, [a, b, c, d, e]),
            (Quintic, Quintic, [a, b, c, d, e, f]),
            (Sextic, Sextic, [a, b, c, d, e, f, g]),
            (Septic, Septic, [a, b, c, d, e, f, g, h]),
            (Octic, Octic, [a, b, c, d, e, f, g, h, i]),
            (Nonic, Nonic, [a, b, c, d, e, f, g, h, i, j]),
            (Arrhenius, Arrhenius, [a, b]),
            (Inverse, Inverse, [a, b]),
            (Logistic, Logistic, [a, b, c]),
            (Gompertz, Gompertz, [a, b, c]),
            (BiExponential, BiExponential, [a1, k1, a2, k2, c]),
            (DampedSinusoid, DampedSinusoid, [a, k, omega, phi, c]),
            (Lorentzian, Lorentzian, [a, x0, gamma, c]),
            (NaturalLog, NaturalLog, [a, b]),
            (FourPl, FourPl, [a, b, c, d]),
            (FivePl, FivePl, [a, b, c, d, m]),
            (MichaelisMenten, MichaelisMenten, [vmax, km]),
            (ExponentialBasic, ExponentialBasic, [a, b, c]),
            (ExponentialLinear, ExponentialLinear, [a, b, c, d]),
            (ExponentialHalfLife, ExponentialHalfLife, [a, b, c]),
            (FallingExponential, FallingExponential, [y0, v0, k]),
            (HyperbolicTangent, HyperbolicTangent, [a, b, c, d]),
            (ArctangentStep, ArctangentStep, [a, b, c, d]),
            (Softplus, Softplus, [a, b, c, d]),
            (Power, Power, [a, b]),
            (Gaussian, Gaussian, [a, b, c]),
            (Rational11, Rational11, [a, b, c, d]),
            (Rational22, Rational22, [a, b, c, d, e]),
            (Rational33, Rational33, [a, b, c, d, e, f, g]),
            (Rational44, Rational44, [a, b, c, d, e, f, g, h, i]),
            (Rational55, Rational55, [a, b, c, d, e, f, g, h, i, j, k]),
            (Emg, Emg, [a, mu, sigma, tau, c]),
            (PseudoVoigt, PseudoVoigt, [a, x0, sigma, gamma, eta, c]),
        }
    };
}

macro_rules! curve_params_family_match {
    ($value:expr, $(($variant:ident, $family:ident, [$($field:ident),+])),+ $(,)?) => {
        match $value {
            $(Self::$variant { .. } => CurveFamily::$family,)+
            _ => unreachable!("saturating trend variants are handled before the macro match"),
        }
    };
}

macro_rules! curve_params_with_values_match {
    ($value:expr, $callback:expr, $(($variant:ident, $family:ident, [$($field:ident),+])),+ $(,)?) => {
        match $value {
            $(
                Self::$variant { $($field),+ } => {
                    let values = [$(*$field),+];
                    $callback(CurveFamily::$family, &values)
                }
            )+
            _ => unreachable!("saturating trend variants are handled before the macro match"),
        }
    };
}

macro_rules! curve_params_from_slice_match {
    ($family_value:expr, $values:expr, $(($variant:ident, $family:ident, [$($field:ident),+])),+ $(,)?) => {
        match $family_value {
            $(
                CurveFamily::$family => {
                    let mut iter = $values.iter().copied();
                    Self::$variant {
                        $(
                            $field: iter
                                .next()
                                .expect("parameter count checked before construction"),
                        )+
                    }
                }
            )+
            _ => unreachable!("saturating trend families are handled before the macro match"),
        }
    };
}

impl CurveParams {
    fn resolve_saturating_trend_tau_grid(
        family: CurveFamily,
        tau_grid: Option<&SaturatingTrendTauGrid>,
    ) -> SaturatingTrendTauGrid {
        let count = family
            .saturating_trend_tau_count()
            .expect("helper is only called for saturating-trend families");
        tau_grid
            .cloned()
            .unwrap_or_else(|| SaturatingTrendTauGrid::default_for_count(count))
    }

    /// Определяет семейство по варианту параметров.
    pub fn family(&self) -> CurveFamily {
        match self {
            Self::SaturatingTrendBasis1 { .. } => CurveFamily::SaturatingTrendBasis1,
            Self::SaturatingTrendBasis2 { .. } => CurveFamily::SaturatingTrendBasis2,
            Self::SaturatingTrendBasis3 { .. } => CurveFamily::SaturatingTrendBasis3,
            Self::SaturatingTrendBasis4 { .. } => CurveFamily::SaturatingTrendBasis4,
            Self::SaturatingTrendBasis5 { .. } => CurveFamily::SaturatingTrendBasis5,
            Self::SaturatingTrendBasis6 { .. } => CurveFamily::SaturatingTrendBasis6,
            _ => curve_params_variants!(curve_params_family_match, self,),
        }
    }

    fn with_family_values<R>(&self, callback: impl FnOnce(CurveFamily, &[f64]) -> R) -> R {
        match self {
            Self::SaturatingTrendBasis1 { c, w1, .. } => {
                let values = [*c, *w1];
                callback(CurveFamily::SaturatingTrendBasis1, &values)
            }
            Self::SaturatingTrendBasis2 { c, w1, w2, .. } => {
                let values = [*c, *w1, *w2];
                callback(CurveFamily::SaturatingTrendBasis2, &values)
            }
            Self::SaturatingTrendBasis3 { c, w1, w2, w3, .. } => {
                let values = [*c, *w1, *w2, *w3];
                callback(CurveFamily::SaturatingTrendBasis3, &values)
            }
            Self::SaturatingTrendBasis4 {
                c, w1, w2, w3, w4, ..
            } => {
                let values = [*c, *w1, *w2, *w3, *w4];
                callback(CurveFamily::SaturatingTrendBasis4, &values)
            }
            Self::SaturatingTrendBasis5 {
                c,
                w1,
                w2,
                w3,
                w4,
                w5,
                ..
            } => {
                let values = [*c, *w1, *w2, *w3, *w4, *w5];
                callback(CurveFamily::SaturatingTrendBasis5, &values)
            }
            Self::SaturatingTrendBasis6 {
                c,
                w1,
                w2,
                w3,
                w4,
                w5,
                w6,
                ..
            } => {
                let values = [*c, *w1, *w2, *w3, *w4, *w5, *w6];
                callback(CurveFamily::SaturatingTrendBasis6, &values)
            }
            _ => curve_params_variants!(curve_params_with_values_match, self, callback,),
        }
    }

    /// Передает канонический срез значений параметров в callback без промежуточной аллокации.
    pub fn with_values<R>(&self, callback: impl FnOnce(&[f64]) -> R) -> R {
        self.with_family_values(|_, values| callback(values))
    }

    /// Передает имена и значения параметров в каноническом порядке без промежуточной аллокации.
    pub fn with_names_and_values<R>(
        &self,
        callback: impl FnOnce(&[&'static str], &[f64]) -> R,
    ) -> R {
        self.with_family_values(|family, values| callback(family.parameter_names(), values))
    }

    /// Возвращает параметры в виде вектора в каноническом порядке.
    pub fn values(&self) -> Vec<f64> {
        self.with_values(|values| values.to_vec())
    }

    /// Вычисляет значение модели для заданного `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        self.with_family_values(|family, values| {
            models::value_at_with_saturating_taus(family, values, x, self.saturating_trend_taus())
        })
    }

    /// Возвращает сетку `τ`, если параметры принадлежат saturating-базису.
    pub fn saturating_trend_tau_grid(&self) -> Option<&SaturatingTrendTauGrid> {
        match self {
            Self::SaturatingTrendBasis1 { taus, .. }
            | Self::SaturatingTrendBasis2 { taus, .. }
            | Self::SaturatingTrendBasis3 { taus, .. }
            | Self::SaturatingTrendBasis4 { taus, .. }
            | Self::SaturatingTrendBasis5 { taus, .. }
            | Self::SaturatingTrendBasis6 { taus, .. } => Some(taus),
            _ => None,
        }
    }

    /// Возвращает активную сетку `τ` как срез.
    pub fn saturating_trend_taus(&self) -> Option<&[f64]> {
        self.saturating_trend_tau_grid()
            .map(SaturatingTrendTauGrid::as_slice)
    }

    /// Конструирует параметры семейства из среза значений.
    ///
    /// Проверяет длину и конечность значений на границе ввода.
    pub fn try_from_slice(family: CurveFamily, values: &[f64]) -> Result<Self, InputError> {
        Self::try_from_slice_with_tau_grid(family, values, None)
    }

    /// Конструирует параметры с явной сеткой `τ` для saturating-базиса.
    pub fn try_from_slice_with_tau_grid(
        family: CurveFamily,
        values: &[f64],
        tau_grid: Option<&SaturatingTrendTauGrid>,
    ) -> Result<Self, InputError> {
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

        if let Some(expected_tau_count) = family.saturating_trend_tau_count()
            && let Some(grid) = tau_grid
            && grid.count() != expected_tau_count
        {
            return Err(InputError::WrongSaturatingTrendTauCount {
                expected_min: expected_tau_count,
                expected_max: expected_tau_count,
                got: grid.count(),
            });
        }

        let params = match family {
            CurveFamily::SaturatingTrendBasis1 => Self::SaturatingTrendBasis1 {
                c: values[0],
                w1: values[1],
                taus: Self::resolve_saturating_trend_tau_grid(family, tau_grid),
            },
            CurveFamily::SaturatingTrendBasis2 => Self::SaturatingTrendBasis2 {
                c: values[0],
                w1: values[1],
                w2: values[2],
                taus: Self::resolve_saturating_trend_tau_grid(family, tau_grid),
            },
            CurveFamily::SaturatingTrendBasis3 => Self::SaturatingTrendBasis3 {
                c: values[0],
                w1: values[1],
                w2: values[2],
                w3: values[3],
                taus: Self::resolve_saturating_trend_tau_grid(family, tau_grid),
            },
            CurveFamily::SaturatingTrendBasis4 => Self::SaturatingTrendBasis4 {
                c: values[0],
                w1: values[1],
                w2: values[2],
                w3: values[3],
                w4: values[4],
                taus: Self::resolve_saturating_trend_tau_grid(family, tau_grid),
            },
            CurveFamily::SaturatingTrendBasis5 => Self::SaturatingTrendBasis5 {
                c: values[0],
                w1: values[1],
                w2: values[2],
                w3: values[3],
                w4: values[4],
                w5: values[5],
                taus: Self::resolve_saturating_trend_tau_grid(family, tau_grid),
            },
            CurveFamily::SaturatingTrendBasis6 => Self::SaturatingTrendBasis6 {
                c: values[0],
                w1: values[1],
                w2: values[2],
                w3: values[3],
                w4: values[4],
                w5: values[5],
                w6: values[6],
                taus: Self::resolve_saturating_trend_tau_grid(family, tau_grid),
            },
            _ => curve_params_variants!(curve_params_from_slice_match, family, values,),
        };

        Ok(params)
    }

    /// Удобная обертка над [`Self::try_from_slice`] для владения `Vec<f64>`.
    pub fn try_from_values(family: CurveFamily, values: Vec<f64>) -> Result<Self, InputError> {
        Self::try_from_slice(family, &values)
    }

    /// Реконструирует параметры из оптимизируемого среза, сохраняя метаданные шаблона.
    pub fn try_from_slice_like(template: &Self, values: &[f64]) -> Result<Self, InputError> {
        Self::try_from_slice_with_tau_grid(
            template.family(),
            values,
            template.saturating_trend_tau_grid(),
        )
    }
}
