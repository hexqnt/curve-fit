//! Типизированное представление параметров всех поддерживаемых семейств кривых.

use crate::models;

use super::{CurveFamily, InputError};
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
            (Emg, Emg, [a, mu, sigma, tau, c]),
            (PseudoVoigt, PseudoVoigt, [a, x0, sigma, gamma, eta, c]),
        }
    };
}

macro_rules! curve_params_family_match {
    ($value:expr, $(($variant:ident, $family:ident, [$($field:ident),+])),+ $(,)?) => {
        match $value {
            $(Self::$variant { .. } => CurveFamily::$family,)+
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
        }
    };
}

impl CurveParams {
    /// Определяет семейство по варианту параметров.
    pub fn family(&self) -> CurveFamily {
        curve_params_variants!(curve_params_family_match, self,)
    }

    fn with_family_values<R>(&self, callback: impl FnOnce(CurveFamily, &[f64]) -> R) -> R {
        curve_params_variants!(curve_params_with_values_match, self, callback,)
    }

    /// Возвращает параметры в виде вектора в каноническом порядке.
    pub fn values(&self) -> Vec<f64> {
        self.with_family_values(|_family, values| values.to_vec())
    }

    /// Вычисляет значение модели для заданного `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        self.with_family_values(|family, values| models::value_at(family, values, x))
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

        Ok(curve_params_variants!(
            curve_params_from_slice_match,
            family,
            values,
        ))
    }

    /// Удобная обертка над [`Self::try_from_slice`] для владения `Vec<f64>`.
    pub fn try_from_values(family: CurveFamily, values: Vec<f64>) -> Result<Self, InputError> {
        Self::try_from_slice(family, &values)
    }
}
