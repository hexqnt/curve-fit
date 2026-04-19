//! Ошибки на границе доменной модели: точки, параметры и конфигурации оптимизаторов.

use std::fmt;

use super::CurveFamily;

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
