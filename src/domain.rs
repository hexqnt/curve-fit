//! Доменная модель для точек, семейств кривых и параметров оптимизации.
//! Здесь сосредоточены типы с инвариантами, которые проверяются на границах системы.

mod error;
mod family;
mod fit_result;
mod optimizer;
mod params;
mod point;
mod saturating_trend;

pub use error::InputError;
pub use family::{CurveFamily, MAX_RATIONAL_DEGREE, MIN_RATIONAL_DEGREE};
pub use fit_result::FitResult;
pub use optimizer::{
    AdamConfig, LbfgsConfig, NelderMeadConfig, NewtonCgConfig, OptimizerConfig, OptimizerMethod,
    SgdConfig, SteepestDescentConfig,
};
pub use params::CurveParams;
pub use point::{Point, Points};
pub use saturating_trend::{
    DEFAULT_SATURATING_TREND_TAUS_YEARS, MAX_SATURATING_TREND_TAU_COUNT,
    MIN_SATURATING_TREND_TAU_COUNT, SaturatingTrendTauGrid,
};

#[cfg(test)]
pub(crate) use family::CURVE_FAMILY_COUNT;

#[cfg(test)]
mod tests;
