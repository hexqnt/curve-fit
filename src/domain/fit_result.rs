//! Доменный результат подгонки параметрической модели.

use super::{CurveFamily, CurveParams};

#[derive(Debug, Clone, PartialEq)]
/// Результат подгонки параметрической модели.
pub struct FitResult {
    pub family: CurveFamily,
    pub params: CurveParams,
    pub mse: f64,
    pub rmse: f64,
    pub iterations: u64,
}
