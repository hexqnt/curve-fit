//! Общие численные константы и короткие вспомогательные обертки для внутренних fit-модулей.

use super::*;

pub(super) const PARAM_EPS: f64 = models::PARAM_EPS;
pub(super) const LARGE_COST: f64 = 1e24;
pub(super) const MAX_POLYNOMIAL_PARAMS: usize = 10;
pub(super) const STEEPEST_DESCENT_GRAD_TOL: f64 = 1e-12;
pub(super) const HESSIAN_FD_REL_STEP: f64 = 1e-4;
pub(super) const HESSIAN_FD_MIN_STEP: f64 = 1e-6;
// Пробуем базовый шаг, затем уменьшаем/увеличиваем его, чтобы переживать локальные NaN/Inf.
pub(super) const FD_STEP_RETRY_FACTORS: [f64; 5] = [1.0, 0.5, 2.0, 0.25, 4.0];
pub(crate) const HESSIAN_DIAGONAL_JITTER: f64 = models::HESSIAN_DIAGONAL_JITTER;

pub(super) fn positive_x(value: f64) -> f64 {
    models::positive_x(value)
}

#[cfg(test)]
pub(super) fn softplus(value: f64) -> f64 {
    models::softplus(value)
}
