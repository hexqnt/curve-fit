//! Строго типизированная сетка времен насыщения для saturating-trend базиса.

use super::InputError;

pub const MIN_SATURATING_TREND_TAU_COUNT: usize = 1;
pub const MAX_SATURATING_TREND_TAU_COUNT: usize = 6;
pub const DEFAULT_SATURATING_TREND_TAUS_YEARS: [f64; MAX_SATURATING_TREND_TAU_COUNT] =
    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

#[derive(Debug, Clone, PartialEq)]
/// Возрастающая положительная сетка `τ`, задаваемая пользователем для saturating-базиса.
pub struct SaturatingTrendTauGrid {
    values: [f64; MAX_SATURATING_TREND_TAU_COUNT],
    count: usize,
}

impl SaturatingTrendTauGrid {
    /// Создает сетку из уже распарсенного набора `τ` и проверяет инварианты.
    pub fn from_values(values: &[f64]) -> Result<Self, InputError> {
        let count = values.len();
        if !(MIN_SATURATING_TREND_TAU_COUNT..=MAX_SATURATING_TREND_TAU_COUNT).contains(&count) {
            return Err(InputError::WrongSaturatingTrendTauCount {
                expected_min: MIN_SATURATING_TREND_TAU_COUNT,
                expected_max: MAX_SATURATING_TREND_TAU_COUNT,
                got: count,
            });
        }

        let mut stored = [0.0; MAX_SATURATING_TREND_TAU_COUNT];
        let mut previous = 0.0;
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(InputError::NonFiniteSaturatingTrendTau { index, value });
            }
            if value <= 0.0 {
                return Err(InputError::NonPositiveSaturatingTrendTau { index, value });
            }
            if index > 0 && value <= previous {
                return Err(InputError::NonIncreasingSaturatingTrendTau {
                    index,
                    previous,
                    value,
                });
            }
            stored[index] = value;
            previous = value;
        }

        Ok(Self {
            values: stored,
            count,
        })
    }

    /// Возвращает дефолтный лог-равномерный префикс заданной длины.
    pub fn default_for_count(count: usize) -> Self {
        let clamped = count.clamp(
            MIN_SATURATING_TREND_TAU_COUNT,
            MAX_SATURATING_TREND_TAU_COUNT,
        );
        Self {
            values: DEFAULT_SATURATING_TREND_TAUS_YEARS,
            count: clamped,
        }
    }

    /// Возвращает число активных `τ`.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Возвращает активный префикс сетки.
    pub fn as_slice(&self) -> &[f64] {
        &self.values[..self.count]
    }
}
