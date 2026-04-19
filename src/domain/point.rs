//! Типы точки и набора точек с инвариантами на конечность координат и минимальный размер.

use super::InputError;
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
