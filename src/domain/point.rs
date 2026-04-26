//! Типы точки и набора точек с инвариантами на конечность координат и минимальный размер.

use super::InputError;
use std::ops::Deref;
use std::sync::Arc;

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
    points: Arc<[Point]>,
}

impl Points {
    /// Возвращает неизменяемый срез точек без дополнительных аллокаций.
    pub fn as_slice(&self) -> &[Point] {
        &self.points
    }

    /// Возвращает итератор по точкам.
    pub fn iter(&self) -> std::slice::Iter<'_, Point> {
        self.points.iter()
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
        let (first, rest) = self
            .points
            .split_first()
            .expect("Points invariant guarantees a non-empty collection");
        rest.iter()
            .fold((first.x(), first.x()), |(min_x, max_x), point| {
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
            points: Arc::from(points),
        })
    }
}

impl<const N: usize> TryFrom<[Point; N]> for Points {
    type Error = InputError;

    fn try_from(points: [Point; N]) -> Result<Self, Self::Error> {
        if N < MIN_POINTS {
            return Err(InputError::TooFewPoints {
                len: N,
                min_required: MIN_POINTS,
            });
        }
        Ok(Self {
            points: Arc::from(points),
        })
    }
}

impl AsRef<[Point]> for Points {
    fn as_ref(&self) -> &[Point] {
        self.as_slice()
    }
}

impl Deref for Points {
    type Target = [Point];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a Points {
    type Item = &'a Point;
    type IntoIter = std::slice::Iter<'a, Point>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
