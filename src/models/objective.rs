use super::common;
use super::{Grad, Hessian, Param, TermGrad, TermHessian, TermValue};

/// Уровень вычисления только значения objective.
pub(crate) trait ObjectiveValue {
    fn value(&self, param: &Param) -> f64;
}

/// Уровень вычисления значения и градиента objective.
pub(crate) trait ObjectiveGrad: ObjectiveValue {
    fn value_grad(&self, param: &Param) -> (f64, Grad);
}

/// Уровень вычисления значения, градиента и гессиана objective.
pub(crate) trait ObjectiveHessian: ObjectiveGrad {
    fn value_grad_raw_hessian(&self, param: &Param) -> (f64, Grad, Hessian);

    fn value_grad_hessian(&self, param: &Param) -> (f64, Grad, Hessian) {
        let (value, gradient, mut hessian) = self.value_grad_raw_hessian(param);
        common::stabilize_hessian(&mut hessian);
        (value, gradient, hessian)
    }
}

/// Objective, собранный из терма (или композиции термов через обертку).
pub(crate) struct CurveObjective<T> {
    parameter_count: usize,
    term: T,
}

impl<T> CurveObjective<T> {
    pub(crate) fn new(parameter_count: usize, term: T) -> Self {
        Self {
            parameter_count,
            term,
        }
    }
}

impl<T> ObjectiveValue for CurveObjective<T>
where
    T: TermValue,
{
    fn value(&self, param: &Param) -> f64 {
        let mut value = 0.0;
        self.term.add_value(param, &mut value);
        value
    }
}

impl<T> ObjectiveGrad for CurveObjective<T>
where
    T: TermGrad,
{
    fn value_grad(&self, param: &Param) -> (f64, Grad) {
        let mut value = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        self.term.add_value_grad(param, &mut value, &mut gradient);
        (value, gradient)
    }
}

impl<T> ObjectiveHessian for CurveObjective<T>
where
    T: TermHessian,
{
    fn value_grad_raw_hessian(&self, param: &Param) -> (f64, Grad, Hessian) {
        let mut value = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        let mut hessian = Hessian::zeros((self.parameter_count, self.parameter_count));
        self.term
            .add_value_grad_hessian(param, &mut value, &mut gradient, &mut hessian);
        (value, gradient, hessian)
    }
}

/// Центр-разностная аппроксимация градиента поверх objective, умеющего считать только значение.
pub(crate) struct CentralDiffGradient<O> {
    inner: O,
    rel_step: f64,
    min_step: f64,
}

impl<O> CentralDiffGradient<O> {
    pub(crate) fn new(inner: O, rel_step: f64, min_step: f64) -> Self {
        Self {
            inner,
            rel_step,
            min_step,
        }
    }
}

impl<O> ObjectiveValue for CentralDiffGradient<O>
where
    O: ObjectiveValue,
{
    fn value(&self, param: &Param) -> f64 {
        self.inner.value(param)
    }
}

impl<O> ObjectiveGrad for CentralDiffGradient<O>
where
    O: ObjectiveValue,
{
    fn value_grad(&self, param: &Param) -> (f64, Grad) {
        let value = self.inner.value(param);
        let mut gradient = vec![0.0; param.len()];
        central_diff_gradient_from_value(
            param,
            self.rel_step,
            self.min_step,
            |probe| self.inner.value(probe),
            &mut gradient,
        );
        (value, gradient)
    }
}

/// Центр-разностная аппроксимация гессиана поверх objective, умеющего считать градиент.
pub(crate) struct CentralDiffHessian<O> {
    inner: O,
    rel_step: f64,
    min_step: f64,
}

impl<O> CentralDiffHessian<O> {
    pub(crate) fn new(inner: O, rel_step: f64, min_step: f64) -> Self {
        Self {
            inner,
            rel_step,
            min_step,
        }
    }
}

impl<O> ObjectiveValue for CentralDiffHessian<O>
where
    O: ObjectiveGrad,
{
    fn value(&self, param: &Param) -> f64 {
        self.inner.value(param)
    }
}

impl<O> ObjectiveGrad for CentralDiffHessian<O>
where
    O: ObjectiveGrad,
{
    fn value_grad(&self, param: &Param) -> (f64, Grad) {
        self.inner.value_grad(param)
    }
}

impl<O> ObjectiveHessian for CentralDiffHessian<O>
where
    O: ObjectiveGrad,
{
    fn value_grad_raw_hessian(&self, param: &Param) -> (f64, Grad, Hessian) {
        let (value, gradient) = self.inner.value_grad(param);
        let hessian = central_diff_hessian_from_gradient(
            param,
            self.rel_step,
            self.min_step,
            |probe, gradient_out| {
                let (_, gradient_probe) = self.inner.value_grad(probe);
                gradient_out.copy_from_slice(&gradient_probe);
            },
        );
        (value, gradient, hessian)
    }
}

#[inline]
fn fd_step(value: f64, rel_step: f64, min_step: f64) -> f64 {
    ((value.abs() + 1.0) * rel_step).max(min_step)
}

pub(crate) fn central_diff_gradient_from_value<F>(
    param: &Param,
    rel_step: f64,
    min_step: f64,
    mut value_at: F,
    gradient: &mut [f64],
) where
    F: FnMut(&Param) -> f64,
{
    debug_assert_eq!(param.len(), gradient.len());
    let mut probe = param.to_vec();
    let mut index = 0;
    while index < param.len() {
        let step = fd_step(param[index], rel_step, min_step);
        probe[index] = param[index] + step;
        let value_plus = value_at(&probe);
        probe[index] = param[index] - step;
        let value_minus = value_at(&probe);
        probe[index] = param[index];
        let value = (value_plus - value_minus) / (2.0 * step);
        gradient[index] = if value.is_finite() { value } else { 0.0 };
        index += 1;
    }
}

pub(crate) fn central_diff_hessian_from_gradient<G>(
    param: &Param,
    rel_step: f64,
    min_step: f64,
    mut gradient_at: G,
) -> Hessian
where
    G: FnMut(&Param, &mut [f64]),
{
    let dimension = param.len();
    let mut hessian = Hessian::zeros((dimension, dimension));
    let mut probe = param.to_vec();
    let mut grad_plus = vec![0.0; dimension];
    let mut grad_minus = vec![0.0; dimension];

    let mut column = 0;
    while column < dimension {
        let step = fd_step(param[column], rel_step, min_step);
        probe[column] = param[column] + step;
        gradient_at(&probe, &mut grad_plus);
        probe[column] = param[column] - step;
        gradient_at(&probe, &mut grad_minus);
        probe[column] = param[column];

        let denom = 2.0 * step;
        let mut row = 0;
        while row < dimension {
            let value = (grad_plus[row] - grad_minus[row]) / denom;
            hessian[[row, column]] = if value.is_finite() { value } else { 0.0 };
            row += 1;
        }
        column += 1;
    }

    hessian
}
