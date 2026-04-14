//! Модели параметрических кривых: значение, градиент и аналитический гессиан.

use crate::domain::{CurveFamily, CurveParams};
use ndarray::Array2;

mod arctangent_step;
mod arrhenius;
mod bi_exponential;
mod common;
mod damped_sinusoid;
mod emg;
mod exponential_basic;
mod exponential_half_life;
mod exponential_linear;
mod falling_exponential;
mod five_pl;
mod four_pl;
mod gaussian;
mod gompertz;
mod hyperbolic_tangent;
mod inverse;
mod logistic;
mod lorentzian;
mod michaelis_menten;
mod natural_log;
mod polynomial;
mod power;
mod pseudo_voigt;
mod rational_11;
mod rational_22;
mod softplus;
#[cfg(test)]
mod test_support;

pub(crate) use common::{HESSIAN_DIAGONAL_JITTER, PARAM_EPS};

const OBJECTIVE_GRADIENT_FD_REL_STEP: f64 = 1e-5;
const OBJECTIVE_GRADIENT_FD_MIN_STEP: f64 = 1e-7;
const OBJECTIVE_HESSIAN_FD_REL_STEP: f64 = 1e-4;
const OBJECTIVE_HESSIAN_FD_MIN_STEP: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradientComputation {
    Analytic,
    NeedsNumerical,
}

/// Аналитическая функция потерь по предсказанию модели.
pub(crate) trait PredictionLoss {
    fn value(&self, prediction: f64, target: f64) -> f64;
    fn d_prediction(&self, prediction: f64, target: f64) -> f64;
    fn d2_prediction(&self, prediction: f64, target: f64) -> f64;
}

/// Уровень вычисления только значения objective.
pub(crate) trait ObjectiveValue {
    fn value(&self, param: &[f64]) -> f64;
}

/// Уровень вычисления значения и градиента objective.
pub(crate) trait ObjectiveGrad: ObjectiveValue {
    fn value_grad(&self, param: &[f64]) -> (f64, Vec<f64>);
}

/// Уровень вычисления значения, градиента и гессиана objective.
pub(crate) trait ObjectiveHessian: ObjectiveGrad {
    fn value_grad_raw_hessian(&self, param: &[f64]) -> (f64, Vec<f64>, Array2<f64>);

    fn value_grad_hessian(&self, param: &[f64]) -> (f64, Vec<f64>, Array2<f64>) {
        let (value, gradient, mut hessian) = self.value_grad_raw_hessian(param);
        common::stabilize_hessian(&mut hessian);
        (value, gradient, hessian)
    }
}

/// Вклад отдельного терма в значение objective.
pub(crate) trait TermValue {
    fn add_value(&self, param: &[f64], value: &mut f64);
}

/// Вклад отдельного терма в значение и градиент objective.
pub(crate) trait TermGrad: TermValue {
    fn add_value_grad(&self, param: &[f64], value: &mut f64, gradient: &mut [f64]);
}

/// Вклад отдельного терма в значение, градиент и raw-гессиан objective.
pub(crate) trait TermHessian: TermGrad {
    fn add_value_grad_hessian(
        &self,
        param: &[f64],
        value: &mut f64,
        gradient: &mut [f64],
        hessian: &mut Array2<f64>,
    );
}

/// Data-term для подгонки параметрической кривой по набору точек.
pub(crate) struct DataTerm<'a, L> {
    family: CurveFamily,
    x_values: &'a [f64],
    y_values: &'a [f64],
    loss: L,
}

impl<'a, L> DataTerm<'a, L> {
    pub(crate) fn new(
        family: CurveFamily,
        x_values: &'a [f64],
        y_values: &'a [f64],
        loss: L,
    ) -> Self {
        Self {
            family,
            x_values,
            y_values,
            loss,
        }
    }
}

impl<L> TermValue for DataTerm<'_, L>
where
    L: PredictionLoss,
{
    fn add_value(&self, param: &[f64], value: &mut f64) {
        *value += objective_value(self.family, self.x_values, self.y_values, param, &self.loss);
    }
}

impl<L> TermGrad for DataTerm<'_, L>
where
    L: PredictionLoss,
{
    fn add_value_grad(&self, param: &[f64], value: &mut f64, gradient: &mut [f64]) {
        let mut local_gradient = vec![0.0; gradient.len()];
        let (local_value, _) = objective_value_grad(
            self.family,
            self.x_values,
            self.y_values,
            param,
            &self.loss,
            &mut local_gradient,
        );

        *value += local_value;
        for (dst, src) in gradient.iter_mut().zip(local_gradient) {
            *dst += src;
        }
    }
}

impl<L> TermHessian for DataTerm<'_, L>
where
    L: PredictionLoss,
{
    fn add_value_grad_hessian(
        &self,
        param: &[f64],
        value: &mut f64,
        gradient: &mut [f64],
        hessian: &mut Array2<f64>,
    ) {
        let mut local_gradient = vec![0.0; gradient.len()];
        let (local_value, _, local_hessian) = objective_value_grad_raw_hessian(
            self.family,
            self.x_values,
            self.y_values,
            param,
            &self.loss,
            &mut local_gradient,
        );

        *value += local_value;
        for (dst, src) in gradient.iter_mut().zip(local_gradient) {
            *dst += src;
        }
        *hessian += &local_hessian;
    }
}

/// Objective, собранный из терма (в текущей реализации используется один `DataTerm`).
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
    fn value(&self, param: &[f64]) -> f64 {
        let mut value = 0.0;
        self.term.add_value(param, &mut value);
        value
    }
}

impl<T> ObjectiveGrad for CurveObjective<T>
where
    T: TermGrad,
{
    fn value_grad(&self, param: &[f64]) -> (f64, Vec<f64>) {
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
    fn value_grad_raw_hessian(&self, param: &[f64]) -> (f64, Vec<f64>, Array2<f64>) {
        let mut value = 0.0;
        let mut gradient = vec![0.0; self.parameter_count];
        let mut hessian = Array2::zeros((self.parameter_count, self.parameter_count));
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
    fn value(&self, param: &[f64]) -> f64 {
        self.inner.value(param)
    }
}

impl<O> ObjectiveGrad for CentralDiffGradient<O>
where
    O: ObjectiveValue,
{
    fn value_grad(&self, param: &[f64]) -> (f64, Vec<f64>) {
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
    fn value(&self, param: &[f64]) -> f64 {
        self.inner.value(param)
    }
}

impl<O> ObjectiveGrad for CentralDiffHessian<O>
where
    O: ObjectiveGrad,
{
    fn value_grad(&self, param: &[f64]) -> (f64, Vec<f64>) {
        self.inner.value_grad(param)
    }
}

impl<O> ObjectiveHessian for CentralDiffHessian<O>
where
    O: ObjectiveGrad,
{
    fn value_grad_raw_hessian(&self, param: &[f64]) -> (f64, Vec<f64>, Array2<f64>) {
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

#[inline]
pub(crate) fn positive_x(value: f64) -> f64 {
    common::positive_x(value)
}

#[cfg(test)]
#[inline]
pub(crate) fn softplus(value: f64) -> f64 {
    common::softplus(value)
}

#[inline]
pub(crate) fn evaluate_raw(family: CurveFamily, param: &[f64], x: f64) -> f64 {
    if family.is_polynomial() {
        return polynomial::eval(param, x);
    }

    match family {
        CurveFamily::Arrhenius => arrhenius::eval(param, x),
        CurveFamily::Inverse => inverse::eval(param, x),
        CurveFamily::Logistic => logistic::eval(param, x),
        CurveFamily::Gompertz => gompertz::eval(param, x),
        CurveFamily::BiExponential => bi_exponential::eval(param, x),
        CurveFamily::DampedSinusoid => damped_sinusoid::eval(param, x),
        CurveFamily::Lorentzian => lorentzian::eval(param, x),
        CurveFamily::NaturalLog => natural_log::eval(param, x),
        CurveFamily::FourPl => four_pl::eval(param, x),
        CurveFamily::FivePl => five_pl::eval(param, x),
        CurveFamily::MichaelisMenten => michaelis_menten::eval(param, x),
        CurveFamily::ExponentialBasic => exponential_basic::eval(param, x),
        CurveFamily::ExponentialLinear => exponential_linear::eval(param, x),
        CurveFamily::ExponentialHalfLife => exponential_half_life::eval(param, x),
        CurveFamily::FallingExponential => falling_exponential::eval(param, x),
        CurveFamily::HyperbolicTangent => hyperbolic_tangent::eval(param, x),
        CurveFamily::ArctangentStep => arctangent_step::eval(param, x),
        CurveFamily::Softplus => softplus::eval(param, x),
        CurveFamily::Power => power::eval(param, x),
        CurveFamily::Gaussian => gaussian::eval(param, x),
        CurveFamily::Rational11 => rational_11::eval(param, x),
        CurveFamily::Rational22 => rational_22::eval(param, x),
        CurveFamily::Emg => emg::eval(param, x),
        CurveFamily::PseudoVoigt => pseudo_voigt::eval(param, x),
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn central_diff_gradient_from_value<F>(
    param: &[f64],
    rel_step: f64,
    min_step: f64,
    mut value_at: F,
    gradient: &mut [f64],
) where
    F: FnMut(&[f64]) -> f64,
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
    param: &[f64],
    rel_step: f64,
    min_step: f64,
    mut gradient_at: G,
) -> Array2<f64>
where
    G: FnMut(&[f64], &mut [f64]),
{
    let dimension = param.len();
    let mut hessian = Array2::zeros((dimension, dimension));
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

fn objective_value_with_loss<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
) -> f64
where
    L: PredictionLoss,
{
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut index = 0;
    while index < x_values.len() {
        let model = family.evaluate_raw(param, x_values[index]);
        let contribution = loss.value(model, y_values[index]);
        if !contribution.is_finite() {
            return f64::INFINITY;
        }
        sum += contribution;
        if !sum.is_finite() {
            return f64::INFINITY;
        }
        index += 1;
    }
    sum / x_values.len() as f64
}

pub(crate) fn objective_value<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
) -> f64
where
    L: PredictionLoss,
{
    objective_value_with_loss(family, x_values, y_values, param, loss)
}

pub(crate) fn evaluate_curve_params(params: &CurveParams, x: f64) -> f64 {
    match params {
        CurveParams::Linear { a, b } => {
            let values = [*a, *b];
            evaluate_raw(CurveFamily::Linear, &values, x)
        }
        CurveParams::Quadratic { a, b, c } => {
            let values = [*a, *b, *c];
            evaluate_raw(CurveFamily::Quadratic, &values, x)
        }
        CurveParams::Cubic { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::Cubic, &values, x)
        }
        CurveParams::Quartic { a, b, c, d, e } => {
            let values = [*a, *b, *c, *d, *e];
            evaluate_raw(CurveFamily::Quartic, &values, x)
        }
        CurveParams::Quintic { a, b, c, d, e, f } => {
            let values = [*a, *b, *c, *d, *e, *f];
            evaluate_raw(CurveFamily::Quintic, &values, x)
        }
        CurveParams::Sextic {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        } => {
            let values = [*a, *b, *c, *d, *e, *f, *g];
            evaluate_raw(CurveFamily::Sextic, &values, x)
        }
        CurveParams::Septic {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
        } => {
            let values = [*a, *b, *c, *d, *e, *f, *g, *h];
            evaluate_raw(CurveFamily::Septic, &values, x)
        }
        CurveParams::Octic {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
        } => {
            let values = [*a, *b, *c, *d, *e, *f, *g, *h, *i];
            evaluate_raw(CurveFamily::Octic, &values, x)
        }
        CurveParams::Nonic {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
            j,
        } => {
            let values = [*a, *b, *c, *d, *e, *f, *g, *h, *i, *j];
            evaluate_raw(CurveFamily::Nonic, &values, x)
        }
        CurveParams::Arrhenius { a, b } => {
            let values = [*a, *b];
            evaluate_raw(CurveFamily::Arrhenius, &values, x)
        }
        CurveParams::Inverse { a, b } => {
            let values = [*a, *b];
            evaluate_raw(CurveFamily::Inverse, &values, x)
        }
        CurveParams::Logistic { a, b, c } => {
            let values = [*a, *b, *c];
            evaluate_raw(CurveFamily::Logistic, &values, x)
        }
        CurveParams::Gompertz { a, b, c } => {
            let values = [*a, *b, *c];
            evaluate_raw(CurveFamily::Gompertz, &values, x)
        }
        CurveParams::BiExponential { a1, k1, a2, k2, c } => {
            let values = [*a1, *k1, *a2, *k2, *c];
            evaluate_raw(CurveFamily::BiExponential, &values, x)
        }
        CurveParams::DampedSinusoid {
            a,
            k,
            omega,
            phi,
            c,
        } => {
            let values = [*a, *k, *omega, *phi, *c];
            evaluate_raw(CurveFamily::DampedSinusoid, &values, x)
        }
        CurveParams::Lorentzian { a, x0, gamma, c } => {
            let values = [*a, *x0, *gamma, *c];
            evaluate_raw(CurveFamily::Lorentzian, &values, x)
        }
        CurveParams::NaturalLog { a, b } => {
            let values = [*a, *b];
            evaluate_raw(CurveFamily::NaturalLog, &values, x)
        }
        CurveParams::FourPl { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::FourPl, &values, x)
        }
        CurveParams::FivePl { a, b, c, d, m } => {
            let values = [*a, *b, *c, *d, *m];
            evaluate_raw(CurveFamily::FivePl, &values, x)
        }
        CurveParams::MichaelisMenten { vmax, km } => {
            let values = [*vmax, *km];
            evaluate_raw(CurveFamily::MichaelisMenten, &values, x)
        }
        CurveParams::ExponentialBasic { a, b, c } => {
            let values = [*a, *b, *c];
            evaluate_raw(CurveFamily::ExponentialBasic, &values, x)
        }
        CurveParams::ExponentialLinear { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::ExponentialLinear, &values, x)
        }
        CurveParams::ExponentialHalfLife { a, b, c } => {
            let values = [*a, *b, *c];
            evaluate_raw(CurveFamily::ExponentialHalfLife, &values, x)
        }
        CurveParams::FallingExponential { y0, v0, k } => {
            let values = [*y0, *v0, *k];
            evaluate_raw(CurveFamily::FallingExponential, &values, x)
        }
        CurveParams::HyperbolicTangent { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::HyperbolicTangent, &values, x)
        }
        CurveParams::ArctangentStep { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::ArctangentStep, &values, x)
        }
        CurveParams::Softplus { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::Softplus, &values, x)
        }
        CurveParams::Power { a, b } => {
            let values = [*a, *b];
            evaluate_raw(CurveFamily::Power, &values, x)
        }
        CurveParams::Gaussian { a, b, c } => {
            let values = [*a, *b, *c];
            evaluate_raw(CurveFamily::Gaussian, &values, x)
        }
        CurveParams::Rational11 { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            evaluate_raw(CurveFamily::Rational11, &values, x)
        }
        CurveParams::Rational22 { a, b, c, d, e } => {
            let values = [*a, *b, *c, *d, *e];
            evaluate_raw(CurveFamily::Rational22, &values, x)
        }
        CurveParams::Emg {
            a,
            mu,
            sigma,
            tau,
            c,
        } => {
            let values = [*a, *mu, *sigma, *tau, *c];
            evaluate_raw(CurveFamily::Emg, &values, x)
        }
        CurveParams::PseudoVoigt {
            a,
            x0,
            sigma,
            gamma,
            eta,
            c,
        } => {
            let values = [*a, *x0, *sigma, *gamma, *eta, *c];
            evaluate_raw(CurveFamily::PseudoVoigt, &values, x)
        }
    }
}

pub(crate) fn accumulate_gradient<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
    gradient: &mut [f64],
) -> GradientComputation
where
    L: PredictionLoss,
{
    if family.is_polynomial() {
        polynomial::accumulate_gradient(x_values, y_values, param, loss, gradient);
        return GradientComputation::Analytic;
    }

    match family {
        CurveFamily::Arrhenius => {
            arrhenius::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Inverse => {
            inverse::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Logistic => {
            logistic::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Gompertz => {
            gompertz::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::BiExponential => {
            bi_exponential::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::DampedSinusoid => {
            damped_sinusoid::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Lorentzian => {
            lorentzian::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::NaturalLog => {
            natural_log::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::FourPl => {
            four_pl::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::FivePl => {
            five_pl::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::MichaelisMenten => {
            michaelis_menten::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::ExponentialBasic => {
            exponential_basic::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::ExponentialLinear => {
            exponential_linear::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::ExponentialHalfLife => {
            exponential_half_life::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::FallingExponential => {
            falling_exponential::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::HyperbolicTangent => {
            hyperbolic_tangent::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::ArctangentStep => {
            arctangent_step::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Softplus => {
            softplus::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Power => {
            power::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Gaussian => {
            gaussian::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Rational11 => {
            rational_11::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Rational22 => {
            rational_22::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        CurveFamily::Emg => {
            // Для EMG аналитический градиент пока не реализован:
            // внешняя логика выполнит численное дифференцирование.
            emg::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::NeedsNumerical
        }
        CurveFamily::PseudoVoigt => {
            pseudo_voigt::accumulate_gradient(x_values, y_values, param, loss, gradient);
            GradientComputation::Analytic
        }
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn analytic_hessian<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
) -> Option<Array2<f64>>
where
    L: PredictionLoss,
{
    if family.is_polynomial() {
        return polynomial::analytic_hessian(x_values, y_values, param, loss);
    }

    match family {
        CurveFamily::Arrhenius => arrhenius::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Inverse => inverse::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Logistic => logistic::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Gompertz => gompertz::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::BiExponential => {
            bi_exponential::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::DampedSinusoid => {
            damped_sinusoid::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::Lorentzian => lorentzian::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::NaturalLog => natural_log::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::FourPl => four_pl::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::FivePl => five_pl::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::MichaelisMenten => {
            michaelis_menten::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::ExponentialBasic => {
            exponential_basic::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::ExponentialLinear => {
            exponential_linear::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::ExponentialHalfLife => {
            exponential_half_life::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::FallingExponential => {
            falling_exponential::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::HyperbolicTangent => {
            hyperbolic_tangent::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::ArctangentStep => {
            arctangent_step::analytic_hessian(x_values, y_values, param, loss)
        }
        CurveFamily::Softplus => softplus::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Power => power::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Gaussian => gaussian::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Rational11 => rational_11::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Rational22 => rational_22::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::Emg => emg::analytic_hessian(x_values, y_values, param, loss),
        CurveFamily::PseudoVoigt => pseudo_voigt::analytic_hessian(x_values, y_values, param, loss),
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn objective_value_grad<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
    gradient: &mut [f64],
) -> (f64, GradientComputation)
where
    L: PredictionLoss,
{
    struct DataValueObjective<'a, L> {
        family: CurveFamily,
        x_values: &'a [f64],
        y_values: &'a [f64],
        loss: &'a L,
    }

    impl<L> ObjectiveValue for DataValueObjective<'_, L>
    where
        L: PredictionLoss,
    {
        fn value(&self, param: &[f64]) -> f64 {
            objective_value_with_loss(self.family, self.x_values, self.y_values, param, self.loss)
        }
    }

    debug_assert_eq!(x_values.len(), y_values.len());
    gradient.fill(0.0);

    let value = objective_value_with_loss(family, x_values, y_values, param, loss);
    let mode = accumulate_gradient(family, x_values, y_values, param, loss, gradient);
    if matches!(mode, GradientComputation::NeedsNumerical) {
        let objective = DataValueObjective {
            family,
            x_values,
            y_values,
            loss,
        };
        let numerical = CentralDiffGradient::new(
            objective,
            OBJECTIVE_GRADIENT_FD_REL_STEP,
            OBJECTIVE_GRADIENT_FD_MIN_STEP,
        );
        let (_, numerical_gradient) = numerical.value_grad(param);
        gradient.copy_from_slice(&numerical_gradient);
        return (value, mode);
    }

    let sample_scale = 1.0 / x_values.len() as f64;
    for value in gradient.iter_mut() {
        *value *= sample_scale;
    }

    (value, mode)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn objective_value_grad_raw_hessian<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
    gradient: &mut [f64],
) -> (f64, GradientComputation, Array2<f64>)
where
    L: PredictionLoss,
{
    struct DataValueGradObjective<'a, L> {
        family: CurveFamily,
        x_values: &'a [f64],
        y_values: &'a [f64],
        loss: &'a L,
    }

    impl<L> ObjectiveValue for DataValueGradObjective<'_, L>
    where
        L: PredictionLoss,
    {
        fn value(&self, param: &[f64]) -> f64 {
            objective_value_with_loss(self.family, self.x_values, self.y_values, param, self.loss)
        }
    }

    impl<L> ObjectiveGrad for DataValueGradObjective<'_, L>
    where
        L: PredictionLoss,
    {
        fn value_grad(&self, param: &[f64]) -> (f64, Vec<f64>) {
            let mut gradient = vec![0.0; param.len()];
            let (value, _) = objective_value_grad(
                self.family,
                self.x_values,
                self.y_values,
                param,
                self.loss,
                &mut gradient,
            );
            (value, gradient)
        }
    }

    let (value, gradient_mode) =
        objective_value_grad(family, x_values, y_values, param, loss, gradient);

    if let Some(hessian) = analytic_hessian(family, x_values, y_values, param, loss) {
        let mut raw_hessian = hessian;
        let mut index = 0;
        while index < raw_hessian.nrows() {
            raw_hessian[[index, index]] -= HESSIAN_DIAGONAL_JITTER;
            index += 1;
        }
        return (value, gradient_mode, raw_hessian);
    }

    let objective = DataValueGradObjective {
        family,
        x_values,
        y_values,
        loss,
    };
    let numerical = CentralDiffHessian::new(
        objective,
        OBJECTIVE_HESSIAN_FD_REL_STEP,
        OBJECTIVE_HESSIAN_FD_MIN_STEP,
    );
    let (_, _, hessian) = numerical.value_grad_raw_hessian(param);
    (value, gradient_mode, hessian)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn objective_value_grad_hessian<L>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
    gradient: &mut [f64],
) -> (f64, GradientComputation, Array2<f64>)
where
    L: PredictionLoss,
{
    let (value, gradient_mode, mut hessian) =
        objective_value_grad_raw_hessian(family, x_values, y_values, param, loss, gradient);
    common::stabilize_hessian(&mut hessian);
    (value, gradient_mode, hessian)
}
