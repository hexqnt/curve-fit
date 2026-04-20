//! Модели параметрических кривых и objective-слой для value/grad/hessian.

use ndarray::Array2;

mod arctangent_step;
mod arrhenius;
mod bi_exponential;
mod common;
mod damped_sinusoid;
mod data_term;
mod dispatch;
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
mod loss;
mod michaelis_menten;
mod natural_log;
mod objective;
mod polynomial;
mod power;
mod pseudo_voigt;
mod rational_11;
mod rational_22;
mod rational_nn;
#[cfg(test)]
mod simd_tests;
mod softplus;
mod term;
#[cfg(test)]
mod test_support;

pub(crate) use common::{HESSIAN_DIAGONAL_JITTER, PARAM_EPS};
pub(crate) use data_term::DataTerm;
pub(crate) use dispatch::value_at;
pub(crate) use loss::PredictionLoss;
pub(crate) use objective::{
    CentralDiffGradient, CentralDiffHessian, CurveObjective, ObjectiveGrad, ObjectiveHessian,
    ObjectiveValue,
};
#[cfg(test)]
pub(crate) use objective::{central_diff_gradient_from_value, central_diff_hessian_from_gradient};
pub(crate) use term::{TermGrad, TermHessian, TermValue};

pub(crate) type Param = [f64];
pub(crate) type Grad = Vec<f64>;
pub(crate) type Hessian = Array2<f64>;

#[inline]
pub(crate) fn positive_x(value: f64) -> f64 {
    common::positive_x(value)
}

#[cfg(test)]
#[inline]
pub(crate) fn softplus(value: f64) -> f64 {
    common::softplus(value)
}
