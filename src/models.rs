//! Модели параметрических кривых и objective-слой для value/grad/hessian.

use crate::domain::{CurveFamily, CurveParams};
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

pub(crate) fn evaluate_curve_params(params: &CurveParams, x: f64) -> f64 {
    match params {
        CurveParams::Linear { a, b } => {
            let values = [*a, *b];
            value_at(CurveFamily::Linear, &values, x)
        }
        CurveParams::Quadratic { a, b, c } => {
            let values = [*a, *b, *c];
            value_at(CurveFamily::Quadratic, &values, x)
        }
        CurveParams::Cubic { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::Cubic, &values, x)
        }
        CurveParams::Quartic { a, b, c, d, e } => {
            let values = [*a, *b, *c, *d, *e];
            value_at(CurveFamily::Quartic, &values, x)
        }
        CurveParams::Quintic { a, b, c, d, e, f } => {
            let values = [*a, *b, *c, *d, *e, *f];
            value_at(CurveFamily::Quintic, &values, x)
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
            value_at(CurveFamily::Sextic, &values, x)
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
            value_at(CurveFamily::Septic, &values, x)
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
            value_at(CurveFamily::Octic, &values, x)
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
            value_at(CurveFamily::Nonic, &values, x)
        }
        CurveParams::Arrhenius { a, b } => {
            let values = [*a, *b];
            value_at(CurveFamily::Arrhenius, &values, x)
        }
        CurveParams::Inverse { a, b } => {
            let values = [*a, *b];
            value_at(CurveFamily::Inverse, &values, x)
        }
        CurveParams::Logistic { a, b, c } => {
            let values = [*a, *b, *c];
            value_at(CurveFamily::Logistic, &values, x)
        }
        CurveParams::Gompertz { a, b, c } => {
            let values = [*a, *b, *c];
            value_at(CurveFamily::Gompertz, &values, x)
        }
        CurveParams::BiExponential { a1, k1, a2, k2, c } => {
            let values = [*a1, *k1, *a2, *k2, *c];
            value_at(CurveFamily::BiExponential, &values, x)
        }
        CurveParams::DampedSinusoid {
            a,
            k,
            omega,
            phi,
            c,
        } => {
            let values = [*a, *k, *omega, *phi, *c];
            value_at(CurveFamily::DampedSinusoid, &values, x)
        }
        CurveParams::Lorentzian { a, x0, gamma, c } => {
            let values = [*a, *x0, *gamma, *c];
            value_at(CurveFamily::Lorentzian, &values, x)
        }
        CurveParams::NaturalLog { a, b } => {
            let values = [*a, *b];
            value_at(CurveFamily::NaturalLog, &values, x)
        }
        CurveParams::FourPl { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::FourPl, &values, x)
        }
        CurveParams::FivePl { a, b, c, d, m } => {
            let values = [*a, *b, *c, *d, *m];
            value_at(CurveFamily::FivePl, &values, x)
        }
        CurveParams::MichaelisMenten { vmax, km } => {
            let values = [*vmax, *km];
            value_at(CurveFamily::MichaelisMenten, &values, x)
        }
        CurveParams::ExponentialBasic { a, b, c } => {
            let values = [*a, *b, *c];
            value_at(CurveFamily::ExponentialBasic, &values, x)
        }
        CurveParams::ExponentialLinear { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::ExponentialLinear, &values, x)
        }
        CurveParams::ExponentialHalfLife { a, b, c } => {
            let values = [*a, *b, *c];
            value_at(CurveFamily::ExponentialHalfLife, &values, x)
        }
        CurveParams::FallingExponential { y0, v0, k } => {
            let values = [*y0, *v0, *k];
            value_at(CurveFamily::FallingExponential, &values, x)
        }
        CurveParams::HyperbolicTangent { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::HyperbolicTangent, &values, x)
        }
        CurveParams::ArctangentStep { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::ArctangentStep, &values, x)
        }
        CurveParams::Softplus { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::Softplus, &values, x)
        }
        CurveParams::Power { a, b } => {
            let values = [*a, *b];
            value_at(CurveFamily::Power, &values, x)
        }
        CurveParams::Gaussian { a, b, c } => {
            let values = [*a, *b, *c];
            value_at(CurveFamily::Gaussian, &values, x)
        }
        CurveParams::Rational11 { a, b, c, d } => {
            let values = [*a, *b, *c, *d];
            value_at(CurveFamily::Rational11, &values, x)
        }
        CurveParams::Rational22 { a, b, c, d, e } => {
            let values = [*a, *b, *c, *d, *e];
            value_at(CurveFamily::Rational22, &values, x)
        }
        CurveParams::Emg {
            a,
            mu,
            sigma,
            tau,
            c,
        } => {
            let values = [*a, *mu, *sigma, *tau, *c];
            value_at(CurveFamily::Emg, &values, x)
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
            value_at(CurveFamily::PseudoVoigt, &values, x)
        }
    }
}
