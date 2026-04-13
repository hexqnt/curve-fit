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
mod tests;

pub(crate) use common::{HESSIAN_DIAGONAL_JITTER, PARAM_EPS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradientComputation {
    Analytic,
    NeedsNumerical,
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
    mut loss_derivative_from_prediction: L,
    gradient: &mut [f64],
) -> GradientComputation
where
    L: FnMut(f64, f64) -> f64,
{
    if family.is_polynomial() {
        polynomial::accumulate_gradient(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            gradient,
        );
        return GradientComputation::Analytic;
    }

    match family {
        CurveFamily::Arrhenius => {
            arrhenius::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Inverse => {
            inverse::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Logistic => {
            logistic::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Gompertz => {
            gompertz::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::BiExponential => {
            bi_exponential::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::DampedSinusoid => {
            damped_sinusoid::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Lorentzian => {
            lorentzian::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::NaturalLog => {
            natural_log::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::FourPl => {
            four_pl::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::FivePl => {
            five_pl::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::MichaelisMenten => {
            michaelis_menten::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::ExponentialBasic => {
            exponential_basic::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::ExponentialLinear => {
            exponential_linear::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::ExponentialHalfLife => {
            exponential_half_life::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::FallingExponential => {
            falling_exponential::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::HyperbolicTangent => {
            hyperbolic_tangent::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::ArctangentStep => {
            arctangent_step::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Softplus => {
            softplus::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Power => {
            power::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Gaussian => {
            gaussian::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Rational11 => {
            rational_11::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Rational22 => {
            rational_22::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        CurveFamily::Emg => {
            // Для EMG аналитический градиент пока не реализован:
            // внешняя логика выполнит численное дифференцирование.
            emg::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::NeedsNumerical
        }
        CurveFamily::PseudoVoigt => {
            pseudo_voigt::accumulate_gradient(
                x_values,
                y_values,
                param,
                &mut loss_derivative_from_prediction,
                gradient,
            );
            GradientComputation::Analytic
        }
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn analytic_hessian<L1, L2>(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    mut loss_derivative_from_prediction: L1,
    mut loss_second_derivative_from_prediction: L2,
) -> Option<Array2<f64>>
where
    L1: FnMut(f64, f64) -> f64,
    L2: FnMut(f64, f64) -> f64,
{
    if family.is_polynomial() {
        return polynomial::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        );
    }

    match family {
        CurveFamily::Arrhenius => arrhenius::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Inverse => inverse::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Logistic => logistic::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Gompertz => gompertz::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::BiExponential => bi_exponential::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::DampedSinusoid => damped_sinusoid::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Lorentzian => lorentzian::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::NaturalLog => natural_log::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::FourPl => four_pl::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::FivePl => five_pl::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::MichaelisMenten => michaelis_menten::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::ExponentialBasic => exponential_basic::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::ExponentialLinear => exponential_linear::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::ExponentialHalfLife => exponential_half_life::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::FallingExponential => falling_exponential::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::HyperbolicTangent => hyperbolic_tangent::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::ArctangentStep => arctangent_step::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Softplus => softplus::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Power => power::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Gaussian => gaussian::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Rational11 => rational_11::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Rational22 => rational_22::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::Emg => emg::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        CurveFamily::PseudoVoigt => pseudo_voigt::analytic_hessian(
            x_values,
            y_values,
            param,
            &mut loss_derivative_from_prediction,
            &mut loss_second_derivative_from_prediction,
        ),
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}
