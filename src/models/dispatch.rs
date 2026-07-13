use crate::domain::{CurveFamily, DEFAULT_SATURATING_TREND_TAUS_YEARS};

use super::{
    Grad, Hessian, Param, PredictionLoss, arctangent_step, arrhenius, bi_exponential,
    damped_sinusoid, emg, exponential_basic, exponential_half_life, exponential_linear,
    falling_exponential, five_pl, four_pl, gaussian, gompertz, hyperbolic_tangent, inverse,
    logistic, lorentzian, michaelis_menten, natural_log, polynomial, power, pseudo_voigt,
    rational_11, rational_22, rational_nn, saturating_trend_basis, softplus,
};

const MAX_MODEL_PARAM_COUNT: usize = 11;
type ValueKernel = fn(&Param, f64) -> f64;
type ValueGradKernel = fn(&Param, f64, &mut [f64]) -> f64;

/// Выбранное один раз ядро модели для пакетной обработки точек.
#[derive(Clone, Copy)]
enum PointKernel {
    Plain {
        value: ValueKernel,
        value_grad: ValueGradKernel,
    },
    Saturating,
}

impl PointKernel {
    fn for_family(family: CurveFamily) -> Self {
        if family.is_polynomial() {
            return Self::Plain {
                value: polynomial::value_at,
                value_grad: polynomial::value_grad_at,
            };
        }

        let (value, value_grad): (ValueKernel, ValueGradKernel) = match family {
            CurveFamily::Arrhenius => (arrhenius::value_at, arrhenius::value_grad_at),
            CurveFamily::Inverse => (inverse::value_at, inverse::value_grad_at),
            CurveFamily::Logistic => (logistic::value_at, logistic::value_grad_at),
            CurveFamily::Gompertz => (gompertz::value_at, gompertz::value_grad_at),
            CurveFamily::BiExponential => (bi_exponential::value_at, bi_exponential::value_grad_at),
            CurveFamily::DampedSinusoid => {
                (damped_sinusoid::value_at, damped_sinusoid::value_grad_at)
            }
            CurveFamily::Lorentzian => (lorentzian::value_at, lorentzian::value_grad_at),
            CurveFamily::NaturalLog => (natural_log::value_at, natural_log::value_grad_at),
            CurveFamily::FourPl => (four_pl::value_at, four_pl::value_grad_at),
            CurveFamily::FivePl => (five_pl::value_at, five_pl::value_grad_at),
            CurveFamily::MichaelisMenten => {
                (michaelis_menten::value_at, michaelis_menten::value_grad_at)
            }
            CurveFamily::ExponentialBasic => (
                exponential_basic::value_at,
                exponential_basic::value_grad_at,
            ),
            CurveFamily::ExponentialLinear => (
                exponential_linear::value_at,
                exponential_linear::value_grad_at,
            ),
            CurveFamily::ExponentialHalfLife => (
                exponential_half_life::value_at,
                exponential_half_life::value_grad_at,
            ),
            CurveFamily::FallingExponential => (
                falling_exponential::value_at,
                falling_exponential::value_grad_at,
            ),
            CurveFamily::HyperbolicTangent => (
                hyperbolic_tangent::value_at,
                hyperbolic_tangent::value_grad_at,
            ),
            CurveFamily::ArctangentStep => {
                (arctangent_step::value_at, arctangent_step::value_grad_at)
            }
            CurveFamily::Softplus => (softplus::value_at, softplus::value_grad_at),
            CurveFamily::Power => (power::value_at, power::value_grad_at),
            CurveFamily::Gaussian => (gaussian::value_at, gaussian::value_grad_at),
            CurveFamily::Rational11 => (rational_11::value_at, rational_11::value_grad_at),
            CurveFamily::Rational22 => (rational_22::value_at, rational_22::value_grad_at),
            CurveFamily::Rational33 | CurveFamily::Rational44 | CurveFamily::Rational55 => {
                (rational_nn::value_at, rational_nn::value_grad_at)
            }
            CurveFamily::Emg => (emg::value_at, emg::value_grad_at),
            CurveFamily::PseudoVoigt => (pseudo_voigt::value_at, pseudo_voigt::value_grad_at),
            CurveFamily::SaturatingTrendBasis1
            | CurveFamily::SaturatingTrendBasis2
            | CurveFamily::SaturatingTrendBasis3
            | CurveFamily::SaturatingTrendBasis4
            | CurveFamily::SaturatingTrendBasis5
            | CurveFamily::SaturatingTrendBasis6 => return Self::Saturating,
            _ => unreachable!("Polynomial families are handled by the guarded branch above"),
        };
        Self::Plain { value, value_grad }
    }

    #[inline]
    fn value(
        self,
        family: CurveFamily,
        param: &Param,
        x: f64,
        saturating_trend_taus: Option<&[f64]>,
    ) -> f64 {
        match self {
            Self::Plain { value, .. } => value(param, x),
            Self::Saturating => saturating_trend_basis::value_at(
                param,
                x,
                saturating_trend_taus_for_family(family, saturating_trend_taus),
            ),
        }
    }

    #[inline]
    fn value_grad(
        self,
        family: CurveFamily,
        param: &Param,
        x: f64,
        saturating_trend_taus: Option<&[f64]>,
        grad: &mut [f64],
    ) -> f64 {
        match self {
            Self::Plain { value_grad, .. } => value_grad(param, x, grad),
            Self::Saturating => saturating_trend_basis::value_grad_at(
                param,
                x,
                saturating_trend_taus_for_family(family, saturating_trend_taus),
                grad,
            ),
        }
    }
}

#[inline]
fn saturating_trend_taus_for_family(
    family: CurveFamily,
    saturating_trend_taus: Option<&[f64]>,
) -> &[f64] {
    if let Some(taus) = saturating_trend_taus {
        return taus;
    }

    let count = family
        .saturating_trend_tau_count()
        .expect("only saturating trend families request tau defaults");
    &DEFAULT_SATURATING_TREND_TAUS_YEARS[..count]
}

#[cfg(test)]
#[inline]
pub(crate) fn value_at(family: CurveFamily, param: &Param, x: f64) -> f64 {
    value_at_with_saturating_taus(family, param, x, None)
}

#[inline]
pub(crate) fn value_at_with_saturating_taus(
    family: CurveFamily,
    param: &Param,
    x: f64,
    saturating_trend_taus: Option<&[f64]>,
) -> f64 {
    PointKernel::for_family(family).value(family, param, x, saturating_trend_taus)
}

pub(crate) fn objective_value(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    saturating_trend_taus: Option<&[f64]>,
    loss: &dyn PredictionLoss,
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }

    let kernel = PointKernel::for_family(family);
    let mut sum = 0.0;
    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let prediction = kernel.value(family, param, x, saturating_trend_taus);
        let contribution = loss.value(prediction, y);
        if !contribution.is_finite() {
            return f64::INFINITY;
        }
        sum += contribution;
        if !sum.is_finite() {
            return f64::INFINITY;
        }
    }

    sum / x_values.len() as f64
}

#[inline]
pub(crate) fn has_analytic_grad(family: CurveFamily) -> bool {
    !matches!(family, CurveFamily::Emg)
}

pub(crate) fn objective_value_grad_analytic(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    saturating_trend_taus: Option<&[f64]>,
    loss: &dyn PredictionLoss,
) -> Option<(f64, Grad)> {
    debug_assert_eq!(x_values.len(), y_values.len());

    if !has_analytic_grad(family) {
        return None;
    }

    let mut gradient = vec![0.0; param.len()];
    if x_values.is_empty() {
        return Some((0.0, gradient));
    }

    debug_assert!(param.len() <= MAX_MODEL_PARAM_COUNT);
    let kernel = PointKernel::for_family(family);
    let mut point_gradient = [0.0; MAX_MODEL_PARAM_COUNT];
    let mut value = 0.0;
    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let prediction = kernel.value_grad(
            family,
            param,
            x,
            saturating_trend_taus,
            &mut point_gradient[..param.len()],
        );
        let contribution = loss.value(prediction, y);
        let derivative = loss.d_prediction(prediction, y);
        if !contribution.is_finite() || !derivative.is_finite() {
            return None;
        }
        value += contribution;
        for (accumulator, point_derivative) in
            gradient.iter_mut().zip(point_gradient.iter().copied())
        {
            let next = *accumulator + derivative * point_derivative;
            if !next.is_finite() {
                return None;
            }
            *accumulator = next;
        }
        if !value.is_finite() {
            return None;
        }
    }

    let sample_scale = 1.0 / x_values.len() as f64;
    value *= sample_scale;
    for gradient_value in &mut gradient {
        *gradient_value *= sample_scale;
    }

    Some((value, gradient))
}

/// Возвращает raw-гессиан параметров модели из внешних производных по предсказанию:
/// `value_first = dF/dŷ`, `value_second = d²F/dŷ²`.
pub(crate) fn model_raw_hessian_from_value_derivatives(
    family: CurveFamily,
    x_values: &[f64],
    param: &Param,
    saturating_trend_taus: Option<&[f64]>,
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Hessian> {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(x_values.len(), value_second.len());

    if x_values.is_empty() {
        return Some(Hessian::zeros((param.len(), param.len())));
    }

    if family.is_polynomial() {
        return polynomial::add_value_grad_raw_hessian(x_values, param, value_first, value_second);
    }

    match family {
        CurveFamily::Arrhenius => {
            arrhenius::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Inverse => {
            inverse::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Logistic => {
            logistic::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Gompertz => {
            gompertz::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::BiExponential => {
            bi_exponential::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::DampedSinusoid => {
            damped_sinusoid::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Lorentzian => {
            lorentzian::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::NaturalLog => {
            natural_log::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::FourPl => {
            four_pl::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::FivePl => {
            five_pl::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::MichaelisMenten => {
            michaelis_menten::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::ExponentialBasic => exponential_basic::add_value_grad_raw_hessian(
            x_values,
            param,
            value_first,
            value_second,
        ),
        CurveFamily::ExponentialLinear => exponential_linear::add_value_grad_raw_hessian(
            x_values,
            param,
            value_first,
            value_second,
        ),
        CurveFamily::ExponentialHalfLife => exponential_half_life::add_value_grad_raw_hessian(
            x_values,
            param,
            value_first,
            value_second,
        ),
        CurveFamily::FallingExponential => falling_exponential::add_value_grad_raw_hessian(
            x_values,
            param,
            value_first,
            value_second,
        ),
        CurveFamily::HyperbolicTangent => hyperbolic_tangent::add_value_grad_raw_hessian(
            x_values,
            param,
            value_first,
            value_second,
        ),
        CurveFamily::ArctangentStep => {
            arctangent_step::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Softplus => {
            softplus::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Power => {
            power::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Gaussian => {
            gaussian::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Rational11 => {
            rational_11::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Rational22 => {
            rational_22::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Rational33 | CurveFamily::Rational44 | CurveFamily::Rational55 => {
            rational_nn::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::Emg => {
            emg::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::PseudoVoigt => {
            pseudo_voigt::add_value_grad_raw_hessian(x_values, param, value_first, value_second)
        }
        CurveFamily::SaturatingTrendBasis1
        | CurveFamily::SaturatingTrendBasis2
        | CurveFamily::SaturatingTrendBasis3
        | CurveFamily::SaturatingTrendBasis4
        | CurveFamily::SaturatingTrendBasis5
        | CurveFamily::SaturatingTrendBasis6 => saturating_trend_basis::add_value_grad_raw_hessian(
            x_values,
            param,
            saturating_trend_taus_for_family(family, saturating_trend_taus),
            value_first,
            value_second,
        ),
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

#[allow(dead_code)]
pub(crate) fn objective_raw_hessian_analytic(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    saturating_trend_taus: Option<&[f64]>,
    loss: &dyn PredictionLoss,
) -> Option<Hessian> {
    debug_assert_eq!(x_values.len(), y_values.len());

    let mut value_first = vec![0.0; x_values.len()];
    let mut value_second = vec![0.0; x_values.len()];
    let kernel = PointKernel::for_family(family);

    for ((&x, &y), (first_out, second_out)) in x_values
        .iter()
        .zip(y_values.iter())
        .zip(value_first.iter_mut().zip(value_second.iter_mut()))
    {
        let prediction = kernel.value(family, param, x, saturating_trend_taus);
        let first_derivative = loss.d_prediction(prediction, y);
        let second_derivative = loss.d2_prediction(prediction, y);
        if !first_derivative.is_finite() || !second_derivative.is_finite() {
            return None;
        }
        *first_out = first_derivative;
        *second_out = second_derivative;
    }

    model_raw_hessian_from_value_derivatives(
        family,
        x_values,
        param,
        saturating_trend_taus,
        &value_first,
        &value_second,
    )
}

pub(crate) fn objective_value_grad_raw_hessian_analytic(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    saturating_trend_taus: Option<&[f64]>,
    loss: &dyn PredictionLoss,
) -> Option<(f64, Grad, Hessian)> {
    debug_assert_eq!(x_values.len(), y_values.len());

    if !has_analytic_grad(family) {
        return None;
    }
    if x_values.is_empty() {
        return Some((
            0.0,
            vec![0.0; param.len()],
            Hessian::zeros((param.len(), param.len())),
        ));
    }

    debug_assert!(param.len() <= MAX_MODEL_PARAM_COUNT);
    let kernel = PointKernel::for_family(family);
    let mut point_gradient = [0.0; MAX_MODEL_PARAM_COUNT];
    let mut gradient = vec![0.0; param.len()];
    let mut value_first = vec![0.0; x_values.len()];
    let mut value_second = vec![0.0; x_values.len()];
    let mut value = 0.0;

    for (index, (&x, &y)) in x_values.iter().zip(y_values.iter()).enumerate() {
        let prediction = kernel.value_grad(
            family,
            param,
            x,
            saturating_trend_taus,
            &mut point_gradient[..param.len()],
        );
        let contribution = loss.value(prediction, y);
        let first = loss.d_prediction(prediction, y);
        let second = loss.d2_prediction(prediction, y);
        if !contribution.is_finite() || !first.is_finite() || !second.is_finite() {
            return None;
        }
        value += contribution;
        value_first[index] = first;
        value_second[index] = second;
        for (accumulator, point_derivative) in
            gradient.iter_mut().zip(point_gradient.iter().copied())
        {
            let next = *accumulator + first * point_derivative;
            if !next.is_finite() {
                return None;
            }
            *accumulator = next;
        }
        if !value.is_finite() {
            return None;
        }
    }

    let sample_scale = 1.0 / x_values.len() as f64;
    value *= sample_scale;
    for gradient_value in &mut gradient {
        *gradient_value *= sample_scale;
    }
    let hessian = model_raw_hessian_from_value_derivatives(
        family,
        x_values,
        param,
        saturating_trend_taus,
        &value_first,
        &value_second,
    )?;
    Some((value, gradient, hessian))
}
