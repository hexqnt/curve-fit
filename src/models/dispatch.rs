use crate::domain::CurveFamily;

use super::{
    Grad, Hessian, Param, PredictionLoss, arctangent_step, arrhenius, bi_exponential,
    damped_sinusoid, emg, exponential_basic, exponential_half_life, exponential_linear,
    falling_exponential, five_pl, four_pl, gaussian, gompertz, hyperbolic_tangent, inverse,
    logistic, lorentzian, michaelis_menten, natural_log, polynomial, power, pseudo_voigt,
    rational_11, rational_22, rational_nn, softplus,
};

#[inline]
pub(crate) fn value_at(family: CurveFamily, param: &Param, x: f64) -> f64 {
    if family.is_polynomial() {
        return polynomial::value_at(param, x);
    }

    match family {
        CurveFamily::Arrhenius => arrhenius::value_at(param, x),
        CurveFamily::Inverse => inverse::value_at(param, x),
        CurveFamily::Logistic => logistic::value_at(param, x),
        CurveFamily::Gompertz => gompertz::value_at(param, x),
        CurveFamily::BiExponential => bi_exponential::value_at(param, x),
        CurveFamily::DampedSinusoid => damped_sinusoid::value_at(param, x),
        CurveFamily::Lorentzian => lorentzian::value_at(param, x),
        CurveFamily::NaturalLog => natural_log::value_at(param, x),
        CurveFamily::FourPl => four_pl::value_at(param, x),
        CurveFamily::FivePl => five_pl::value_at(param, x),
        CurveFamily::MichaelisMenten => michaelis_menten::value_at(param, x),
        CurveFamily::ExponentialBasic => exponential_basic::value_at(param, x),
        CurveFamily::ExponentialLinear => exponential_linear::value_at(param, x),
        CurveFamily::ExponentialHalfLife => exponential_half_life::value_at(param, x),
        CurveFamily::FallingExponential => falling_exponential::value_at(param, x),
        CurveFamily::HyperbolicTangent => hyperbolic_tangent::value_at(param, x),
        CurveFamily::ArctangentStep => arctangent_step::value_at(param, x),
        CurveFamily::Softplus => softplus::value_at(param, x),
        CurveFamily::Power => power::value_at(param, x),
        CurveFamily::Gaussian => gaussian::value_at(param, x),
        CurveFamily::Rational11 => rational_11::value_at(param, x),
        CurveFamily::Rational22 => rational_22::value_at(param, x),
        CurveFamily::Rational33 | CurveFamily::Rational44 | CurveFamily::Rational55 => {
            rational_nn::value_at(param, x)
        }
        CurveFamily::Emg => emg::value_at(param, x),
        CurveFamily::PseudoVoigt => pseudo_voigt::value_at(param, x),
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn objective_value(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    loss: &dyn PredictionLoss,
) -> f64 {
    debug_assert_eq!(x_values.len(), y_values.len());
    if x_values.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut index = 0;
    while index < x_values.len() {
        let prediction = value_at(family, param, x_values[index]);
        let contribution = loss.value(prediction, y_values[index]);
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

#[inline]
pub(crate) fn has_analytic_grad(family: CurveFamily) -> bool {
    !matches!(family, CurveFamily::Emg)
}

/// Добавляет несмасштабированный вклад в градиент параметров модели:
/// `dF/dθ = (dF/dŷ) * (dŷ/dθ)`.
///
/// Здесь `value_first` — внешняя производная `dF/dŷ` по каждой точке.
/// Это позволяет использовать один и тот же model-kernel как для loss,
/// так и для произвольного downstream-звена в цепочке.
pub(crate) fn add_model_grad_unscaled(
    family: CurveFamily,
    x_values: &[f64],
    param: &Param,
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    if family.is_polynomial() {
        polynomial::add_value_grad(x_values, param, value_first, gradient);
        return;
    }

    match family {
        CurveFamily::Arrhenius => arrhenius::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::Inverse => inverse::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::Logistic => logistic::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::Gompertz => gompertz::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::BiExponential => {
            bi_exponential::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::DampedSinusoid => {
            damped_sinusoid::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::Lorentzian => {
            lorentzian::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::NaturalLog => {
            natural_log::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::FourPl => four_pl::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::FivePl => five_pl::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::MichaelisMenten => {
            michaelis_menten::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::ExponentialBasic => {
            exponential_basic::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::ExponentialLinear => {
            exponential_linear::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::ExponentialHalfLife => {
            exponential_half_life::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::FallingExponential => {
            falling_exponential::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::HyperbolicTangent => {
            hyperbolic_tangent::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::ArctangentStep => {
            arctangent_step::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::Softplus => softplus::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::Power => power::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::Gaussian => gaussian::add_value_grad(x_values, param, value_first, gradient),
        CurveFamily::Rational11 => {
            rational_11::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::Rational22 => {
            rational_22::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::Rational33 | CurveFamily::Rational44 | CurveFamily::Rational55 => {
            rational_nn::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::PseudoVoigt => {
            pseudo_voigt::add_value_grad(x_values, param, value_first, gradient)
        }
        CurveFamily::Emg => emg::add_value_grad(x_values, param, value_first, gradient),
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn objective_value_grad_analytic(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    loss: &dyn PredictionLoss,
) -> Option<(f64, Grad)> {
    debug_assert_eq!(x_values.len(), y_values.len());

    if !has_analytic_grad(family) {
        return None;
    }

    let value = objective_value(family, x_values, y_values, param, loss);
    let mut gradient = vec![0.0; param.len()];
    if !x_values.is_empty() {
        // dF/dy_hat для каждой точки, где F — вклад функции потерь.
        // Вычисляем отдельно от model-kernel'а:
        // это сохраняет разделение ответственности model vs loss.
        let mut value_first = vec![0.0; x_values.len()];
        let mut index = 0;
        while index < x_values.len() {
            let prediction = value_at(family, param, x_values[index]);
            let derivative = loss.d_prediction(prediction, y_values[index]);
            if !derivative.is_finite() {
                return None;
            }
            value_first[index] = derivative;
            index += 1;
        }

        add_model_grad_unscaled(family, x_values, param, &value_first, &mut gradient);
        let sample_scale = 1.0 / x_values.len() as f64;
        for gradient_value in &mut gradient {
            *gradient_value *= sample_scale;
        }
    }

    Some((value, gradient))
}

/// Возвращает raw-гессиан параметров модели из внешних производных по предсказанию:
/// `value_first = dF/dŷ`, `value_second = d²F/dŷ²`.
pub(crate) fn model_raw_hessian_from_value_derivatives(
    family: CurveFamily,
    x_values: &[f64],
    param: &Param,
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
        _ => unreachable!("Polynomial families are handled by the guarded branch above"),
    }
}

pub(crate) fn objective_raw_hessian_analytic(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    loss: &dyn PredictionLoss,
) -> Option<Hessian> {
    debug_assert_eq!(x_values.len(), y_values.len());

    let mut value_first = vec![0.0; x_values.len()];
    let mut value_second = vec![0.0; x_values.len()];

    let mut index = 0;
    while index < x_values.len() {
        let prediction = value_at(family, param, x_values[index]);
        let first_derivative = loss.d_prediction(prediction, y_values[index]);
        let second_derivative = loss.d2_prediction(prediction, y_values[index]);
        if !first_derivative.is_finite() || !second_derivative.is_finite() {
            return None;
        }
        value_first[index] = first_derivative;
        value_second[index] = second_derivative;
        index += 1;
    }

    model_raw_hessian_from_value_derivatives(family, x_values, param, &value_first, &value_second)
}

pub(crate) fn objective_value_grad_raw_hessian_analytic(
    family: CurveFamily,
    x_values: &[f64],
    y_values: &[f64],
    param: &Param,
    loss: &dyn PredictionLoss,
) -> Option<(f64, Grad, Hessian)> {
    debug_assert_eq!(x_values.len(), y_values.len());

    let (value, gradient) = objective_value_grad_analytic(family, x_values, y_values, param, loss)?;
    let hessian = objective_raw_hessian_analytic(family, x_values, y_values, param, loss)?;
    Some((value, gradient, hessian))
}
