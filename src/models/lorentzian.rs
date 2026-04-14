use super::common::positive_param_with_derivative;
use ndarray::Array2;

/// Вычисляет лоренцев пик:
/// `f(x) = baseline + amplitude / (1 + ((x - x0) / gamma)^2)`,
/// где:
/// - `amplitude` — амплитуда пика,
/// - `x0` — центр пика,
/// - `gamma` — полуширина (параметризована положительным преобразованием),
/// - `baseline` — базовый уровень.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let x0 = param[1];
    let gamma_raw = param[2];
    let baseline = param[3];
    let (gamma, _) = positive_param_with_derivative(gamma_raw);
    let u = (x - x0) / gamma;
    baseline + amplitude / (1.0 + u * u)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 4);

    let amplitude = param[0];
    let x0 = param[1];
    let gamma_raw = param[2];
    let baseline = param[3];
    let (gamma, d_gamma_raw) = positive_param_with_derivative(gamma_raw);
    let u = (x - x0) / gamma;
    let den = 1.0 + u * u;
    let inv_den = 1.0 / den;
    let common = 2.0 * amplitude / (den * den * gamma);

    grad[0] = inv_den;
    grad[1] = common * u;
    grad[2] = common * u * u * d_gamma_raw;
    grad[3] = 1.0;

    baseline + amplitude * inv_den
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 4];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    _x_values: &[f64],
    _param: &[f64],
    _value_first: &[f64],
    _value_second: &[f64],
) -> Option<Array2<f64>> {
    None
}
