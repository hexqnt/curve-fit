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

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let amplitude = param[0];
    let x0 = param[1];
    let gamma_raw = param[2];
    let (gamma, d_gamma_raw) = positive_param_with_derivative(gamma_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let u = (x - x0) / gamma;
        let den = 1.0 + u * u;
        let inv_den = 1.0 / den;
        let residual = value_first[index];
        let common = 2.0 * amplitude / (den * den * gamma);

        gradient[0] += residual * inv_den;
        gradient[1] += residual * (common * u);
        gradient[2] += residual * (common * u * u) * d_gamma_raw;
        gradient[3] += residual;
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
