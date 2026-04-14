use super::common::positive_param_with_derivative;
use ndarray::Array2;

const LN_2: f64 = std::f64::consts::LN_2;

/// Вычисляет экспоненциальную модель через период полураспада:
/// `f(x) = offset + amplitude * exp(-ln(2) * x / half_life)`,
/// где:
/// - `offset` — базовый уровень,
/// - `amplitude` — амплитуда экспоненциальной части,
/// - `half_life` — период полураспада (параметризован положительным преобразованием).
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let amplitude = param[1];
    let half_life_raw = param[2];
    let (half_life, _) = positive_param_with_derivative(half_life_raw);
    let exponent = -LN_2 * x / half_life;
    offset + amplitude * exponent.exp()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let offset = param[0];
    let amplitude = param[1];
    let half_life_raw = param[2];
    let (half_life, d_c_raw) = positive_param_with_derivative(half_life_raw);
    let exponent = -LN_2 * x / half_life;
    let pow = exponent.exp();
    let d_model_d_c = amplitude * pow * LN_2 * x / (half_life * half_life);

    grad[0] = 1.0;
    grad[1] = pow;
    grad[2] = d_model_d_c * d_c_raw;

    offset + amplitude * pow
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 3];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
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
