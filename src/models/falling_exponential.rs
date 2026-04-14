use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет кривую экспоненциального спада:
/// `f(x) = y0 - (v0 / k) * (1 - exp(-k * x))`,
/// где:
/// - `y0` — начальный уровень,
/// - `v0` — масштаб скорости спада,
/// - `k` — коэффициент спада (параметризован как ненулевой).
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let y0 = param[0];
    let v0 = param[1];
    let k_raw = param[2];
    let (k, _) = non_zero_param_with_derivative(k_raw);
    let one_minus_exp = -(-k * x).exp_m1();
    y0 - (v0 / k) * one_minus_exp
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let y0 = param[0];
    let v0 = param[1];
    let k_raw = param[2];
    let (k, d_k_raw) = non_zero_param_with_derivative(k_raw);
    let exp_part = (-k * x).exp();
    let one_minus_exp = -(-k * x).exp_m1();
    let d_model_d_v0 = -one_minus_exp / k;
    let d_model_d_k = v0 * (one_minus_exp - k * x * exp_part) / (k * k);

    grad[0] = 1.0;
    grad[1] = d_model_d_v0;
    grad[2] = d_model_d_k * d_k_raw;

    y0 - (v0 / k) * one_minus_exp
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
