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

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let v0 = param[1];
    let k_raw = param[2];
    let (k, d_k_raw) = non_zero_param_with_derivative(k_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let exp_part = (-k * x).exp();
        let one_minus_exp = -(-k * x).exp_m1();
        let residual = value_first[index];
        let d_model_d_v0 = -one_minus_exp / k;
        let d_model_d_k = v0 * (one_minus_exp - k * x * exp_part) / (k * k);

        gradient[0] += residual;
        gradient[1] += residual * d_model_d_v0;
        gradient[2] += residual * d_model_d_k * d_k_raw;
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
