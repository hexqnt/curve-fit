use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет кривую экспоненциального спада:
/// `f(x) = y0 - (v0 / k) * (1 - exp(-k * x))`,
/// где:
/// - `y0` — начальный уровень,
/// - `v0` — масштаб скорости спада,
/// - `k` — коэффициент спада (параметризован как ненулевой).
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let y0 = param[0];
    let v0 = param[1];
    let k_raw = param[2];
    let (k, _) = non_zero_param_with_derivative(k_raw);
    let one_minus_exp = -(-k * x).exp_m1();
    y0 - (v0 / k) * one_minus_exp
}

pub(super) fn accumulate_gradient<L>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_derivative_from_prediction: &mut L,
    gradient: &mut [f64],
) where
    L: FnMut(f64, f64) -> f64,
{
    debug_assert_eq!(x_values.len(), y_values.len());
    let y0 = param[0];
    let v0 = param[1];
    let k_raw = param[2];
    let (k, d_k_raw) = non_zero_param_with_derivative(k_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let exp_part = (-k * x).exp();
        let one_minus_exp = -(-k * x).exp_m1();
        let model = y0 - (v0 / k) * one_minus_exp;
        let residual = loss_derivative_from_prediction(model, y);
        let d_model_d_v0 = -one_minus_exp / k;
        let d_model_d_k = v0 * (one_minus_exp - k * x * exp_part) / (k * k);

        gradient[0] += residual;
        gradient[1] += residual * d_model_d_v0;
        gradient[2] += residual * d_model_d_k * d_k_raw;
        index += 1;
    }
}

pub(super) fn analytic_hessian<L1, L2>(
    _x_values: &[f64],
    _y_values: &[f64],
    _param: &[f64],
    _loss_derivative_from_prediction: &mut L1,
    _loss_second_derivative_from_prediction: &mut L2,
) -> Option<Array2<f64>>
where
    L1: FnMut(f64, f64) -> f64,
    L2: FnMut(f64, f64) -> f64,
{
    None
}
