use super::common::positive_param_with_derivative;
use ndarray::Array2;

/// Вычисляет гауссову кривую:
/// `f(x) = amplitude * exp(-(x - mean)^2 / (2 * sigma^2))`,
/// где:
/// - `amplitude` — амплитуда,
/// - `mean` — центр пика,
/// - `sigma` — ширина (параметризована положительным преобразованием).
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let mean = param[1];
    let sigma_raw = param[2];
    let (sigma, _) = positive_param_with_derivative(sigma_raw);
    let delta = x - mean;
    amplitude * (-(delta * delta) / (2.0 * sigma * sigma)).exp()
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let amplitude = param[0];
    let mean = param[1];
    let sigma_raw = param[2];
    let (sigma, d_c_raw) = positive_param_with_derivative(sigma_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let c2 = sigma * sigma;
        let delta = x - mean;
        let exp_part = (-(delta * delta) / (2.0 * c2)).exp();
        let residual = value_first[index];
        let d_model_d_a = exp_part;
        let d_model_d_b = amplitude * exp_part * delta / c2;
        let d_model_d_c = amplitude * exp_part * delta * delta / (c2 * sigma);

        gradient[0] += residual * d_model_d_a;
        gradient[1] += residual * d_model_d_b;
        gradient[2] += residual * d_model_d_c * d_c_raw;
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
