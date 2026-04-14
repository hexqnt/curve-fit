use super::common::positive_param_with_derivative;
use ndarray::Array2;

/// Вычисляет гауссову кривую:
/// `f(x) = amplitude * exp(-(x - mean)^2 / (2 * sigma^2))`,
/// где:
/// - `amplitude` — амплитуда,
/// - `mean` — центр пика,
/// - `sigma` — ширина (параметризована положительным преобразованием).
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let mean = param[1];
    let sigma_raw = param[2];
    let (sigma, _) = positive_param_with_derivative(sigma_raw);
    let delta = x - mean;
    amplitude * (-(delta * delta) / (2.0 * sigma * sigma)).exp()
}

pub(super) fn accumulate_gradient<L>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
    gradient: &mut [f64],
) where
    L: super::PredictionLoss,
{
    debug_assert_eq!(x_values.len(), y_values.len());
    let amplitude = param[0];
    let mean = param[1];
    let sigma_raw = param[2];
    let (sigma, d_c_raw) = positive_param_with_derivative(sigma_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let c2 = sigma * sigma;
        let delta = x - mean;
        let exp_part = (-(delta * delta) / (2.0 * c2)).exp();
        let model = amplitude * exp_part;
        let residual = loss.d_prediction(model, y);
        let d_model_d_a = exp_part;
        let d_model_d_b = amplitude * exp_part * delta / c2;
        let d_model_d_c = amplitude * exp_part * delta * delta / (c2 * sigma);

        gradient[0] += residual * d_model_d_a;
        gradient[1] += residual * d_model_d_b;
        gradient[2] += residual * d_model_d_c * d_c_raw;
        index += 1;
    }
}

pub(super) fn analytic_hessian<L>(
    _x_values: &[f64],
    _y_values: &[f64],
    _param: &[f64],
    _loss: &L,
) -> Option<Array2<f64>>
where
    L: super::PredictionLoss,
{
    None
}
