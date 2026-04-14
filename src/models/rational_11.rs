use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет рациональную функцию порядка (1, 1):
/// `f(x) = offset + (num_slope * x + num_offset) / (1 + den_slope * x)`,
/// где:
/// - `num_slope`, `num_offset` — коэффициенты числителя,
/// - `den_slope` — коэффициент знаменателя,
/// - `offset` — вертикальный сдвиг.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let num_slope = param[0];
    let num_offset = param[1];
    let den_slope = param[2];
    let offset = param[3];
    let numerator = num_slope * x + num_offset;
    let denominator_raw = 1.0 + den_slope * x;
    let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
    offset + numerator / denominator
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
    let num_slope = param[0];
    let num_offset = param[1];
    let den_slope = param[2];
    let offset = param[3];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let numerator = num_slope * x + num_offset;
        let denominator_raw = 1.0 + den_slope * x;
        let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);
        let model = offset + numerator / denominator;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * (x / denominator);
        gradient[1] += residual * (1.0 / denominator);
        gradient[2] += residual * (-numerator * x / (denominator * denominator)) * d_den_raw;
        gradient[3] += residual;
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
