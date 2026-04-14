use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет модель Михаэлиса-Ментен:
/// `f(x) = vmax * x / (x + km)`,
/// где:
/// - `vmax` — максимальная скорость,
/// - `km` — константа Михаэлиса.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let vmax = param[0];
    let km = param[1];
    let (denominator, _) = non_zero_param_with_derivative(x + km);
    vmax * x / denominator
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
    let vmax = param[0];
    let km = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let (denominator, d_den_d_km) = non_zero_param_with_derivative(x + km);
        let model = vmax * x / denominator;
        let residual = loss_derivative_from_prediction(model, y);
        let d_model_d_vmax = x / denominator;
        let d_model_d_km = -vmax * x / (denominator * denominator) * d_den_d_km;

        gradient[0] += residual * d_model_d_vmax;
        gradient[1] += residual * d_model_d_km;
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
