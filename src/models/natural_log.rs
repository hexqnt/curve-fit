use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;

/// Вычисляет логарифмическую зависимость:
/// `f(x) = scale * ln(x / x_scale)`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `x_scale` — масштаб по оси `x` (параметризован положительным преобразованием).
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let scale = param[0];
    let x_scale_raw = param[1];
    let x = positive_x(x);
    let (x_scale, _) = positive_param_with_derivative(x_scale_raw);
    scale * (x / x_scale).ln()
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
    let scale = param[0];
    let x_scale_raw = param[1];
    let (x_scale, d_b_raw) = positive_param_with_derivative(x_scale_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let ln_term = (x / x_scale).ln();
        let model = scale * ln_term;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * ln_term;
        gradient[1] += residual * (-scale / x_scale) * d_b_raw;
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
