use super::common::non_zero_param_with_derivative;
use ndarray::Array2;

/// Вычисляет рациональную функцию порядка (2, 2):
/// `f(x) = (num_quad * x^2 + num_linear * x + num_const) / (1 + den_linear * x + den_quad * x^2)`,
/// где:
/// - `num_quad`, `num_linear`, `num_const` — коэффициенты числителя,
/// - `den_linear`, `den_quad` — коэффициенты знаменателя.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let num_quad = param[0];
    let num_linear = param[1];
    let num_const = param[2];
    let den_linear = param[3];
    let den_quad = param[4];
    let x2 = x * x;
    let numerator = num_quad * x2 + num_linear * x + num_const;
    let denominator_raw = 1.0 + den_linear * x + den_quad * x2;
    let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
    numerator / denominator
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
    let num_quad = param[0];
    let num_linear = param[1];
    let num_const = param[2];
    let den_linear = param[3];
    let den_quad = param[4];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let x2 = x * x;
        let numerator = num_quad * x2 + num_linear * x + num_const;
        let denominator_raw = 1.0 + den_linear * x + den_quad * x2;
        let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);
        let model = numerator / denominator;
        let residual = loss.d_prediction(model, y);

        gradient[0] += residual * (x2 / denominator);
        gradient[1] += residual * (x / denominator);
        gradient[2] += residual * (1.0 / denominator);
        gradient[3] += residual * (-numerator * x / (denominator * denominator)) * d_den_raw;
        gradient[4] += residual * (-numerator * x2 / (denominator * denominator)) * d_den_raw;
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
