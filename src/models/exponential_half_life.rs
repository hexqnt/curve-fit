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
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let amplitude = param[1];
    let half_life_raw = param[2];
    let (half_life, _) = positive_param_with_derivative(half_life_raw);
    let exponent = -LN_2 * x / half_life;
    offset + amplitude * exponent.exp()
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
    let offset = param[0];
    let amplitude = param[1];
    let half_life_raw = param[2];
    let (half_life, d_c_raw) = positive_param_with_derivative(half_life_raw);

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let exponent = -LN_2 * x / half_life;
        let pow = exponent.exp();
        let model = offset + amplitude * pow;
        let residual = loss.d_prediction(model, y);
        let d_model_d_c = amplitude * pow * LN_2 * x / (half_life * half_life);

        gradient[0] += residual;
        gradient[1] += residual * pow;
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
