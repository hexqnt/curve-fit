use ndarray::Array2;

/// Вычисляет затухающую синусоиду:
/// `f(x) = amplitude * exp(-damping * x) * sin(omega * x + phi) + offset`,
/// где:
/// - `amplitude` — начальная амплитуда,
/// - `damping` — коэффициент затухания,
/// - `omega` — угловая частота,
/// - `phi` — фазовый сдвиг,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];
    let offset = param[4];
    amplitude * (-damping * x).exp() * (omega * x + phi).sin() + offset
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
    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];
    let offset = param[4];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let exp_part = (-damping * x).exp();
        let angle = omega * x + phi;
        let sin_part = angle.sin();
        let cos_part = angle.cos();
        let model = amplitude * exp_part * sin_part + offset;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * exp_part * sin_part;
        gradient[1] += residual * (-amplitude * x * exp_part * sin_part);
        gradient[2] += residual * (amplitude * exp_part * cos_part * x);
        gradient[3] += residual * (amplitude * exp_part * cos_part);
        gradient[4] += residual;
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
