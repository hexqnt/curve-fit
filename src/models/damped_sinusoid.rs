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
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];
    let offset = param[4];
    amplitude * (-damping * x).exp() * (omega * x + phi).sin() + offset
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 5);

    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];
    let offset = param[4];
    let exp_part = (-damping * x).exp();
    let angle = omega * x + phi;
    let sin_part = angle.sin();
    let cos_part = angle.cos();

    grad[0] = exp_part * sin_part;
    grad[1] = -amplitude * x * exp_part * sin_part;
    grad[2] = amplitude * exp_part * cos_part * x;
    grad[3] = amplitude * exp_part * cos_part;
    grad[4] = 1.0;

    amplitude * exp_part * sin_part + offset
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 5];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
        gradient[4] += upstream * point_grad[4];
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
