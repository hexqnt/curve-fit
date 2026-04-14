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

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let amplitude = param[0];
    let damping = param[1];
    let omega = param[2];
    let phi = param[3];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let exp_part = (-damping * x).exp();
        let angle = omega * x + phi;
        let sin_part = angle.sin();
        let cos_part = angle.cos();
        let residual = value_first[index];

        gradient[0] += residual * exp_part * sin_part;
        gradient[1] += residual * (-amplitude * x * exp_part * sin_part);
        gradient[2] += residual * (amplitude * exp_part * cos_part * x);
        gradient[3] += residual * (amplitude * exp_part * cos_part);
        gradient[4] += residual;
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
