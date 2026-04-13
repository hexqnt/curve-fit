use ndarray::Array2;
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    param[0] * (-param[1] * x).exp() * (param[2] * x + param[3]).sin() + param[4]
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

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let exp_part = (-param[1] * x).exp();
        let angle = param[2] * x + param[3];
        let sin_part = angle.sin();
        let cos_part = angle.cos();
        let model = param[0] * exp_part * sin_part + param[4];
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * exp_part * sin_part;
        gradient[1] += residual * (-param[0] * x * exp_part * sin_part);
        gradient[2] += residual * (param[0] * exp_part * cos_part * x);
        gradient[3] += residual * (param[0] * exp_part * cos_part);
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
