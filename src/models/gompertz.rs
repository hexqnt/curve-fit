use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, stabilize_hessian};
use ndarray::Array2;

/// Вычисляет кривую Гомпертца:
/// `f(x) = upper_asymptote * exp(-exp(-growth_rate * (x - x0)))`,
/// где:
/// - `upper_asymptote` — верхняя асимптота,
/// - `growth_rate` — скорость роста,
/// - `x0` — положение точки перегиба.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let upper_asymptote = param[0];
    let growth_rate = param[1];
    let x0 = param[2];
    let inner = (-growth_rate * (x - x0)).exp();
    upper_asymptote * (-inner).exp()
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
    let upper_asymptote = param[0];
    let growth_rate = param[1];
    let x0 = param[2];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let x_centered = x - x0;
        let exp_inner = (-growth_rate * x_centered).exp();
        let exp_outer = (-exp_inner).exp();
        let model = upper_asymptote * exp_outer;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * exp_outer;
        gradient[1] += residual * (upper_asymptote * exp_outer * exp_inner * x_centered);
        gradient[2] += residual * (-upper_asymptote * exp_outer * exp_inner * growth_rate);
        index += 1;
    }
}

pub(super) fn analytic_hessian<L1, L2>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss_derivative_from_prediction: &mut L1,
    loss_second_derivative_from_prediction: &mut L2,
) -> Option<Array2<f64>>
where
    L1: FnMut(f64, f64) -> f64,
    L2: FnMut(f64, f64) -> f64,
{
    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((3, 3));
    let upper_asymptote = param[0];
    let growth_rate = param[1];
    let x0 = param[2];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let y = y_values[index];
        let u = x - x0;
        let exp_inner = (-growth_rate * u).exp();
        let exp_outer = (-exp_inner).exp();
        let exp_product = exp_outer * exp_inner;
        let d2_shape_dz2 = exp_product * (exp_inner - 1.0);
        let model = upper_asymptote * exp_outer;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss_derivative_from_prediction(model, y);
        let loss_second = loss_second_derivative_from_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a = exp_outer;
        let jac_b = upper_asymptote * exp_product * u;
        let jac_c = -upper_asymptote * exp_product * growth_rate;

        let d2_model_dadb = exp_product * u;
        let d2_model_dadc = -exp_product * growth_rate;
        let d2_model_dbdb = upper_asymptote * d2_shape_dz2 * u * u;
        let d2_model_dbdc = -upper_asymptote * (growth_rate * u * d2_shape_dz2 + exp_product);
        let d2_model_dcdc = upper_asymptote * d2_shape_dz2 * growth_rate * growth_rate;

        hessian[[0, 0]] += loss_second * jac_a * jac_a;
        hessian[[0, 1]] += loss_second * jac_a * jac_b + loss_first * d2_model_dadb;
        hessian[[0, 2]] += loss_second * jac_a * jac_c + loss_first * d2_model_dadc;
        hessian[[1, 1]] += loss_second * jac_b * jac_b + loss_first * d2_model_dbdb;
        hessian[[1, 2]] += loss_second * jac_b * jac_c + loss_first * d2_model_dbdc;
        hessian[[2, 2]] += loss_second * jac_c * jac_c + loss_first * d2_model_dcdc;
        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    stabilize_hessian(&mut hessian);
    Some(hessian)
}
