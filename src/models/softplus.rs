use super::common::{
    is_finite_non_negative, scale_and_mirror_upper_hessian, sigmoid, softplus as math_softplus,
    stabilize_hessian,
};
use ndarray::Array2;

/// Вычисляет softplus-переход:
/// `f(x) = amplitude * softplus(slope * (x - x0)) + offset`,
/// где:
/// - `amplitude` — масштаб перехода,
/// - `slope` — крутизна перехода,
/// - `x0` — центр перехода по оси `x`,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];
    amplitude * math_softplus(slope * (x - x0)) + offset
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
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let z = slope * (x - x0);
        let softplus_z = math_softplus(z);
        let sigma_z = sigmoid(z);
        let model = amplitude * softplus_z + offset;
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * softplus_z;
        gradient[1] += residual * (amplitude * sigma_z * (x - x0));
        gradient[2] += residual * (-amplitude * sigma_z * slope);
        gradient[3] += residual;
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
    if param.len() != 4 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((4, 4));
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let y = y_values[index];
        let u = x - x0;
        let z = slope * u;
        let softplus_z = math_softplus(z);
        let sigma_z = sigmoid(z);
        let d2_shape_dz2 = sigma_z * (1.0 - sigma_z);
        let model = amplitude * softplus_z + offset;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss_derivative_from_prediction(model, y);
        let loss_second = loss_second_derivative_from_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a = softplus_z;
        let jac_b = amplitude * sigma_z * u;
        let jac_c = -amplitude * sigma_z * slope;
        let jac_d = 1.0;

        let d2_model_dadb = sigma_z * u;
        let d2_model_dadc = -sigma_z * slope;
        let d2_model_dbdb = amplitude * d2_shape_dz2 * u * u;
        let d2_model_dbdc = -amplitude * (slope * u * d2_shape_dz2 + sigma_z);
        let d2_model_dcdc = amplitude * d2_shape_dz2 * slope * slope;

        hessian[[0, 0]] += loss_second * jac_a * jac_a;
        hessian[[0, 1]] += loss_second * jac_a * jac_b + loss_first * d2_model_dadb;
        hessian[[0, 2]] += loss_second * jac_a * jac_c + loss_first * d2_model_dadc;
        hessian[[0, 3]] += loss_second * jac_a * jac_d;

        hessian[[1, 1]] += loss_second * jac_b * jac_b + loss_first * d2_model_dbdb;
        hessian[[1, 2]] += loss_second * jac_b * jac_c + loss_first * d2_model_dbdc;
        hessian[[1, 3]] += loss_second * jac_b * jac_d;

        hessian[[2, 2]] += loss_second * jac_c * jac_c + loss_first * d2_model_dcdc;
        hessian[[2, 3]] += loss_second * jac_c * jac_d;

        hessian[[3, 3]] += loss_second * jac_d * jac_d;
        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    stabilize_hessian(&mut hessian);
    Some(hessian)
}
