use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, stabilize_hessian};
use ndarray::Array2;

#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    param[0] * (-param[1] * x).exp() + param[2] * (-param[3] * x).exp() + param[4]
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
        let exp1 = (-param[1] * x).exp();
        let exp2 = (-param[3] * x).exp();
        let model = param[0] * exp1 + param[2] * exp2 + param[4];
        let residual = loss_derivative_from_prediction(model, y);

        gradient[0] += residual * exp1;
        gradient[1] += residual * (-param[0] * x * exp1);
        gradient[2] += residual * exp2;
        gradient[3] += residual * (-param[2] * x * exp2);
        gradient[4] += residual;
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
    if param.len() != 5 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((5, 5));

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let y = y_values[index];
        let exp1 = (-param[1] * x).exp();
        let exp2 = (-param[3] * x).exp();
        let model = param[0] * exp1 + param[2] * exp2 + param[4];
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss_derivative_from_prediction(model, y);
        let loss_second = loss_second_derivative_from_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a1 = exp1;
        let jac_k1 = -param[0] * x * exp1;
        let jac_a2 = exp2;
        let jac_k2 = -param[2] * x * exp2;
        let jac_c = 1.0;

        let d2_model_da1dk1 = -x * exp1;
        let d2_model_dk1dk1 = param[0] * x * x * exp1;
        let d2_model_da2dk2 = -x * exp2;
        let d2_model_dk2dk2 = param[2] * x * x * exp2;

        hessian[[0, 0]] += loss_second * jac_a1 * jac_a1;
        hessian[[0, 1]] += loss_second * jac_a1 * jac_k1 + loss_first * d2_model_da1dk1;
        hessian[[0, 2]] += loss_second * jac_a1 * jac_a2;
        hessian[[0, 3]] += loss_second * jac_a1 * jac_k2;
        hessian[[0, 4]] += loss_second * jac_a1 * jac_c;

        hessian[[1, 1]] += loss_second * jac_k1 * jac_k1 + loss_first * d2_model_dk1dk1;
        hessian[[1, 2]] += loss_second * jac_k1 * jac_a2;
        hessian[[1, 3]] += loss_second * jac_k1 * jac_k2;
        hessian[[1, 4]] += loss_second * jac_k1 * jac_c;

        hessian[[2, 2]] += loss_second * jac_a2 * jac_a2;
        hessian[[2, 3]] += loss_second * jac_a2 * jac_k2 + loss_first * d2_model_da2dk2;
        hessian[[2, 4]] += loss_second * jac_a2 * jac_c;

        hessian[[3, 3]] += loss_second * jac_k2 * jac_k2 + loss_first * d2_model_dk2dk2;
        hessian[[3, 4]] += loss_second * jac_k2 * jac_c;

        hessian[[4, 4]] += loss_second * jac_c * jac_c;
        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    stabilize_hessian(&mut hessian);
    Some(hessian)
}
