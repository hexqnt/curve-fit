use super::common::{
    is_finite_non_negative, scale_and_mirror_upper_hessian, sigmoid, softplus as math_softplus,
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
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];
    amplitude * math_softplus(slope * (x - x0)) + offset
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let z = slope * (x - x0);
        let softplus_z = math_softplus(z);
        let sigma_z = sigmoid(z);
        let residual = value_first[index];

        gradient[0] += residual * softplus_z;
        gradient[1] += residual * (amplitude * sigma_z * (x - x0));
        gradient[2] += residual * (-amplitude * sigma_z * slope);
        gradient[3] += residual;
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != 4 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((4, 4));
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let u = x - x0;
        let z = slope * u;
        let softplus_z = math_softplus(z);
        let sigma_z = sigmoid(z);
        let d2_shape_dz2 = sigma_z * (1.0 - sigma_z);
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
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

        hessian[[0, 0]] += value_second * jac_a * jac_a;
        hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
        hessian[[0, 2]] += value_second * jac_a * jac_c + value_first * d2_model_dadc;
        hessian[[0, 3]] += value_second * jac_a * jac_d;

        hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
        hessian[[1, 2]] += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
        hessian[[1, 3]] += value_second * jac_b * jac_d;

        hessian[[2, 2]] += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
        hessian[[2, 3]] += value_second * jac_c * jac_d;

        hessian[[3, 3]] += value_second * jac_d * jac_d;
        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::value_at;
    use crate::domain::CurveFamily;
    use crate::models::test_support::assert_near;
    use crate::models::{
        softplus, test_support::assert_family_gradient_and_hessian_match_numerical_reference,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[1.3, 0.6, -0.4, 0.2], 0.6);
        let expected = 1.3 * softplus(0.6) + 0.2;
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Softplus,
            &[-2.0, -1.1, -0.4, 0.3, 1.0, 1.9],
            &[1.3, 0.7, 0.2, 0.2],
            &[1.0, 0.5, -0.1, 0.0],
            4e-5,
            1e-3,
        );
    }
}
