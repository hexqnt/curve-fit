use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет арктангенс-ступень:
/// `f(x) = amplitude * atan(slope * (x - x0)) + offset`,
/// где:
/// - `amplitude` — амплитуда перехода,
/// - `slope` — крутизна перехода,
/// - `x0` — центр перехода по оси `x`,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];
    amplitude * (slope * (x - x0)).atan() + offset
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
        let atan_z = z.atan();
        let inv_den = 1.0 / (1.0 + z * z);
        let residual = value_first[index];

        gradient[0] += residual * atan_z;
        gradient[1] += residual * (amplitude * (x - x0) * inv_den);
        gradient[2] += residual * (-amplitude * slope * inv_den);
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
        let atan_z = z.atan();
        let inv_den = 1.0 / (1.0 + z * z);
        let d2_shape_dz2 = -2.0 * z * inv_den * inv_den;
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = atan_z;
        let jac_b = amplitude * inv_den * u;
        let jac_c = -amplitude * inv_den * slope;
        let jac_d = 1.0;

        let d2_model_dadb = inv_den * u;
        let d2_model_dadc = -inv_den * slope;
        let d2_model_dbdb = amplitude * d2_shape_dz2 * u * u;
        let d2_model_dbdc = amplitude * (d2_shape_dz2 * (-slope) * u - inv_den);
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
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[1.2, 0.5, -0.2, 0.3], 0.8);
        assert_near(value, 1.2 * 0.5_f64.atan() + 0.3, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::ArctangentStep,
            &[-2.2, -1.3, -0.5, 0.2, 1.1, 2.1],
            &[2.1, 0.8, 0.3, 0.4],
            &[1.7, 0.5, -0.2, 0.1],
            4e-5,
            1e-3,
        );
    }
}
