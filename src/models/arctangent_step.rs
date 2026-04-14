use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, stabilize_hessian};
use ndarray::Array2;

/// Вычисляет арктангенс-ступень:
/// `f(x) = amplitude * atan(slope * (x - x0)) + offset`,
/// где:
/// - `amplitude` — амплитуда перехода,
/// - `slope` — крутизна перехода,
/// - `x0` — центр перехода по оси `x`,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];
    amplitude * (slope * (x - x0)).atan() + offset
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
    let amplitude = param[0];
    let slope = param[1];
    let x0 = param[2];
    let offset = param[3];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let z = slope * (x - x0);
        let atan_z = z.atan();
        let inv_den = 1.0 / (1.0 + z * z);
        let model = amplitude * atan_z + offset;
        let residual = loss.d_prediction(model, y);

        gradient[0] += residual * atan_z;
        gradient[1] += residual * (amplitude * (x - x0) * inv_den);
        gradient[2] += residual * (-amplitude * slope * inv_den);
        gradient[3] += residual;
        index += 1;
    }
}

pub(super) fn analytic_hessian<L>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
) -> Option<Array2<f64>>
where
    L: super::PredictionLoss,
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
        let atan_z = z.atan();
        let inv_den = 1.0 / (1.0 + z * z);
        let d2_shape_dz2 = -2.0 * z * inv_den * inv_den;
        let model = amplitude * atan_z + offset;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss.d_prediction(model, y);
        let loss_second = loss.d2_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
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

#[cfg(test)]
mod tests {
    use super::eval;
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = eval(&[1.2, 0.5, -0.2, 0.3], 0.8);
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
