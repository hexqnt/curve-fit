use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, stabilize_hessian};
use ndarray::Array2;

/// Вычисляет базовую экспоненциальную кривую:
/// `f(x) = offset + amplitude * exp(-decay_rate * x)`,
/// где:
/// - `offset` — базовый уровень,
/// - `amplitude` — амплитуда экспоненциальной части,
/// - `decay_rate` — коэффициент затухания.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];
    offset + amplitude * (-decay_rate * x).exp()
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
    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];
        let exp_part = (-decay_rate * x).exp();
        let model = offset + amplitude * exp_part;
        let residual = loss.d_prediction(model, y);
        gradient[0] += residual;
        gradient[1] += residual * exp_part;
        gradient[2] += residual * (-amplitude * x * exp_part);
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
    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((3, 3));
    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let y = y_values[index];
        let exp_part = (-decay_rate * x).exp();
        let model = offset + amplitude * exp_part;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss.d_prediction(model, y);
        let loss_second = loss.d2_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a = 1.0;
        let jac_b = exp_part;
        let jac_c = -amplitude * x * exp_part;
        let d2_model_dbdc = -x * exp_part;
        let d2_model_dcdc = amplitude * x * x * exp_part;

        hessian[[0, 0]] += loss_second * jac_a * jac_a;
        hessian[[0, 1]] += loss_second * jac_a * jac_b;
        hessian[[0, 2]] += loss_second * jac_a * jac_c;
        hessian[[1, 1]] += loss_second * jac_b * jac_b;
        hessian[[1, 2]] += loss_second * jac_b * jac_c + loss_first * d2_model_dbdc;
        hessian[[2, 2]] += loss_second * jac_c * jac_c + loss_first * d2_model_dcdc;

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
        let value = eval(&[0.2, 1.5, 0.4], 2.0);
        let expected = 0.2 + 1.5 * (-0.8_f64).exp();
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::ExponentialBasic,
            &[-1.0, -0.2, 0.3, 1.1, 2.0],
            &[0.8, 1.4, 0.6],
            &[0.5, 1.1, 0.3],
            2e-5,
            3e-4,
        );
    }
}
