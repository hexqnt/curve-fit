use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет базовую экспоненциальную кривую:
/// `f(x) = offset + amplitude * exp(-decay_rate * x)`,
/// где:
/// - `offset` — базовый уровень,
/// - `amplitude` — амплитуда экспоненциальной части,
/// - `decay_rate` — коэффициент затухания.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];
    offset + amplitude * (-decay_rate * x).exp()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let offset = param[0];
    let amplitude = param[1];
    let decay_rate = param[2];
    let exp_part = (-decay_rate * x).exp();

    grad[0] = 1.0;
    grad[1] = exp_part;
    grad[2] = -amplitude * x * exp_part;

    offset + amplitude * exp_part
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 3];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((3, 3));
    let amplitude = param[1];
    let decay_rate = param[2];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let exp_part = (-decay_rate * x).exp();
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = 1.0;
        let jac_b = exp_part;
        let jac_c = -amplitude * x * exp_part;
        let d2_model_dbdc = -x * exp_part;
        let d2_model_dcdc = amplitude * x * x * exp_part;

        hessian[[0, 0]] += value_second * jac_a * jac_a;
        hessian[[0, 1]] += value_second * jac_a * jac_b;
        hessian[[0, 2]] += value_second * jac_a * jac_c;
        hessian[[1, 1]] += value_second * jac_b * jac_b;
        hessian[[1, 2]] += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
        hessian[[2, 2]] += value_second * jac_c * jac_c + value_first * d2_model_dcdc;

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
        let value = value_at(&[0.2, 1.5, 0.4], 2.0);
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
