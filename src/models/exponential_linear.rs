use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет экспоненциально-линейную модель:
/// `f(x) = exp_amplitude * exp(exp_rate * x) + linear_slope * x + offset`,
/// где:
/// - `exp_amplitude` — амплитуда экспоненциальной части,
/// - `exp_rate` — показатель экспоненциального роста/затухания,
/// - `linear_slope` — наклон линейной части,
/// - `offset` — свободный член.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let exp_amplitude = param[0];
    let exp_rate = param[1];
    let linear_slope = param[2];
    let offset = param[3];
    exp_amplitude * (exp_rate * x).exp() + linear_slope * x + offset
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 4);

    let exp_amplitude = param[0];
    let exp_rate = param[1];
    let linear_slope = param[2];
    let offset = param[3];
    let exp_part = (exp_rate * x).exp();

    grad[0] = exp_part;
    grad[1] = exp_amplitude * exp_part * x;
    grad[2] = x;
    grad[3] = 1.0;

    exp_amplitude * exp_part + linear_slope * x + offset
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 4];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        gradient[3] += upstream * point_grad[3];
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
    let exp_amplitude = param[0];
    let exp_rate = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let exp_part = (exp_rate * x).exp();
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = exp_part;
        let jac_b = exp_amplitude * x * exp_part;
        let jac_c = x;
        let jac_d = 1.0;
        let d2_model_dadb = x * exp_part;
        let d2_model_dbdb = exp_amplitude * x * x * exp_part;

        hessian[[0, 0]] += value_second * jac_a * jac_a;
        hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
        hessian[[0, 2]] += value_second * jac_a * jac_c;
        hessian[[0, 3]] += value_second * jac_a * jac_d;

        hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
        hessian[[1, 2]] += value_second * jac_b * jac_c;
        hessian[[1, 3]] += value_second * jac_b * jac_d;

        hessian[[2, 2]] += value_second * jac_c * jac_c;
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
        let value = value_at(&[1.2, 0.3, -0.4, 0.1], 2.0);
        let expected = 1.2 * 0.6_f64.exp() - 0.8 + 0.1;
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::ExponentialLinear,
            &[-1.2, -0.5, 0.0, 0.7, 1.4],
            &[1.4, 0.35, -0.4, 0.2],
            &[1.0, 0.2, -0.2, 0.0],
            3e-5,
            6e-4,
        );
    }
}
