use super::common::{is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет степенную зависимость:
/// `f(x) = scale * x^exponent`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `exponent` — показатель степени.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let scale = param[0];
    let exponent = param[1];
    scale * positive_x(x).powf(exponent)
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let scale = param[0];
    let exponent = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let pow = x.powf(exponent);
        let residual = value_first[index];
        gradient[0] += residual * pow;
        gradient[1] += residual * scale * pow * x.ln();
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != 2 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((2, 2));
    let scale = param[0];
    let exponent = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let log_x = x.ln();
        let pow = x.powf(exponent);
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = pow;
        let jac_b = scale * pow * log_x;
        let d2_model_dadb = pow * log_x;
        let d2_model_dbdb = scale * pow * log_x * log_x;

        hessian[[0, 0]] += value_second * jac_a * jac_a;
        hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
        hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
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
        let value = value_at(&[2.0, 1.5], 4.0);
        assert_near(value, 16.0, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Power,
            &[0.3, 0.8, 1.2, 2.5, 4.0],
            &[1.1, 0.8],
            &[0.8, 0.5],
            3e-5,
            4e-4,
        );
    }
}
