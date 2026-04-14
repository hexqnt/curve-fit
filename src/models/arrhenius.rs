use super::common::{is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет кривую Аррениуса:
/// `f(x) = prefactor * exp(temp_coeff / x)`,
/// где:
/// - `prefactor` — масштабный коэффициент,
/// - `temp_coeff` — параметр температурной чувствительности.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let x = positive_x(x);
    let prefactor = param[0];
    let temp_coeff = param[1];
    prefactor * (temp_coeff / x).exp()
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let prefactor = param[0];
    let temp_coeff = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let exp_term = (temp_coeff / x).exp();
        let residual = value_first[index];
        gradient[0] += residual * exp_term;
        gradient[1] += residual * (prefactor * exp_term / x);
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
    let prefactor = param[0];
    let temp_coeff = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let exp_term = (temp_coeff / x).exp();
        let inv_x = 1.0 / x;
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = exp_term;
        let jac_b = prefactor * exp_term * inv_x;
        let d2_model_dadb = exp_term * inv_x;
        let d2_model_dbdb = prefactor * exp_term * inv_x * inv_x;

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
        let value = value_at(&[1.2, 0.5], 2.0);
        assert_near(value, 1.2 * (0.25_f64).exp(), 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Arrhenius,
            &[0.4, 0.8, 1.4, 2.5, 4.0],
            &[1.5, 0.9],
            &[1.2, 0.5],
            2e-5,
            3e-4,
        );
    }
}
