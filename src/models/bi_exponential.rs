use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет сумму двух экспонент:
/// `f(x) = a1 * exp(-k1 * x) + a2 * exp(-k2 * x) + offset`,
/// где:
/// - `a1`, `a2` — амплитуды экспоненциальных компонент,
/// - `k1`, `k2` — коэффициенты затухания компонент,
/// - `offset` — вертикальный сдвиг.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let a1 = param[0];
    let k1 = param[1];
    let a2 = param[2];
    let k2 = param[3];
    let offset = param[4];
    a1 * (-k1 * x).exp() + a2 * (-k2 * x).exp() + offset
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    let a1 = param[0];
    let k1 = param[1];
    let a2 = param[2];
    let k2 = param[3];

    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let exp1 = (-k1 * x).exp();
        let exp2 = (-k2 * x).exp();
        let residual = value_first[index];

        gradient[0] += residual * exp1;
        gradient[1] += residual * (-a1 * x * exp1);
        gradient[2] += residual * exp2;
        gradient[3] += residual * (-a2 * x * exp2);
        gradient[4] += residual;
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != 5 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((5, 5));
    let a1 = param[0];
    let k1 = param[1];
    let a2 = param[2];
    let k2 = param[3];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let exp1 = (-k1 * x).exp();
        let exp2 = (-k2 * x).exp();
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a1 = exp1;
        let jac_k1 = -a1 * x * exp1;
        let jac_a2 = exp2;
        let jac_k2 = -a2 * x * exp2;
        let jac_c = 1.0;

        let d2_model_da1dk1 = -x * exp1;
        let d2_model_dk1dk1 = a1 * x * x * exp1;
        let d2_model_da2dk2 = -x * exp2;
        let d2_model_dk2dk2 = a2 * x * x * exp2;

        hessian[[0, 0]] += value_second * jac_a1 * jac_a1;
        hessian[[0, 1]] += value_second * jac_a1 * jac_k1 + value_first * d2_model_da1dk1;
        hessian[[0, 2]] += value_second * jac_a1 * jac_a2;
        hessian[[0, 3]] += value_second * jac_a1 * jac_k2;
        hessian[[0, 4]] += value_second * jac_a1 * jac_c;

        hessian[[1, 1]] += value_second * jac_k1 * jac_k1 + value_first * d2_model_dk1dk1;
        hessian[[1, 2]] += value_second * jac_k1 * jac_a2;
        hessian[[1, 3]] += value_second * jac_k1 * jac_k2;
        hessian[[1, 4]] += value_second * jac_k1 * jac_c;

        hessian[[2, 2]] += value_second * jac_a2 * jac_a2;
        hessian[[2, 3]] += value_second * jac_a2 * jac_k2 + value_first * d2_model_da2dk2;
        hessian[[2, 4]] += value_second * jac_a2 * jac_c;

        hessian[[3, 3]] += value_second * jac_k2 * jac_k2 + value_first * d2_model_dk2dk2;
        hessian[[3, 4]] += value_second * jac_k2 * jac_c;

        hessian[[4, 4]] += value_second * jac_c * jac_c;
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
        let value = value_at(&[1.0, 0.4, 0.5, 0.2, -0.1], 1.5);
        let expected = (-0.6_f64).exp() + 0.5 * (-0.3_f64).exp() - 0.1;
        assert_near(value, expected, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::BiExponential,
            &[-0.8, -0.1, 0.3, 0.9, 1.8, 2.7],
            &[1.2, 0.7, 0.5, 0.25, -0.3],
            &[0.9, 0.4, 0.4, 0.1, -0.1],
            5e-5,
            2e-3,
        );
    }
}
