use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет кривую Гомпертца:
/// `f(x) = upper_asymptote * exp(-exp(-growth_rate * (x - x0)))`,
/// где:
/// - `upper_asymptote` — верхняя асимптота,
/// - `growth_rate` — скорость роста,
/// - `x0` — положение точки перегиба.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let upper_asymptote = param[0];
    let growth_rate = param[1];
    let x0 = param[2];
    let inner = (-growth_rate * (x - x0)).exp();
    upper_asymptote * (-inner).exp()
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let upper_asymptote = param[0];
    let growth_rate = param[1];
    let x0 = param[2];
    let x_centered = x - x0;
    let exp_inner = (-growth_rate * x_centered).exp();
    let exp_outer = (-exp_inner).exp();

    grad[0] = exp_outer;
    grad[1] = upper_asymptote * exp_outer * exp_inner * x_centered;
    grad[2] = -upper_asymptote * exp_outer * exp_inner * growth_rate;

    upper_asymptote * exp_outer
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
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(x_values.len(), value_second.len());

    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((3, 3)));
    }

    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((3, 3));
    let upper_asymptote = param[0];
    let growth_rate = param[1];
    let x0 = param[2];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let u = x - x0;
        let exp_inner = (-growth_rate * u).exp();
        let exp_outer = (-exp_inner).exp();
        let exp_product = exp_outer * exp_inner;
        let d2_shape_dz2 = exp_product * (exp_inner - 1.0);
        let model = upper_asymptote * exp_outer;
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = exp_outer;
        let jac_b = upper_asymptote * exp_product * u;
        let jac_c = -upper_asymptote * exp_product * growth_rate;

        let d2_model_dadb = exp_product * u;
        let d2_model_dadc = -exp_product * growth_rate;
        let d2_model_dbdb = upper_asymptote * d2_shape_dz2 * u * u;
        let d2_model_dbdc = -upper_asymptote * (growth_rate * u * d2_shape_dz2 + exp_product);
        let d2_model_dcdc = upper_asymptote * d2_shape_dz2 * growth_rate * growth_rate;

        hessian[[0, 0]] += value_second * jac_a * jac_a;
        hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
        hessian[[0, 2]] += value_second * jac_a * jac_c + value_first * d2_model_dadc;
        hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
        hessian[[1, 2]] += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
        hessian[[2, 2]] += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
        index += 1;
    }

    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::{add_value_grad_raw_hessian, value_at};
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[1.7, 0.8, -0.2], 0.3);
        let inner = (-0.8_f64 * 0.5).exp();
        assert_near(value, 1.7 * (-inner).exp(), 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Gompertz,
            &[-1.5, -0.8, -0.2, 0.6, 1.4, 2.3],
            &[1.9, 0.9, 0.2],
            &[1.4, 0.6, -0.2],
            4e-5,
            8e-4,
        );
    }

    #[test]
    fn raw_hessian_is_zero_for_empty_dataset() {
        let hessian = add_value_grad_raw_hessian(&[], &[1.0, 0.8, 0.0], &[], &[])
            .expect("empty dataset must produce zero hessian");
        assert_eq!(hessian.shape(), &[3, 3]);
        assert!(hessian.iter().all(|&value| value == 0.0));
    }
}
