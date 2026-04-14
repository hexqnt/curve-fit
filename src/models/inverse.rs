use super::common::{is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет обратную зависимость:
/// `f(x) = offset + scale / x`,
/// где:
/// - `offset` — базовый уровень,
/// - `scale` — коэффициент обратной компоненты.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let scale = param[1];
    offset + scale / positive_x(x)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 2);

    let offset = param[0];
    let scale = param[1];
    let inv_x = 1.0 / positive_x(x);

    grad[0] = 1.0;
    grad[1] = inv_x;

    offset + scale * inv_x
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = [0.0; 2];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);
        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    _value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    debug_assert_eq!(x_values.len(), value_second.len());

    if param.len() != 2 {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((2, 2)));
    }

    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((2, 2));
    let offset = param[0];
    let scale = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let inv_x = 1.0 / x;
        let model = offset + scale * inv_x;
        if !model.is_finite() {
            return None;
        }

        let weight = value_second[index];
        if !is_finite_non_negative(weight) {
            return None;
        }

        hessian[[0, 0]] += weight;
        hessian[[0, 1]] += weight * inv_x;
        hessian[[1, 1]] += weight * inv_x * inv_x;
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
        let value = value_at(&[1.25, -0.6], 2.0);
        assert_near(value, 0.95, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Inverse,
            &[1.0, 2.0, 4.0, 8.0],
            &[1.0, 0.5],
            &[0.9, 0.3],
            2e-5,
            2e-4,
        );
    }

    #[test]
    fn raw_hessian_is_zero_for_empty_dataset() {
        let hessian = add_value_grad_raw_hessian(&[], &[1.0, 0.5], &[], &[])
            .expect("empty dataset must produce zero hessian");
        assert_eq!(hessian.shape(), &[2, 2]);
        assert!(hessian.iter().all(|&value| value == 0.0));
    }
}
