use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian};
use ndarray::Array2;

/// Вычисляет полином в форме Горнера:
/// `f(x) = p0 * x^n + p1 * x^(n-1) + ... + pn`,
/// где `param = [p0, p1, ..., pn]` — коэффициенты от старшей степени к младшей.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    param
        .iter()
        .copied()
        .fold(0.0, |acc, coefficient| acc * x + coefficient)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), param.len());

    let value = value_at(param, x);
    let mut basis = 1.0;
    for grad_value in grad.iter_mut().rev() {
        *grad_value = basis;
        basis *= x;
    }

    value
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    let mut point_grad = vec![0.0; gradient.len()];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        for (gradient_value, point_grad_value) in gradient.iter_mut().zip(point_grad.iter()) {
            *gradient_value += upstream * point_grad_value;
        }
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    _value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    let dimension = param.len();
    if dimension == 0 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((dimension, dimension));
    let mut basis = vec![0.0; dimension];

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let model = value_at(param, x);
        if !model.is_finite() {
            return None;
        }

        let weight = value_second[index];
        if !is_finite_non_negative(weight) {
            return None;
        }

        let mut basis_index = dimension;
        let mut power = 1.0;
        while basis_index > 0 {
            basis_index -= 1;
            basis[basis_index] = power;
            power *= x;
        }

        let mut row = 0;
        while row < dimension {
            let basis_row = basis[row];
            let mut column = row;
            while column < dimension {
                hessian[[row, column]] += weight * basis_row * basis[column];
                column += 1;
            }
            row += 1;
        }

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
        let value = value_at(&[2.0, -3.0, 1.0], 4.0);
        assert_near(value, 21.0, 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Linear,
            &[-1.0, 0.0, 2.0, 3.5],
            &[1.5, -0.25],
            &[0.3, -0.7],
            2e-5,
            2e-4,
        );
    }
}
