use super::common::{
    is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian, stabilize_hessian,
};
use ndarray::Array2;

/// Вычисляет обратную зависимость:
/// `f(x) = offset + scale / x`,
/// где:
/// - `offset` — базовый уровень,
/// - `scale` — коэффициент обратной компоненты.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let offset = param[0];
    let scale = param[1];
    offset + scale / positive_x(x)
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
    let scale = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let model = offset + scale / x;
        let residual = loss.d_prediction(model, y);
        gradient[0] += residual;
        gradient[1] += residual / x;
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
    if param.len() != 2 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((2, 2));
    let offset = param[0];
    let scale = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let inv_x = 1.0 / x;
        let model = offset + scale * inv_x;
        if !model.is_finite() {
            return None;
        }

        let weight = loss.d2_prediction(model, y);
        if !is_finite_non_negative(weight) {
            return None;
        }

        hessian[[0, 0]] += weight;
        hessian[[0, 1]] += weight * inv_x;
        hessian[[1, 1]] += weight * inv_x * inv_x;
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
        let value = eval(&[1.25, -0.6], 2.0);
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
}
