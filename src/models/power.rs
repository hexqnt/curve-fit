use super::common::{
    is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian, stabilize_hessian,
};
use ndarray::Array2;

/// Вычисляет степенную зависимость:
/// `f(x) = scale * x^exponent`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `exponent` — показатель степени.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let scale = param[0];
    let exponent = param[1];
    scale * positive_x(x).powf(exponent)
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
    let scale = param[0];
    let exponent = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let pow = x.powf(exponent);
        let model = scale * pow;
        let residual = loss.d_prediction(model, y);
        gradient[0] += residual * pow;
        gradient[1] += residual * scale * pow * x.ln();
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
    let scale = param[0];
    let exponent = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let log_x = x.ln();
        let pow = x.powf(exponent);
        let model = scale * pow;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss.d_prediction(model, y);
        let loss_second = loss.d2_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a = pow;
        let jac_b = scale * pow * log_x;
        let d2_model_dadb = pow * log_x;
        let d2_model_dbdb = scale * pow * log_x * log_x;

        hessian[[0, 0]] += loss_second * jac_a * jac_a;
        hessian[[0, 1]] += loss_second * jac_a * jac_b + loss_first * d2_model_dadb;
        hessian[[1, 1]] += loss_second * jac_b * jac_b + loss_first * d2_model_dbdb;
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
        let value = eval(&[2.0, 1.5], 4.0);
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
