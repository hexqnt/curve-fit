use super::common::{
    is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian, stabilize_hessian,
};
use ndarray::Array2;

/// Вычисляет кривую Аррениуса:
/// `f(x) = prefactor * exp(temp_coeff / x)`,
/// где:
/// - `prefactor` — масштабный коэффициент,
/// - `temp_coeff` — параметр температурной чувствительности.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let x = positive_x(x);
    let prefactor = param[0];
    let temp_coeff = param[1];
    prefactor * (temp_coeff / x).exp()
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
    let prefactor = param[0];
    let temp_coeff = param[1];

    let mut index = 0;
    while index < x_values.len() {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let exp_term = (temp_coeff / x).exp();
        let model = prefactor * exp_term;
        let residual = loss.d_prediction(model, y);
        gradient[0] += residual * exp_term;
        gradient[1] += residual * (prefactor * exp_term / x);
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
    let prefactor = param[0];
    let temp_coeff = param[1];

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let y = y_values[index];
        let exp_term = (temp_coeff / x).exp();
        let inv_x = 1.0 / x;
        let model = prefactor * exp_term;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss.d_prediction(model, y);
        let loss_second = loss.d2_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        let jac_a = exp_term;
        let jac_b = prefactor * exp_term * inv_x;
        let d2_model_dadb = exp_term * inv_x;
        let d2_model_dbdb = prefactor * exp_term * inv_x * inv_x;

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
        let value = eval(&[1.2, 0.5], 2.0);
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
