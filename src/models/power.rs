use super::common::{is_finite_non_negative, positive_x, scale_and_mirror_upper_hessian};
use ndarray::Array2;

const PARAM_COUNT: usize = 2;

#[derive(Clone, Copy)]
struct Params<T> {
    scale: T,
    exponent: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [scale, exponent]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self { scale, exponent }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        self.scale * positive_x(x).powf(self.exponent)
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let x = positive_x(x);
        let pow = x.powf(self.exponent);

        grad[0] = pow;
        grad[1] = self.scale * pow * x.ln();

        self.scale * pow
    }
}

/// Вычисляет степенную зависимость:
/// `f(x) = scale * x^exponent`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `exponent` — показатель степени.
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    Params::parse(param).value_at(x)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    Params::parse(param).value_grad_at(x, grad)
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());
    let params = Params::parse(param);

    let mut point_grad = [0.0; PARAM_COUNT];
    for (&x, &upstream) in x_values.iter().zip(value_first.iter()) {
        params.value_grad_at(x, &mut point_grad);

        for (gradient_value, point_grad_value) in gradient.iter_mut().zip(point_grad.iter()) {
            *gradient_value += upstream * point_grad_value;
        }
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    if param.len() != PARAM_COUNT {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;
    let mut hessian = Array2::zeros((PARAM_COUNT, PARAM_COUNT));
    let params = Params::parse(param);

    let mut index = 0;
    while index < sample_count {
        let x = positive_x(x_values[index]);
        let log_x = x.ln();
        let pow = x.powf(params.exponent);
        let model = params.value_at(x);
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        let jac_a = pow;
        let jac_b = params.scale * pow * log_x;
        let d2_model_dadb = pow * log_x;
        let d2_model_dbdb = params.scale * pow * log_x * log_x;

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
