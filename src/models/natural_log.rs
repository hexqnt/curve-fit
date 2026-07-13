use std::simd::StdFloat;

use ndarray::Array2;

use super::common::{Vf64, positive_param_with_derivative_simd, positive_x_simd};
use super::common::{positive_param_with_derivative, positive_x};

const PARAM_COUNT: usize = 2;

#[derive(Clone, Copy)]
struct Params<T> {
    scale: T,
    x_scale_raw: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [scale, x_scale_raw]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self { scale, x_scale_raw }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            scale: Vf64::splat(self.scale),
            x_scale_raw: Vf64::splat(self.x_scale_raw),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let x = positive_x(x);
        let (x_scale, _) = positive_param_with_derivative(self.x_scale_raw);
        self.scale * (x / x_scale).ln()
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let x = positive_x(x);
        let (x_scale, d_b_raw) = positive_param_with_derivative(self.x_scale_raw);
        let ln_term = (x / x_scale).ln();

        grad[0] = ln_term;
        grad[1] = (-self.scale / x_scale) * d_b_raw;

        self.scale * ln_term
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let x = positive_x_simd(x);
        let (x_scale, _) = positive_param_with_derivative_simd(self.x_scale_raw);
        self.scale * (x / x_scale).ln()
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x = positive_x_simd(x);
        let (x_scale, d_b_raw) = positive_param_with_derivative_simd(self.x_scale_raw);
        let ln_term = (x / x_scale).ln();

        grad[0] = ln_term;
        grad[1] = (-self.scale / x_scale) * d_b_raw;

        self.scale * ln_term
    }
}
/// Вычисляет логарифмическую зависимость:
/// `f(x) = scale * ln(x / x_scale)`,
/// где:
/// - `scale` — масштабный коэффициент,
/// - `x_scale` — масштаб по оси `x` (параметризован положительным преобразованием).
///
/// Значение `x` предварительно ограничивается снизу через `positive_x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    Params::parse(param).value_at(x)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_simd_at(param: &[f64], x: Vf64) -> Vf64 {
    Params::parse(param).simd().value_at(x)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    Params::parse(param).value_grad_at(x, grad)
}

#[allow(dead_code)]
#[inline]
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 2]) -> Vf64 {
    Params::parse(param).simd().value_grad_at(x, grad)
}

pub(super) fn add_value_grad_raw_hessian(
    _x_values: &[f64],
    _param: &[f64],
    _value_first: &[f64],
    _value_second: &[f64],
) -> Option<Array2<f64>> {
    None
}
