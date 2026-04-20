use super::common::non_zero_param_with_derivative;
use super::common::{Vf64, non_zero_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct Params<T> {
    num_quad: T,
    num_linear: T,
    num_const: T,
    den_linear: T,
    den_quad: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [num_quad, num_linear, num_const, den_linear, den_quad]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            num_quad,
            num_linear,
            num_const,
            den_linear,
            den_quad,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            num_quad: Vf64::splat(self.num_quad),
            num_linear: Vf64::splat(self.num_linear),
            num_const: Vf64::splat(self.num_const),
            den_linear: Vf64::splat(self.den_linear),
            den_quad: Vf64::splat(self.den_quad),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let x2 = x * x;
        let numerator = self.num_quad * x2 + self.num_linear * x + self.num_const;
        let denominator_raw = 1.0 + self.den_linear * x + self.den_quad * x2;
        let (denominator, _) = non_zero_param_with_derivative(denominator_raw);
        numerator / denominator
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let x2 = x * x;
        let numerator = self.num_quad * x2 + self.num_linear * x + self.num_const;
        let denominator_raw = 1.0 + self.den_linear * x + self.den_quad * x2;
        let (denominator, d_den_raw) = non_zero_param_with_derivative(denominator_raw);

        grad[0] = x2 / denominator;
        grad[1] = x / denominator;
        grad[2] = 1.0 / denominator;
        grad[3] = (-numerator * x / (denominator * denominator)) * d_den_raw;
        grad[4] = (-numerator * x2 / (denominator * denominator)) * d_den_raw;

        numerator / denominator
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let x2 = x * x;
        let numerator = self.num_quad * x2 + self.num_linear * x + self.num_const;
        let denominator_raw = Vf64::splat(1.0) + self.den_linear * x + self.den_quad * x2;
        let (denominator, _) = non_zero_param_with_derivative_simd(denominator_raw);
        numerator / denominator
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let x2 = x * x;
        let numerator = self.num_quad * x2 + self.num_linear * x + self.num_const;
        let denominator_raw = Vf64::splat(1.0) + self.den_linear * x + self.den_quad * x2;
        let (denominator, d_den_raw) = non_zero_param_with_derivative_simd(denominator_raw);

        grad[0] = x2 / denominator;
        grad[1] = x / denominator;
        grad[2] = Vf64::splat(1.0) / denominator;
        grad[3] = (-numerator * x / (denominator * denominator)) * d_den_raw;
        grad[4] = (-numerator * x2 / (denominator * denominator)) * d_den_raw;

        numerator / denominator
    }
}

/// Вычисляет рациональную функцию порядка (2, 2):
/// `f(x) = (num_quad * x^2 + num_linear * x + num_const) / (1 + den_linear * x + den_quad * x^2)`,
/// где:
/// - `num_quad`, `num_linear`, `num_const` — коэффициенты числителя,
/// - `den_linear`, `den_quad` — коэффициенты знаменателя.
///
/// Знаменатель параметризуется через `non_zero_param_with_derivative`.
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
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 5]) -> Vf64 {
    Params::parse(param).simd().value_grad_at(x, grad)
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
    let params_simd = params.simd();

    {
        let (x_chunks, x_tail) = x_values.as_chunks::<{ Vf64::LEN }>();
        let (value_first_chunks, value_first_tail) = value_first.as_chunks::<{ Vf64::LEN }>();
        debug_assert_eq!(x_chunks.len(), value_first_chunks.len());
        debug_assert_eq!(x_tail.len(), value_first_tail.len());

        let mut point_grad = [Vf64::splat(0.0); PARAM_COUNT];
        let mut gradient_accum = [Vf64::splat(0.0); PARAM_COUNT];

        for (x_chunk, value_first_chunk) in x_chunks.iter().zip(value_first_chunks.iter()) {
            let x = Vf64::from_array(*x_chunk);
            let upstream = Vf64::from_array(*value_first_chunk);
            params_simd.value_grad_at(x, &mut point_grad);

            for (gradient_value, point_grad_value) in
                gradient_accum.iter_mut().zip(point_grad.iter().copied())
            {
                *gradient_value += upstream * point_grad_value;
            }
        }

        for (gradient_value, accum_value) in gradient.iter_mut().zip(gradient_accum.iter().copied())
        {
            *gradient_value += accum_value.reduce_sum();
        }

        let mut point_grad = [0.0; PARAM_COUNT];
        for (&x, &upstream) in x_tail.iter().zip(value_first_tail.iter()) {
            params.value_grad_at(x, &mut point_grad);

            for (gradient_value, point_grad_value) in gradient.iter_mut().zip(point_grad.iter()) {
                *gradient_value += upstream * point_grad_value;
            }
        }
    }
}

pub(super) fn add_value_grad_raw_hessian(
    _x_values: &[f64],
    _param: &[f64],
    _value_first: &[f64],
    _value_second: &[f64],
) -> Option<Array2<f64>> {
    None
}
