use super::common::positive_param_with_derivative;
use super::common::{Vf64, positive_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 3;

#[derive(Clone, Copy)]
struct Params<T> {
    amplitude: T,
    mean: T,
    sigma_raw: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [amplitude, mean, sigma_raw]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            amplitude,
            mean,
            sigma_raw,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            amplitude: Vf64::splat(self.amplitude),
            mean: Vf64::splat(self.mean),
            sigma_raw: Vf64::splat(self.sigma_raw),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let (sigma, _) = positive_param_with_derivative(self.sigma_raw);
        let delta = x - self.mean;
        self.amplitude * (-(delta * delta) / (2.0 * sigma * sigma)).exp()
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let (sigma, d_c_raw) = positive_param_with_derivative(self.sigma_raw);
        let c2 = sigma * sigma;
        let delta = x - self.mean;
        let exp_part = (-(delta * delta) / (2.0 * c2)).exp();
        let d_model_d_a = exp_part;
        let d_model_d_b = self.amplitude * exp_part * delta / c2;
        let d_model_d_c = self.amplitude * exp_part * delta * delta / (c2 * sigma);

        grad[0] = d_model_d_a;
        grad[1] = d_model_d_b;
        grad[2] = d_model_d_c * d_c_raw;

        self.amplitude * exp_part
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let (sigma, _) = positive_param_with_derivative_simd(self.sigma_raw);
        let delta = x - self.mean;
        self.amplitude * (-(delta * delta) / (Vf64::splat(2.0) * sigma * sigma)).exp()
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let (sigma, d_c_raw) = positive_param_with_derivative_simd(self.sigma_raw);
        let c2 = sigma * sigma;
        let delta = x - self.mean;
        let exp_part = (-(delta * delta) / (Vf64::splat(2.0) * c2)).exp();
        let d_model_d_a = exp_part;
        let d_model_d_b = self.amplitude * exp_part * delta / c2;
        let d_model_d_c = self.amplitude * exp_part * delta * delta / (c2 * sigma);

        grad[0] = d_model_d_a;
        grad[1] = d_model_d_b;
        grad[2] = d_model_d_c * d_c_raw;

        self.amplitude * exp_part
    }
}
/// Вычисляет гауссову кривую:
/// `f(x) = amplitude * exp(-(x - mean)^2 / (2 * sigma^2))`,
/// где:
/// - `amplitude` — амплитуда,
/// - `mean` — центр пика,
/// - `sigma` — ширина (параметризована положительным преобразованием).
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
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; 3]) -> Vf64 {
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
