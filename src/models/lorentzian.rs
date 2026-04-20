use super::common::positive_param_with_derivative;
use super::common::{Vf64, positive_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 4;

#[derive(Clone, Copy)]
struct Params {
    amplitude: f64,
    x0: f64,
    gamma_raw: f64,
    baseline: f64,
}

impl Params {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [amplitude, x0, gamma_raw, baseline]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            amplitude,
            x0,
            gamma_raw,
            baseline,
        }
    }

    #[inline]
    fn simd(self) -> SimdParams {
        SimdParams {
            amplitude: Vf64::splat(self.amplitude),
            x0: Vf64::splat(self.x0),
            gamma_raw: Vf64::splat(self.gamma_raw),
            baseline: Vf64::splat(self.baseline),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let (gamma, _) = positive_param_with_derivative(self.gamma_raw);
        let u = (x - self.x0) / gamma;
        self.baseline + self.amplitude / (1.0 + u * u)
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let (gamma, d_gamma_raw) = positive_param_with_derivative(self.gamma_raw);
        let u = (x - self.x0) / gamma;
        let den = 1.0 + u * u;
        let inv_den = 1.0 / den;
        let common = 2.0 * self.amplitude / (den * den * gamma);

        grad[0] = inv_den;
        grad[1] = common * u;
        grad[2] = common * u * u * d_gamma_raw;
        grad[3] = 1.0;

        self.baseline + self.amplitude * inv_den
    }
}

#[derive(Clone, Copy)]
struct SimdParams {
    amplitude: Vf64,
    x0: Vf64,
    gamma_raw: Vf64,
    baseline: Vf64,
}

impl SimdParams {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let (gamma, _) = positive_param_with_derivative_simd(self.gamma_raw);
        let u = (x - self.x0) / gamma;
        self.baseline + self.amplitude / (Vf64::splat(1.0) + u * u)
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let (gamma, d_gamma_raw) = positive_param_with_derivative_simd(self.gamma_raw);
        let u = (x - self.x0) / gamma;
        let den = Vf64::splat(1.0) + u * u;
        let inv_den = Vf64::splat(1.0) / den;
        let common = Vf64::splat(2.0) * self.amplitude / (den * den * gamma);

        grad[0] = inv_den;
        grad[1] = common * u;
        grad[2] = common * u * u * d_gamma_raw;
        grad[3] = Vf64::splat(1.0);

        self.baseline + self.amplitude * inv_den
    }
}

/// Вычисляет лоренцев пик:
/// `f(x) = baseline + amplitude / (1 + ((x - x0) / gamma)^2)`,
/// где:
/// - `amplitude` — амплитуда пика,
/// - `x0` — центр пика,
/// - `gamma` — полуширина (параметризована положительным преобразованием),
/// - `baseline` — базовый уровень.
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
pub(super) fn value_grad_simd_at(param: &[f64], x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
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
