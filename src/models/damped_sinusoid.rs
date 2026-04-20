use super::common::Vf64;
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct Params<T> {
    amplitude: T,
    damping: T,
    omega: T,
    phi: T,
    offset: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [amplitude, damping, omega, phi, offset]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            amplitude,
            damping,
            omega,
            phi,
            offset,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            amplitude: Vf64::splat(self.amplitude),
            damping: Vf64::splat(self.damping),
            omega: Vf64::splat(self.omega),
            phi: Vf64::splat(self.phi),
            offset: Vf64::splat(self.offset),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        self.amplitude * (-self.damping * x).exp() * (self.omega * x + self.phi).sin() + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let exp_part = (-self.damping * x).exp();
        let angle = self.omega * x + self.phi;
        let sin_part = angle.sin();
        let cos_part = angle.cos();

        grad[0] = exp_part * sin_part;
        grad[1] = -self.amplitude * x * exp_part * sin_part;
        grad[2] = self.amplitude * exp_part * cos_part * x;
        grad[3] = self.amplitude * exp_part * cos_part;
        grad[4] = 1.0;

        self.amplitude * exp_part * sin_part + self.offset
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        self.amplitude * (-self.damping * x).exp() * (self.omega * x + self.phi).sin() + self.offset
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let exp_part = (-self.damping * x).exp();
        let angle = self.omega * x + self.phi;
        let sin_part = angle.sin();
        let cos_part = angle.cos();

        grad[0] = exp_part * sin_part;
        grad[1] = -self.amplitude * x * exp_part * sin_part;
        grad[2] = self.amplitude * exp_part * cos_part * x;
        grad[3] = self.amplitude * exp_part * cos_part;
        grad[4] = Vf64::splat(1.0);

        self.amplitude * exp_part * sin_part + self.offset
    }
}

/// Вычисляет затухающую синусоиду:
/// `f(x) = amplitude * exp(-damping * x) * sin(omega * x + phi) + offset`,
/// где:
/// - `amplitude` — начальная амплитуда,
/// - `damping` — коэффициент затухания,
/// - `omega` — угловая частота,
/// - `phi` — фазовый сдвиг,
/// - `offset` — вертикальный сдвиг.
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
