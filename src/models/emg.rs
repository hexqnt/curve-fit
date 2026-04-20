use super::common::{Vf64, erfc_approx_simd, positive_param_with_derivative_simd};
use super::common::{erfc_approx, positive_param_with_derivative};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct Params<T> {
    amplitude: T,
    mu: T,
    sigma: T,
    tau: T,
    baseline: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [amplitude, mu, sigma, tau, baseline]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            amplitude,
            mu,
            sigma,
            tau,
            baseline,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            amplitude: Vf64::splat(self.amplitude),
            mu: Vf64::splat(self.mu),
            sigma: Vf64::splat(self.sigma),
            tau: Vf64::splat(self.tau),
            baseline: Vf64::splat(self.baseline),
        }
    }

    #[inline]
    fn eval_right(self, x: f64) -> f64 {
        let (sigma, _) = positive_param_with_derivative(self.sigma);
        let (tau, _) = positive_param_with_derivative(self.tau);
        let delta = x - self.mu;
        let z = (sigma / tau - delta / sigma) / std::f64::consts::SQRT_2;
        let exponent = sigma * sigma / (2.0 * tau * tau) - delta / tau;
        self.baseline + (self.amplitude / (2.0 * tau)) * exponent.exp() * erfc_approx(z)
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        if self.tau.is_sign_negative() {
            self.eval_right(2.0 * self.mu - x)
        } else {
            self.eval_right(x)
        }
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);
        grad.fill(0.0);
        self.value_at(x)
    }
}

impl Params<Vf64> {
    #[inline]
    fn eval_right(self, x: Vf64) -> Vf64 {
        let (sigma, _) = positive_param_with_derivative_simd(self.sigma);
        let (tau, _) = positive_param_with_derivative_simd(self.tau);
        let delta = x - self.mu;
        let z = (sigma / tau - delta / sigma) / Vf64::splat(std::f64::consts::SQRT_2);
        let exponent = sigma * sigma / (Vf64::splat(2.0) * tau * tau) - delta / tau;
        self.baseline
            + (self.amplitude / (Vf64::splat(2.0) * tau)) * exponent.exp() * erfc_approx_simd(z)
    }

    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        if self.tau.to_array()[0].is_sign_negative() {
            self.eval_right(Vf64::splat(2.0) * self.mu - x)
        } else {
            self.eval_right(x)
        }
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        grad.fill(Vf64::splat(0.0));
        self.value_at(x)
    }
}

/// Вычисляет экспоненциально-модифицированную гауссиану (EMG):
/// `f(x) = baseline + (amplitude / (2 * tau)) * exp(sigma^2 / (2 * tau^2) - (x - mu) / tau) * erfc(z)`,
/// где:
/// - `amplitude` — амплитуда,
/// - `mu` — центр гауссовой части,
/// - `sigma` — ширина гауссовой части,
/// - `tau` — экспоненциальная постоянная,
/// - `baseline` — вертикальный сдвиг.
///
/// Для `tau < 0` используется отражение по `mu`, что даёт хвост в противоположную сторону.
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
