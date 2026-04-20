use super::common::{Vf64, positive_param_with_derivative_simd, sigmoid_simd};
use super::common::{positive_param_with_derivative, sigmoid};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

const PARAM_COUNT: usize = 6;

#[derive(Clone, Copy)]
struct Params<T> {
    amplitude: T,
    x0: T,
    sigma_raw: T,
    gamma_raw: T,
    eta_raw: T,
    baseline: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [amplitude, x0, sigma_raw, gamma_raw, eta_raw, baseline]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            amplitude,
            x0,
            sigma_raw,
            gamma_raw,
            eta_raw,
            baseline,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            amplitude: Vf64::splat(self.amplitude),
            x0: Vf64::splat(self.x0),
            sigma_raw: Vf64::splat(self.sigma_raw),
            gamma_raw: Vf64::splat(self.gamma_raw),
            eta_raw: Vf64::splat(self.eta_raw),
            baseline: Vf64::splat(self.baseline),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let (sigma, _) = positive_param_with_derivative(self.sigma_raw);
        let (gamma, _) = positive_param_with_derivative(self.gamma_raw);
        let eta = sigmoid(self.eta_raw);
        let delta = x - self.x0;
        let gaussian = (-(delta * delta) / (2.0 * sigma * sigma)).exp();
        let lorentzian = 1.0 / (1.0 + (delta / gamma).powi(2));
        self.baseline + self.amplitude * (eta * gaussian + (1.0 - eta) * lorentzian)
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let (sigma, d_sigma_raw) = positive_param_with_derivative(self.sigma_raw);
        let (gamma, d_gamma_raw) = positive_param_with_derivative(self.gamma_raw);
        let eta = sigmoid(self.eta_raw);
        let eta_prime = eta * (1.0 - eta);
        let delta = x - self.x0;

        let sigma2 = sigma * sigma;
        let gaussian = (-(delta * delta) / (2.0 * sigma2)).exp();
        let d_gaussian_dx0 = gaussian * delta / sigma2;
        let d_gaussian_d_sigma = gaussian * delta * delta / (sigma2 * sigma);

        let u = delta / gamma;
        let den = 1.0 + u * u;
        let lorentzian = 1.0 / den;
        let den2 = den * den;
        let d_lorentzian_dx0 = 2.0 * u / (den2 * gamma);
        let d_lorentzian_d_gamma = 2.0 * u * u / (den2 * gamma);

        let mix = eta * gaussian + (1.0 - eta) * lorentzian;

        grad[0] = mix;
        grad[1] = self.amplitude * (eta * d_gaussian_dx0 + (1.0 - eta) * d_lorentzian_dx0);
        grad[2] = self.amplitude * eta * d_gaussian_d_sigma * d_sigma_raw;
        grad[3] = self.amplitude * (1.0 - eta) * d_lorentzian_d_gamma * d_gamma_raw;
        grad[4] = self.amplitude * (gaussian - lorentzian) * eta_prime;
        grad[5] = 1.0;

        self.baseline + self.amplitude * mix
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let (sigma, _) = positive_param_with_derivative_simd(self.sigma_raw);
        let (gamma, _) = positive_param_with_derivative_simd(self.gamma_raw);
        let eta = sigmoid_simd(self.eta_raw);
        let delta = x - self.x0;
        let gaussian = (-(delta * delta) / (Vf64::splat(2.0) * sigma * sigma)).exp();
        let u = delta / gamma;
        let lorentzian = Vf64::splat(1.0) / (Vf64::splat(1.0) + u * u);
        self.baseline + self.amplitude * (eta * gaussian + (Vf64::splat(1.0) - eta) * lorentzian)
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let (sigma, d_sigma_raw) = positive_param_with_derivative_simd(self.sigma_raw);
        let (gamma, d_gamma_raw) = positive_param_with_derivative_simd(self.gamma_raw);
        let eta = sigmoid_simd(self.eta_raw);
        let eta_prime = eta * (Vf64::splat(1.0) - eta);
        let delta = x - self.x0;

        let sigma2 = sigma * sigma;
        let gaussian = (-(delta * delta) / (Vf64::splat(2.0) * sigma2)).exp();
        let d_gaussian_dx0 = gaussian * delta / sigma2;
        let d_gaussian_d_sigma = gaussian * delta * delta / (sigma2 * sigma);

        let u = delta / gamma;
        let den = Vf64::splat(1.0) + u * u;
        let lorentzian = Vf64::splat(1.0) / den;
        let den2 = den * den;
        let d_lorentzian_dx0 = Vf64::splat(2.0) * u / (den2 * gamma);
        let d_lorentzian_d_gamma = Vf64::splat(2.0) * u * u / (den2 * gamma);

        let mix = eta * gaussian + (Vf64::splat(1.0) - eta) * lorentzian;

        grad[0] = mix;
        grad[1] =
            self.amplitude * (eta * d_gaussian_dx0 + (Vf64::splat(1.0) - eta) * d_lorentzian_dx0);
        grad[2] = self.amplitude * eta * d_gaussian_d_sigma * d_sigma_raw;
        grad[3] = self.amplitude * (Vf64::splat(1.0) - eta) * d_lorentzian_d_gamma * d_gamma_raw;
        grad[4] = self.amplitude * (gaussian - lorentzian) * eta_prime;
        grad[5] = Vf64::splat(1.0);

        self.baseline + self.amplitude * mix
    }
}

/// Вычисляет псевдо-Войт профиль:
/// `f(x) = baseline + amplitude * (eta * G(x) + (1 - eta) * L(x))`,
/// где:
/// - `amplitude` — амплитуда,
/// - `x0` — центр пика,
/// - `sigma` — ширина гауссовой части (положительный параметр),
/// - `gamma` — ширина лоренцевой части (положительный параметр),
/// - `eta` — вес смешивания `G/L` (через сигмоиду),
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
