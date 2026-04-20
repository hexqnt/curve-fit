use super::common::positive_param_with_derivative;
use super::common::{Vf64, positive_param_with_derivative_simd};
use ndarray::Array2;
use std::simd::StdFloat;
use std::simd::num::SimdFloat;

const LN_2: f64 = std::f64::consts::LN_2;
const PARAM_COUNT: usize = 3;

#[derive(Clone, Copy)]
struct Params<T> {
    offset: T,
    amplitude: T,
    half_life_raw: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [offset, amplitude, half_life_raw]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            offset,
            amplitude,
            half_life_raw,
        }
    }

    #[inline]
    fn simd(self) -> Params<Vf64> {
        Params::<Vf64> {
            offset: Vf64::splat(self.offset),
            amplitude: Vf64::splat(self.amplitude),
            half_life_raw: Vf64::splat(self.half_life_raw),
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let (half_life, _) = positive_param_with_derivative(self.half_life_raw);
        let exponent = -LN_2 * x / half_life;
        self.offset + self.amplitude * exponent.exp()
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let (half_life, d_c_raw) = positive_param_with_derivative(self.half_life_raw);
        let exponent = -LN_2 * x / half_life;
        let pow = exponent.exp();
        let d_model_d_c = self.amplitude * pow * LN_2 * x / (half_life * half_life);

        grad[0] = 1.0;
        grad[1] = pow;
        grad[2] = d_model_d_c * d_c_raw;

        self.offset + self.amplitude * pow
    }
}

impl Params<Vf64> {
    #[inline]
    fn value_at(self, x: Vf64) -> Vf64 {
        let (half_life, _) = positive_param_with_derivative_simd(self.half_life_raw);
        let exponent = -Vf64::splat(LN_2) * x / half_life;
        self.offset + self.amplitude * exponent.exp()
    }

    #[inline]
    fn value_grad_at(self, x: Vf64, grad: &mut [Vf64; PARAM_COUNT]) -> Vf64 {
        let (half_life, d_c_raw) = positive_param_with_derivative_simd(self.half_life_raw);
        let exponent = -Vf64::splat(LN_2) * x / half_life;
        let pow = exponent.exp();
        let d_model_d_c = self.amplitude * pow * Vf64::splat(LN_2) * x / (half_life * half_life);

        grad[0] = Vf64::splat(1.0);
        grad[1] = pow;
        grad[2] = d_model_d_c * d_c_raw;

        self.offset + self.amplitude * pow
    }
}

/// Вычисляет экспоненциальную модель через период полураспада:
/// `f(x) = offset + amplitude * exp(-ln(2) * x / half_life)`,
/// где:
/// - `offset` — базовый уровень,
/// - `amplitude` — амплитуда экспоненциальной части,
/// - `half_life` — период полураспада (параметризован положительным преобразованием).
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
