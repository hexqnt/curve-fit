use super::common::{positive_param_with_derivative, positive_x};
use ndarray::Array2;

const PARAM_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct Params<T> {
    top: T,
    hill_slope: T,
    ec50_raw: T,
    bottom: T,
    asymmetry_raw: T,
}

impl Params<f64> {
    #[inline]
    fn parse(param: &[f64]) -> Self {
        let [top, hill_slope, ec50_raw, bottom, asymmetry_raw]: [f64; PARAM_COUNT] = param
            .try_into()
            .unwrap_or_else(|_| panic!("expected {} params", PARAM_COUNT));
        Self {
            top,
            hill_slope,
            ec50_raw,
            bottom,
            asymmetry_raw,
        }
    }

    #[inline]
    fn value_at(self, x: f64) -> f64 {
        let x = positive_x(x);
        let (ec50, _) = positive_param_with_derivative(self.ec50_raw);
        let (asymmetry, _) = positive_param_with_derivative(self.asymmetry_raw);
        let ratio = x / ec50;
        let pow = ratio.powf(self.hill_slope);
        self.bottom + (self.top - self.bottom) * (1.0 + pow).powf(-asymmetry)
    }

    #[inline]
    fn value_grad_at(self, x: f64, grad: &mut [f64]) -> f64 {
        debug_assert_eq!(grad.len(), PARAM_COUNT);

        let x = positive_x(x);
        let (ec50, d_c_raw) = positive_param_with_derivative(self.ec50_raw);
        let (asymmetry, d_m_raw) = positive_param_with_derivative(self.asymmetry_raw);
        let ratio = x / ec50;
        let pow = ratio.powf(self.hill_slope);
        let den = 1.0 + pow;
        let inv = den.powf(-asymmetry);
        let d_pow_db = pow * ratio.ln();
        let d_pow_dc = -pow * self.hill_slope / ec50;
        let d_inv_db = -asymmetry * den.powf(-asymmetry - 1.0) * d_pow_db;
        let d_inv_dc = -asymmetry * den.powf(-asymmetry - 1.0) * d_pow_dc;
        let d_inv_dm = -inv * den.ln();

        grad[0] = inv;
        grad[1] = (self.top - self.bottom) * d_inv_db;
        grad[2] = (self.top - self.bottom) * d_inv_dc * d_c_raw;
        grad[3] = 1.0 - inv;
        grad[4] = (self.top - self.bottom) * d_inv_dm * d_m_raw;

        self.bottom + (self.top - self.bottom) * inv
    }
}

/// Вычисляет пяти-параметрическую логистическую кривую (5PL):
/// `f(x) = bottom + (top - bottom) * (1 + (x / ec50)^hill_slope)^(-asymmetry)`,
/// где:
/// - `top` — верхняя асимптота,
/// - `hill_slope` — крутизна,
/// - `ec50` — точка перегиба (параметризована положительным преобразованием),
/// - `bottom` — нижняя асимптота,
/// - `asymmetry` — параметр асимметрии (параметризован положительным преобразованием).
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
    _x_values: &[f64],
    _param: &[f64],
    _value_first: &[f64],
    _value_second: &[f64],
) -> Option<Array2<f64>> {
    None
}
