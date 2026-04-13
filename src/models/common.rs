use ndarray::Array2;

pub(crate) const PARAM_EPS: f64 = 1e-9;
pub(crate) const HESSIAN_DIAGONAL_JITTER: f64 = 1e-9;

#[inline]
pub(crate) fn positive_x(value: f64) -> f64 {
    value.max(PARAM_EPS)
}

#[inline]
pub(crate) fn positive_param_with_derivative(value: f64) -> (f64, f64) {
    if value.abs() >= PARAM_EPS {
        (value.abs(), value.signum())
    } else {
        (PARAM_EPS, 0.0)
    }
}

#[inline]
pub(crate) fn non_zero_param_with_derivative(value: f64) -> (f64, f64) {
    if value.abs() >= PARAM_EPS {
        (value, 1.0)
    } else if value.is_sign_negative() {
        (-PARAM_EPS, 0.0)
    } else {
        (PARAM_EPS, 0.0)
    }
}

#[inline]
pub(crate) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[inline]
pub(crate) fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

#[inline]
pub(crate) fn erf_approx(value: f64) -> f64 {
    let sign = value.signum();
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
        * t
        + 0.254829592)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[inline]
pub(crate) fn erfc_approx(value: f64) -> f64 {
    1.0 - erf_approx(value)
}

#[inline]
pub(crate) fn is_finite_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

pub(crate) fn scale_and_mirror_upper_hessian(hessian: &mut Array2<f64>, scale: f64) {
    let dimension = hessian.nrows();
    debug_assert_eq!(dimension, hessian.ncols());
    let mut row = 0;
    while row < dimension {
        let mut column = row;
        while column < dimension {
            let value = hessian[[row, column]] * scale;
            hessian[[row, column]] = value;
            hessian[[column, row]] = value;
            column += 1;
        }
        row += 1;
    }
}

pub(crate) fn stabilize_hessian(hessian: &mut Array2<f64>) {
    let dimension = hessian.nrows();
    debug_assert_eq!(dimension, hessian.ncols());
    let mut row = 0;
    while row < dimension {
        let mut column = row + 1;
        while column < dimension {
            let value = 0.5 * (hessian[[row, column]] + hessian[[column, row]]);
            hessian[[row, column]] = value;
            hessian[[column, row]] = value;
            column += 1;
        }
        if !hessian[[row, row]].is_finite() {
            hessian[[row, row]] = 0.0;
        }
        hessian[[row, row]] += HESSIAN_DIAGONAL_JITTER;
        row += 1;
    }
}
