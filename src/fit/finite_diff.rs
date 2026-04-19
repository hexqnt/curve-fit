//! Численное дифференцирование и стабилизация аппроксимированной Гессианы.

use super::*;

pub(super) fn gradient_l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[inline]
pub(super) fn finite_central_difference(
    value_plus: f64,
    value_minus: f64,
    step: f64,
) -> Option<f64> {
    if !value_plus.is_finite() || !value_minus.is_finite() {
        return None;
    }
    let derivative = (value_plus - value_minus) / (2.0 * step);
    if derivative.is_finite() {
        Some(derivative)
    } else {
        None
    }
}

#[inline]
pub(super) fn finite_array1(values: &Array1<f64>) -> bool {
    values.iter().all(|value| value.is_finite())
}

pub(super) fn vec_to_array1(values: &[f64]) -> Array1<f64> {
    Array1::from_vec(values.to_vec())
}

pub(super) fn array1_as_slice(values: &Array1<f64>) -> &[f64] {
    values
        .as_slice()
        .expect("Array1 parameters must have contiguous memory layout")
}

pub(super) fn stabilize_hessian(hessian: &mut Array2<f64>) {
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
        // Добавляем небольшой jitter на диагональ, чтобы численный шум реже ломал Newton-CG.
        hessian[[row, row]] += HESSIAN_DIAGONAL_JITTER;
        row += 1;
    }
}

pub(super) fn numerical_hessian_from_gradient<O>(
    problem: &O,
    param: &Array1<f64>,
) -> Result<Array2<f64>, argmin::core::Error>
where
    O: Gradient<Param = Array1<f64>, Gradient = Array1<f64>>,
{
    let param_slice = array1_as_slice(param);
    let dimension = param_slice.len();
    let mut hessian = Array2::zeros((dimension, dimension));
    let mut probe = param.clone();
    let mut column_values = vec![0.0; dimension];

    for column in 0..dimension {
        let base_step =
            ((param_slice[column].abs() + 1.0) * HESSIAN_FD_REL_STEP).max(HESSIAN_FD_MIN_STEP);
        let mut computed = false;
        for factor in FD_STEP_RETRY_FACTORS {
            let step = base_step * factor;
            probe[column] = param[column] + step;
            let grad_plus = problem.gradient(&probe)?;
            probe[column] = param[column] - step;
            let grad_minus = problem.gradient(&probe)?;
            probe[column] = param[column];

            if !finite_array1(&grad_plus) || !finite_array1(&grad_minus) {
                continue;
            }

            let denom = 2.0 * step;
            let mut column_is_finite = true;
            for row in 0..dimension {
                let value = (grad_plus[row] - grad_minus[row]) / denom;
                if !value.is_finite() {
                    column_is_finite = false;
                    break;
                }
                column_values[row] = value;
            }

            if column_is_finite {
                for row in 0..dimension {
                    hessian[[row, column]] = column_values[row];
                }
                computed = true;
                break;
            }
        }

        if !computed {
            for row in 0..dimension {
                hessian[[row, column]] = 0.0;
            }
        }
    }

    stabilize_hessian(&mut hessian);
    Ok(hessian)
}
