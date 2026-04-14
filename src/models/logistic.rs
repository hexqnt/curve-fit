use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, sigmoid};
use ndarray::Array2;

/// Вычисляет значение логистической кривой:
/// `f(x) = upper_asymptote / (1 + exp(-slope * (x - x0)))`,
/// где:
/// - `upper_asymptote` — амплитуда (верхняя асимптота),
/// - `slope` — крутизна перехода,
/// - `x0` — положение точки перегиба по оси `x`.
#[inline]
pub(super) fn value_at(param: &[f64], x: f64) -> f64 {
    let upper_asymptote = param[0];
    let slope = param[1];
    let x0 = param[2];
    let z = slope * (x - x0);
    upper_asymptote * sigmoid(z)
}

#[inline]
pub(super) fn value_grad_at(param: &[f64], x: f64, grad: &mut [f64]) -> f64 {
    debug_assert_eq!(grad.len(), 3);

    let upper_asymptote = param[0];
    let slope = param[1];
    let x0 = param[2];
    let z = slope * (x - x0);
    let s = sigmoid(z);
    let ds_dz = s * (1.0 - s);

    grad[0] = s;
    grad[1] = upper_asymptote * ds_dz * (x - x0);
    grad[2] = -upper_asymptote * ds_dz * slope;

    upper_asymptote * s
}

pub(super) fn add_value_grad(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    gradient: &mut [f64],
) {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(gradient.len(), param.len());

    // Для каждой точки считаем вклад в градиент скалярной цели по параметрам
    // логистической модели. Используем цепное правило:
    // dF/dθ = (dF/dŷ) * (dŷ/dθ), где F — любой downstream-скаляр
    // (например, loss или выход следующего звена в цепочке).
    let mut point_grad = [0.0; 3];
    let mut index = 0;
    while index < x_values.len() {
        let upstream = value_first[index];
        value_grad_at(param, x_values[index], &mut point_grad);

        gradient[0] += upstream * point_grad[0];
        gradient[1] += upstream * point_grad[1];
        gradient[2] += upstream * point_grad[2];
        index += 1;
    }
}

pub(super) fn add_value_grad_raw_hessian(
    x_values: &[f64],
    param: &[f64],
    value_first: &[f64],
    value_second: &[f64],
) -> Option<Array2<f64>> {
    debug_assert_eq!(x_values.len(), value_first.len());
    debug_assert_eq!(x_values.len(), value_second.len());

    // Модель должна иметь ровно 3 параметра: upper_asymptote, slope, x0.
    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    if sample_count == 0 {
        return Some(Array2::zeros((3, 3)));
    }

    let sample_scale = 1.0 / sample_count as f64;
    let upper_asymptote = param[0];
    let slope = param[1];
    let x0 = param[2];

    // Заполняем только верхний треугольник, затем отзеркаливаем.
    let mut hessian = Array2::zeros((3, 3));

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];

        // u = x - x0, z = slope * u, s = sigmoid(z).
        let u = x - x0;
        let z = slope * u;
        let s = sigmoid(z);
        let ds_dz = s * (1.0 - s);
        // Вторая производная сигмоиды по z:
        // d²s/dz² = ds/dz * (1 - 2s).
        let d2s_dz2 = ds_dz * (1.0 - 2.0 * s);
        let model = upper_asymptote * s;
        if !model.is_finite() {
            return None;
        }

        let value_first = value_first[index];
        let value_second = value_second[index];
        if !value_first.is_finite() || !is_finite_non_negative(value_second) {
            return None;
        }

        // Якобиан модели по параметрам.
        let jac_a = s;
        let jac_b = upper_asymptote * ds_dz * u;
        let jac_c = -upper_asymptote * ds_dz * slope;

        // Вторые частные производные модели ŷ по параметрам.
        // Эти слагаемые нужны в части dF/dŷ * d²ŷ/dθi dθj.
        let d2_model_dadb = ds_dz * u;
        let d2_model_dadc = -ds_dz * slope;
        let d2_model_dbdb = upper_asymptote * d2s_dz2 * u * u;
        let d2_model_dbdc = -upper_asymptote * (slope * u * d2s_dz2 + ds_dz);
        let d2_model_dcdc = upper_asymptote * d2s_dz2 * slope * slope;

        // Для скалярной функции по одной точке:
        // Hij = (d²F/dŷ²) * (dŷ/dθi) * (dŷ/dθj) + (dF/dŷ) * (d²ŷ/dθi dθj).
        hessian[[0, 0]] += value_second * jac_a * jac_a;
        hessian[[0, 1]] += value_second * jac_a * jac_b + value_first * d2_model_dadb;
        hessian[[0, 2]] += value_second * jac_a * jac_c + value_first * d2_model_dadc;
        hessian[[1, 1]] += value_second * jac_b * jac_b + value_first * d2_model_dbdb;
        hessian[[1, 2]] += value_second * jac_b * jac_c + value_first * d2_model_dbdc;
        hessian[[2, 2]] += value_second * jac_c * jac_c + value_first * d2_model_dcdc;
        index += 1;
    }

    // Приводим к среднему по выборке и достраиваем симметричный нижний треугольник.
    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::value_at;
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = value_at(&[2.0, 1.5, 0.5], 1.5);
        assert_near(value, 2.0 / (1.0 + (-1.5_f64).exp()), 1e-12);
    }

    #[test]
    fn derivatives_match_numerical_reference() {
        assert_family_gradient_and_hessian_match_numerical_reference(
            CurveFamily::Logistic,
            &[-2.0, -1.0, -0.3, 0.4, 1.1, 2.0],
            &[2.2, 1.1, 0.3],
            &[1.8, 0.8, -0.1],
            3e-5,
            6e-4,
        );
    }
}
