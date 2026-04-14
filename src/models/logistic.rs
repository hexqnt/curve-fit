use super::common::{is_finite_non_negative, scale_and_mirror_upper_hessian, stabilize_hessian};
use ndarray::Array2;

/// Вычисляет значение логистической кривой:
/// `f(x) = upper_asymptote / (1 + exp(-slope * (x - x0)))`,
/// где:
/// - `upper_asymptote` — амплитуда (верхняя асимптота),
/// - `slope` — крутизна перехода,
/// - `x0` — положение точки перегиба по оси `x`.
#[inline]
pub(super) fn eval(param: &[f64], x: f64) -> f64 {
    let upper_asymptote = param[0];
    let slope = param[1];
    let x0 = param[2];
    let exponent = -slope * (x - x0);
    upper_asymptote / (1.0 + exponent.exp())
}

pub(super) fn accumulate_gradient<L>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
    gradient: &mut [f64],
) where
    L: super::PredictionLoss,
{
    debug_assert_eq!(x_values.len(), y_values.len());
    let upper_asymptote = param[0];
    let slope = param[1];
    let x0 = param[2];

    // Для каждой точки считаем вклад в градиент функции потерь по параметрам
    // логистической модели. Используем цепное правило:
    // dL/dθ = (dL/dŷ) * (dŷ/dθ).
    let mut index = 0;
    while index < x_values.len() {
        let x = x_values[index];
        let y = y_values[index];

        // Обозначения:
        // z = slope * (x - x0), s = sigmoid(z), ŷ = upper_asymptote * s.
        // Тогда dŷ/d(upper_asymptote) = s.
        let z = slope * (x - x0);
        let s = 1.0 / (1.0 + (-z).exp());
        let model = upper_asymptote * s;

        // Внешняя производная потерь по предсказанию модели.
        let residual = loss.d_prediction(model, y);

        // Производная сигмоиды по аргументу z:
        // ds/dz = s * (1 - s).
        let ds_dz = s * (1.0 - s);

        // dŷ/d(upper_asymptote) = s
        gradient[0] += residual * s;
        // dŷ/d(slope) = upper_asymptote * ds/dz * (x - x0)
        gradient[1] += residual * (upper_asymptote * ds_dz * (x - x0));
        // dŷ/d(x0) = upper_asymptote * ds/dz * d(slope * (x - x0))/d(x0) = -upper_asymptote * ds/dz * slope
        gradient[2] += residual * (upper_asymptote * ds_dz * (-slope));
        index += 1;
    }
}

pub(super) fn analytic_hessian<L>(
    x_values: &[f64],
    y_values: &[f64],
    param: &[f64],
    loss: &L,
) -> Option<Array2<f64>>
where
    L: super::PredictionLoss,
{
    // Модель должна иметь ровно 3 параметра: upper_asymptote, slope, x0.
    if param.len() != 3 {
        return None;
    }

    let sample_count = x_values.len();
    let sample_scale = 1.0 / sample_count as f64;

    // Заполняем только верхний треугольник, затем отзеркаливаем.
    let mut hessian = Array2::zeros((3, 3));

    let mut index = 0;
    while index < sample_count {
        let x = x_values[index];
        let y = y_values[index];

        // Локальные обозначения для краткости формул:
        // upper_asymptote — амплитуда, slope — крутизна, x0 — сдвиг по x.
        let upper_asymptote = param[0];
        let slope = param[1];
        let x0 = param[2];

        // u = x - x0, z = slope * u, s = sigmoid(z).
        let u = x - x0;
        let z = slope * u;
        let s = 1.0 / (1.0 + (-z).exp());
        let ds_dz = s * (1.0 - s);
        // Вторая производная сигмоиды по z:
        // d²s/dz² = ds/dz * (1 - 2s).
        let d2s_dz2 = ds_dz * (1.0 - 2.0 * s);
        let model = upper_asymptote * s;
        if !model.is_finite() {
            return None;
        }

        let loss_first = loss.d_prediction(model, y);
        let loss_second = loss.d2_prediction(model, y);
        if !loss_first.is_finite() || !is_finite_non_negative(loss_second) {
            return None;
        }

        // Якобиан модели по параметрам.
        let jac_a = s;
        let jac_b = upper_asymptote * ds_dz * u;
        let jac_c = -upper_asymptote * ds_dz * slope;

        // Вторые частные производные модели ŷ по параметрам.
        // Эти слагаемые нужны в части loss_first * d²ŷ/dθi dθj.
        let d2_model_dadb = ds_dz * u;
        let d2_model_dadc = -ds_dz * slope;
        let d2_model_dbdb = upper_asymptote * d2s_dz2 * u * u;
        let d2_model_dbdc = -upper_asymptote * (slope * u * d2s_dz2 + ds_dz);
        let d2_model_dcdc = upper_asymptote * d2s_dz2 * slope * slope;

        // Для скалярной потери по одной точке:
        // Hij = (d²L/dŷ²) * (dŷ/dθi) * (dŷ/dθj) + (dL/dŷ) * (d²ŷ/dθi dθj).
        hessian[[0, 0]] += loss_second * jac_a * jac_a;
        hessian[[0, 1]] += loss_second * jac_a * jac_b + loss_first * d2_model_dadb;
        hessian[[0, 2]] += loss_second * jac_a * jac_c + loss_first * d2_model_dadc;
        hessian[[1, 1]] += loss_second * jac_b * jac_b + loss_first * d2_model_dbdb;
        hessian[[1, 2]] += loss_second * jac_b * jac_c + loss_first * d2_model_dbdc;
        hessian[[2, 2]] += loss_second * jac_c * jac_c + loss_first * d2_model_dcdc;
        index += 1;
    }

    // Приводим к среднему по выборке, достраиваем симметричный нижний треугольник
    // и мягко стабилизируем матрицу для численной устойчивости оптимизации.
    scale_and_mirror_upper_hessian(&mut hessian, sample_scale);
    stabilize_hessian(&mut hessian);
    Some(hessian)
}

#[cfg(test)]
mod tests {
    use super::eval;
    use crate::domain::CurveFamily;
    use crate::models::test_support::{
        assert_family_gradient_and_hessian_match_numerical_reference, assert_near,
    };

    #[test]
    fn value_matches_known_example() {
        let value = eval(&[2.0, 1.5, 0.5], 1.5);
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
