use super::*;

const NORMALIZATION_SCALE_EPS: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Масштабирующая нормализация параметрических моделей без сдвига.
///
/// Нормализует данные по формулам:
/// - `x_norm = x / x_scale`
/// - `y_norm = y / y_scale`
///
/// Такой вид сохраняет структуру всех поддерживаемых семейств и позволяет
/// детерминированно преобразовывать параметры туда-обратно.
pub(super) struct ParametricNormalization {
    x_scale: f64,
    y_scale: f64,
}

impl ParametricNormalization {
    /// Строит коэффициенты нормализации по максимальным абсолютным значениям `x` и `y`.
    pub(super) fn try_from_points(points: &Points) -> Result<Self, String> {
        let mut max_abs_x = 0.0_f64;
        let mut max_abs_y = 0.0_f64;
        for point in points.as_slice() {
            max_abs_x = max_abs_x.max(point.x().abs());
            max_abs_y = max_abs_y.max(point.y().abs());
        }

        let x_scale = max_abs_x.max(NORMALIZATION_SCALE_EPS);
        let y_scale = max_abs_y.max(NORMALIZATION_SCALE_EPS);
        if !x_scale.is_finite() || !y_scale.is_finite() {
            return Err("Normalization scale must be finite".to_string());
        }

        Ok(Self { x_scale, y_scale })
    }

    /// Нормализует точки для внутреннего фиттинга.
    pub(super) fn normalize_points(self, points: &Points) -> Result<Points, String> {
        let mut normalized = Vec::with_capacity(points.len());
        for point in points.as_slice() {
            let x = point.x() / self.x_scale;
            let y = point.y() / self.y_scale;
            let normalized_point = Point::try_new(x, y)
                .map_err(|error| format!("Normalized point must be finite: {error}"))?;
            normalized.push(normalized_point);
        }

        Points::try_from(normalized)
            .map_err(|error| format!("Normalized points are invalid: {error}"))
    }

    /// Переводит параметры из исходного масштаба в нормализованный.
    pub(super) fn normalize_params(self, params: &CurveParams) -> Result<CurveParams, String> {
        self.transform_params(params, true)
    }

    /// Переводит параметры из нормализованного масштаба обратно в исходный.
    pub(super) fn denormalize_params(self, params: &CurveParams) -> Result<CurveParams, String> {
        self.transform_params(params, false)
    }

    fn transform_params(
        self,
        params: &CurveParams,
        to_normalized: bool,
    ) -> Result<CurveParams, String> {
        let family = params.family();
        let mut values = params.values();
        let x_scale = self.x_scale;
        let y_scale = self.y_scale;

        if family.is_polynomial() {
            let degree = values.len().saturating_sub(1);
            for (index, value) in values.iter_mut().enumerate() {
                let power = (degree - index) as i32;
                let x_factor = x_scale.powi(power);
                if to_normalized {
                    *value = *value * x_factor / y_scale;
                } else {
                    *value = *value * y_scale / x_factor;
                }
            }
        } else {
            match family {
                CurveFamily::Arrhenius => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= x_scale;
                    }
                }
                CurveFamily::Inverse => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= x_scale * y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= x_scale * y_scale;
                    }
                }
                CurveFamily::Logistic => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] *= x_scale;
                        values[2] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] /= x_scale;
                        values[2] *= x_scale;
                    }
                }
                CurveFamily::Gompertz => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] *= x_scale;
                        values[2] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] /= x_scale;
                        values[2] *= x_scale;
                    }
                }
                CurveFamily::BiExponential => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] *= x_scale;
                        values[2] /= y_scale;
                        values[3] *= x_scale;
                        values[4] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] /= x_scale;
                        values[2] *= y_scale;
                        values[3] /= x_scale;
                        values[4] *= y_scale;
                    }
                }
                CurveFamily::DampedSinusoid => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] *= x_scale;
                        values[2] *= x_scale;
                        values[4] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] /= x_scale;
                        values[2] /= x_scale;
                        values[4] *= y_scale;
                    }
                }
                CurveFamily::Lorentzian => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= x_scale;
                        values[2] /= x_scale;
                        values[3] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= x_scale;
                        values[2] *= x_scale;
                        values[3] *= y_scale;
                    }
                }
                CurveFamily::NaturalLog => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= x_scale;
                    }
                }
                CurveFamily::FourPl => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[2] /= x_scale;
                        values[3] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[2] *= x_scale;
                        values[3] *= y_scale;
                    }
                }
                CurveFamily::FivePl => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[2] /= x_scale;
                        values[3] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[2] *= x_scale;
                        values[3] *= y_scale;
                    }
                }
                CurveFamily::MichaelisMenten => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= x_scale;
                    }
                }
                CurveFamily::ExponentialBasic => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= y_scale;
                        values[2] *= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= y_scale;
                        values[2] /= x_scale;
                    }
                }
                CurveFamily::ExponentialLinear => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] *= x_scale;
                        values[2] = values[2] * x_scale / y_scale;
                        values[3] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] /= x_scale;
                        values[2] = values[2] * y_scale / x_scale;
                        values[3] *= y_scale;
                    }
                }
                CurveFamily::ExponentialHalfLife => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= y_scale;
                        values[2] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= y_scale;
                        values[2] *= x_scale;
                    }
                }
                CurveFamily::FallingExponential => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] = values[1] * x_scale / y_scale;
                        values[2] *= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] = values[1] * y_scale / x_scale;
                        values[2] /= x_scale;
                    }
                }
                CurveFamily::HyperbolicTangent
                | CurveFamily::ArctangentStep
                | CurveFamily::Softplus => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] *= x_scale;
                        values[2] /= x_scale;
                        values[3] /= y_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] /= x_scale;
                        values[2] *= x_scale;
                        values[3] *= y_scale;
                    }
                }
                CurveFamily::Power => {
                    let power = values[1];
                    let x_factor = x_scale.powf(power);
                    if to_normalized {
                        values[0] = values[0] * x_factor / y_scale;
                    } else {
                        values[0] = values[0] * y_scale / x_factor;
                    }
                }
                CurveFamily::Gaussian => {
                    if to_normalized {
                        values[0] /= y_scale;
                        values[1] /= x_scale;
                        values[2] /= x_scale;
                    } else {
                        values[0] *= y_scale;
                        values[1] *= x_scale;
                        values[2] *= x_scale;
                    }
                }
                CurveFamily::Linear
                | CurveFamily::Quadratic
                | CurveFamily::Cubic
                | CurveFamily::Quartic
                | CurveFamily::Quintic
                | CurveFamily::Sextic
                | CurveFamily::Septic
                | CurveFamily::Octic
                | CurveFamily::Nonic => unreachable!("handled by polynomial branch above"),
            }
        }

        CurveParams::try_from_values(family, values).map_err(|error| {
            if to_normalized {
                format!("Failed to normalize parameters: {error}")
            } else {
                format!("Failed to denormalize parameters: {error}")
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_values_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() <= tolerance,
                "expected {e}, got {a}, tolerance={tolerance}"
            );
        }
    }

    fn sample_points() -> Points {
        Points::try_from(vec![
            Point::try_new(-3.0, 12.0).unwrap(),
            Point::try_new(-1.0, -8.0).unwrap(),
            Point::try_new(2.0, 4.0).unwrap(),
            Point::try_new(5.0, -2.0).unwrap(),
        ])
        .unwrap()
    }

    fn strictly_positive_points() -> Points {
        Points::try_from(vec![
            Point::try_new(0.5, 1.2).unwrap(),
            Point::try_new(1.0, 2.5).unwrap(),
            Point::try_new(2.0, 4.0).unwrap(),
            Point::try_new(3.5, 5.1).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn normalize_points_scales_xy_components() {
        let points = sample_points();
        let normalization = ParametricNormalization::try_from_points(&points).unwrap();
        let normalized = normalization.normalize_points(&points).unwrap();

        let xs = normalized
            .as_slice()
            .iter()
            .map(|point| point.x().abs())
            .collect::<Vec<_>>();
        let ys = normalized
            .as_slice()
            .iter()
            .map(|point| point.y().abs())
            .collect::<Vec<_>>();

        let max_x = xs.into_iter().fold(0.0, f64::max);
        let max_y = ys.into_iter().fold(0.0, f64::max);
        assert!((max_x - 1.0).abs() <= 1e-12);
        assert!((max_y - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn all_families_roundtrip_params_after_normalization() {
        let points = strictly_positive_points();
        let normalization = ParametricNormalization::try_from_points(&points).unwrap();

        for family in CurveFamily::ALL {
            let original = family.default_params();
            let normalized = normalization.normalize_params(&original).unwrap();
            let restored = normalization.denormalize_params(&normalized).unwrap();
            assert_values_close(&restored.values(), &original.values(), 1e-9);
        }
    }
}
