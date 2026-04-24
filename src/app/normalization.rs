//! Масштабная нормализация параметрических задач до более устойчивого численного диапазона.

use super::*;

const NORMALIZATION_SCALE_EPS: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScaleDirection {
    ToNormalized,
    FromNormalized,
}

impl ScaleDirection {
    fn apply(self, value: f64, factor: f64) -> f64 {
        match self {
            Self::ToNormalized => value * factor,
            Self::FromNormalized => value / factor,
        }
    }

    fn error_prefix(self) -> &'static str {
        match self {
            Self::ToNormalized => "Failed to normalize parameters",
            Self::FromNormalized => "Failed to denormalize parameters",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParamScale {
    x_exp: i32,
    y_exp: i32,
}

impl ParamScale {
    const fn new(x_exp: i32, y_exp: i32) -> Self {
        Self { x_exp, y_exp }
    }
}

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
        let (max_abs_x, max_abs_y) = points.iter().fold((0.0_f64, 0.0_f64), |acc, point| {
            (acc.0.max(point.x().abs()), acc.1.max(point.y().abs()))
        });

        let x_scale = max_abs_x.max(NORMALIZATION_SCALE_EPS);
        let y_scale = max_abs_y.max(NORMALIZATION_SCALE_EPS);
        if !x_scale.is_finite() || !y_scale.is_finite() {
            return Err("Normalization scale must be finite".to_string());
        }

        Ok(Self { x_scale, y_scale })
    }

    /// Нормализует точки для внутреннего фиттинга.
    pub(super) fn normalize_points(self, points: &Points) -> Result<Points, String> {
        let normalized = points
            .iter()
            .map(|point| {
                Point::try_new(point.x() / self.x_scale, point.y() / self.y_scale)
                    .map_err(|error| format!("Normalized point must be finite: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Points::try_from(normalized)
            .map_err(|error| format!("Normalized points are invalid: {error}"))
    }

    /// Переводит параметры из исходного масштаба в нормализованный.
    pub(super) fn normalize_params(self, params: &CurveParams) -> Result<CurveParams, String> {
        self.transform_params(params, ScaleDirection::ToNormalized)
    }

    /// Переводит параметры из нормализованного масштаба обратно в исходный.
    pub(super) fn denormalize_params(self, params: &CurveParams) -> Result<CurveParams, String> {
        self.transform_params(params, ScaleDirection::FromNormalized)
    }

    fn transform_params(
        self,
        params: &CurveParams,
        direction: ScaleDirection,
    ) -> Result<CurveParams, String> {
        let family = params.family();
        let mut values = params.values();

        if family.is_polynomial() {
            self.transform_polynomial_params(&mut values, direction);
        } else if let Some(degree) = family.rational_degree() {
            self.transform_rational_params(&mut values, degree, direction);
        } else if let Some(scales) = Self::static_param_scales(family) {
            if values.len() != scales.len() {
                return Err(format!(
                    "{}: internal scale map mismatch for {} (values={}, scales={})",
                    direction.error_prefix(),
                    family.label(),
                    values.len(),
                    scales.len()
                ));
            }
            for (value, scale) in values.iter_mut().zip(scales.iter().copied()) {
                self.transform_value(value, scale, direction);
            }
        } else {
            match family {
                CurveFamily::Power => self.transform_power_params(&mut values, direction),
                CurveFamily::Linear
                | CurveFamily::Quadratic
                | CurveFamily::Cubic
                | CurveFamily::Quartic
                | CurveFamily::Quintic
                | CurveFamily::Sextic
                | CurveFamily::Septic
                | CurveFamily::Octic
                | CurveFamily::Nonic => unreachable!("handled by polynomial branch above"),
                CurveFamily::Rational11
                | CurveFamily::Rational22
                | CurveFamily::Rational33
                | CurveFamily::Rational44
                | CurveFamily::Rational55 => unreachable!("handled by rational branch above"),
                _ => unreachable!("handled by static scale table above"),
            }
        }

        CurveParams::try_from_values(family, values)
            .map_err(|error| format!("{}: {error}", direction.error_prefix()))
    }

    fn transform_polynomial_params(self, values: &mut [f64], direction: ScaleDirection) {
        let degree = values.len() - 1;
        for (index, value) in values.iter_mut().enumerate() {
            let power = (degree - index) as i32;
            self.transform_value(value, ParamScale::new(power, -1), direction);
        }
    }

    fn transform_power_params(self, values: &mut [f64], direction: ScaleDirection) {
        let power = values[1];
        let factor = self.x_scale.powf(power) / self.y_scale;
        values[0] = direction.apply(values[0], factor);
    }

    fn transform_rational_params(
        self,
        values: &mut [f64],
        degree: usize,
        direction: ScaleDirection,
    ) {
        if degree == 1 {
            self.transform_value(&mut values[0], ParamScale::new(1, -1), direction);
            self.transform_value(&mut values[1], ParamScale::new(0, -1), direction);
            self.transform_value(&mut values[2], ParamScale::new(1, 0), direction);
            self.transform_value(&mut values[3], ParamScale::new(0, -1), direction);
            return;
        }

        let numerator_len = degree + 1;
        for (index, value) in values.iter_mut().enumerate().take(numerator_len) {
            let power = (degree - index) as i32;
            self.transform_value(value, ParamScale::new(power, -1), direction);
        }

        for (index, value) in values.iter_mut().enumerate().skip(numerator_len) {
            let power = (index - degree) as i32;
            self.transform_value(value, ParamScale::new(power, 0), direction);
        }
    }

    fn transform_value(self, value: &mut f64, scale: ParamScale, direction: ScaleDirection) {
        let factor = self.scale_factor(scale);
        *value = direction.apply(*value, factor);
    }

    fn scale_factor(self, scale: ParamScale) -> f64 {
        self.x_scale.powi(scale.x_exp) * self.y_scale.powi(scale.y_exp)
    }

    fn static_param_scales(family: CurveFamily) -> Option<&'static [ParamScale]> {
        const ARRHENIUS_SCALES: [ParamScale; 2] = [ParamScale::new(0, -1), ParamScale::new(-1, 0)];
        const INVERSE_SCALES: [ParamScale; 2] = [ParamScale::new(0, -1), ParamScale::new(-1, -1)];
        const LOGISTIC_SCALES: [ParamScale; 3] = [
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
            ParamScale::new(-1, 0),
        ];
        const BI_EXPONENTIAL_SCALES: [ParamScale; 5] = [
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
            ParamScale::new(0, -1),
        ];
        const DAMPED_SINUSOID_SCALES: [ParamScale; 5] = [
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
            ParamScale::new(1, 0),
            ParamScale::new(0, 0),
            ParamScale::new(0, -1),
        ];
        const LORENTZIAN_SCALES: [ParamScale; 4] = [
            ParamScale::new(0, -1),
            ParamScale::new(-1, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(0, -1),
        ];
        const NATURAL_LOG_SCALES: [ParamScale; 2] =
            [ParamScale::new(0, -1), ParamScale::new(-1, 0)];
        const FOUR_PL_SCALES: [ParamScale; 4] = [
            ParamScale::new(0, -1),
            ParamScale::new(0, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(0, -1),
        ];
        const FIVE_PL_SCALES: [ParamScale; 5] = [
            ParamScale::new(0, -1),
            ParamScale::new(0, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(0, -1),
            ParamScale::new(0, 0),
        ];
        const EXPONENTIAL_BASIC_SCALES: [ParamScale; 3] = [
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
        ];
        const EXPONENTIAL_LINEAR_SCALES: [ParamScale; 4] = [
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
            ParamScale::new(1, -1),
            ParamScale::new(0, -1),
        ];
        const EXPONENTIAL_HALF_LIFE_SCALES: [ParamScale; 3] = [
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(-1, 0),
        ];
        const FALLING_EXPONENTIAL_SCALES: [ParamScale; 3] = [
            ParamScale::new(0, -1),
            ParamScale::new(1, -1),
            ParamScale::new(1, 0),
        ];
        const STEP_LIKE_SCALES: [ParamScale; 4] = [
            ParamScale::new(0, -1),
            ParamScale::new(1, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(0, -1),
        ];
        const GAUSSIAN_SCALES: [ParamScale; 3] = [
            ParamScale::new(0, -1),
            ParamScale::new(-1, 0),
            ParamScale::new(-1, 0),
        ];
        const EMG_SCALES: [ParamScale; 5] = [
            ParamScale::new(-1, -1),
            ParamScale::new(-1, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(0, -1),
        ];
        const PSEUDO_VOIGT_SCALES: [ParamScale; 6] = [
            ParamScale::new(0, -1),
            ParamScale::new(-1, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(-1, 0),
            ParamScale::new(0, 0),
            ParamScale::new(0, -1),
        ];
        const SATURATING_TREND_BASIS_SCALES: [ParamScale; 7] = [
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
            ParamScale::new(0, -1),
        ];

        match family {
            CurveFamily::Arrhenius => Some(&ARRHENIUS_SCALES),
            CurveFamily::Inverse => Some(&INVERSE_SCALES),
            CurveFamily::Logistic | CurveFamily::Gompertz => Some(&LOGISTIC_SCALES),
            CurveFamily::BiExponential => Some(&BI_EXPONENTIAL_SCALES),
            CurveFamily::DampedSinusoid => Some(&DAMPED_SINUSOID_SCALES),
            CurveFamily::Lorentzian => Some(&LORENTZIAN_SCALES),
            CurveFamily::NaturalLog | CurveFamily::MichaelisMenten => Some(&NATURAL_LOG_SCALES),
            CurveFamily::FourPl => Some(&FOUR_PL_SCALES),
            CurveFamily::FivePl => Some(&FIVE_PL_SCALES),
            CurveFamily::ExponentialBasic => Some(&EXPONENTIAL_BASIC_SCALES),
            CurveFamily::ExponentialLinear => Some(&EXPONENTIAL_LINEAR_SCALES),
            CurveFamily::ExponentialHalfLife => Some(&EXPONENTIAL_HALF_LIFE_SCALES),
            CurveFamily::FallingExponential => Some(&FALLING_EXPONENTIAL_SCALES),
            CurveFamily::HyperbolicTangent
            | CurveFamily::ArctangentStep
            | CurveFamily::Softplus => Some(&STEP_LIKE_SCALES),
            CurveFamily::Gaussian => Some(&GAUSSIAN_SCALES),
            CurveFamily::Emg => Some(&EMG_SCALES),
            CurveFamily::PseudoVoigt => Some(&PSEUDO_VOIGT_SCALES),
            CurveFamily::SaturatingTrendBasis1 => Some(&SATURATING_TREND_BASIS_SCALES[..2]),
            CurveFamily::SaturatingTrendBasis2 => Some(&SATURATING_TREND_BASIS_SCALES[..3]),
            CurveFamily::SaturatingTrendBasis3 => Some(&SATURATING_TREND_BASIS_SCALES[..4]),
            CurveFamily::SaturatingTrendBasis4 => Some(&SATURATING_TREND_BASIS_SCALES[..5]),
            CurveFamily::SaturatingTrendBasis5 => Some(&SATURATING_TREND_BASIS_SCALES[..6]),
            CurveFamily::SaturatingTrendBasis6 => Some(&SATURATING_TREND_BASIS_SCALES[..7]),
            CurveFamily::Power
            | CurveFamily::Linear
            | CurveFamily::Quadratic
            | CurveFamily::Cubic
            | CurveFamily::Quartic
            | CurveFamily::Quintic
            | CurveFamily::Sextic
            | CurveFamily::Septic
            | CurveFamily::Octic
            | CurveFamily::Nonic
            | CurveFamily::Rational11
            | CurveFamily::Rational22
            | CurveFamily::Rational33
            | CurveFamily::Rational44
            | CurveFamily::Rational55 => None,
        }
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
