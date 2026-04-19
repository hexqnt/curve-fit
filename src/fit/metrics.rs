//! Расчет loss и итоговых метрик качества с опциональной квантизацией наблюдений.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Целевая метрика, по которой оптимизатор минимизирует ошибку.
pub enum OptimizationLossMetric {
    #[default]
    Mse,
    Mae,
    SoftL1,
}

impl OptimizationLossMetric {
    /// Полный список вариантов для UI и переборов.
    pub const ALL: [Self; 3] = [Self::Mse, Self::Mae, Self::SoftL1];

    /// Короткое имя метрики для подписи в легенде.
    pub fn id(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::Mae => "mae",
            Self::SoftL1 => "soft_l1",
        }
    }

    pub(super) fn value_from_residual(self, residual: f64) -> f64 {
        match self {
            Self::Mse => residual * residual,
            Self::Mae => residual.abs(),
            Self::SoftL1 => 2.0 * ((1.0 + residual * residual).sqrt() - 1.0),
        }
    }

    pub(super) fn residual_derivative(self, residual: f64) -> f64 {
        match self {
            Self::Mse => 2.0 * residual,
            Self::Mae => {
                if residual > 0.0 {
                    1.0
                } else if residual < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::SoftL1 => 2.0 * residual / (1.0 + residual * residual).sqrt(),
        }
    }

    pub(super) fn residual_second_derivative(self, residual: f64) -> f64 {
        match self {
            Self::Mse => 2.0,
            Self::Mae => 0.0,
            Self::SoftL1 => 2.0 / (1.0 + residual * residual).powf(1.5),
        }
    }
}

/// Значение по умолчанию для числа знаков после запятой в режиме квантизации метрик.
pub(crate) const DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Число знаков после запятой для квантизации метрик.
pub(crate) struct MetricQuantizationDecimalPlaces(u8);

impl MetricQuantizationDecimalPlaces {
    pub(crate) const MIN: u8 = 0;
    pub(crate) const MAX: u8 = 15;

    pub(crate) fn try_new(value: u8) -> Result<Self, String> {
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(format!(
                "Metric quantization decimal places must be in range {}..={}, got {value}",
                Self::MIN,
                Self::MAX
            ));
        }
        Ok(Self(value))
    }

    pub(crate) fn get(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Конфигурация квантизации для расчета objective и метрик.
pub(crate) enum MetricQuantization {
    #[default]
    Disabled,
    Enabled(MetricQuantizationDecimalPlaces),
}

impl MetricQuantization {
    pub(crate) fn from_ui_state(enabled: bool, decimal_places: u8) -> Result<Self, String> {
        if !enabled {
            return Ok(Self::Disabled);
        }
        Ok(Self::Enabled(MetricQuantizationDecimalPlaces::try_new(
            decimal_places,
        )?))
    }
}

/// Внутренний вспомогательный тип, применяющий квантизацию к `predicted/observed`.
#[derive(Debug, Clone, Copy)]
pub(super) enum ResidualQuantizer {
    Disabled,
    Enabled { scale: f64 },
}

impl ResidualQuantizer {
    pub(super) fn new(metric_quantization: MetricQuantization) -> Self {
        match metric_quantization {
            MetricQuantization::Disabled => Self::Disabled,
            MetricQuantization::Enabled(decimal_places) => Self::Enabled {
                scale: 10_f64.powi(decimal_places.get() as i32),
            },
        }
    }

    #[inline]
    pub(super) fn quantize_value(self, value: f64) -> f64 {
        match self {
            Self::Disabled => value,
            Self::Enabled { scale } => (value * scale).round() / scale,
        }
    }

    #[inline]
    pub(super) fn residual(self, predicted: f64, observed: f64) -> f64 {
        self.quantize_value(predicted) - self.quantize_value(observed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Снимок метрик на одной итерации оптимизации.
pub struct IterationMetricSnapshot {
    pub loss: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub soft_l1: f64,
    pub r2: f64,
    pub max_abs_error: f64,
}

/// Вычисляет базовые метрики качества подгонки: `(MSE, RMSE)`.
pub fn calculate_metrics(points: &Points, params: &CurveParams) -> (f64, f64) {
    calculate_metrics_with_quantization(points, params, MetricQuantization::Disabled)
}

pub(crate) fn calculate_metrics_with_quantization(
    points: &Points,
    params: &CurveParams,
    metric_quantization: MetricQuantization,
) -> (f64, f64) {
    let scalar = calculate_scalar_metrics_from_evaluator(points, metric_quantization, |x| {
        params.evaluate(x)
    });
    (scalar.mse, scalar.rmse)
}

#[cfg(test)]
pub(crate) fn calculate_iteration_metrics(
    points: &Points,
    params: &CurveParams,
    loss_metric: OptimizationLossMetric,
) -> IterationMetricSnapshot {
    calculate_iteration_metrics_with_quantization(
        points,
        params,
        loss_metric,
        MetricQuantization::Disabled,
    )
}

pub(crate) fn calculate_iteration_metrics_with_quantization(
    points: &Points,
    params: &CurveParams,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
) -> IterationMetricSnapshot {
    calculate_iteration_metrics_from_evaluator(points, loss_metric, metric_quantization, |x| {
        params.evaluate(x)
    })
}

pub(super) fn calculate_iteration_metrics_from_evaluator<F>(
    points: &Points,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    evaluate: F,
) -> IterationMetricSnapshot
where
    F: FnMut(f64) -> f64,
{
    let scalar = calculate_scalar_metrics_from_evaluator(points, metric_quantization, evaluate);
    IterationMetricSnapshot {
        loss: scalar_loss_value(loss_metric, &scalar),
        mse: scalar.mse,
        rmse: scalar.rmse,
        mae: scalar.mae,
        soft_l1: scalar.soft_l1,
        r2: scalar.r2,
        max_abs_error: scalar.max_abs_error,
    }
}

pub(super) struct ScalarMetrics {
    pub(super) mse: f64,
    pub(super) rmse: f64,
    pub(super) mae: f64,
    pub(super) soft_l1: f64,
    pub(super) r2: f64,
    pub(super) max_abs_error: f64,
}

pub(super) fn scalar_loss_value(
    loss_metric: OptimizationLossMetric,
    metrics: &ScalarMetrics,
) -> f64 {
    match loss_metric {
        OptimizationLossMetric::Mse => metrics.mse,
        OptimizationLossMetric::Mae => metrics.mae,
        OptimizationLossMetric::SoftL1 => metrics.soft_l1,
    }
}

pub(super) fn calculate_scalar_metrics_from_evaluator<F>(
    points: &Points,
    metric_quantization: MetricQuantization,
    mut evaluate: F,
) -> ScalarMetrics
where
    F: FnMut(f64) -> f64,
{
    let quantizer = ResidualQuantizer::new(metric_quantization);
    let sample_count = points.len() as f64;
    let y_mean = points
        .as_slice()
        .iter()
        .map(|point| quantizer.quantize_value(point.y()))
        .sum::<f64>()
        / sample_count;

    let mut sse = 0.0;
    let mut sae = 0.0;
    let mut soft_l1_sum = 0.0;
    let mut max_abs_error = 0.0_f64;
    for point in points.as_slice() {
        let residual = quantizer.residual(evaluate(point.x()), point.y());
        let abs_residual = residual.abs();
        sse += residual * residual;
        sae += abs_residual;
        soft_l1_sum += OptimizationLossMetric::SoftL1.value_from_residual(residual);
        max_abs_error = max_abs_error.max(abs_residual);
    }

    let sst = points
        .as_slice()
        .iter()
        .map(|point| {
            let centered = quantizer.quantize_value(point.y()) - y_mean;
            centered * centered
        })
        .sum::<f64>();
    let mse = sse / sample_count;
    let rmse = mse.sqrt();
    let mae = sae / sample_count;
    let soft_l1 = soft_l1_sum / sample_count;
    let r2 = if sst <= 1e-15 {
        if sse <= 1e-15 { 1.0 } else { 0.0 }
    } else {
        1.0 - sse / sst
    };

    ScalarMetrics {
        mse,
        rmse,
        mae,
        soft_l1,
        r2,
        max_abs_error,
    }
}

pub(super) struct EvaluatorMetrics {
    pub(super) mse: f64,
    pub(super) rmse: f64,
    pub(super) mae: f64,
    pub(super) soft_l1: f64,
    pub(super) r2: f64,
    pub(super) max_abs_error: f64,
    pub(super) residuals: Vec<[f64; 2]>,
}

pub(super) fn calculate_metrics_from_evaluator<F>(
    points: &Points,
    metric_quantization: MetricQuantization,
    mut evaluate: F,
) -> EvaluatorMetrics
where
    F: FnMut(f64) -> f64,
{
    let scalar =
        calculate_scalar_metrics_from_evaluator(points, metric_quantization, &mut evaluate);

    let mut residuals = Vec::with_capacity(points.len());
    for point in points.as_slice() {
        let residual = evaluate(point.x()) - point.y();
        residuals.push([point.x(), residual]);
    }

    EvaluatorMetrics {
        mse: scalar.mse,
        rmse: scalar.rmse,
        mae: scalar.mae,
        soft_l1: scalar.soft_l1,
        r2: scalar.r2,
        max_abs_error: scalar.max_abs_error,
        residuals,
    }
}
use super::*;
