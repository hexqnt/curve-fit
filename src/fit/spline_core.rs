//! Общая инфраструктура подготовки, оценки и расчета метрик для сплайнов.

use super::*;

#[derive(Debug, Clone, PartialEq)]
/// Подробный результат подгонки сплайна.
pub struct SplineResult {
    pub knots: Vec<[f64; 2]>,
    pub curve: Vec<[f64; 2]>,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub max_abs_error: f64,
    pub residuals: Vec<[f64; 2]>,
    pub iterations: u64,
}

/// Число узлов сплайна по умолчанию.
pub const DEFAULT_SPLINE_KNOTS: usize = 8;
/// Число сэмплов кривой для визуализации по умолчанию.
pub const DEFAULT_SPLINE_SAMPLES: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Стратегия сокращения исходных точек до фиксированного числа узлов.
pub enum SplineKnotStrategy {
    #[default]
    BinMean,
    BinMedian,
}

impl SplineKnotStrategy {
    /// Полный список стратегий для UI и переборов.
    pub const ALL: [Self; 2] = [Self::BinMean, Self::BinMedian];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Политика экстраполяции сплайна вне диапазона узлов.
pub enum SplineExtrapolation {
    #[default]
    Linear,
    Clamp,
}

impl SplineExtrapolation {
    /// Полный список вариантов экстраполяции.
    pub const ALL: [Self; 2] = [Self::Linear, Self::Clamp];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Политика обработки повторяющихся `x` перед построением сплайна.
pub enum SplineDuplicateXPolicy {
    #[default]
    Error,
    MeanY,
    MedianY,
    FirstY,
}

impl SplineDuplicateXPolicy {
    /// Полный список вариантов обработки повторов.
    pub const ALL: [Self; 4] = [Self::Error, Self::MeanY, Self::MedianY, Self::FirstY];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Конфигурация построения и оптимизации сплайна.
pub struct SplineConfig {
    pub knots: usize,
    pub samples: usize,
    pub knot_strategy: SplineKnotStrategy,
    pub extrapolation: SplineExtrapolation,
    pub duplicate_x_policy: SplineDuplicateXPolicy,
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            knots: DEFAULT_SPLINE_KNOTS,
            samples: DEFAULT_SPLINE_SAMPLES,
            knot_strategy: SplineKnotStrategy::default(),
            extrapolation: SplineExtrapolation::default(),
            duplicate_x_policy: SplineDuplicateXPolicy::default(),
        }
    }
}

impl SplineConfig {
    /// Нормализует конфигурацию, обеспечивая минимально допустимые значения.
    pub fn normalized(self) -> Self {
        Self {
            knots: self.knots.max(2),
            samples: self.samples.max(2),
            knot_strategy: self.knot_strategy,
            extrapolation: self.extrapolation,
            duplicate_x_policy: self.duplicate_x_policy,
        }
    }
}

fn aggregate_duplicate_y(values: &[[f64; 2]], policy: SplineDuplicateXPolicy) -> f64 {
    match policy {
        SplineDuplicateXPolicy::Error | SplineDuplicateXPolicy::FirstY => values[0][1],
        SplineDuplicateXPolicy::MeanY => {
            values.iter().map(|point| point[1]).sum::<f64>() / values.len() as f64
        }
        SplineDuplicateXPolicy::MedianY => {
            let mut sorted = values.iter().map(|point| point[1]).collect::<Vec<_>>();
            sorted.sort_by(|a, b| a.total_cmp(b));
            median_of_sorted(&sorted)
        }
    }
}

pub(super) fn sorted_points_with_duplicate_policy(
    points: &Points,
    policy: SplineDuplicateXPolicy,
) -> Result<Vec<[f64; 2]>, FitError> {
    let mut sorted = points
        .iter()
        .map(|point| [point.x(), point.y()])
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));

    if sorted.len() < 2 {
        return Ok(sorted);
    }

    let mut deduplicated = Vec::with_capacity(sorted.len());
    let mut index = 0;
    while index < sorted.len() {
        let x = sorted[index][0];
        let mut next = index + 1;
        // Считаем почти равные x дублями, чтобы сгладить эффект шумов округления.
        while next < sorted.len() && (sorted[next][0] - x).abs() <= 1e-12 {
            next += 1;
        }

        let duplicate_count = next - index;
        if duplicate_count == 1 {
            deduplicated.push(sorted[index]);
            index = next;
            continue;
        }

        if matches!(policy, SplineDuplicateXPolicy::Error) {
            return Err(FitError::InvalidSplineInput(format!(
                "Spline requires unique x values, but found duplicate x={x}"
            )));
        }

        deduplicated.push([x, aggregate_duplicate_y(&sorted[index..next], policy)]);
        index = next;
    }

    Ok(deduplicated)
}

const MIN_LINEAR_SPLINE_KNOTS: usize = 2;
const MIN_MONOTONE_SPLINE_KNOTS: usize = 2;
const MIN_NATURAL_SPLINE_KNOTS: usize = 3;
const MIN_AKIMA_SPLINE_KNOTS: usize = 5;
const SPLINE_FD_REL_STEP: f64 = 1e-6;
const SPLINE_FD_MIN_STEP: f64 = 1e-7;
const SPLINE_CURVE_DOMAIN_PADDING_RATIO: f64 = 0.1;
const SPLINE_CURVE_DOMAIN_EPS: f64 = 1e-9;
const SPLINE_CURVE_DOMAIN_FALLBACK_PADDING: f64 = 1.0;

/// Нормализованный вид сплайновых семейств внутри алгоритмов оптимизации.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplineFamilyKind {
    Linear,
    MonotoneCubic,
    NaturalCubic,
    Akima,
}

impl SplineFamilyKind {
    fn name(self) -> &'static str {
        match self {
            Self::Linear => "Linear spline",
            Self::MonotoneCubic => "Monotone cubic spline",
            Self::NaturalCubic => "Natural cubic spline",
            Self::Akima => "Akima spline",
        }
    }

    fn min_knots(self) -> usize {
        match self {
            Self::Linear => MIN_LINEAR_SPLINE_KNOTS,
            Self::MonotoneCubic => MIN_MONOTONE_SPLINE_KNOTS,
            Self::NaturalCubic => MIN_NATURAL_SPLINE_KNOTS,
            Self::Akima => MIN_AKIMA_SPLINE_KNOTS,
        }
    }
}

/// Вычислитель значения сплайна после подготовки узлов и производных.
pub(super) enum SplineEvaluator {
    Linear {
        extrapolation: SplineExtrapolation,
    },
    CubicHermite {
        derivatives: Vec<f64>,
        extrapolation: SplineExtrapolation,
    },
    NaturalCubic {
        second_derivatives: Vec<f64>,
        extrapolation: SplineExtrapolation,
    },
}

impl SplineEvaluator {
    pub(super) fn evaluate(&self, knots: &[[f64; 2]], x: f64) -> f64 {
        match self {
            Self::Linear { extrapolation } => evaluate_linear_spline(knots, x, *extrapolation),
            Self::CubicHermite {
                derivatives,
                extrapolation,
            } => evaluate_cubic_hermite_spline(knots, derivatives, x, *extrapolation),
            Self::NaturalCubic {
                second_derivatives,
                extrapolation,
            } => evaluate_natural_cubic_spline(knots, second_derivatives, x, *extrapolation),
        }
    }
}

fn build_spline_evaluator(
    family: SplineFamilyKind,
    knots: &[[f64; 2]],
    extrapolation: SplineExtrapolation,
) -> Result<SplineEvaluator, FitError> {
    match family {
        SplineFamilyKind::Linear => Ok(SplineEvaluator::Linear { extrapolation }),
        SplineFamilyKind::MonotoneCubic => {
            let derivatives = build_monotone_cubic_derivatives(knots)?;
            Ok(SplineEvaluator::CubicHermite {
                derivatives,
                extrapolation,
            })
        }
        SplineFamilyKind::NaturalCubic => {
            let second_derivatives = build_natural_cubic_second_derivatives(knots)?;
            Ok(SplineEvaluator::NaturalCubic {
                second_derivatives,
                extrapolation,
            })
        }
        SplineFamilyKind::Akima => {
            let derivatives = build_akima_derivatives(knots)?;
            Ok(SplineEvaluator::CubicHermite {
                derivatives,
                extrapolation,
            })
        }
    }
}

pub(super) fn spline_lbfgs_config() -> LbfgsConfig {
    LbfgsConfig::try_new(7, 150, 1e-6, 1e-10, 1e-4, 0.9, 1e-12, 1.0, 1e-10).unwrap_or_default()
}

pub(super) fn materialize_spline_knots(knot_x: &[f64], knot_y: &[f64]) -> Vec<[f64; 2]> {
    let mut knots = Vec::with_capacity(knot_x.len());
    materialize_spline_knots_into(knot_x, knot_y, &mut knots);
    knots
}

pub(super) fn materialize_spline_knots_into(
    knot_x: &[f64],
    knot_y: &[f64],
    out: &mut Vec<[f64; 2]>,
) {
    debug_assert_eq!(knot_x.len(), knot_y.len());
    out.clear();
    out.reserve(knot_x.len());
    out.extend(
        knot_x
            .iter()
            .copied()
            .zip(knot_y.iter().copied())
            .map(|(x, y)| [x, y]),
    );
}

pub(super) struct SplineProblem {
    family: SplineFamilyKind,
    knot_x: Box<[f64]>,
    points: Points,
    extrapolation: SplineExtrapolation,
    loss_metric: OptimizationLossMetric,
    residual_quantizer: ResidualQuantizer,
}

impl SplineProblem {
    pub(super) fn new(
        family: SplineFamilyKind,
        initial_knots: &[[f64; 2]],
        points: &Points,
        extrapolation: SplineExtrapolation,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Self {
        let knot_x = initial_knots
            .iter()
            .map(|point| point[0])
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            family,
            knot_x,
            points: points.clone(),
            extrapolation,
            loss_metric,
            residual_quantizer: ResidualQuantizer::new(metric_quantization),
        }
    }

    fn evaluate_objective(&self, knot_y: &[f64]) -> f64 {
        if knot_y.len() != self.knot_x.len() {
            return LARGE_COST;
        }

        let mut knot_buffer = Vec::with_capacity(self.knot_x.len());
        self.evaluate_objective_with_knot_buffer(knot_y, &mut knot_buffer)
    }

    fn evaluate_objective_with_knot_buffer(
        &self,
        knot_y: &[f64],
        knot_buffer: &mut Vec<[f64; 2]>,
    ) -> f64 {
        if knot_y.len() != self.knot_x.len() {
            return LARGE_COST;
        }

        materialize_spline_knots_into(self.knot_x.as_ref(), knot_y, knot_buffer);
        self.evaluate_objective_from_knots(knot_buffer)
    }

    fn evaluate_objective_from_knots(&self, knots: &[[f64; 2]]) -> f64 {
        let evaluator = match build_spline_evaluator(self.family, knots, self.extrapolation) {
            Ok(evaluator) => evaluator,
            Err(_) => return LARGE_COST,
        };
        self.accumulate_objective(|x| evaluator.evaluate(knots, x))
    }

    fn accumulate_objective(&self, mut evaluate: impl FnMut(f64) -> f64) -> f64 {
        let mut objective_sum = 0.0;
        let mut max_abs_residual = 0.0_f64;
        for point in &self.points {
            let prediction = self.residual_quantizer.quantize_value(evaluate(point.x()));
            let observed = self.residual_quantizer.quantize_value(point.y());
            let residual = prediction - observed;
            if !residual.is_finite() {
                return LARGE_COST;
            }
            if self.loss_metric == OptimizationLossMetric::Chebyshev {
                max_abs_residual = max_abs_residual.max(residual.abs());
            } else {
                objective_sum += self.loss_metric.value_from_prediction(prediction, observed);
                if !objective_sum.is_finite() {
                    return LARGE_COST;
                }
            }
        }
        if self.loss_metric == OptimizationLossMetric::Chebyshev {
            max_abs_residual
        } else {
            objective_sum / self.points.len() as f64
        }
    }
}

impl CostFunction for SplineProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.evaluate_objective(array1_as_slice(param)))
    }
}

impl Gradient for SplineProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let param_slice = array1_as_slice(param);
        let mut probe = param.clone();
        let mut gradient = Array1::zeros(param_slice.len());
        let mut knot_buffer = Vec::with_capacity(self.knot_x.len());
        for (index, gradient_value) in gradient.iter_mut().enumerate() {
            // Численный градиент по центральной схеме конечной разности.
            let base_step =
                ((param_slice[index].abs() + 1.0) * SPLINE_FD_REL_STEP).max(SPLINE_FD_MIN_STEP);
            let derivative = FD_STEP_RETRY_FACTORS.iter().copied().find_map(|factor| {
                let step = base_step * factor;
                probe[index] = param_slice[index] + step;
                let cost_plus = self
                    .evaluate_objective_with_knot_buffer(array1_as_slice(&probe), &mut knot_buffer);
                probe[index] = param_slice[index] - step;
                let cost_minus = self
                    .evaluate_objective_with_knot_buffer(array1_as_slice(&probe), &mut knot_buffer);
                probe[index] = param_slice[index];
                finite_central_difference(cost_plus, cost_minus, step)
            });
            *gradient_value = derivative.unwrap_or(LARGE_COST);
        }
        Ok(gradient)
    }
}

impl Hessian for SplineProblem {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        numerical_hessian_from_gradient(self, param)
    }
}

fn median_of_sorted(values: &[f64]) -> f64 {
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn median_x_of_sorted_window(window: &[[f64; 2]]) -> f64 {
    let mid = window.len() / 2;
    if window.len().is_multiple_of(2) {
        0.5 * (window[mid - 1][0] + window[mid][0])
    } else {
        window[mid][0]
    }
}

pub(super) fn approximate_spline_knots(
    sorted: &[[f64; 2]],
    max_knots: usize,
    strategy: SplineKnotStrategy,
) -> Vec<[f64; 2]> {
    let max_knots = max_knots.max(2);
    let target = sorted.len().min(max_knots);
    if sorted.len() <= target {
        return sorted.to_vec();
    }

    let n = sorted.len();
    let mut knots = Vec::with_capacity(target);
    let mut y_values = Vec::new();
    for i in 0..target {
        let start = (i * n) / target;
        let end = ((i + 1) * n).div_ceil(target);
        let end = end.min(n);
        let window = &sorted[start..end];

        let knot = match strategy {
            SplineKnotStrategy::BinMean => {
                let count = window.len() as f64;
                let x_mean = window.iter().map(|p| p[0]).sum::<f64>() / count;
                let y_mean = window.iter().map(|p| p[1]).sum::<f64>() / count;
                [x_mean, y_mean]
            }
            SplineKnotStrategy::BinMedian => {
                y_values.clear();
                y_values.extend(window.iter().map(|p| p[1]));
                y_values.sort_by(|a, b| a.total_cmp(b));
                [
                    median_x_of_sorted_window(window),
                    median_of_sorted(&y_values),
                ]
            }
        };
        knots.push(knot);
    }

    // Гарантируем монотонность x после агрегации бинов (защита от округлений float).
    knots.sort_by(|a, b| a[0].total_cmp(&b[0]));
    knots
}

fn slope_between(p0: [f64; 2], p1: [f64; 2]) -> f64 {
    (p1[1] - p0[1]) / (p1[0] - p0[0])
}

pub(super) fn evaluate_linear_spline(
    sorted: &[[f64; 2]],
    x: f64,
    extrapolation: SplineExtrapolation,
) -> f64 {
    let last = sorted.len() - 1;
    if x <= sorted[0][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[0][1],
            SplineExtrapolation::Linear => {
                let slope = slope_between(sorted[0], sorted[1]);
                sorted[0][1] + slope * (x - sorted[0][0])
            }
        };
    }
    if x >= sorted[last][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[last][1],
            SplineExtrapolation::Linear => {
                let slope = slope_between(sorted[last - 1], sorted[last]);
                sorted[last][1] + slope * (x - sorted[last][0])
            }
        };
    }

    let upper = sorted.partition_point(|point| point[0] < x);
    let index = upper.saturating_sub(1).min(last - 1);
    let x0 = sorted[index][0];
    let y0 = sorted[index][1];
    let x1 = sorted[index + 1][0];
    let y1 = sorted[index + 1][1];
    let t = (x - x0) / (x1 - x0);
    y0 * (1.0 - t) + y1 * t
}

fn evaluate_cubic_hermite_spline(
    sorted: &[[f64; 2]],
    derivatives: &[f64],
    x: f64,
    extrapolation: SplineExtrapolation,
) -> f64 {
    let last = sorted.len() - 1;
    if x <= sorted[0][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[0][1],
            SplineExtrapolation::Linear => sorted[0][1] + derivatives[0] * (x - sorted[0][0]),
        };
    }
    if x >= sorted[last][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[last][1],
            SplineExtrapolation::Linear => {
                sorted[last][1] + derivatives[last] * (x - sorted[last][0])
            }
        };
    }

    let upper = sorted.partition_point(|point| point[0] < x);
    let index = upper.saturating_sub(1).min(last - 1);
    let x0 = sorted[index][0];
    let y0 = sorted[index][1];
    let x1 = sorted[index + 1][0];
    let y1 = sorted[index + 1][1];
    let h = x1 - x0;
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    h00 * y0 + h10 * h * derivatives[index] + h01 * y1 + h11 * h * derivatives[index + 1]
}

fn natural_left_slope(sorted: &[[f64; 2]], second_derivatives: &[f64]) -> f64 {
    let h = sorted[1][0] - sorted[0][0];
    (sorted[1][1] - sorted[0][1]) / h
        - h * (2.0 * second_derivatives[0] + second_derivatives[1]) / 6.0
}

fn natural_right_slope(sorted: &[[f64; 2]], second_derivatives: &[f64]) -> f64 {
    let n = sorted.len();
    let h = sorted[n - 1][0] - sorted[n - 2][0];
    (sorted[n - 1][1] - sorted[n - 2][1]) / h
        + h * (2.0 * second_derivatives[n - 1] + second_derivatives[n - 2]) / 6.0
}

fn build_monotone_cubic_derivatives(sorted: &[[f64; 2]]) -> Result<Vec<f64>, FitError> {
    let n = sorted.len();
    if n < 2 {
        return Err(FitError::InvalidSplineInput(
            "Monotone cubic spline requires at least 2 points".to_string(),
        ));
    }
    if n == 2 {
        let dx = sorted[1][0] - sorted[0][0];
        let slope = (sorted[1][1] - sorted[0][1]) / dx;
        return Ok(vec![slope, slope]);
    }

    let mut h = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);
    for window in sorted.windows(2) {
        let dx = window[1][0] - window[0][0];
        h.push(dx);
        delta.push((window[1][1] - window[0][1]) / dx);
    }

    let mut derivatives = vec![0.0; n];
    for i in 1..n - 1 {
        let delta_prev = delta[i - 1];
        let delta_next = delta[i];
        if delta_prev * delta_next <= 0.0 {
            derivatives[i] = 0.0;
            continue;
        }

        let h_prev = h[i - 1];
        let h_next = h[i];
        let w1 = 2.0 * h_next + h_prev;
        let w2 = h_next + 2.0 * h_prev;
        derivatives[i] = (w1 + w2) / (w1 / delta_prev + w2 / delta_next);
    }

    let h0 = h[0];
    let h1 = h[1];
    let d0 = delta[0];
    let d1 = delta[1];
    let mut first = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if first.signum() != d0.signum() {
        first = 0.0;
    } else if d0.signum() != d1.signum() && first.abs() > 3.0 * d0.abs() {
        first = 3.0 * d0;
    }
    derivatives[0] = first;

    let hn2 = h[n - 2];
    let hn3 = h[n - 3];
    let dn2 = delta[n - 2];
    let dn3 = delta[n - 3];
    let mut last = ((2.0 * hn2 + hn3) * dn2 - hn2 * dn3) / (hn2 + hn3);
    if last.signum() != dn2.signum() {
        last = 0.0;
    } else if dn2.signum() != dn3.signum() && last.abs() > 3.0 * dn2.abs() {
        last = 3.0 * dn2;
    }
    derivatives[n - 1] = last;

    Ok(derivatives)
}

fn build_akima_derivatives(sorted: &[[f64; 2]]) -> Result<Vec<f64>, FitError> {
    let n = sorted.len();
    if n < 5 {
        return Err(FitError::InvalidSplineInput(
            "Akima spline requires at least 5 points".to_string(),
        ));
    }

    let mut slopes = Vec::with_capacity(n - 1);
    for window in sorted.windows(2) {
        let dx = window[1][0] - window[0][0];
        slopes.push((window[1][1] - window[0][1]) / dx);
    }

    let slope_at = |index: isize| -> f64 {
        let len = slopes.len() as isize;
        let last = (len - 1) as usize;
        let right_linear = 2.0 * slopes[last] - slopes[last - 1];
        let right_quadratic = 3.0 * slopes[last] - 2.0 * slopes[last - 1];
        if (0..len).contains(&index) {
            return slopes[index as usize];
        }
        if index < -2 {
            return 3.0 * slopes[0] - 2.0 * slopes[1];
        }
        if index == -1 {
            return 2.0 * slopes[0] - slopes[1];
        }
        if index == -2 {
            return 3.0 * slopes[0] - 2.0 * slopes[1];
        }
        if index == len {
            return right_linear;
        }
        if index == len + 1 {
            return right_quadratic;
        }
        right_quadratic
    };

    let mut derivatives = vec![0.0; n];
    for (i, derivative) in derivatives.iter_mut().enumerate() {
        let i = i as isize;
        let m_im2 = slope_at(i - 2);
        let m_im1 = slope_at(i - 1);
        let m_i = slope_at(i);
        let m_ip1 = slope_at(i + 1);
        let w1 = (m_ip1 - m_i).abs();
        let w2 = (m_im1 - m_im2).abs();
        let weight_sum = w1 + w2;
        *derivative = if weight_sum > 1e-12 {
            (w1 * m_im1 + w2 * m_i) / weight_sum
        } else {
            0.5 * (m_im1 + m_i)
        };
    }

    Ok(derivatives)
}

fn build_natural_cubic_second_derivatives(sorted: &[[f64; 2]]) -> Result<Vec<f64>, FitError> {
    let n = sorted.len();
    if n < 3 {
        return Err(FitError::InvalidSplineInput(
            "Natural cubic spline requires at least 3 points".to_string(),
        ));
    }

    let interior_len = n - 2;
    let mut lower = vec![0.0; interior_len];
    let mut diagonal = vec![0.0; interior_len];
    let mut upper = vec![0.0; interior_len];
    let mut rhs = vec![0.0; interior_len];

    for interior_index in 0..interior_len {
        let i = interior_index + 1;
        let h_prev = sorted[i][0] - sorted[i - 1][0];
        let h_next = sorted[i + 1][0] - sorted[i][0];
        lower[interior_index] = h_prev;
        diagonal[interior_index] = 2.0 * (h_prev + h_next);
        upper[interior_index] = h_next;
        rhs[interior_index] = 6.0
            * ((sorted[i + 1][1] - sorted[i][1]) / h_next
                - (sorted[i][1] - sorted[i - 1][1]) / h_prev);
    }

    for i in 1..interior_len {
        let factor = lower[i] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    // Решаем трехдиагональную систему методом Томаса.
    let mut interior = vec![0.0; interior_len];
    interior[interior_len - 1] = rhs[interior_len - 1] / diagonal[interior_len - 1];
    for i in (0..interior_len - 1).rev() {
        interior[i] = (rhs[i] - upper[i] * interior[i + 1]) / diagonal[i];
    }

    let mut second_derivatives = vec![0.0; n];
    for (index, value) in interior.into_iter().enumerate() {
        second_derivatives[index + 1] = value;
    }
    Ok(second_derivatives)
}

fn evaluate_natural_cubic_spline(
    sorted: &[[f64; 2]],
    second_derivatives: &[f64],
    x: f64,
    extrapolation: SplineExtrapolation,
) -> f64 {
    let last = sorted.len() - 1;
    if x <= sorted[0][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[0][1],
            SplineExtrapolation::Linear => {
                sorted[0][1] + natural_left_slope(sorted, second_derivatives) * (x - sorted[0][0])
            }
        };
    }
    if x >= sorted[last][0] {
        return match extrapolation {
            SplineExtrapolation::Clamp => sorted[last][1],
            SplineExtrapolation::Linear => {
                sorted[last][1]
                    + natural_right_slope(sorted, second_derivatives) * (x - sorted[last][0])
            }
        };
    }

    let upper = sorted.partition_point(|point| point[0] < x);
    let index = upper.saturating_sub(1).min(last - 1);
    let x0 = sorted[index][0];
    let y0 = sorted[index][1];
    let x1 = sorted[index + 1][0];
    let y1 = sorted[index + 1][1];
    let h = x1 - x0;
    let a = (x1 - x) / h;
    let b = (x - x0) / h;

    a * y0
        + b * y1
        + ((a * a * a - a) * second_derivatives[index]
            + (b * b * b - b) * second_derivatives[index + 1])
            * (h * h / 6.0)
}

fn sample_sorted_curve<F>(samples: usize, x_bounds: [f64; 2], mut evaluate: F) -> Vec<[f64; 2]>
where
    F: FnMut(f64) -> f64,
{
    let sample_count = samples.max(2);
    let x_min = x_bounds[0];
    let x_max = x_bounds[1];
    let span = x_max - x_min;

    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count - 1) as f64;
            let x = x_min + span * t;
            [x, evaluate(x)]
        })
        .collect()
}

pub(super) fn expanded_spline_curve_x_bounds(x_min: f64, x_max: f64) -> [f64; 2] {
    // Повторяем правило из UI-графика: небольшой запас по X,
    // чтобы линия не "обрубалась" на крайних наблюдениях.
    if (x_max - x_min).abs() < SPLINE_CURVE_DOMAIN_EPS {
        [
            x_min - SPLINE_CURVE_DOMAIN_FALLBACK_PADDING,
            x_max + SPLINE_CURVE_DOMAIN_FALLBACK_PADDING,
        ]
    } else {
        let padding = (x_max - x_min) * SPLINE_CURVE_DOMAIN_PADDING_RATIO;
        [x_min - padding, x_max + padding]
    }
}

fn ensure_min_knot_count(
    knots: usize,
    min_knots: usize,
    spline_name: &str,
) -> Result<(), FitError> {
    if knots < min_knots {
        return Err(FitError::InvalidSplineInput(format!(
            "{spline_name} requires at least {min_knots} knots"
        )));
    }
    Ok(())
}

pub(super) struct PreparedSplineInputs {
    pub(super) config: SplineConfig,
    pub(super) knot_x: Box<[f64]>,
    pub(super) initial_y: Vec<f64>,
    pub(super) curve_x_bounds: [f64; 2],
}

pub(super) fn prepare_spline_inputs(
    points: &Points,
    config: SplineConfig,
    family: SplineFamilyKind,
    initial_knot_y: Option<&[f64]>,
) -> Result<PreparedSplineInputs, FitError> {
    let config = config.normalized();
    if points.len() >= family.min_knots() {
        ensure_min_knot_count(config.knots, family.min_knots(), family.name())?;
    }

    let sorted = sorted_points_with_duplicate_policy(points, config.duplicate_x_policy)?;
    let initial_knots = approximate_spline_knots(&sorted, config.knots, config.knot_strategy);
    if initial_knots.len() < family.min_knots() {
        return Err(FitError::InvalidSplineInput(format!(
            "{} requires at least {} points",
            family.name(),
            family.min_knots()
        )));
    }
    let curve_x_bounds = expanded_spline_curve_x_bounds(sorted[0][0], sorted[sorted.len() - 1][0]);

    let knot_x = initial_knots
        .iter()
        .map(|point| point[0])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let initial_y_default = initial_knots
        .iter()
        .map(|point| point[1])
        .collect::<Vec<_>>();
    let initial_y = if let Some(values) = initial_knot_y {
        let expected = knot_x.len();
        if values.len() != expected {
            return Err(FitError::InvalidSplineInput(format!(
                "Spline initialization expects {expected} values, got {}",
                values.len()
            )));
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(FitError::InvalidSplineInput(format!(
                    "Spline initialization value at index {index} must be finite, got {value}"
                )));
            }
        }
        values.to_vec()
    } else {
        initial_y_default
    };

    Ok(PreparedSplineInputs {
        config,
        knot_x,
        initial_y,
        curve_x_bounds,
    })
}

pub(super) struct SplineFinalizeContext<'a> {
    pub(super) points: &'a Points,
    pub(super) family: SplineFamilyKind,
    pub(super) config: SplineConfig,
    pub(super) knot_x: &'a [f64],
    pub(super) curve_x_bounds: [f64; 2],
    pub(super) loss_metric: OptimizationLossMetric,
    pub(super) metric_quantization: MetricQuantization,
}

pub(super) fn build_spline_result_from_knot_y(
    context: &SplineFinalizeContext<'_>,
    knot_y: &[f64],
    iterations: u64,
) -> Result<(SplineResult, IterationMetricSnapshot), FitError> {
    let built = build_spline_curve_from_knot_y(
        context.family,
        context.config.extrapolation,
        context.config.samples,
        context.knot_x,
        knot_y,
        context.curve_x_bounds,
    )?;
    let metrics =
        calculate_metrics_from_evaluator(context.points, context.metric_quantization, |x| {
            built.evaluator.evaluate(&built.knots, x)
        });

    let result = SplineResult {
        knots: built.knots,
        curve: built.curve,
        mse: metrics.mse,
        rmse: metrics.rmse,
        mae: metrics.mae,
        r2: metrics.r2,
        max_abs_error: metrics.max_abs_error,
        residuals: metrics.residuals,
        iterations,
    };
    let iteration_metrics = IterationMetricSnapshot {
        loss: match context.loss_metric {
            OptimizationLossMetric::Mse => metrics.mse,
            OptimizationLossMetric::Mae => metrics.mae,
            OptimizationLossMetric::SoftL1 => metrics.soft_l1,
            OptimizationLossMetric::Chebyshev => metrics.max_abs_error,
            OptimizationLossMetric::Msle => metrics.msle,
        },
        mse: metrics.mse,
        rmse: metrics.rmse,
        mae: metrics.mae,
        soft_l1: metrics.soft_l1,
        r2: metrics.r2,
        max_abs_error: metrics.max_abs_error,
    };

    Ok((result, iteration_metrics))
}

/// Строит стартовую кривую сплайна из пользовательской инициализации.
///
/// Используется UI для формирования replay-кадра `iteration = 0` до запуска оптимизации.
pub(crate) fn build_spline_initial_curve_from_knot_y(
    points: &Points,
    family: SplineFamilyKind,
    config: SplineConfig,
    knot_y: &[f64],
) -> Result<Vec<[f64; 2]>, FitError> {
    let prepared = prepare_spline_inputs(points, config, family, Some(knot_y))?;
    let built = build_spline_curve_from_knot_y(
        family,
        prepared.config.extrapolation,
        prepared.config.samples,
        prepared.knot_x.as_ref(),
        &prepared.initial_y,
        prepared.curve_x_bounds,
    )?;
    Ok(built.curve)
}

pub(super) struct BuiltSplineCurve {
    pub(super) knots: Vec<[f64; 2]>,
    pub(super) evaluator: SplineEvaluator,
    pub(super) curve: Vec<[f64; 2]>,
}

pub(super) fn build_spline_curve_from_knot_y(
    family: SplineFamilyKind,
    extrapolation: SplineExtrapolation,
    samples: usize,
    knot_x: &[f64],
    knot_y: &[f64],
    curve_x_bounds: [f64; 2],
) -> Result<BuiltSplineCurve, FitError> {
    let knots = materialize_spline_knots(knot_x, knot_y);
    let evaluator = build_spline_evaluator(family, &knots, extrapolation)?;
    let curve = sample_sorted_curve(samples, curve_x_bounds, |x| evaluator.evaluate(&knots, x));
    Ok(BuiltSplineCurve {
        knots,
        evaluator,
        curve,
    })
}
