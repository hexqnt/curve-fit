use super::*;

#[derive(Debug, Clone, Default)]
/// Накопленные диагностические ряды по итерациям оптимизации.
pub(super) struct IterationDiagnostics {
    pub(super) family: Option<CurveFamily>,
    pub(super) spline_parameter_count: Option<usize>,
    pub(super) loss_points: Vec<[f64; 2]>,
    pub(super) mse_points: Vec<[f64; 2]>,
    pub(super) rmse_points: Vec<[f64; 2]>,
    pub(super) mae_points: Vec<[f64; 2]>,
    pub(super) soft_l1_points: Vec<[f64; 2]>,
    pub(super) r2_abs_points: Vec<[f64; 2]>,
    pub(super) max_abs_error_points: Vec<[f64; 2]>,
    pub(super) parameter_names: Vec<String>,
    pub(super) parameter_series: Vec<Vec<[f64; 2]>>,
}

impl IterationDiagnostics {
    pub(super) fn clear(&mut self) {
        self.family = None;
        self.spline_parameter_count = None;
        self.clear_metric_points();
        self.parameter_names.clear();
        self.parameter_series.clear();
    }

    pub(super) fn initialize(
        &mut self,
        points: &Points,
        params: &CurveParams,
        loss_metric: OptimizationLossMetric,
    ) {
        let family = params.family();
        self.reset_for_family(family);
        let metrics = calculate_iteration_metrics(points, params, loss_metric);
        self.append(0, metrics, params);
    }

    pub(super) fn append(
        &mut self,
        iteration: u64,
        metrics: IterationMetricSnapshot,
        params: &CurveParams,
    ) {
        let family = params.family();
        if self.family != Some(family) || self.spline_parameter_count.is_some() {
            self.reset_for_family(family);
        }

        let values = params.values();
        if values.len() != self.parameter_series.len() {
            self.reset_for_family(family);
        }

        let iteration = iteration as f64;
        self.upsert_metrics(iteration, metrics);
        for (series, value) in self.parameter_series.iter_mut().zip(values) {
            upsert_iteration_point(series, iteration, value);
        }
    }

    pub(super) fn append_spline(
        &mut self,
        iteration: u64,
        metrics: IterationMetricSnapshot,
        knot_y: &[f64],
    ) {
        let parameter_count = knot_y.len();
        if self.family.is_some() || self.spline_parameter_count != Some(parameter_count) {
            self.reset_for_spline(parameter_count);
        }

        let iteration = iteration as f64;
        self.upsert_metrics(iteration, metrics);
        for (series, value) in self.parameter_series.iter_mut().zip(knot_y.iter().copied()) {
            upsert_iteration_point(series, iteration, value);
        }
    }

    fn upsert_metrics(&mut self, iteration: f64, metrics: IterationMetricSnapshot) {
        upsert_iteration_point(&mut self.loss_points, iteration, metrics.loss);
        upsert_iteration_point(&mut self.mse_points, iteration, metrics.mse);
        upsert_iteration_point(&mut self.rmse_points, iteration, metrics.rmse);
        upsert_iteration_point(&mut self.mae_points, iteration, metrics.mae);
        upsert_iteration_point(&mut self.soft_l1_points, iteration, metrics.soft_l1);
        upsert_iteration_point(&mut self.r2_abs_points, iteration, metrics.r2.abs());
        upsert_iteration_point(
            &mut self.max_abs_error_points,
            iteration,
            metrics.max_abs_error,
        );
    }

    fn reset_for_family(&mut self, family: CurveFamily) {
        self.family = Some(family);
        self.spline_parameter_count = None;
        self.clear_metric_points();
        self.parameter_names = family
            .parameter_names()
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        self.parameter_series = (0..self.parameter_names.len())
            .map(|_| Vec::new())
            .collect();
    }

    fn reset_for_spline(&mut self, parameter_count: usize) {
        self.family = None;
        self.spline_parameter_count = Some(parameter_count);
        self.clear_metric_points();
        self.parameter_names = (0..parameter_count)
            .map(|index| format!("knot_y[{index}]"))
            .collect();
        self.parameter_series = (0..self.parameter_names.len())
            .map(|_| Vec::new())
            .collect();
    }

    /// Очищает только временные ряды метрик, не трогая метаданные параметров.
    fn clear_metric_points(&mut self) {
        self.loss_points.clear();
        self.mse_points.clear();
        self.rmse_points.clear();
        self.mae_points.clear();
        self.soft_l1_points.clear();
        self.r2_abs_points.clear();
        self.max_abs_error_points.clear();
    }
}

/// Добавляет точку в ряд или обновляет последнюю, если итерация совпадает.
fn upsert_iteration_point(series: &mut Vec<[f64; 2]>, iteration: f64, value: f64) {
    if let Some(last) = series.last_mut()
        && (last[0] - iteration).abs() <= f64::EPSILON
    {
        last[1] = value;
    } else {
        series.push([iteration, value]);
    }
}

pub(super) fn diagnostics_plot_y_axis_width(
    plot_response: &PlotResponse<()>,
    plot_slot_left: f32,
) -> f32 {
    (plot_response.response.rect.left() - plot_slot_left).max(0.0)
}
