use super::*;

#[derive(Debug, Clone, Default)]
pub(super) struct IterationDiagnostics {
    pub(super) family: Option<CurveFamily>,
    pub(super) spline_parameter_count: Option<usize>,
    pub(super) loss_mse_points: Vec<[f64; 2]>,
    pub(super) parameter_names: Vec<String>,
    pub(super) parameter_series: Vec<Vec<[f64; 2]>>,
}

impl IterationDiagnostics {
    pub(super) fn clear(&mut self) {
        self.family = None;
        self.spline_parameter_count = None;
        self.loss_mse_points.clear();
        self.parameter_names.clear();
        self.parameter_series.clear();
    }

    pub(super) fn initialize(&mut self, points: &Points, params: &CurveParams) {
        let family = params.family();
        self.reset_for_family(family);
        let (mse, _) = calculate_metrics(points, params);
        self.append(0, mse, params);
    }

    pub(super) fn append(&mut self, iteration: u64, mse: f64, params: &CurveParams) {
        let family = params.family();
        if self.family != Some(family) || self.spline_parameter_count.is_some() {
            self.reset_for_family(family);
        }

        let values = params.values();
        if values.len() != self.parameter_series.len() {
            self.reset_for_family(family);
        }

        let iteration = iteration as f64;
        upsert_iteration_point(&mut self.loss_mse_points, iteration, mse);
        for (series, value) in self.parameter_series.iter_mut().zip(values) {
            upsert_iteration_point(series, iteration, value);
        }
    }

    pub(super) fn append_spline(&mut self, iteration: u64, mse: f64, knot_y: &[f64]) {
        let parameter_count = knot_y.len();
        if self.family.is_some() || self.spline_parameter_count != Some(parameter_count) {
            self.reset_for_spline(parameter_count);
        }

        let iteration = iteration as f64;
        upsert_iteration_point(&mut self.loss_mse_points, iteration, mse);
        for (series, value) in self.parameter_series.iter_mut().zip(knot_y.iter().copied()) {
            upsert_iteration_point(series, iteration, value);
        }
    }

    fn reset_for_family(&mut self, family: CurveFamily) {
        self.family = Some(family);
        self.spline_parameter_count = None;
        self.loss_mse_points.clear();
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
        self.loss_mse_points.clear();
        self.parameter_names = (0..parameter_count)
            .map(|index| format!("knot_y[{index}]"))
            .collect();
        self.parameter_series = (0..self.parameter_names.len())
            .map(|_| Vec::new())
            .collect();
    }
}

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
