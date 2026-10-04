//! Общие UI-хелперы и разбиение интерфейса по специализированным панелям.

use super::*;

mod diagnostics_panel;
mod family_params;
mod formula_window;
mod header_bar;
mod optimizer_panel;
mod plot_panel;
mod points_editor_panel;
mod result_panel;
mod status_panel;

const SPLINE_KNOT_INPUTS_MAX_HEIGHT: f32 = 180.0;
const RESULT_PARAMS_MAX_HEIGHT: f32 = 190.0;
const SETTINGS_PANEL_SCROLL_ID: &str = "settings_panel_scroll";
const DIAGNOSTICS_SERIES_ID_LOSS: &str = "diagnostics_series_loss";
const DIAGNOSTICS_SERIES_ID_MSE: &str = "diagnostics_series_mse";
const DIAGNOSTICS_SERIES_ID_RMSE: &str = "diagnostics_series_rmse";
const DIAGNOSTICS_SERIES_ID_MAE: &str = "diagnostics_series_mae";
const DIAGNOSTICS_SERIES_ID_SOFT_L1: &str = "diagnostics_series_soft_l1";
const DIAGNOSTICS_SERIES_ID_R2_ABS: &str = "diagnostics_series_r2_abs";
const DIAGNOSTICS_SERIES_ID_MAX_ABS: &str = "diagnostics_series_max_abs";
const DIAGNOSTICS_SELECTED_ITERATION_MARKER_ID_LOSS: &str =
    "diagnostics_selected_iteration_marker_loss";
const DIAGNOSTICS_SELECTED_ITERATION_MARKER_ID_PARAMS: &str =
    "diagnostics_selected_iteration_marker_params";

impl CurveFitApp {
    pub(super) fn setup_side_panel_content(ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing = style::ITEM_SPACING;
        ui.set_width(ui.available_width());
    }

    pub(super) fn right_side_panel_scroll_area(
        ui: &mut egui::Ui,
        add_contents: impl FnOnce(&mut egui::Ui),
    ) {
        egui::ScrollArea::vertical()
            .id_salt(SETTINGS_PANEL_SCROLL_ID)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                Self::setup_side_panel_content(ui);
                add_contents(ui);
            });
    }

    pub(super) fn ui_model_selector_compact(&mut self, ui: &mut egui::Ui) {
        family_params::ui_model_selector_compact(self, ui);
    }

    pub(super) fn ui_optimizer_action_button_compact(&mut self, ui: &mut egui::Ui) {
        optimizer_panel::ui_optimizer_action_button_compact(self, ui);
    }

    pub(super) fn ui_optimization_metric_selector_compact(&mut self, ui: &mut egui::Ui) {
        status_panel::ui_optimization_metric_selector_compact(self, ui);
    }

    pub(super) fn next_unit_random(&mut self) -> f64 {
        self.spray_seed = self
            .spray_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let value = self.spray_seed >> 11;
        value as f64 / ((1_u64 << 53) as f64)
    }

    pub(super) fn next_uniform_unit_disk_offset(&mut self) -> [f64; 2] {
        let radial = self.next_unit_random().sqrt();
        let angle = TAU * self.next_unit_random();
        [radial * angle.cos(), radial * angle.sin()]
    }

    pub(super) fn next_gaussian_unit_disk_offset(&mut self) -> [f64; 2] {
        loop {
            let u = self.next_unit_random();
            let radial = SPRAY_GAUSSIAN_SIGMA * (-2.0 * (1.0 - u).ln()).sqrt();
            if radial <= 1.0 {
                let angle = TAU * self.next_unit_random();
                return [radial * angle.cos(), radial * angle.sin()];
            }
        }
    }

    pub(super) fn next_spray_unit_disk_offset(&mut self) -> [f64; 2] {
        match self.spray_brush {
            SprayBrush::Uniform => self.next_uniform_unit_disk_offset(),
            SprayBrush::Gaussian => self.next_gaussian_unit_disk_offset(),
        }
    }

    fn reset_spray_rate_state(&mut self) {
        self.spray_points_budget = 0.0;
        self.spray_last_emit_at = None;
    }

    pub(super) fn next_spray_points_to_add(&mut self, now: Instant) -> usize {
        let elapsed_seconds = self
            .spray_last_emit_at
            .map(|last_emit_at| now.saturating_duration_since(last_emit_at).as_secs_f64())
            .unwrap_or(1.0 / SPRAY_REFERENCE_FPS);
        self.spray_last_emit_at = Some(now);

        self.spray_points_budget += self.spray_points_per_second as f64 * elapsed_seconds;
        let points_to_add = self.spray_points_budget.floor() as usize;
        self.spray_points_budget -= points_to_add as f64;
        points_to_add
    }

    pub(super) fn ui_header(&mut self, ui: &mut egui::Ui) {
        header_bar::ui_header(self, ui);
    }

    pub(super) fn ui_status_bar(&mut self, ui: &mut egui::Ui) {
        header_bar::ui_status_bar(self, ui);
    }

    pub(super) fn ui_tools(&mut self, ui: &mut egui::Ui) {
        points_editor_panel::ui_tools(self, ui);
    }

    pub(super) fn ui_point_layers(&mut self, ui: &mut egui::Ui) {
        points_editor_panel::ui_point_layers(self, ui);
    }

    pub(super) fn ui_points_editor(&mut self, ui: &mut egui::Ui) {
        points_editor_panel::ui_points_editor(self, ui);
    }

    pub(super) fn ui_family_and_params(&mut self, ui: &mut egui::Ui) {
        family_params::ui_family_and_params(self, ui);
    }

    pub(super) fn ui_formula_window(&mut self, ctx: &egui::Context) {
        formula_window::ui_formula_window(self, ctx);
    }

    pub(super) fn ui_optimizer(&mut self, ui: &mut egui::Ui) {
        optimizer_panel::ui_optimizer(self, ui);
    }

    pub(super) fn ui_optimization_metric(&mut self, ui: &mut egui::Ui) {
        status_panel::ui_optimization_metric(self, ui);
    }

    pub(super) fn ui_status(&self, ui: &mut egui::Ui) {
        status_panel::ui_status(self, ui);
    }

    pub(super) fn ui_plot(&mut self, ui: &mut egui::Ui, height: f32) {
        plot_panel::ui_plot(self, ui, height);
    }

    pub(super) fn ui_iteration_diagnostics(&mut self, ui: &mut egui::Ui) {
        diagnostics_panel::ui_iteration_diagnostics(self, ui);
    }

    pub(super) fn ui_result(&mut self, ui: &mut egui::Ui) {
        result_panel::ui_result(self, ui);
    }
}

fn diagnostics_hidden_non_loss_series_ids() -> [egui::Id; 6] {
    [
        egui::Id::new(DIAGNOSTICS_SERIES_ID_MSE),
        egui::Id::new(DIAGNOSTICS_SERIES_ID_RMSE),
        egui::Id::new(DIAGNOSTICS_SERIES_ID_MAE),
        egui::Id::new(DIAGNOSTICS_SERIES_ID_SOFT_L1),
        egui::Id::new(DIAGNOSTICS_SERIES_ID_R2_ABS),
        egui::Id::new(DIAGNOSTICS_SERIES_ID_MAX_ABS),
    ]
}
