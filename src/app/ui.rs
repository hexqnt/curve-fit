//! Общие UI-хелперы и разбиение интерфейса по специализированным панелям.

use super::*;
use std::borrow::Cow;

mod diagnostics_panel;
mod family_params;
mod formula_window;
mod header_bar;
mod optimizer_panel;
mod plot_panel;
mod points_editor_panel;
mod result_panel;
mod status_panel;

const PANEL_CARD_CORNER_RADIUS: u8 = 7;
const PANEL_CARD_OUTER_MARGIN_Y: i8 = 4;
const PANEL_CARD_INNER_MARGIN_X: i8 = 10;
const PANEL_CARD_INNER_MARGIN_Y: i8 = 8;
const SPLINE_KNOT_INPUTS_MAX_HEIGHT: f32 = 180.0;
const RESULT_PARAMS_MAX_HEIGHT: f32 = 190.0;
const COLLAPSING_ICON_SCALE: f32 = 1.5;
const COLLAPSING_HEADER_TEXT_OFFSET_X: f32 = 4.0;
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
    pub(super) fn panel_card_frame(ui: &egui::Ui) -> egui::Frame {
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::symmetric(
                PANEL_CARD_INNER_MARGIN_X,
                PANEL_CARD_INNER_MARGIN_Y,
            ))
            .outer_margin(egui::Margin::symmetric(0, PANEL_CARD_OUTER_MARGIN_Y))
            .corner_radius(egui::CornerRadius::same(PANEL_CARD_CORNER_RADIUS))
            .fill(ui.visuals().faint_bg_color)
            .stroke(egui::Stroke::new(
                1.0_f32,
                ui.visuals().widgets.noninteractive.bg_stroke.color,
            ))
    }

    pub(super) fn panel_card_collapsible(
        ui: &mut egui::Ui,
        id_salt: impl std::hash::Hash,
        title: impl Into<egui::WidgetText>,
        add_body: impl FnOnce(&mut egui::Ui),
    ) {
        Self::panel_card_collapsible_with_collapsed_trailing(
            &mut (),
            ui,
            id_salt,
            title,
            |_, ui| {
                add_body(ui);
            },
            |_, _| {},
        );
    }

    pub(super) fn panel_card_collapsible_with_collapsed_trailing<State>(
        state: &mut State,
        ui: &mut egui::Ui,
        id_salt: impl std::hash::Hash,
        title: impl Into<egui::WidgetText>,
        add_body: impl FnOnce(&mut State, &mut egui::Ui),
        add_collapsed_trailing: impl FnOnce(&mut State, &mut egui::Ui),
    ) {
        let title = title.into().heading();
        Self::panel_card_frame(ui).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.scope(|ui| {
                ui.spacing_mut().indent += COLLAPSING_HEADER_TEXT_OFFSET_X;

                let collapsing_id = ui.make_persistent_id(id_salt);
                let mut collapsing_state =
                    egui::containers::collapsing_header::CollapsingState::load_with_default_open(
                        ui.ctx(),
                        collapsing_id,
                        true,
                    );
                let is_collapsed = !collapsing_state.is_open();

                ui.horizontal(|ui| {
                    let previous_item_spacing = ui.spacing().item_spacing;
                    ui.spacing_mut().item_spacing.x = 0.0;
                    collapsing_state.show_toggle_button(ui, paint_enlarged_collapsing_icon);
                    ui.spacing_mut().item_spacing = previous_item_spacing;

                    let mut title_response =
                        ui.add(egui::Label::new(title.clone()).sense(egui::Sense::click()));
                    if title_response.clicked() {
                        collapsing_state.toggle(ui);
                        title_response.mark_changed();
                    }

                    if is_collapsed {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            add_collapsed_trailing(state, ui);
                        });
                    }
                });

                let _ = collapsing_state.show_body_unindented(ui, |ui| {
                    add_body(state, ui);
                });
            });
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

    pub(super) fn action_button_style(
        ui: &egui::Ui,
        is_stop: bool,
    ) -> (egui::Color32, egui::Stroke, egui::Color32) {
        if is_stop {
            if ui.visuals().dark_mode {
                (
                    egui::Color32::from_rgb(120, 58, 49),
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(199, 99, 82)),
                    egui::Color32::from_rgb(255, 238, 232),
                )
            } else {
                (
                    egui::Color32::from_rgb(235, 208, 198),
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(194, 106, 85)),
                    egui::Color32::from_rgb(94, 37, 23),
                )
            }
        } else if ui.visuals().dark_mode {
            (
                egui::Color32::from_rgb(20, 94, 128),
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(98, 199, 232)),
                egui::Color32::from_rgb(227, 247, 255),
            )
        } else {
            (
                egui::Color32::from_rgb(182, 224, 241),
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(68, 146, 178)),
                egui::Color32::from_rgb(13, 67, 86),
            )
        }
    }

    fn toggle_switch(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
        let desired_size = ui.spacing().interact_size.y * egui::vec2(1.50, 0.8);
        let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        if response.clicked() {
            *on = !*on;
            response.mark_changed();
        }

        response.widget_info(|| {
            egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, "")
        });

        if ui.is_rect_visible(rect) {
            let how_on = ui.ctx().animate_bool_responsive(response.id, *on);
            let visuals = ui.style().interact_selectable(&response, *on);
            let rect = rect.expand(visuals.expansion);
            let radius = 0.5 * rect.height();

            ui.painter().rect(
                rect,
                radius,
                visuals.bg_fill,
                visuals.bg_stroke,
                egui::StrokeKind::Inside,
            );

            let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
            let center = egui::pos2(circle_x, rect.center().y);
            ui.painter()
                .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
        }

        response
    }

    fn toggle_switch_labeled(
        ui: &mut egui::Ui,
        on: &mut bool,
        label: impl Into<egui::WidgetText>,
    ) -> egui::Response {
        ui.horizontal(|ui| {
            let switch_response = Self::toggle_switch(ui, on);
            ui.label(label);
            switch_response
        })
        .inner
    }

    fn info_hover(response: egui::Response, text: impl AsRef<str>) -> egui::Response {
        let lines = text
            .as_ref()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        response.on_hover_ui(|ui| {
            ui.set_max_width(380.0);
            ui.spacing_mut().item_spacing.y = 3.0;
            match lines.as_slice() {
                [] => {}
                [single] => {
                    ui.label(*single);
                }
                [title, details @ ..] => {
                    ui.label(egui::RichText::new(*title).strong());
                    for line in details {
                        ui.label(egui::RichText::new(*line).small());
                    }
                }
            }
        })
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

fn paint_enlarged_collapsing_icon(ui: &mut egui::Ui, openness: f32, response: &egui::Response) {
    let enlarged_rect = egui::Rect::from_center_size(
        response.rect.center(),
        response.rect.size() * COLLAPSING_ICON_SCALE,
    )
    .translate(egui::vec2(-0.5 * COLLAPSING_HEADER_TEXT_OFFSET_X, 0.0));
    let enlarged_response = response.clone().with_new_rect(enlarged_rect);
    egui::containers::collapsing_header::paint_default_icon(ui, openness, &enlarged_response);
}

fn formula_preview_text(formula: &str, max_chars: usize) -> Cow<'_, str> {
    if max_chars <= 3 {
        return Cow::Borrowed("...");
    }

    let total_chars = formula.chars().count();
    if total_chars <= max_chars {
        return Cow::Borrowed(formula);
    }

    let mut preview = String::with_capacity(max_chars + 3);
    preview.extend(formula.chars().take(max_chars - 3));
    preview.push_str("...");
    Cow::Owned(preview)
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
