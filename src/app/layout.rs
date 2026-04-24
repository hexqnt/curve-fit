//! Главный `eframe`-цикл и раскладка верхней, боковых и нижней панелей приложения.

use super::*;
impl eframe::App for CurveFitApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        Self::apply_visual_style(ctx);
        self.poll_fit_worker(ctx);
        self.maybe_run_pending_auto_refit();
        self.tick_replay(ctx);
        self.poll_points_clipboard_import(ctx);
        self.poll_clipboard_copy(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_points_file_import_dialog(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_fit_export_save_dialog(ctx);
        self.maybe_refresh_points_cache_after_debounce();

        if !self.fit_in_progress {
            let undo_requested = ctx.input(|input| {
                input.modifiers.command && !input.modifiers.shift && input.key_pressed(egui::Key::Z)
            });
            let redo_requested = ctx.input(|input| {
                input.modifiers.command
                    && (input.key_pressed(egui::Key::Y)
                        || (input.modifiers.shift && input.key_pressed(egui::Key::Z)))
            });
            if undo_requested {
                self.undo_points_edit();
            } else if redo_requested {
                self.redo_points_edit();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let panel_style = ctx.global_style();
        let panel_style = panel_style.as_ref();

        egui::Panel::top("header_panel")
            .frame(Self::top_bottom_panel_frame(panel_style))
            .show_inside(ui, |ui| {
                self.ui_header(ui);
            });

        egui::Panel::bottom("status_bar")
            .frame(Self::top_bottom_panel_frame(panel_style))
            .show_inside(ui, |ui| {
                self.ui_status_bar(ui);
            });

        if self.panel.show_left {
            egui::Panel::left("points_panel")
                .default_size(LEFT_PANEL_DEFAULT_WIDTH)
                .min_size(LEFT_PANEL_MIN_WIDTH)
                .resizable(true)
                .frame(Self::side_panel_frame(panel_style))
                .show_inside(ui, |ui| {
                    let language = self.ui_language;
                    Self::setup_side_panel_content(ui);
                    Self::panel_card_collapsible(
                        ui,
                        "left_section_tools",
                        tr(language, "Tools", "Инструменты"),
                        |ui| {
                            self.ui_tools(ui);
                        },
                    );
                    Self::panel_card_collapsible(
                        ui,
                        "left_section_points",
                        tr(language, "Input Points", "Точки"),
                        |ui| {
                            self.ui_points_editor(ui);
                        },
                    );
                });
        }

        if self.panel.show_right {
            egui::Panel::right("settings_panel")
                .default_size(RIGHT_PANEL_DEFAULT_WIDTH)
                .min_size(RIGHT_PANEL_MIN_WIDTH)
                .resizable(true)
                .frame(Self::side_panel_frame(panel_style))
                .show_inside(ui, |ui| {
                    let language = self.ui_language;
                    Self::right_side_panel_scroll_area(ui, |ui| {
                        Self::panel_card_collapsible_with_collapsed_trailing(
                            self,
                            ui,
                            "right_section_model",
                            tr(language, "Model", "Модель"),
                            CurveFitApp::ui_family_and_params,
                            CurveFitApp::ui_model_selector_compact,
                        );
                        Self::panel_card_collapsible_with_collapsed_trailing(
                            self,
                            ui,
                            "right_section_metric",
                            tr(language, "Optimization metric", "Метрика оптимизации"),
                            CurveFitApp::ui_optimization_metric,
                            CurveFitApp::ui_optimization_metric_selector_compact,
                        );
                        Self::panel_card_collapsible_with_collapsed_trailing(
                            self,
                            ui,
                            "right_section_optimizer",
                            tr(language, "Optimizer", "Оптимизатор"),
                            CurveFitApp::ui_optimizer,
                            CurveFitApp::ui_optimizer_action_button_compact,
                        );
                        Self::panel_card_collapsible(
                            ui,
                            "right_section_result",
                            tr(language, "Result", "Результат"),
                            |ui| {
                                self.ui_result(ui);
                            },
                        );
                    });
                });
            self.track_right_panel_fit_changes_and_maybe_refit();
        }

        self.ui_formula_window(&ctx);

        if self.panel.show_diagnostics {
            egui::Panel::bottom("diagnostics_panel")
                .resizable(true)
                .default_size(DIAGNOSTICS_PANEL_DEFAULT_HEIGHT)
                .min_size(DIAGNOSTICS_PANEL_MIN_HEIGHT)
                .frame(Self::top_bottom_panel_frame(panel_style))
                .show_inside(ui, |ui| {
                    let available_height = ui.available_height();
                    ui.set_height(available_height);
                    self.ui_iteration_diagnostics(ui);
                });
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.ui_plot(ui, ui.available_height().max(2.0));
        });
    }
}
