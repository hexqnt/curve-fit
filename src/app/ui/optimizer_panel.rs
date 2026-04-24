//! Панель выбора оптимизатора, preset-ов и настроек запуска подгонки.

use super::*;

const COMPACT_FIT_BUTTON_WIDTH: f32 = 118.0;
const COMPACT_FIT_BUTTON_HEIGHT: f32 = 30.0;
const FULL_FIT_BUTTON_HEIGHT: f32 = 34.0;

fn summary_row(ui: &mut egui::Ui, label: &str, value: impl std::fmt::Display) {
    ui.label(label);
    ui.monospace(value.to_string());
    ui.end_row();
}

fn ui_log_slider(
    ui: &mut egui::Ui,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    text: &'static str,
) {
    ui.add(egui::Slider::new(value, range).logarithmic(true).text(text));
}

fn ui_wolfe_line_search_sliders(
    ui: &mut egui::Ui,
    c1: &mut f64,
    c2: &mut f64,
    step_min: &mut f64,
    step_max: &mut f64,
    width_tolerance: &mut f64,
) {
    ui_log_slider(ui, c1, C1_MIN..=0.2, "c1");
    ui.add(egui::Slider::new(c2, 0.1..=C2_MAX).text("c2"));
    ui_log_slider(ui, step_min, STEP_MIN_MIN..=1.0, "step_min");
    ui_log_slider(ui, step_max, 1e-6..=STEP_MAX_MAX, "step_max");
    ui_log_slider(ui, width_tolerance, 1e-14..=1e-3, "width_tolerance");
}

fn edit_optimizer_inputs<T>(
    ui: &mut egui::Ui,
    inputs: &mut T,
    preset: &mut OptimizerPreset,
    edit_ui: impl FnOnce(&mut egui::Ui, &mut T),
    normalize: impl FnOnce(&mut T),
) where
    T: Clone + PartialEq,
{
    let before = inputs.clone();
    edit_ui(ui, inputs);
    normalize(inputs);
    if *inputs != before {
        *preset = OptimizerPreset::Custom;
    }
}

pub(super) fn ui_optimizer(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let icon_tint = ui.visuals().text_color();
    egui::ComboBox::from_label(tr(language, "Method", "Метод"))
        .selected_text(optimizer_method_label(language, app.optimizer_method))
        .show_ui(ui, |ui| {
            for method in OptimizerMethod::ALL {
                ui.selectable_value(
                    &mut app.optimizer_method,
                    method,
                    optimizer_method_label(language, method),
                );
            }
        });
    ui.horizontal_wrapped(|ui| {
        let basic_response = ui.selectable_value(
            &mut app.optimizer_mode,
            OptimizerUiMode::Basic,
            tr(language, "Basic", "Базовый"),
        );
        let _ = CurveFitApp::info_hover(
            basic_response,
            optimizer_mode_hint(language, OptimizerUiMode::Basic),
        );
        let advanced_response = ui.selectable_value(
            &mut app.optimizer_mode,
            OptimizerUiMode::Advanced,
            tr(language, "Advanced", "Продвинутый"),
        );
        let _ = CurveFitApp::info_hover(
            advanced_response,
            optimizer_mode_hint(language, OptimizerUiMode::Advanced),
        );
    });

    if app.optimizer_mode == OptimizerUiMode::Basic {
        let previous_preset = app.selected_optimizer_preset();
        let mut selected_preset = previous_preset;
        egui::ComboBox::from_label(tr(language, "Preset", "Пресет"))
            .selected_text(optimizer_preset_label(language, selected_preset))
            .show_ui(ui, |ui| {
                for preset in OptimizerPreset::ALL {
                    ui.selectable_value(
                        &mut selected_preset,
                        preset,
                        optimizer_preset_label(language, preset),
                    );
                }
                if selected_preset == OptimizerPreset::Custom {
                    ui.add_enabled(
                        false,
                        egui::Button::new(optimizer_preset_label(
                            language,
                            OptimizerPreset::Custom,
                        )),
                    );
                }
            });
        if selected_preset != previous_preset {
            if selected_preset == OptimizerPreset::Custom {
                app.set_selected_optimizer_preset(OptimizerPreset::Custom);
            } else {
                app.apply_selected_optimizer_preset(selected_preset);
            }
        }
        ui.add_space(2.0);
        egui::Grid::new("optimizer_basic_summary")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| match app.optimizer_method {
                OptimizerMethod::Lbfgs => {
                    summary_row(ui, "history_size", app.lbfgs_inputs.history_size);
                    summary_row(ui, "max_iters", app.lbfgs_inputs.max_iters);
                    summary_row(ui, "tol_grad", format!("{:.2e}", app.lbfgs_inputs.tol_grad));
                    summary_row(ui, "tol_cost", format!("{:.2e}", app.lbfgs_inputs.tol_cost));
                }
                OptimizerMethod::NelderMead => {
                    summary_row(ui, "max_iters", app.nelder_mead_inputs.max_iters);
                    summary_row(
                        ui,
                        "simplex_scale",
                        format!("{:.3}", app.nelder_mead_inputs.simplex_scale),
                    );
                    summary_row(
                        ui,
                        "sd_tolerance",
                        format!("{:.2e}", app.nelder_mead_inputs.sd_tolerance),
                    );
                }
                OptimizerMethod::SteepestDescent => {
                    summary_row(ui, "max_iters", app.steepest_descent_inputs.max_iters);
                    summary_row(ui, "c1", format!("{:.2e}", app.steepest_descent_inputs.c1));
                    summary_row(ui, "c2", format!("{:.3}", app.steepest_descent_inputs.c2));
                    summary_row(
                        ui,
                        "width_tolerance",
                        format!("{:.2e}", app.steepest_descent_inputs.width_tolerance),
                    );
                }
                OptimizerMethod::NewtonCg => {
                    summary_row(ui, "max_iters", app.newton_cg_inputs.max_iters);
                    summary_row(ui, "tol", format!("{:.2e}", app.newton_cg_inputs.tol));
                    summary_row(
                        ui,
                        "curvature_threshold",
                        format!("{:.2e}", app.newton_cg_inputs.curvature_threshold),
                    );
                    summary_row(ui, "c1", format!("{:.2e}", app.newton_cg_inputs.c1));
                    summary_row(ui, "c2", format!("{:.3}", app.newton_cg_inputs.c2));
                }
                OptimizerMethod::Sgd => {
                    summary_row(ui, "max_iters", app.sgd_inputs.max_iters);
                    summary_row(
                        ui,
                        "learning_rate",
                        format!("{:.2e}", app.sgd_inputs.learning_rate),
                    );
                }
                OptimizerMethod::Adam => {
                    summary_row(ui, "max_iters", app.adam_inputs.max_iters);
                    summary_row(
                        ui,
                        "learning_rate",
                        format!("{:.2e}", app.adam_inputs.learning_rate),
                    );
                }
            });
    } else {
        match app.optimizer_method {
            OptimizerMethod::Lbfgs => {
                edit_optimizer_inputs(
                    ui,
                    &mut app.lbfgs_inputs,
                    &mut app.lbfgs_preset,
                    |ui, inputs| {
                        ui.add(
                            egui::Slider::new(&mut inputs.history_size, 1..=50)
                                .text("history_size"),
                        );
                        ui.add(
                            egui::Slider::new(&mut inputs.max_iters, 10..=10_000).text("max_iters"),
                        );
                        ui_log_slider(ui, &mut inputs.tol_grad, 1e-12..=1e-2, "tol_grad");
                        ui_log_slider(ui, &mut inputs.tol_cost, 1e-14..=1e-4, "tol_cost");
                        ui_wolfe_line_search_sliders(
                            ui,
                            &mut inputs.c1,
                            &mut inputs.c2,
                            &mut inputs.step_min,
                            &mut inputs.step_max,
                            &mut inputs.width_tolerance,
                        );
                    },
                    LbfgsInputState::normalize_after_ui,
                );
            }
            OptimizerMethod::NelderMead => {
                edit_optimizer_inputs(
                    ui,
                    &mut app.nelder_mead_inputs,
                    &mut app.nelder_mead_preset,
                    |ui, inputs| {
                        ui.add(
                            egui::Slider::new(&mut inputs.max_iters, 10..=10_000).text("max_iters"),
                        );
                        ui_log_slider(ui, &mut inputs.simplex_scale, 1e-4..=1.0, "simplex_scale");
                        ui_log_slider(ui, &mut inputs.sd_tolerance, 1e-14..=1e-2, "sd_tolerance");
                        ui_log_slider(ui, &mut inputs.alpha, 1e-3..=5.0, "alpha");
                        ui_log_slider(ui, &mut inputs.gamma, 1.0001..=5.0, "gamma");
                        ui_log_slider(ui, &mut inputs.rho, 1e-4..=0.5, "rho");
                        ui_log_slider(ui, &mut inputs.sigma, 1e-4..=1.0, "sigma");
                    },
                    NelderMeadInputState::normalize_after_ui,
                );
            }
            OptimizerMethod::SteepestDescent => {
                edit_optimizer_inputs(
                    ui,
                    &mut app.steepest_descent_inputs,
                    &mut app.steepest_descent_preset,
                    |ui, inputs| {
                        ui.add(
                            egui::Slider::new(&mut inputs.max_iters, 10..=10_000).text("max_iters"),
                        );
                        ui_wolfe_line_search_sliders(
                            ui,
                            &mut inputs.c1,
                            &mut inputs.c2,
                            &mut inputs.step_min,
                            &mut inputs.step_max,
                            &mut inputs.width_tolerance,
                        );
                    },
                    SteepestDescentInputState::normalize_after_ui,
                );
            }
            OptimizerMethod::NewtonCg => {
                edit_optimizer_inputs(
                    ui,
                    &mut app.newton_cg_inputs,
                    &mut app.newton_cg_preset,
                    |ui, inputs| {
                        ui.add(
                            egui::Slider::new(&mut inputs.max_iters, 10..=10_000).text("max_iters"),
                        );
                        ui_log_slider(ui, &mut inputs.tol, 1e-14..=1e-2, "tol");
                        ui.add(
                            egui::Slider::new(&mut inputs.curvature_threshold, 0.0..=1e-2)
                                .logarithmic(true)
                                .smallest_positive(1e-14)
                                .text("curvature_threshold"),
                        );
                        ui_wolfe_line_search_sliders(
                            ui,
                            &mut inputs.c1,
                            &mut inputs.c2,
                            &mut inputs.step_min,
                            &mut inputs.step_max,
                            &mut inputs.width_tolerance,
                        );
                    },
                    NewtonCgInputState::normalize_after_ui,
                );
            }
            OptimizerMethod::Sgd => {
                edit_optimizer_inputs(
                    ui,
                    &mut app.sgd_inputs,
                    &mut app.sgd_preset,
                    |ui, inputs| {
                        ui.add(
                            egui::Slider::new(&mut inputs.max_iters, 10..=10_000).text("max_iters"),
                        );
                        ui_log_slider(ui, &mut inputs.learning_rate, 1e-6..=1.0, "learning_rate");
                    },
                    SgdInputState::normalize_after_ui,
                );
            }
            OptimizerMethod::Adam => {
                edit_optimizer_inputs(
                    ui,
                    &mut app.adam_inputs,
                    &mut app.adam_preset,
                    |ui, inputs| {
                        ui.add(
                            egui::Slider::new(&mut inputs.max_iters, 10..=10_000).text("max_iters"),
                        );
                        ui_log_slider(ui, &mut inputs.learning_rate, 1e-6..=1.0, "learning_rate");
                    },
                    AdamInputState::normalize_after_ui,
                );
            }
        }
    }

    if ui
        .add(egui::Button::image_and_text(
            reset_icon_image(icon_tint),
            tr(language, "Reset Defaults", "Сбросить по умолчанию"),
        ))
        .clicked()
    {
        app.apply_selected_optimizer_preset(OptimizerPreset::Balanced);
    }

    ui.separator();
    ui_fit_action_button(app, ui, false, true);
    if app.fit_in_progress
        && let Some(iteration) = app.fit_preview_iteration
    {
        ui.label(format!(
            "{}: {iteration}",
            tr(app.ui_language, "Iteration", "Итерация")
        ));
    }
}

pub(super) fn ui_optimizer_action_button_compact(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    ui_fit_action_button(app, ui, true, false);
}

fn ui_fit_action_button(
    app: &mut CurveFitApp,
    ui: &mut egui::Ui,
    compact: bool,
    show_auto_refit_toggle: bool,
) {
    let (fill, stroke, text_color) = CurveFitApp::action_button_style(ui, app.fit_in_progress);
    let (icon, text) = if app.fit_in_progress {
        (
            stop_icon_image(text_color),
            tr(app.ui_language, "Stop", "Стоп"),
        )
    } else {
        (
            fit_icon_image(text_color),
            tr(app.ui_language, "Fit", "Фитинг"),
        )
    };

    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if show_auto_refit_toggle {
                let auto_refit_response = CurveFitApp::toggle_switch_labeled(
                    ui,
                    &mut app.auto_refit_enabled,
                    tr(app.ui_language, "Auto-refit", "Авто-рефит"),
                );
                let _ =
                    CurveFitApp::info_hover(auto_refit_response, auto_refit_hint(app.ui_language));
            }

            let min_size = if compact {
                egui::vec2(COMPACT_FIT_BUTTON_WIDTH, COMPACT_FIT_BUTTON_HEIGHT)
            } else {
                egui::vec2(ui.available_width(), FULL_FIT_BUTTON_HEIGHT)
            };
            let action_button = egui::Button::image_and_text(
                icon,
                egui::RichText::new(text).strong().color(text_color),
            )
            .min_size(min_size)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(egui::CornerRadius::same(UI_CORNER_RADIUS + 1));
            if ui.add(action_button).clicked() {
                if app.fit_in_progress {
                    app.request_stop_fit();
                } else {
                    app.run_fit();
                }
            }
        });
    });
}

fn optimizer_mode_hint(language: UiLanguage, mode: OptimizerUiMode) -> &'static str {
    match mode {
        OptimizerUiMode::Basic => tr(
            language,
            "Basic mode\n- Choose a preset to balance speed vs stability\n- Good default for quick model comparison\n- Switch to Advanced for per-parameter tuning",
            "Базовый режим\n- Выберите пресет для баланса скорости и устойчивости\n- Хороший режим по умолчанию для быстрого сравнения моделей\n- Для точной настройки параметров перейдите в Продвинутый",
        ),
        OptimizerUiMode::Advanced => tr(
            language,
            "Advanced mode\n- Tune solver parameters directly with sliders\n- Any manual change switches preset to Custom\n- Use this mode when convergence is slow or unstable",
            "Продвинутый режим\n- Параметры решателя настраиваются напрямую бегунками\n- Любое ручное изменение переводит пресет в Custom\n- Используйте, если сходимость медленная или нестабильная",
        ),
    }
}

fn auto_refit_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Auto-refit\n- Re-runs fit when right-panel fit settings change\n- If fit is already running, one rerun is queued after completion\n- Applies to model/initial parameters, optimization metric, and optimizer settings",
        "Авто-рефит\n- Повторно запускает фит при изменении fit-настроек в правой панели\n- Если фит уже выполняется, один перезапуск ставится в очередь после завершения\n- Применяется к модели/начальным параметрам, метрике оптимизации и настройкам оптимизатора",
    )
}
