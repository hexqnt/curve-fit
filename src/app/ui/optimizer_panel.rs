use super::*;

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
        ui.selectable_value(
            &mut app.optimizer_mode,
            OptimizerUiMode::Basic,
            tr(language, "Basic", "Базовый"),
        );
        ui.selectable_value(
            &mut app.optimizer_mode,
            OptimizerUiMode::Advanced,
            tr(language, "Advanced", "Продвинутый"),
        );
        CurveFitApp::info_tooltip(ui, optimizer_mode_hint(language, app.optimizer_mode));
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
                    ui.label("history_size");
                    ui.monospace(app.lbfgs_inputs.history_size.to_string());
                    ui.end_row();
                    ui.label("max_iters");
                    ui.monospace(app.lbfgs_inputs.max_iters.to_string());
                    ui.end_row();
                    ui.label("tol_grad");
                    ui.monospace(format!("{:.2e}", app.lbfgs_inputs.tol_grad));
                    ui.end_row();
                    ui.label("tol_cost");
                    ui.monospace(format!("{:.2e}", app.lbfgs_inputs.tol_cost));
                    ui.end_row();
                }
                OptimizerMethod::NelderMead => {
                    ui.label("max_iters");
                    ui.monospace(app.nelder_mead_inputs.max_iters.to_string());
                    ui.end_row();
                    ui.label("simplex_scale");
                    ui.monospace(format!("{:.3}", app.nelder_mead_inputs.simplex_scale));
                    ui.end_row();
                    ui.label("sd_tolerance");
                    ui.monospace(format!("{:.2e}", app.nelder_mead_inputs.sd_tolerance));
                    ui.end_row();
                }
                OptimizerMethod::SteepestDescent => {
                    ui.label("max_iters");
                    ui.monospace(app.steepest_descent_inputs.max_iters.to_string());
                    ui.end_row();
                    ui.label("c1");
                    ui.monospace(format!("{:.2e}", app.steepest_descent_inputs.c1));
                    ui.end_row();
                    ui.label("c2");
                    ui.monospace(format!("{:.3}", app.steepest_descent_inputs.c2));
                    ui.end_row();
                    ui.label("width_tolerance");
                    ui.monospace(format!(
                        "{:.2e}",
                        app.steepest_descent_inputs.width_tolerance
                    ));
                    ui.end_row();
                }
                OptimizerMethod::NewtonCg => {
                    ui.label("max_iters");
                    ui.monospace(app.newton_cg_inputs.max_iters.to_string());
                    ui.end_row();
                    ui.label("tol");
                    ui.monospace(format!("{:.2e}", app.newton_cg_inputs.tol));
                    ui.end_row();
                    ui.label("curvature_threshold");
                    ui.monospace(format!("{:.2e}", app.newton_cg_inputs.curvature_threshold));
                    ui.end_row();
                    ui.label("c1");
                    ui.monospace(format!("{:.2e}", app.newton_cg_inputs.c1));
                    ui.end_row();
                    ui.label("c2");
                    ui.monospace(format!("{:.3}", app.newton_cg_inputs.c2));
                    ui.end_row();
                }
                OptimizerMethod::Sgd => {
                    ui.label("max_iters");
                    ui.monospace(app.sgd_inputs.max_iters.to_string());
                    ui.end_row();
                    ui.label("learning_rate");
                    ui.monospace(format!("{:.2e}", app.sgd_inputs.learning_rate));
                    ui.end_row();
                }
                OptimizerMethod::Adam => {
                    ui.label("max_iters");
                    ui.monospace(app.adam_inputs.max_iters.to_string());
                    ui.end_row();
                    ui.label("learning_rate");
                    ui.monospace(format!("{:.2e}", app.adam_inputs.learning_rate));
                    ui.end_row();
                }
            });
    } else {
        match app.optimizer_method {
            OptimizerMethod::Lbfgs => {
                let before = app.lbfgs_inputs.clone();
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.history_size, 1..=50)
                        .text("history_size"),
                );
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.max_iters, 10..=10_000)
                        .text("max_iters"),
                );
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.tol_grad, 1e-12..=1e-2)
                        .logarithmic(true)
                        .text("tol_grad"),
                );
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.tol_cost, 1e-14..=1e-4)
                        .logarithmic(true)
                        .text("tol_cost"),
                );
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.c1, C1_MIN..=0.2)
                        .logarithmic(true)
                        .text("c1"),
                );
                ui.add(egui::Slider::new(&mut app.lbfgs_inputs.c2, 0.1..=C2_MAX).text("c2"));
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.step_min, STEP_MIN_MIN..=1.0)
                        .logarithmic(true)
                        .text("step_min"),
                );
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.step_max, 1e-6..=STEP_MAX_MAX)
                        .logarithmic(true)
                        .text("step_max"),
                );
                ui.add(
                    egui::Slider::new(&mut app.lbfgs_inputs.width_tolerance, 1e-14..=1e-3)
                        .logarithmic(true)
                        .text("width_tolerance"),
                );

                app.lbfgs_inputs.normalize_after_ui();
                if app.lbfgs_inputs != before {
                    app.lbfgs_preset = OptimizerPreset::Custom;
                }
            }
            OptimizerMethod::NelderMead => {
                let before = app.nelder_mead_inputs.clone();
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.max_iters, 10..=10_000)
                        .text("max_iters"),
                );
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.simplex_scale, 1e-4..=1.0)
                        .logarithmic(true)
                        .text("simplex_scale"),
                );
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.sd_tolerance, 1e-14..=1e-2)
                        .logarithmic(true)
                        .text("sd_tolerance"),
                );
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.alpha, 1e-3..=5.0)
                        .logarithmic(true)
                        .text("alpha"),
                );
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.gamma, 1.0001..=5.0)
                        .logarithmic(true)
                        .text("gamma"),
                );
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.rho, 1e-4..=0.5)
                        .logarithmic(true)
                        .text("rho"),
                );
                ui.add(
                    egui::Slider::new(&mut app.nelder_mead_inputs.sigma, 1e-4..=1.0)
                        .logarithmic(true)
                        .text("sigma"),
                );

                app.nelder_mead_inputs.normalize_after_ui();
                if app.nelder_mead_inputs != before {
                    app.nelder_mead_preset = OptimizerPreset::Custom;
                }
            }
            OptimizerMethod::SteepestDescent => {
                let before = app.steepest_descent_inputs.clone();
                ui.add(
                    egui::Slider::new(&mut app.steepest_descent_inputs.max_iters, 10..=10_000)
                        .text("max_iters"),
                );
                ui.add(
                    egui::Slider::new(&mut app.steepest_descent_inputs.c1, C1_MIN..=0.2)
                        .logarithmic(true)
                        .text("c1"),
                );
                ui.add(
                    egui::Slider::new(&mut app.steepest_descent_inputs.c2, 0.1..=C2_MAX).text("c2"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut app.steepest_descent_inputs.step_min,
                        STEP_MIN_MIN..=1.0,
                    )
                    .logarithmic(true)
                    .text("step_min"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut app.steepest_descent_inputs.step_max,
                        1e-6..=STEP_MAX_MAX,
                    )
                    .logarithmic(true)
                    .text("step_max"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut app.steepest_descent_inputs.width_tolerance,
                        1e-14..=1e-3,
                    )
                    .logarithmic(true)
                    .text("width_tolerance"),
                );

                app.steepest_descent_inputs.normalize_after_ui();
                if app.steepest_descent_inputs != before {
                    app.steepest_descent_preset = OptimizerPreset::Custom;
                }
            }
            OptimizerMethod::NewtonCg => {
                let before = app.newton_cg_inputs.clone();
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.max_iters, 10..=10_000)
                        .text("max_iters"),
                );
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.tol, 1e-14..=1e-2)
                        .logarithmic(true)
                        .text("tol"),
                );
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.curvature_threshold, 0.0..=1e-2)
                        .logarithmic(true)
                        .smallest_positive(1e-14)
                        .text("curvature_threshold"),
                );
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.c1, C1_MIN..=0.2)
                        .logarithmic(true)
                        .text("c1"),
                );
                ui.add(egui::Slider::new(&mut app.newton_cg_inputs.c2, 0.1..=C2_MAX).text("c2"));
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.step_min, STEP_MIN_MIN..=1.0)
                        .logarithmic(true)
                        .text("step_min"),
                );
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.step_max, 1e-6..=STEP_MAX_MAX)
                        .logarithmic(true)
                        .text("step_max"),
                );
                ui.add(
                    egui::Slider::new(&mut app.newton_cg_inputs.width_tolerance, 1e-14..=1e-3)
                        .logarithmic(true)
                        .text("width_tolerance"),
                );

                app.newton_cg_inputs.normalize_after_ui();
                if app.newton_cg_inputs != before {
                    app.newton_cg_preset = OptimizerPreset::Custom;
                }
            }
            OptimizerMethod::Sgd => {
                let before = app.sgd_inputs.clone();
                ui.add(
                    egui::Slider::new(&mut app.sgd_inputs.max_iters, 10..=10_000).text("max_iters"),
                );
                ui.add(
                    egui::Slider::new(&mut app.sgd_inputs.learning_rate, 1e-6..=1.0)
                        .logarithmic(true)
                        .text("learning_rate"),
                );

                app.sgd_inputs.normalize_after_ui();
                if app.sgd_inputs != before {
                    app.sgd_preset = OptimizerPreset::Custom;
                }
            }
            OptimizerMethod::Adam => {
                let before = app.adam_inputs.clone();
                ui.add(
                    egui::Slider::new(&mut app.adam_inputs.max_iters, 10..=10_000)
                        .text("max_iters"),
                );
                ui.add(
                    egui::Slider::new(&mut app.adam_inputs.learning_rate, 1e-6..=1.0)
                        .logarithmic(true)
                        .text("learning_rate"),
                );

                app.adam_inputs.normalize_after_ui();
                if app.adam_inputs != before {
                    app.adam_preset = OptimizerPreset::Custom;
                }
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
    let action_button =
        egui::Button::image_and_text(icon, egui::RichText::new(text).strong().color(text_color))
            .min_size(egui::vec2(ui.available_width(), 34.0))
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
    if app.fit_in_progress
        && let Some(iteration) = app.fit_preview_iteration
    {
        ui.label(format!(
            "{}: {iteration}",
            tr(app.ui_language, "Iteration", "Итерация")
        ));
    }
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
