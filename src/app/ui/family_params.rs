use super::*;

pub(super) fn ui_family_and_params(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let can_edit_params = !app.fit_in_progress;
    let icon_tint = ui.visuals().text_color();

    let previous_model = app.selected_model;
    ui.add_enabled_ui(can_edit_params, |ui| {
        egui::ComboBox::from_label(tr(language, "Model type", "Тип модели"))
            .selected_text(model_choice_label(language, app.selected_model))
            .show_ui(ui, |ui| {
                ui.set_min_width(280.0);
                let mut is_first_group = true;
                for group in ModelGroup::ALL {
                    if !is_first_group {
                        ui.separator();
                    }
                    is_first_group = false;
                    ui.label(egui::RichText::new(model_group_label(language, group)).strong());
                    for model in ModelChoice::ALL {
                        if model_group(model) != group {
                            continue;
                        }
                        let model_label = model_choice_label(language, model);
                        let response =
                            ui.selectable_label(app.selected_model == model, model_label);
                        if response.clicked() {
                            app.selected_model = model;
                        }
                    }
                }
            });
    });

    let mut params_need_sync = false;
    if previous_model != app.selected_model {
        params_need_sync = true;
    }

    if app.selected_model.is_polynomial() {
        let previous_degree = app.polynomial_degree;
        ui.add_enabled(
            can_edit_params,
            egui::Slider::new(&mut app.polynomial_degree, 1..=9).text(tr(
                language,
                "Degree",
                "Степень",
            )),
        );
        if previous_degree != app.polynomial_degree {
            params_need_sync = true;
        }
    }

    if params_need_sync {
        app.sync_parameter_inputs();
        app.clear_fit_outputs();
    }

    let formula_info = model_formula_info(language, app.selected_model, app.polynomial_degree);
    let plain_formula = formula_plain_text(&formula_info.full_formula);
    let formula_preview = formula_preview_text(&plain_formula, 78);
    ui.add_space(2.0);
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(10, 8))
        .corner_radius(egui::CornerRadius::same(PANEL_CARD_CORNER_RADIUS))
        .fill(ui.visuals().extreme_bg_color)
        .stroke(egui::Stroke::new(
            1.0,
            ui.visuals().widgets.noninteractive.bg_stroke.color,
        ))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(tr(language, "Model Formula", "Формула модели")).strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::image(open_formula_icon_image(icon_tint)))
                        .on_hover_text(tr(language, "Open formula", "Открыть формулу"))
                        .clicked()
                    {
                        app.panel.show_formula_window = true;
                    }
                });
            });
            ui.monospace(formula_preview.as_ref());
            ui.label(
                egui::RichText::new(tr(
                    language,
                    "Preview only. Open in a separate window to inspect long formulas.",
                    "Показано превью. Откройте отдельное окно для длинных формул.",
                ))
                .small(),
            );
            ui.label(egui::RichText::new(formula_info.notes).small());
        });

    if let Some(family) = app.resolved_model().parametric_family() {
        let mut method_to_apply = None;
        let mut apply_fitted_init = false;
        ui.horizontal_wrapped(|ui| {
            ui.label(tr(language, "Initial parameters", "Начальные параметры"));
            ui.add_enabled_ui(can_edit_params, |ui| {
                ui.menu_button(
                    tr(language, "+ Initialize", "+ Инициализация"),
                    |ui| {
                        let fitted_init_available = app.has_fitted_params_for_family(family);
                        if fitted_init_available {
                            if ui
                                .button(tr(language, "From fitted model", "Из обученной модели"))
                                .clicked()
                            {
                                apply_fitted_init = true;
                                ui.close();
                            }
                        } else {
                            ui.add_enabled(
                                false,
                                egui::Button::new(tr(
                                    language,
                                    "From fitted model (fit this model first)",
                                    "Из обученной модели (сначала обучите эту модель)",
                                )),
                            );
                        }
                        ui.separator();
                        for method in ParamInitMethod::ALL {
                            if method.is_supported_for_family(family) {
                                if ui
                                    .button(param_init_method_label(language, method))
                                    .clicked()
                                {
                                    method_to_apply = Some(method);
                                    ui.close();
                                }
                            } else {
                                ui.add_enabled(
                                    false,
                                    egui::Button::new(param_init_method_disabled_label(
                                        language, method,
                                    )),
                                );
                            }
                        }
                    },
                );
            });
        });

        if apply_fitted_init {
            app.apply_fitted_param_init();
        } else if let Some(method) = method_to_apply {
            app.apply_param_init_method(method);
        }

        egui::Grid::new("parametric_initial_params_grid")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 6.0))
            .show(ui, |ui| {
                for (index, parameter_name) in family.parameter_names().iter().enumerate() {
                    ui.label(*parameter_name);
                    ui.add_enabled(
                        can_edit_params,
                        egui::TextEdit::singleline(&mut app.parameter_inputs[index])
                            .desired_width(120.0),
                    );
                    ui.end_row();
                }
            });
    } else {
        ui.label(tr(
            language,
            "Spline models are non-parametric, but they optimize knot y-values as parameters.",
            "Сплайны непараметрические, но оптимизируют knot y как параметры.",
        ));
        ui.add_space(4.0);
        let min_knots = app
            .resolved_model()
            .spline_min_knots()
            .expect("non-parametric branch guarantees spline model");
        app.spline_knots = app.spline_knots.max(min_knots);
        app.sync_spline_initial_knot_y_inputs(app.spline_knots);
        let mut spline_method_to_apply = None;

        ui.horizontal_wrapped(|ui| {
            ui.label(tr(language, "Initial parameters", "Начальные параметры"));
            ui.add_enabled_ui(can_edit_params, |ui| {
                ui.menu_button(
                    tr(language, "+ Initialize", "+ Инициализация"),
                    |ui| {
                        for method in ParamInitMethod::ALL {
                            if ui
                                .button(param_init_method_label(language, method))
                                .clicked()
                            {
                                spline_method_to_apply = Some(method);
                                ui.close();
                            }
                        }
                    },
                );
            });
        });

        if let Some(method) = spline_method_to_apply {
            app.apply_spline_param_init_method(method);
        }

        ui.add_enabled_ui(can_edit_params, |ui| {
            ui.add(
                egui::Slider::new(&mut app.spline_knots, min_knots..=40).text(tr(
                    language,
                    "Knot count",
                    "Число узлов",
                )),
            );
            egui::ComboBox::from_label(tr(language, "Knot reduction", "Редукция узлов"))
                .selected_text(spline_knot_strategy_label(
                    language,
                    app.spline_knot_strategy,
                ))
                .show_ui(ui, |ui| {
                    for strategy in SplineKnotStrategy::ALL {
                        ui.selectable_value(
                            &mut app.spline_knot_strategy,
                            strategy,
                            spline_knot_strategy_label(language, strategy),
                        );
                    }
                });
            egui::ComboBox::from_label(tr(language, "Extrapolation", "Экстраполяция"))
                .selected_text(spline_extrapolation_label(
                    language,
                    app.spline_extrapolation,
                ))
                .show_ui(ui, |ui| {
                    for extrapolation in SplineExtrapolation::ALL {
                        ui.selectable_value(
                            &mut app.spline_extrapolation,
                            extrapolation,
                            spline_extrapolation_label(language, extrapolation),
                        );
                    }
                });
            egui::ComboBox::from_label(tr(language, "Duplicate x", "Дубли x"))
                .selected_text(spline_duplicate_policy_label(
                    language,
                    app.spline_duplicate_x_policy,
                ))
                .show_ui(ui, |ui| {
                    for policy in SplineDuplicateXPolicy::ALL {
                        ui.selectable_value(
                            &mut app.spline_duplicate_x_policy,
                            policy,
                            spline_duplicate_policy_label(language, policy),
                        );
                    }
                });
        });
        app.sync_spline_initial_knot_y_inputs(app.spline_knots);
        ui.label(format!(
            "{}: {}",
            tr(
                language,
                "Target spline parameter count",
                "Целевое число параметров сплайна"
            ),
            app.spline_knots
        ));
        ui.label(tr(
            language,
            "Initial knot y values",
            "Начальные значения knot y",
        ));
        egui::ScrollArea::vertical()
            .id_salt("spline_knot_y_inputs_scroll")
            .max_height(SPLINE_KNOT_INPUTS_MAX_HEIGHT)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("spline_initial_knot_y_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 6.0))
                    .show(ui, |ui| {
                        for (index, value) in
                            app.spline_initial_knot_y_inputs.iter_mut().enumerate()
                        {
                            ui.label(format!("knot_y[{index}]"));
                            ui.add_enabled(
                                can_edit_params,
                                egui::TextEdit::singleline(value).desired_width(120.0),
                            );
                            ui.end_row();
                        }
                    });
            });
        ui.label(egui::RichText::new(tr(
                language,
                "More knots means better fit, less smoothing; fewer knots means stronger smoothing.",
                "Больше узлов — более точная подгонка, меньше сглаживания; меньше узлов — более сильное сглаживание.",
            )).small());
        ui.label(egui::RichText::new(tr(
                language,
                "When x-values contain duplicates you can merge them automatically instead of failing.",
                "При повторяющихся x можно автоматически объединять точки вместо ошибки.",
            )).small());
        ui.label(
            egui::RichText::new(tr(
                language,
                "Sample density is selected automatically from knot count and data size.",
                "Плотность сэмплирования выбирается автоматически по числу узлов и размеру данных.",
            ))
            .small(),
        );
    }
}
