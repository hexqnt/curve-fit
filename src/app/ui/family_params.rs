//! Выбор модели и начальных параметров на правой панели.

use super::*;

const COLLAPSED_MODEL_SELECTOR_WIDTH: f32 = 170.0;
const COLLAPSED_MODEL_SELECTOR_MENU_MIN_WIDTH: f32 = 260.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelSelectorMode {
    Full,
    Compact,
}

pub(super) fn ui_family_and_params(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let can_edit_params = !app.fit_in_progress;

    let mut params_need_sync = ui_model_selector(app, ui, can_edit_params, ModelSelectorMode::Full);
    let mut tau_grid_changed = false;

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
    if app.selected_model.is_rational() {
        let previous_degree = app.rational_degree;
        ui.add_enabled(
            can_edit_params,
            egui::Slider::new(
                &mut app.rational_degree,
                MIN_RATIONAL_DEGREE..=MAX_RATIONAL_DEGREE,
            )
            .text(tr(language, "Rational degree", "Степень рациональной")),
        );
        if previous_degree != app.rational_degree {
            params_need_sync = true;
        }
    }
    if app.selected_model.is_saturating_trend_basis() {
        let previous_count = app.saturating_trend_tau_count;
        ui.add_enabled(
            can_edit_params,
            egui::Slider::new(
                &mut app.saturating_trend_tau_count,
                MIN_SATURATING_TREND_TAU_COUNT..=MAX_SATURATING_TREND_TAU_COUNT,
            )
            .text(tr(language, "Tau count", "Число tau")),
        );
        if previous_count != app.saturating_trend_tau_count {
            params_need_sync = true;
        }
        app.ensure_saturating_trend_tau_inputs_cover_count();
        ui.horizontal(|ui| {
            let reset_tau_grid = ui.add_enabled(
                can_edit_params,
                egui::Button::new(tr(language, "Reset tau grid", "Сбросить сетку tau")),
            );
            if reset_tau_grid.clicked() {
                app.set_saturating_trend_tau_inputs(&DEFAULT_SATURATING_TREND_TAUS_YEARS);
                tau_grid_changed = true;
            }
            let _ = CurveFitApp::info_hover(
                reset_tau_grid,
                tr(
                    language,
                    "Restores default increasing tau grid: 0.25, 0.5, 1, 2, 4, 8",
                    "Восстанавливает сетку tau по умолчанию: 0.25, 0.5, 1, 2, 4, 8",
                ),
            );
        });
        ui.label(tr(language, "Tau grid (years)", "Сетка tau (в годах)"));
        egui::Grid::new("saturating_trend_tau_grid_inputs")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 6.0))
            .show(ui, |ui| {
                for index in 0..app.saturating_trend_tau_count {
                    ui.label(format!("tau{}", index + 1));
                    let response = ui.add_enabled(
                        can_edit_params,
                        egui::TextEdit::singleline(&mut app.saturating_trend_tau_inputs[index])
                            .desired_width(120.0),
                    );
                    if response.changed() {
                        tau_grid_changed = true;
                    }
                    ui.end_row();
                }
            });
    }

    if params_need_sync {
        app.sync_parameter_inputs();
        app.clear_fit_outputs();
    } else if tau_grid_changed {
        app.clear_fit_outputs();
    }

    if let Some(family) = app.resolved_model().parametric_family() {
        let mut method_to_apply = None;
        let mut apply_fitted_init = false;
        ui.horizontal_wrapped(|ui| {
            let init_label_response =
                ui.label(tr(language, "Initial parameters", "Начальные параметры"));
            let _ = CurveFitApp::info_hover(init_label_response, parametric_init_hint(language));
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
                                let unavailable_label = format!(
                                    "{} ({})",
                                    param_init_method_label(language, method),
                                    tr(language, "not available", "недоступно")
                                );
                                let unavailable_response =
                                    ui.add_enabled(false, egui::Button::new(unavailable_label));
                                let _ = CurveFitApp::info_hover(
                                    unavailable_response,
                                    param_init_method_disabled_label(language, method),
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
        let Some(min_knots) = app.resolved_model().spline_min_knots() else {
            app.status = Some(StatusMessage::Error(
                "Selected model has no spline configuration".to_string(),
            ));
            return;
        };
        app.spline_knots = app.spline_knots.max(min_knots);
        app.sync_spline_initial_knot_y_inputs(app.spline_knots);
        let mut spline_method_to_apply = None;

        ui.horizontal_wrapped(|ui| {
            let init_label_response =
                ui.label(tr(language, "Initial parameters", "Начальные параметры"));
            let _ = CurveFitApp::info_hover(init_label_response, spline_init_hint(language));
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
            )
            .on_hover_text(knot_count_hint(language));
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
                })
                .response
                .on_hover_text(knot_reduction_hint(language));
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
                })
                .response
                .on_hover_text(extrapolation_hint(language));
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
                })
                .response
                .on_hover_text(duplicate_x_hint(language));
        });
        app.sync_spline_initial_knot_y_inputs(app.spline_knots);
        ui.horizontal_wrapped(|ui| {
            let spline_sampling_response = ui.label(format!(
                "{}: {}",
                tr(
                    language,
                    "Target spline parameter count",
                    "Целевое число параметров сплайна"
                ),
                app.spline_knots
            ));
            let _ =
                CurveFitApp::info_hover(spline_sampling_response, spline_sampling_hint(language));
        });
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
    }
}

pub(super) fn ui_model_selector_compact(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let can_edit_params = !app.fit_in_progress;
    if ui_model_selector(app, ui, can_edit_params, ModelSelectorMode::Compact) {
        app.sync_parameter_inputs();
        app.clear_fit_outputs();
    }
}

fn ui_model_selector(
    app: &mut CurveFitApp,
    ui: &mut egui::Ui,
    can_edit_params: bool,
    mode: ModelSelectorMode,
) -> bool {
    let language = app.ui_language;
    let previous_model = app.selected_model;
    ui.add_enabled_ui(can_edit_params, |ui| {
        let selected_text = model_choice_label(language, app.selected_model);
        match mode {
            ModelSelectorMode::Full => {
                egui::ComboBox::from_label(tr(language, "Model type", "Тип модели"))
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        ui.set_min_width(280.0);
                        ui_model_selector_menu(app, ui, language);
                    });
            }
            ModelSelectorMode::Compact => {
                egui::ComboBox::from_id_salt("collapsed_header_model_selector")
                    .selected_text(selected_text)
                    .width(COLLAPSED_MODEL_SELECTOR_WIDTH)
                    .show_ui(ui, |ui| {
                        ui.set_min_width(COLLAPSED_MODEL_SELECTOR_MENU_MIN_WIDTH);
                        ui_model_selector_menu(app, ui, language);
                    });
            }
        }
    });
    app.selected_model != previous_model
}

fn ui_model_selector_menu(app: &mut CurveFitApp, ui: &mut egui::Ui, language: UiLanguage) {
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
            let response = ui.selectable_label(app.selected_model == model, model_label);
            if response.clicked() {
                app.selected_model = model;
            }
        }
    }
}

fn parametric_init_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Initial parameters (parametric models)\n- These values are the optimizer starting point\n- +Initialize can fill defaults/data-based/randomized values\n- \"From fitted model\" reuses parameters from the latest fit of the same family",
        "Начальные параметры (параметрические модели)\n- Эти значения являются стартовой точкой оптимизатора\n- +Инициализация может подставить значения по умолчанию/по данным/случайно\n- \"Из обученной модели\" берёт параметры из последнего фитинга того же семейства",
    )
}

fn spline_init_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Initial parameters (spline models)\n- Spline is non-parametric, but optimizer still tunes knot y-values\n- +Initialize sets starting knot_y values\n- Better initialization usually reduces iteration count",
        "Начальные параметры (сплайны)\n- Сплайн непараметрический, но оптимизатор всё равно настраивает knot y\n- +Инициализация задаёт стартовые значения knot_y\n- Более удачная инициализация обычно снижает число итераций",
    )
}

fn knot_count_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Knot count\n- More knots: more flexible fit, weaker smoothing\n- Fewer knots: stronger smoothing, simpler curve",
        "Число узлов\n- Больше узлов: более гибкая подгонка, слабее сглаживание\n- Меньше узлов: сильнее сглаживание, проще кривая",
    )
}

fn knot_reduction_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Knot reduction strategy\n- Bin mean: smoother on clean data, more sensitive to outliers\n- Bin median: more robust to spikes/noise",
        "Стратегия редукции узлов\n- Среднее по окнам: обычно более гладко на чистых данных, но чувствительнее к выбросам\n- Медиана по окнам: устойчивее к пикам и шуму",
    )
}

fn extrapolation_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Extrapolation outside knot range\n- Clamp to edge: keep boundary y-value\n- Linear: continue using boundary slope",
        "Экстраполяция вне диапазона узлов\n- Фиксация на краю: удерживать граничное значение y\n- Линейная: продолжать по граничному наклону",
    )
}

fn duplicate_x_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Duplicate x handling\n- Error: fail on repeated x\n- Merge by mean/median: aggregate duplicates before fitting\n- Keep first y: preserve earliest sample per x",
        "Обработка дублей x\n- Error: завершить с ошибкой при повторяющихся x\n- Слияние по mean/median: агрегировать дубли до фитинга\n- Keep first y: оставить первый y для каждого x",
    )
}

fn spline_sampling_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Spline sampling density\n- Selected automatically from knot count and data size\n- Increased as needed, but capped to keep UI responsive",
        "Плотность сэмплирования сплайна\n- Выбирается автоматически по числу узлов и объёму данных\n- Увеличивается по необходимости, но ограничивается для отзывчивого UI",
    )
}
