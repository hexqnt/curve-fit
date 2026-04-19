//! Отображение метрик, параметров и действий экспорта для готового результата.

use super::*;

pub(super) fn ui_result(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    if app.fit_in_progress {
        ui.label(tr(
            language,
            "Fitting in progress. Replay starts after optimization completes.",
            "Подгонка в процессе. Промотка начнется после завершения оптимизации.",
        ));
        if let Some(iteration) = app.fit_preview_iteration {
            ui.label(format!(
                "{}: {iteration}",
                tr(language, "Iteration", "Итерация")
            ));
        }
        if let Some(params) = &app.fit_preview_params {
            ui.separator();
            ui.label(tr(language, "Current parameters", "Текущие параметры"));
            for (name, value) in params
                .family()
                .parameter_names()
                .iter()
                .zip(params.values())
            {
                ui.label(format!("{name} = {value:.8}"));
            }
        }
        return;
    }

    if app.has_fit_export_record() {
        ui.horizontal_wrapped(|ui| {
            let copy_response = CurveFitApp::info_hover(
                ui.button(tr(language, "Copy JSON", "Скопировать JSON")),
                result_json_copy_tooltip(language),
            );
            if copy_response.clicked() {
                app.copy_fit_export_json(ui.ctx());
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let save_response = CurveFitApp::info_hover(
                    ui.button(tr(language, "Save JSON", "Сохранить JSON")),
                    result_json_save_tooltip(language),
                );
                if save_response.clicked() {
                    app.request_fit_export_save_json();
                }
            }
        });
        ui.add_space(2.0);
    }

    let metrics = app.result_metrics.unwrap_or_else(|| {
        if let Some(result) = &app.fit_result {
            ExtendedMetrics {
                mse: result.mse,
                rmse: result.rmse,
                ..ExtendedMetrics::default()
            }
        } else if let Some(result) = &app.spline_result {
            ExtendedMetrics {
                mse: result.mse,
                rmse: result.rmse,
                mae: result.mae,
                r2: result.r2,
                max_abs_error: result.max_abs_error,
            }
        } else {
            ExtendedMetrics::default()
        }
    });

    if let Some(result) = &app.fit_result {
        ui.label(format!(
            "{}: {}",
            tr(language, "Family", "Семейство"),
            family_label(language, result.family)
        ));
        ui.add_space(2.0);
        ui.label(egui::RichText::new(tr(language, "Quality metrics", "Метрики качества")).strong());
        egui::Grid::new("result_quality_grid_parametric")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                ui.label("MSE");
                ui.monospace(format!("{:.8}", metrics.mse));
                ui.end_row();
                ui.label("RMSE");
                ui.monospace(format!("{:.8}", metrics.rmse));
                ui.end_row();
                ui.label("MAE");
                ui.monospace(format!("{:.8}", metrics.mae));
                ui.end_row();
                ui.label("R²");
                ui.monospace(format!("{:.8}", metrics.r2));
                ui.end_row();
                ui.label(tr(language, "Max |error|", "Макс |ошибка|"));
                ui.monospace(format!("{:.8}", metrics.max_abs_error));
                ui.end_row();
            });
        ui.add_space(2.0);
        ui.label(egui::RichText::new(tr(language, "Convergence", "Сходимость")).strong());
        egui::Grid::new("result_convergence_grid_parametric")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                ui.label(tr(language, "Iterations", "Итерации"));
                ui.monospace(result.iterations.to_string());
                ui.end_row();
            });
        ui.add_space(2.0);
        ui.label(tr(language, "Parameters", "Параметры"));
        egui::ScrollArea::vertical()
            .id_salt("result_parametric_params_scroll")
            .max_height(RESULT_PARAMS_MAX_HEIGHT)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("result_parametric_params_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        for (name, value) in result
                            .family
                            .parameter_names()
                            .iter()
                            .zip(result.params.values())
                        {
                            ui.monospace(*name);
                            ui.monospace(format!("{value:.8}"));
                            ui.end_row();
                        }
                    });
            });
    } else if let Some(result) = &app.spline_result {
        ui.label(format!(
            "{}: {}",
            tr(language, "Family", "Семейство"),
            model_choice_label(language, app.selected_model)
        ));
        ui.add_space(2.0);
        ui.label(egui::RichText::new(tr(language, "Quality metrics", "Метрики качества")).strong());
        egui::Grid::new("result_quality_grid_spline")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                ui.label("MSE");
                ui.monospace(format!("{:.8}", metrics.mse));
                ui.end_row();
                ui.label("RMSE");
                ui.monospace(format!("{:.8}", metrics.rmse));
                ui.end_row();
                ui.label("MAE");
                ui.monospace(format!("{:.8}", metrics.mae));
                ui.end_row();
                ui.label("R²");
                ui.monospace(format!("{:.8}", metrics.r2));
                ui.end_row();
                ui.label(tr(language, "Max |error|", "Макс |ошибка|"));
                ui.monospace(format!("{:.8}", metrics.max_abs_error));
                ui.end_row();
            });
        ui.add_space(2.0);
        ui.label(egui::RichText::new(tr(language, "Convergence", "Сходимость")).strong());
        egui::Grid::new("result_convergence_grid_spline")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                ui.label(tr(language, "Iterations", "Итерации"));
                ui.monospace(result.iterations.to_string());
                ui.end_row();
                ui.label(tr(language, "Parameters", "Параметры"));
                ui.monospace(result.knots.len().to_string());
                ui.end_row();
            });
        ui.add_space(2.0);
        ui.label(tr(language, "Parameters", "Параметры"));
        egui::ScrollArea::vertical()
            .id_salt("result_spline_params_scroll")
            .max_height(RESULT_PARAMS_MAX_HEIGHT)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("result_spline_params_grid")
                    .num_columns(3)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        ui.strong("knot");
                        ui.strong("x");
                        ui.strong("y");
                        ui.end_row();
                        for (index, knot) in result.knots.iter().enumerate() {
                            ui.monospace(format!("[{index}]"));
                            ui.monospace(format!("{:.8}", knot[0]));
                            ui.monospace(format!("{:.8}", knot[1]));
                            ui.end_row();
                        }
                    });
            });
    } else {
        ui.label(tr(
            language,
            "Run Fit to see optimization results.",
            "Нажмите Fit, чтобы увидеть результат оптимизации.",
        ));
    }
}

fn result_json_copy_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Copy JSON\n- Copies serialized fit result to clipboard\n- Includes model info, input summary, metrics, convergence and parameters",
        "Скопировать JSON\n- Копирует сериализованный результат фита в буфер обмена\n- Содержит модель, сводку по входу, метрики, сходимость и параметры",
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn result_json_save_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Save JSON\n- Opens file dialog and saves serialized fit result as .json",
        "Сохранить JSON\n- Открывает диалог и сохраняет сериализованный результат фита в .json",
    )
}
