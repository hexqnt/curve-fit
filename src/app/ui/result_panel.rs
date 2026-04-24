//! Отображение метрик, параметров и действий экспорта для готового результата.

use super::*;
use std::fmt::Write as _;

fn section_title(ui: &mut egui::Ui, title: &str) {
    ui.label(egui::RichText::new(title).strong());
}

fn grid_row(ui: &mut egui::Ui, label: &str, value: impl std::fmt::Display) {
    ui.label(label);
    ui.monospace(value.to_string());
    ui.end_row();
}

fn format_f64_list(values: &[f64], precision: usize) -> String {
    let mut formatted = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            formatted.push_str(", ");
        }
        let _ = write!(formatted, "{value:.precision$}");
    }
    formatted
}

fn ui_quality_metrics_grid(
    ui: &mut egui::Ui,
    language: UiLanguage,
    grid_id: &str,
    metrics: ExtendedMetrics,
) {
    section_title(ui, tr(language, "Quality metrics", "Метрики качества"));
    egui::Grid::new(grid_id)
        .num_columns(2)
        .spacing(egui::vec2(8.0, 4.0))
        .show(ui, |ui| {
            grid_row(ui, "MSE", format!("{:.8}", metrics.mse));
            grid_row(ui, "RMSE", format!("{:.8}", metrics.rmse));
            grid_row(ui, "MAE", format!("{:.8}", metrics.mae));
            grid_row(ui, "R²", format!("{:.8}", metrics.r2));
            grid_row(
                ui,
                tr(language, "Max |error|", "Макс |ошибка|"),
                format!("{:.8}", metrics.max_abs_error),
            );
        });
}

fn ui_parametric_fit_preview(ui: &mut egui::Ui, language: UiLanguage, params: &CurveParams) {
    ui.separator();
    ui.label(tr(language, "Current parameters", "Текущие параметры"));
    if let Some(taus) = params.saturating_trend_taus() {
        ui.label(format!(
            "{}: {}",
            tr(language, "Tau grid", "Сетка tau"),
            format_f64_list(taus, 4)
        ));
    }
    params.with_names_and_values(|names, values| {
        for (name, value) in names.iter().zip(values.iter()) {
            ui.label(format!("{name} = {value:.8}"));
        }
    });
}

fn resolved_result_metrics(app: &CurveFitApp) -> ExtendedMetrics {
    app.result_metrics.unwrap_or_else(|| {
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
    })
}

fn ui_result_export_actions(app: &mut CurveFitApp, ui: &mut egui::Ui, language: UiLanguage) {
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

fn ui_parametric_result(
    ui: &mut egui::Ui,
    language: UiLanguage,
    result: &FitResult,
    metrics: ExtendedMetrics,
) {
    ui.label(format!(
        "{}: {}",
        tr(language, "Family", "Семейство"),
        family_label(language, result.family)
    ));
    ui.add_space(2.0);
    ui_quality_metrics_grid(ui, language, "result_quality_grid_parametric", metrics);
    ui.add_space(2.0);

    section_title(ui, tr(language, "Convergence", "Сходимость"));
    egui::Grid::new("result_convergence_grid_parametric")
        .num_columns(2)
        .spacing(egui::vec2(8.0, 4.0))
        .show(ui, |ui| {
            grid_row(
                ui,
                tr(language, "Iterations", "Итерации"),
                result.iterations,
            );
        });
    ui.add_space(2.0);

    if let Some(taus) = result.params.saturating_trend_taus() {
        ui.label(tr(language, "Tau grid", "Сетка tau"));
        ui.monospace(format_f64_list(taus, 8));
        ui.add_space(2.0);
    }

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
                    result.params.with_names_and_values(|names, values| {
                        for (name, value) in names.iter().zip(values.iter()) {
                            grid_row(ui, name, format!("{value:.8}"));
                        }
                    });
                });
        });
}

fn ui_spline_result(
    ui: &mut egui::Ui,
    language: UiLanguage,
    result: &SplineResult,
    selected_model: ModelChoice,
    metrics: ExtendedMetrics,
) {
    ui.label(format!(
        "{}: {}",
        tr(language, "Family", "Семейство"),
        model_choice_label(language, selected_model)
    ));
    ui.add_space(2.0);
    ui_quality_metrics_grid(ui, language, "result_quality_grid_spline", metrics);
    ui.add_space(2.0);

    section_title(ui, tr(language, "Convergence", "Сходимость"));
    egui::Grid::new("result_convergence_grid_spline")
        .num_columns(2)
        .spacing(egui::vec2(8.0, 4.0))
        .show(ui, |ui| {
            grid_row(
                ui,
                tr(language, "Iterations", "Итерации"),
                result.iterations,
            );
            grid_row(
                ui,
                tr(language, "Parameters", "Параметры"),
                result.knots.len(),
            );
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
}

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
            ui_parametric_fit_preview(ui, language, params);
        }
        return;
    }

    if app.has_fit_export_record() {
        ui_result_export_actions(app, ui, language);
    }

    let metrics = resolved_result_metrics(app);

    if let Some(result) = &app.fit_result {
        ui_parametric_result(ui, language, result, metrics);
    } else if let Some(result) = &app.spline_result {
        ui_spline_result(ui, language, result, app.selected_model, metrics);
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
