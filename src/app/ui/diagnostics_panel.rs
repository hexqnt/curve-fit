//! Графики loss, параметров и остаточных ошибок по истории итераций.

use super::*;

pub(super) fn ui_iteration_diagnostics(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let previous_tab = app.panel.diagnostics_tab;
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut app.panel.diagnostics_tab, DiagnosticsTab::Loss, "Loss");
        ui.selectable_value(
            &mut app.panel.diagnostics_tab,
            DiagnosticsTab::Residuals,
            tr(language, "Residuals", "Остатки"),
        );
    });
    if app.panel.diagnostics_tab != previous_tab {
        app.panel.diagnostics_shared_axis_width = 0.0;
    }
    ui.add_space(2.0);

    let (loss_color, residual_color, zero_color) = if ui.visuals().dark_mode {
        (
            egui::Color32::from_rgb(245, 126, 95),
            egui::Color32::from_rgb(106, 198, 230),
            egui::Color32::from_rgb(131, 147, 160),
        )
    } else {
        (
            egui::Color32::from_rgb(181, 93, 67),
            egui::Color32::from_rgb(40, 131, 165),
            egui::Color32::from_rgb(139, 151, 160),
        )
    };

    match app.panel.diagnostics_tab {
        DiagnosticsTab::Loss => ui_loss_diagnostics(app, ui, language, loss_color),
        DiagnosticsTab::Residuals => {
            ui_residuals_diagnostics(app, ui, language, residual_color, zero_color)
        }
    }
}

fn ui_loss_diagnostics(
    app: &mut CurveFitApp,
    ui: &mut egui::Ui,
    language: UiLanguage,
    loss_color: egui::Color32,
) {
    if app.iteration_diagnostics.loss_points.is_empty() {
        ui.label(tr(
            language,
            "Run Fit to collect iteration history.",
            "Запустите фитинг, чтобы получить историю итераций.",
        ));
        app.panel.diagnostics_shared_axis_width = 0.0;
        return;
    }

    let available_height = ui.available_height().max(2.0);
    let spacing = ui.spacing().item_spacing.y;
    let plot_count = 2.0;
    let total_spacing = spacing * (plot_count - 1.0);
    let plot_height = ((available_height - total_spacing).max(2.0)) / plot_count;
    let mut iteration_x_min = f64::INFINITY;
    let mut iteration_x_max = f64::NEG_INFINITY;
    for [iteration, _] in &app.iteration_diagnostics.loss_points {
        iteration_x_min = iteration_x_min.min(*iteration);
        iteration_x_max = iteration_x_max.max(*iteration);
    }
    for series in &app.iteration_diagnostics.parameter_series {
        for [iteration, _] in series {
            iteration_x_min = iteration_x_min.min(*iteration);
            iteration_x_max = iteration_x_max.max(*iteration);
        }
    }
    if !iteration_x_min.is_finite() || !iteration_x_max.is_finite() {
        iteration_x_min = 0.0;
        iteration_x_max = 1.0;
    }
    if (iteration_x_max - iteration_x_min).abs() <= f64::EPSILON {
        let padding = iteration_x_min.abs().max(1.0) * 0.05;
        iteration_x_min -= padding;
        iteration_x_max += padding;
    }
    let hidden_non_loss_ids = if app.panel.diagnostics_hide_non_loss_by_default_pending {
        app.panel.diagnostics_hide_non_loss_by_default_pending = false;
        Some(diagnostics_hidden_non_loss_series_ids().to_vec())
    } else {
        None
    };
    let selected_iteration_x = app
        .replay_selected_iteration()
        .map(|iteration| iteration as f64);
    let selected_iteration_marker_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
    let mut running_axis_width = app.panel.diagnostics_shared_axis_width.max(1.0);

    {
        let loss_points = &app.iteration_diagnostics.loss_points;
        let mse_points = &app.iteration_diagnostics.mse_points;
        let rmse_points = &app.iteration_diagnostics.rmse_points;
        let mae_points = &app.iteration_diagnostics.mae_points;
        let soft_l1_points = &app.iteration_diagnostics.soft_l1_points;
        let r2_abs_points = &app.iteration_diagnostics.r2_abs_points;
        let max_abs_error_points = &app.iteration_diagnostics.max_abs_error_points;
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), plot_height),
            egui::Layout::left_to_right(egui::Align::Min),
            |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let plot_slot_left = ui.max_rect().left();
                let mut legend = Legend::default().background_alpha(0.55);
                if let Some(hidden_ids) = hidden_non_loss_ids.as_ref() {
                    legend = legend.hidden_items(hidden_ids.iter().copied());
                }
                let plot_response = Plot::new("loss_metrics_plot")
                    .height(plot_height)
                    .legend(legend)
                    .link_axis("diagnostics_iter_x_link", [true, false])
                    // Не используем default_x_bounds: он фиксирует X после первого кадра.
                    // При начальной точке iteration=0 это «замораживает» ось около нуля.
                    .include_x(iteration_x_min)
                    .include_x(iteration_x_max)
                    .auto_bounds([true, true])
                    .y_axis_min_width(running_axis_width)
                    .show_grid([true, true])
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_double_click_reset(false)
                    .allow_boxed_zoom(false)
                    .show(ui, |plot_ui| {
                        let loss_name = format!("loss({})", app.fit_loss_metric.id());
                        plot_ui.line(
                            Line::new(
                                loss_name,
                                PlotPoints::from_iter(loss_points.iter().copied()),
                            )
                            .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_LOSS))
                            .width(1.9_f32)
                            .color(loss_color),
                        );
                        if app.fit_loss_metric != OptimizationLossMetric::Mse {
                            plot_ui.line(
                                Line::new(
                                    "MSE (L2)",
                                    PlotPoints::from_iter(mse_points.iter().copied()),
                                )
                                .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_MSE))
                                .width(1.5_f32),
                            );
                        }
                        plot_ui.line(
                            Line::new("RMSE", PlotPoints::from_iter(rmse_points.iter().copied()))
                                .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_RMSE))
                                .width(1.5_f32),
                        );
                        if app.fit_loss_metric != OptimizationLossMetric::Mae {
                            plot_ui.line(
                                Line::new(
                                    "MAE (L1)",
                                    PlotPoints::from_iter(mae_points.iter().copied()),
                                )
                                .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_MAE))
                                .width(1.5_f32),
                            );
                        }
                        if app.fit_loss_metric != OptimizationLossMetric::SoftL1 {
                            plot_ui.line(
                                Line::new(
                                    "soft_l1",
                                    PlotPoints::from_iter(soft_l1_points.iter().copied()),
                                )
                                .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_SOFT_L1))
                                .width(1.5_f32),
                            );
                        }
                        plot_ui.line(
                            Line::new("|R2|", PlotPoints::from_iter(r2_abs_points.iter().copied()))
                                .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_R2_ABS))
                                .width(1.5_f32),
                        );
                        plot_ui.line(
                            Line::new(
                                "MaxAbsError",
                                PlotPoints::from_iter(max_abs_error_points.iter().copied()),
                            )
                            .id(egui::Id::new(DIAGNOSTICS_SERIES_ID_MAX_ABS))
                            .width(1.5_f32),
                        );
                        if let Some(selected_iteration_x) = selected_iteration_x {
                            plot_ui.vline(
                                VLine::new("", selected_iteration_x)
                                    .id(egui::Id::new(
                                        DIAGNOSTICS_SELECTED_ITERATION_MARKER_ID_LOSS,
                                    ))
                                    .width(1.0_f32)
                                    .color(selected_iteration_marker_color)
                                    .style(LineStyle::dashed_dense())
                                    .allow_hover(false),
                            );
                        }
                    });
                let axis_width = diagnostics_plot_y_axis_width(&plot_response, plot_slot_left);
                running_axis_width = running_axis_width.max(axis_width);
            },
        );
    }

    {
        let parameter_names = &app.iteration_diagnostics.parameter_names;
        let parameter_series = &app.iteration_diagnostics.parameter_series;
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), plot_height),
            egui::Layout::left_to_right(egui::Align::Min),
            |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let plot_slot_left = ui.max_rect().left();
                let plot_response = Plot::new("parameter_iteration_plot")
                    .height(plot_height)
                    .legend(Legend::default().background_alpha(0.55))
                    .link_axis("diagnostics_iter_x_link", [true, false])
                    .include_x(iteration_x_min)
                    .include_x(iteration_x_max)
                    .auto_bounds([true, true])
                    .y_axis_min_width(running_axis_width)
                    .show_grid([true, true])
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_double_click_reset(false)
                    .allow_boxed_zoom(false)
                    .show(ui, |plot_ui| {
                        for (name, series) in parameter_names.iter().zip(parameter_series.iter()) {
                            plot_ui.line(
                                Line::new(
                                    name.clone(),
                                    PlotPoints::from_iter(series.iter().copied()),
                                )
                                .width(1.7_f32),
                            );
                        }
                        if let Some(selected_iteration_x) = selected_iteration_x {
                            plot_ui.vline(
                                VLine::new("", selected_iteration_x)
                                    .id(egui::Id::new(
                                        DIAGNOSTICS_SELECTED_ITERATION_MARKER_ID_PARAMS,
                                    ))
                                    .width(1.0_f32)
                                    .color(selected_iteration_marker_color)
                                    .style(LineStyle::dashed_dense())
                                    .allow_hover(false),
                            );
                        }
                    });
                let axis_width = diagnostics_plot_y_axis_width(&plot_response, plot_slot_left);
                running_axis_width = running_axis_width.max(axis_width);
            },
        );
    }

    app.panel.diagnostics_shared_axis_width = running_axis_width;
}

fn ui_residuals_diagnostics(
    app: &mut CurveFitApp,
    ui: &mut egui::Ui,
    language: UiLanguage,
    residual_color: egui::Color32,
    zero_color: egui::Color32,
) {
    if app.residual_plot_points.is_empty() {
        ui.label(tr(
            language,
            "Residuals will be available after fit completes.",
            "Остатки появятся после завершения фитинга.",
        ));
        app.panel.diagnostics_shared_axis_width = 0.0;
        return;
    }

    let residual_points = &app.residual_plot_points;
    let x_min = residual_points
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let x_max = residual_points
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let zero_line = [[x_min, 0.0], [x_max, 0.0]];
    let available_height = ui.available_height().max(2.0);
    let mut running_axis_width = app.panel.diagnostics_shared_axis_width.max(1.0);

    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), available_height),
        egui::Layout::left_to_right(egui::Align::Min),
        |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let plot_slot_left = ui.max_rect().left();
            let plot_response = Plot::new("residuals_diagnostics_plot")
                .height(available_height)
                .legend(Legend::default().background_alpha(0.55))
                .y_axis_min_width(running_axis_width)
                .show_grid([true, true])
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_double_click_reset(false)
                .allow_boxed_zoom(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(
                            tr(language, "Zero", "Ноль"),
                            PlotPoints::from_iter(zero_line),
                        )
                        .width(1.2_f32)
                        .color(zero_color),
                    );
                    plot_ui.points(
                        PlotPointsItem::new(
                            tr(language, "Residuals", "Остатки"),
                            residual_points.as_slice(),
                        )
                        .radius(2.3_f32)
                        .color(residual_color),
                    );
                });
            let axis_width = diagnostics_plot_y_axis_width(&plot_response, plot_slot_left);
            running_axis_width = running_axis_width.max(axis_width);
        },
    );

    app.panel.diagnostics_shared_axis_width = running_axis_width;
}
