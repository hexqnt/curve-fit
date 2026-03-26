use super::*;

pub(super) fn add_point_from_plot(app: &mut CurveFitApp, x: f64, y: f64) {
    let mut points = match app.parse_points_for_edit() {
        Ok(points) => points,
        Err(error) => {
            app.status = Some(StatusMessage::Error(error));
            return;
        }
    };

    match Point::try_new(x, y) {
        Ok(point) => {
            points.push(point);
            app.write_points_text(&points, true);
        }
        Err(error) => {
            app.status = Some(StatusMessage::Error(error.to_string()));
        }
    }
}

pub(super) fn spray_points_from_plot(
    app: &mut CurveFitApp,
    center_x: f64,
    center_y: f64,
    radius_x: f64,
    radius_y: f64,
    points_to_add: usize,
) {
    if points_to_add == 0 {
        return;
    }

    let mut points = match app.parse_points_for_edit() {
        Ok(points) => points,
        Err(error) => {
            app.status = Some(StatusMessage::Error(error));
            return;
        }
    };

    for _ in 0..points_to_add {
        let [offset_x, offset_y] = app.next_spray_unit_disk_offset();
        let x = center_x + offset_x * radius_x;
        let y = center_y + offset_y * radius_y;
        if let Ok(point) = Point::try_new(x, y) {
            points.push(point);
        }
    }

    app.write_points_text(&points, false);
}

pub(super) fn erase_points_from_plot(
    app: &mut CurveFitApp,
    center_x: f64,
    center_y: f64,
    radius_x: f64,
    radius_y: f64,
) {
    let mut points = match app.parse_points_for_edit() {
        Ok(points) => points,
        Err(error) => {
            app.status = Some(StatusMessage::Error(error));
            return;
        }
    };

    if radius_x <= 0.0 || radius_y <= 0.0 {
        return;
    }

    points.retain(|point| {
        let dx = (point.x() - center_x) / radius_x;
        let dy = (point.y() - center_y) / radius_y;
        dx * dx + dy * dy > 1.0
    });

    app.write_points_text(&points, false);
}

fn plot_position_from_screen(
    plot_response: &PlotResponse<()>,
    screen_pos: egui::Pos2,
) -> Option<PlotPoint> {
    // Инструменты рисования должны работать только внутри области данных графика.
    // Иначе курсор над осями/легендой даёт координаты вне ожидаемого диапазона.
    if !plot_response.transform.frame().contains(screen_pos) {
        return None;
    }
    Some(plot_response.transform.value_from_position(screen_pos))
}

pub(super) fn handle_plot_tools(app: &mut CurveFitApp, plot_response: &PlotResponse<()>) {
    if app.fit_in_progress {
        app.reset_spray_rate_state();
        return;
    }

    let response = &plot_response.response;
    let is_continuous_tool = matches!(app.plot_tool, PlotTool::Spray | PlotTool::Eraser);
    let primary_down_on_plot = response.is_pointer_button_down_on();
    if is_continuous_tool && primary_down_on_plot {
        if app.active_tool_bounds.is_none() {
            app.push_points_undo_snapshot(app.points.text.clone());
        }
        app.active_tool_bounds
            .get_or_insert(*plot_response.transform.bounds());
    } else {
        app.active_tool_bounds = None;
    }

    let spray_active = app.plot_tool == PlotTool::Spray && primary_down_on_plot;
    if spray_active {
        // Просим следующий кадр, чтобы поддерживать стабильный points/sec даже без движения мыши.
        response.ctx.request_repaint();
    } else {
        app.reset_spray_rate_state();
    }

    let bounds = plot_response.transform.bounds();
    let radius_x_scale = bounds.width().abs().max(1e-6);
    let radius_y_scale = bounds.height().abs().max(1e-6);

    match app.plot_tool {
        PlotTool::None => {}
        PlotTool::SinglePoint => {
            if response.clicked_by(egui::PointerButton::Primary)
                && let Some(screen_pos) = response.interact_pointer_pos()
                && let Some(plot_pos) = plot_position_from_screen(plot_response, screen_pos)
            {
                add_point_from_plot(app, plot_pos.x, plot_pos.y);
            }
        }
        PlotTool::Spray => {
            if primary_down_on_plot
                && let Some(screen_pos) = response.interact_pointer_pos()
                && let Some(plot_pos) = plot_position_from_screen(plot_response, screen_pos)
            {
                let points_to_add = app.next_spray_points_to_add(Instant::now());
                let radius_x = app.spray_radius_rel * radius_x_scale;
                let radius_y = app.spray_radius_rel * radius_y_scale;
                spray_points_from_plot(
                    app,
                    plot_pos.x,
                    plot_pos.y,
                    radius_x,
                    radius_y,
                    points_to_add,
                );
            }
        }
        PlotTool::Eraser => {
            if primary_down_on_plot
                && let Some(screen_pos) = response.interact_pointer_pos()
                && let Some(plot_pos) = plot_position_from_screen(plot_response, screen_pos)
            {
                let radius_x = app.eraser_radius_rel * radius_x_scale;
                let radius_y = app.eraser_radius_rel * radius_y_scale;
                erase_points_from_plot(app, plot_pos.x, plot_pos.y, radius_x, radius_y);
            }
        }
    }
}

pub(super) fn ui_plot(app: &mut CurveFitApp, ui: &mut egui::Ui, height: f32) {
    let language = app.ui_language;
    let points = Arc::clone(&app.points_cache().plot_points);
    let points_slice = points.as_ref();
    let (x_min, x_max) = plot_domain(points_slice);
    let navigation_mode = matches!(app.plot_tool, PlotTool::None);
    let spline_curve = app.spline_plot_curve.clone();
    let spline_curve_slice = spline_curve.as_deref();
    let sampled_curve = if spline_curve_slice.is_none() {
        let active_params = app
            .fit_preview_params
            .clone()
            .or_else(|| app.fit_result.as_ref().map(|result| result.params.clone()));
        active_params
            .map(|params| app.cached_sampled_curve(&params, x_min, x_max, PARAMETRIC_PLOT_SAMPLES))
    } else {
        None
    };
    let fitted_curve_points = spline_curve_slice.or(sampled_curve.as_deref());
    let fitted_line_name = if spline_curve_slice.is_some() {
        if let Some(iteration) = app.fit_preview_iteration {
            format!(
                "{} ({})",
                model_choice_label(language, app.selected_model),
                format_args!("{} {iteration}", tr(language, "iter", "итер."))
            )
        } else {
            model_choice_label(language, app.selected_model).to_string()
        }
    } else if let Some(iteration) = app.fit_preview_iteration {
        format!(
            "{} ({})",
            tr(language, "Fitted", "Фитинг"),
            format_args!("{} {iteration}", tr(language, "iter", "итер."))
        )
    } else {
        tr(language, "Fitted", "Фитинг").to_string()
    };
    let content_bounds = fit_bounds_for_content(points_slice, fitted_curve_points);
    let fit_bounds = if app.fit_to_content_requested {
        content_bounds
    } else {
        None
    };
    let center_bounds = if app.center_origin_requested {
        let (span_x, span_y) = app
            .last_plot_bounds
            .or(content_bounds)
            .map(|bounds| (bounds.width().abs(), bounds.height().abs()))
            .unwrap_or((2.0, 2.0));
        let half_x = span_x.max(1e-6) * 0.5;
        let half_y = span_y.max(1e-6) * 0.5;
        Some(PlotBounds::from_min_max(
            [-half_x, -half_y],
            [half_x, half_y],
        ))
    } else {
        None
    };
    let origin_bottom_left_bounds = if app.origin_bottom_left_requested {
        let (max_x, max_y) = app
            .last_plot_bounds
            .or(content_bounds)
            .map(|bounds| {
                let span_x = bounds.width().abs().max(1e-6);
                let span_y = bounds.height().abs().max(1e-6);
                // Если текущий видимый диапазон полностью в положительной зоне,
                // сохраняем правую/верхнюю границы, чтобы не «отрезать» содержимое.
                (
                    bounds.max()[0].max(span_x).max(1e-6),
                    bounds.max()[1].max(span_y).max(1e-6),
                )
            })
            .unwrap_or((2.0, 2.0));
        Some(PlotBounds::from_min_max([0.0, 0.0], [max_x, max_y]))
    } else {
        None
    };
    let locked_tool_bounds = app.active_tool_bounds;
    let (samples_color, fitted_color) = if ui.visuals().dark_mode {
        (
            egui::Color32::from_rgb(232, 140, 96),
            egui::Color32::from_rgb(96, 204, 238),
        )
    } else {
        (
            egui::Color32::from_rgb(184, 87, 53),
            egui::Color32::from_rgb(24, 126, 165),
        )
    };

    let plot_response = Plot::new("fit_plot")
        .height(height)
        .legend(Legend::default().background_alpha(0.55))
        .show_axes([true, true])
        .show_grid([true, true])
        .allow_drag(navigation_mode)
        .allow_zoom(navigation_mode)
        .allow_scroll(navigation_mode)
        .allow_double_click_reset(navigation_mode)
        .allow_boxed_zoom(navigation_mode)
        .show(ui, |plot_ui| {
            if let Some(bounds) = locked_tool_bounds {
                plot_ui.set_plot_bounds(bounds);
            }
            if let Some(bounds) = fit_bounds {
                plot_ui.set_plot_bounds(bounds);
            }
            if let Some(bounds) = center_bounds {
                plot_ui.set_plot_bounds(bounds);
            }
            if let Some(bounds) = origin_bottom_left_bounds {
                plot_ui.set_plot_bounds(bounds);
            }
            if !points_slice.is_empty() {
                plot_ui.points(
                    PlotPointsItem::new(tr(language, "Samples", "Точки"), points_slice)
                        .radius(2.8)
                        .color(samples_color),
                );
            }
            if let Some(fitted) = spline_curve_slice {
                plot_ui.line(
                    Line::new(fitted_line_name.clone(), fitted)
                        .width(2.2)
                        .color(fitted_color),
                );
            } else if let Some(fitted) = sampled_curve.as_deref() {
                plot_ui.line(
                    Line::new(fitted_line_name.clone(), fitted)
                        .width(2.2)
                        .color(fitted_color),
                );
            }
        });

    let bounds = plot_response.transform.bounds();
    app.last_plot_bounds = Some(*bounds);

    if app.fit_to_content_requested {
        app.fit_to_content_requested = false;
    }
    if app.center_origin_requested {
        app.center_origin_requested = false;
    }
    if app.origin_bottom_left_requested {
        app.origin_bottom_left_requested = false;
    }

    handle_plot_tools(app, &plot_response);
}
