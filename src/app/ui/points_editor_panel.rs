use super::*;

pub(super) fn ui_tools(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let icon_tint = ui.visuals().text_color();

    let tools = [
        PlotTool::None,
        PlotTool::SinglePoint,
        PlotTool::Spray,
        PlotTool::Eraser,
    ];
    let tool_width = ((ui.available_width() - ui.spacing().item_spacing.x).max(120.0)) * 0.5;
    egui::Grid::new("plot_tools_grid")
        .num_columns(2)
        .spacing(egui::vec2(ui.spacing().item_spacing.x, 6.0))
        .show(ui, |ui| {
            for (index, tool) in tools.into_iter().enumerate() {
                let selected = app.plot_tool == tool;
                let button = egui::Button::image_and_text(
                    tool_icon_image(tool, icon_tint),
                    tool_label(language, tool),
                )
                .selected(selected)
                .min_size(egui::vec2(tool_width, 0.0));
                let response = ui
                    .add(button)
                    .on_hover_text(tool_usage_hint(language, tool));
                if response.clicked() {
                    app.plot_tool = tool;
                }
                if index % 2 == 1 {
                    ui.end_row();
                }
            }
        });

    ui.add_space(2.0);
    match app.plot_tool {
        PlotTool::None | PlotTool::SinglePoint => {}
        PlotTool::Spray => {
            ui.add(
                egui::Slider::new(&mut app.spray_points_per_second, 10..=1_000)
                    .logarithmic(true)
                    .text(tr(language, "Rate, px/s", "Скорость, т/с")),
            );
            ui.add(
                egui::Slider::new(&mut app.spray_radius_rel, 0.002..=0.2)
                    .logarithmic(true)
                    .text(tr(language, "Radius", "Радиус")),
            );
            ui.horizontal_wrapped(|ui| {
                ui.label(tr(language, "Brush", "Кисть"));
                CurveFitApp::info_tooltip(ui, spray_brush_hint(language));
                ui.selectable_value(
                    &mut app.spray_brush,
                    SprayBrush::Uniform,
                    spray_brush_label(language, SprayBrush::Uniform),
                );
                ui.selectable_value(
                    &mut app.spray_brush,
                    SprayBrush::Gaussian,
                    spray_brush_label(language, SprayBrush::Gaussian),
                );
            });
        }
        PlotTool::Eraser => {
            ui.add(
                egui::Slider::new(&mut app.eraser_radius_rel, 0.002..=0.2)
                    .logarithmic(true)
                    .text(tr(language, "Radius", "Радиус")),
            );
        }
    }
}

pub(super) fn ui_points_editor(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let icon_tint = ui.visuals().text_color();
    let can_edit_points = !app.fit_in_progress;
    let (parse_error_line, valid_points_count, parse_error_message) = {
        let cache = app.points_cache();
        let valid_points_count = cache.parsed_points.as_ref().ok().map(Vec::len);
        let parse_error_message = cache.parsed_points.as_ref().err().cloned();
        (
            cache.parse_error_line,
            valid_points_count,
            parse_error_message,
        )
    };
    ui.horizontal_wrapped(|ui| {
        CurveFitApp::info_tooltip(ui, points_input_hint(language));
        if let Some(count) = valid_points_count {
            ui.label(
                egui::RichText::new(format!(
                    "{}: {count}",
                    tr(language, "Valid points", "Валидных точек")
                ))
                .small(),
            );
        }
    });
    if let Some(line) = parse_error_line {
        ui.colored_label(
            ui.visuals().error_fg_color,
            format!(
                "{} {line}",
                tr(language, "Parse error at line", "Ошибка парсинга в строке")
            ),
        );
    }
    let can_fill_with_residuals = can_edit_points && !app.residual_plot_points.is_empty();
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(
                can_edit_points && !app.points.undo_stack.is_empty(),
                egui::Button::image_and_text(
                    undo_icon_image(icon_tint),
                    tr(language, "Undo", "Отменить"),
                ),
            )
            .clicked()
        {
            app.undo_points_edit();
        }
        if ui
            .add_enabled(
                can_edit_points && !app.points.redo_stack.is_empty(),
                egui::Button::image_and_text(
                    redo_icon_image(icon_tint),
                    tr(language, "Redo", "Повторить"),
                ),
            )
            .clicked()
        {
            app.redo_points_edit();
        }
        if ui
            .add_enabled(
                can_edit_points,
                egui::Button::image_and_text(
                    clear_icon_image(icon_tint),
                    tr(language, "Clear", "Очистить"),
                ),
            )
            .clicked()
        {
            app.clear_points_text(true);
            app.clear_fit_outputs();
            app.status = Some(StatusMessage::Cleared);
        }
        ui.add_enabled_ui(can_edit_points, |ui| {
            ui.menu_button(tr(language, "Actions", "Действия"), |ui| {
                if ui
                    .add_enabled(
                        can_fill_with_residuals,
                        egui::Button::new(tr(
                            language,
                            "Fill with residuals",
                            "Заполнить остатками",
                        )),
                    )
                    .clicked()
                {
                    app.fill_points_with_residuals();
                    ui.close();
                }
            });
        });
    });

    let hint = tr(
        language,
        "Example:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
        "Пример:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
    );
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace).max(1.0);
    let body_height = ui.text_style_height(&egui::TextStyle::Body).max(1.0);
    let small_height = ui.text_style_height(&egui::TextStyle::Small).max(1.0);
    // Резервируем место под нижний блок (переключатель нормализации и служебные подписи),
    // чтобы поле ввода точек не вытесняло его за пределы видимой области.
    let footer_reserved_height = body_height
        + small_height
        + if app.fit_in_progress {
            body_height
        } else {
            0.0
        }
        + ui.spacing().item_spacing.y * 4.0
        + 16.0;
    let text_height = (ui.available_height() - footer_reserved_height).max(row_height * 6.0);
    let desired_rows = (text_height / row_height).floor().max(1.0) as usize;
    egui::ScrollArea::vertical()
        .id_salt("points_text_scroll")
        .max_height(text_height)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let text_width = ui.available_width();
            let before_edit = app.points.text.clone();
            let mut layouter = move |ui: &egui::Ui,
                                     text: &dyn egui::TextBuffer,
                                     wrap_width: f32|
                  -> std::sync::Arc<egui::Galley> {
                let mut job = egui::text::LayoutJob::default();
                job.wrap.max_width = wrap_width;
                let text_color = ui.visuals().text_color();
                let font_id = egui::TextStyle::Monospace.resolve(ui.style());
                let error_bg = if ui.visuals().dark_mode {
                    egui::Color32::from_rgb(70, 26, 26)
                } else {
                    egui::Color32::from_rgb(255, 230, 230)
                };
                for (index, line) in text.as_str().split_inclusive('\n').enumerate() {
                    let mut format = egui::TextFormat {
                        font_id: font_id.clone(),
                        color: text_color,
                        ..Default::default()
                    };
                    if parse_error_line == Some(index + 1) {
                        format.background = error_bg;
                    }
                    job.append(line, 0.0, format);
                }
                if text.as_str().is_empty() {
                    job.append(
                        "",
                        0.0,
                        egui::TextFormat {
                            font_id,
                            color: text_color,
                            ..Default::default()
                        },
                    );
                }
                ui.fonts_mut(|fonts| fonts.layout_job(job))
            };
            let response = ui.add(
                egui::TextEdit::multiline(&mut app.points.text)
                    .desired_width(text_width)
                    .desired_rows(desired_rows)
                    .font(egui::TextStyle::Monospace)
                    .hint_text(hint)
                    .layouter(&mut layouter)
                    .interactive(can_edit_points),
            );
            if response.changed() {
                app.push_points_undo_snapshot(before_edit);
                app.points.redo_stack.clear();
                app.invalidate_points_cache();
            }
        });

    if let Some(error) = parse_error_message {
        ui.colored_label(
            ui.visuals().error_fg_color,
            format!("{POINTS_PARSE_ERROR_PREFIX}{error}"),
        );
    }

    ui.separator();
    ui.add_enabled_ui(can_edit_points, |ui| {
        ui.horizontal_wrapped(|ui| {
            CurveFitApp::toggle_switch_labeled(
                ui,
                &mut app.normalize_parametric_data,
                tr(
                    language,
                    "Normalize x/y before fit (parametric models)",
                    "Нормализовать x/y перед фитингом (параметрические модели)",
                ),
            );
            CurveFitApp::info_tooltip(ui, normalization_hint(language));
        });
    });

    if app.fit_in_progress {
        ui.label(tr(
            language,
            "Point editing is disabled while fitting is running.",
            "Редактирование точек отключено во время подгонки.",
        ));
    }
}

fn tool_usage_hint(language: UiLanguage, tool: PlotTool) -> &'static str {
    match tool {
        PlotTool::None => tr(
            language,
            "Navigation mode\n- Drag to pan the plot\n- Use wheel/trackpad to zoom\n- Double-click resets view bounds",
            "Режим навигации\n- Перетаскивание двигает график\n- Колесо/трекпад меняют масштаб\n- Двойной клик сбрасывает вид в границы данных",
        ),
        PlotTool::SinglePoint => tr(
            language,
            "Single point tool\n- Left click on plot to add one sample\n- Best for precise manual placement",
            "Инструмент одной точки\n- Левый клик по графику добавляет одну точку\n- Подходит для точного ручного ввода",
        ),
        PlotTool::Spray => tr(
            language,
            "Spray tool\n- Hold left mouse button to add a stream of points\n- Rate controls points per second\n- Radius controls spread around cursor",
            "Инструмент распыления\n- Зажмите левую кнопку, чтобы добавлять поток точек\n- Скорость задаёт число точек в секунду\n- Радиус задаёт разброс вокруг курсора",
        ),
        PlotTool::Eraser => tr(
            language,
            "Eraser tool\n- Hold left mouse button to remove points\n- Radius controls erase area around cursor",
            "Ластик\n- Зажмите левую кнопку, чтобы удалять точки\n- Радиус задаёт область стирания вокруг курсора",
        ),
    }
}

fn points_input_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Input format\n- One point per line: x y\n- Separators: space, tab, or ';'\n- Decimal comma is accepted (e.g. 1,25)\n- Empty lines are ignored",
        "Формат ввода\n- Одна точка на строку: x y\n- Разделители: пробел, табуляция или ';'\n- Десятичная запятая поддерживается (например, 1,25)\n- Пустые строки игнорируются",
    )
}

fn spray_brush_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Brush distribution\n- Uniform: equal probability inside circle\n- Gaussian: denser near center, softer edges",
        "Распределение кисти\n- Равномерная: одинаковая плотность внутри круга\n- Гауссова: выше плотность в центре, мягче по краям",
    )
}

fn normalization_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Parametric normalization\n- Fit runs on normalized x/y for better numerical conditioning\n- Displayed parameters and metrics remain in original units\n- Useful when x and y scales differ significantly",
        "Нормализация параметрических данных\n- Фитинг выполняется на нормализованных x/y для лучшей численной устойчивости\n- Параметры и метрики в интерфейсе остаются в исходных единицах\n- Полезно при сильно разных масштабах x и y",
    )
}
