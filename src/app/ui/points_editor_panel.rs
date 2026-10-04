//! Панель инструментов и текстовый редактор исходных точек.

use std::borrow::Cow;

use super::*;
use crate::app::widgets::{
    LayerAction, PointLayerRow, PointsTextEdit, ToggleSwitch, ToolbarButton,
    with_toolbar_hover_style,
};

const LAYER_VISIBILITY_COLUMN_WIDTH: f32 = 34.0;
const LAYER_COLOR_COLUMN_WIDTH: f32 = 72.0;
const LAYER_COUNT_COLUMN_WIDTH: f32 = 36.0;

pub(super) fn ui_tools(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let icon_tint = ui.visuals().text_color();

    let tools = [
        PlotTool::None,
        PlotTool::SinglePoint,
        PlotTool::Dotted,
        PlotTool::Spray,
        PlotTool::Eraser,
    ];
    with_toolbar_hover_style(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = style::TOOLBAR_BUTTON_SPACING;
            for tool in tools {
                let selected = app.plot_tool == tool;
                let button = ToolbarButton::new(
                    tool_icon_image(tool, icon_tint),
                    tool_usage_hint(language, tool),
                )
                .selected(selected)
                .frame(true);
                let response = ui.add(button);
                if response.clicked() {
                    app.plot_tool = tool;
                }
            }
        });
    });

    ui.add_space(style::SECTION_GAP);
    match app.plot_tool {
        PlotTool::None | PlotTool::SinglePoint | PlotTool::Dotted => {}
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
                let uniform_response = ui.selectable_value(
                    &mut app.spray_brush,
                    SprayBrush::Uniform,
                    spray_brush_label(language, SprayBrush::Uniform),
                );
                let _ = widgets::info_hover(
                    uniform_response,
                    spray_brush_mode_hint(language, SprayBrush::Uniform),
                );
                let gaussian_response = ui.selectable_value(
                    &mut app.spray_brush,
                    SprayBrush::Gaussian,
                    spray_brush_label(language, SprayBrush::Gaussian),
                );
                let _ = widgets::info_hover(
                    gaussian_response,
                    spray_brush_mode_hint(language, SprayBrush::Gaussian),
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

pub(super) fn ui_point_layers(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let icon_tint = ui.visuals().text_color();
    let can_edit_layers = !app.fit_in_progress;

    let clipboard_available = !app.clipboard_import_in_progress();
    let mut toolbar_action = None;
    with_toolbar_hover_style(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = style::TOOLBAR_BUTTON_SPACING;
            for (action, icon, hint, available) in [
                (
                    LayerAction::New,
                    layer_new_icon_image(icon_tint),
                    layer_new_tooltip(language),
                    true,
                ),
                (
                    LayerAction::Duplicate,
                    layer_duplicate_icon_image(icon_tint),
                    layer_duplicate_tooltip(language),
                    true,
                ),
                (
                    LayerAction::Paste,
                    clipboard_import_icon_image(icon_tint),
                    layer_clipboard_tooltip(language),
                    clipboard_available,
                ),
                (
                    LayerAction::Clear,
                    clear_icon_image(icon_tint),
                    layer_clear_tooltip(language),
                    true,
                ),
                (
                    LayerAction::Delete,
                    layer_delete_icon_image(icon_tint),
                    layer_delete_tooltip(language),
                    true,
                ),
            ] {
                if ui
                    .add_enabled(can_edit_layers && available, ToolbarButton::new(icon, hint))
                    .clicked()
                {
                    toolbar_action = Some(action);
                }
            }
        });
    });
    if let Some(action) = toolbar_action {
        apply_layer_action(app, ui.ctx(), action);
    }

    ui.add_space(style::SECTION_GAP);
    let total_rows = app.point_layers.layers.len();
    let clipboard_available = !app.clipboard_import_in_progress();
    let mut pending_action = None;
    egui_extras::TableBuilder::new(ui)
        .id_salt("point_layers_list")
        .striped(false)
        .sense(egui::Sense::click())
        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
        .min_scrolled_height(0.0)
        .max_scroll_height(180.0)
        .auto_shrink([false, true])
        .column(egui_extras::Column::exact(LAYER_VISIBILITY_COLUMN_WIDTH))
        .column(egui_extras::Column::exact(LAYER_COLOR_COLUMN_WIDTH))
        .column(egui_extras::Column::remainder().at_least(48.0).clip(true))
        .column(egui_extras::Column::exact(LAYER_COUNT_COLUMN_WIDTH))
        .body(|body| {
            body.rows(PointLayerRow::HEIGHT, total_rows, |row| {
                let layer = &mut app.point_layers.layers[row.index()];
                let layer_id = layer.id;
                let selected = layer_id == app.point_layers.selected_id;
                let cache = points_editor_cache_with_policy(&mut layer.points, false);
                let point_count = cache
                    .parsed_points
                    .as_ref()
                    .map(Vec::len)
                    .unwrap_or_else(|_| cache.plot_points.len());
                let has_error = cache.parsed_points.is_err();
                let result = PointLayerRow::new(layer, language)
                    .selected(selected)
                    .editable(can_edit_layers)
                    .clipboard_available(clipboard_available)
                    .point_count(point_count, has_error)
                    .show(row);
                if result.select {
                    app.point_layers.select(layer_id);
                }
                if let Some(action) = result.action {
                    pending_action = Some((layer_id, action));
                }
            });
        });
    // Команды применяются после построения таблицы, чтобы удаление не меняло индексы строк.
    if let Some((layer_id, action)) = pending_action {
        app.point_layers.select(layer_id);
        apply_layer_action(app, ui.ctx(), action);
    }
}

fn apply_layer_action(app: &mut CurveFitApp, ctx: &egui::Context, action: LayerAction) {
    match action {
        LayerAction::New => {
            app.create_empty_point_layer();
            return;
        }
        LayerAction::Paste => {
            app.request_points_clipboard_import(ctx);
            return;
        }
        LayerAction::Duplicate => {
            app.duplicate_selected_point_layer();
        }
        LayerAction::Clear => app.clear_points_text(true),
        LayerAction::Delete => app.delete_selected_point_layer(),
        LayerAction::ToggleVisibility => {
            let layer = app.selected_layer_mut();
            layer.visible = !layer.visible;
            app.refresh_status_after_points_edit();
        }
        LayerAction::ShowOnly => {
            if !app.point_layers.show_only(app.selected_layer().id) {
                return;
            }
            app.refresh_status_after_points_edit();
        }
    }
    app.clear_fit_outputs();
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
    let can_move_points_to_positive_xy = can_edit_points && app.can_move_points_to_positive_xy();
    let can_import_from_clipboard = can_edit_points && !app.clipboard_import_in_progress();
    #[cfg(not(target_arch = "wasm32"))]
    let can_import_from_file = can_edit_points && !app.points_file_import_in_progress();
    with_toolbar_hover_style(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = style::TOOLBAR_BUTTON_SPACING;
            let undo_response = ui.add_enabled(
                can_edit_points && !app.selected_points_editor().undo_stack.is_empty(),
                ToolbarButton::new(undo_icon_image(icon_tint), undo_tooltip(language)),
            );
            if undo_response.clicked() {
                app.undo_points_edit();
            }
            let redo_response = ui.add_enabled(
                can_edit_points && !app.selected_points_editor().redo_stack.is_empty(),
                ToolbarButton::new(redo_icon_image(icon_tint), redo_tooltip(language)),
            );
            if redo_response.clicked() {
                app.redo_points_edit();
            }
            let import_response = ui.add_enabled(
                can_import_from_clipboard,
                ToolbarButton::new(
                    clipboard_import_icon_image(icon_tint),
                    clipboard_import_tooltip(language),
                ),
            );
            if import_response.clicked() {
                app.request_points_clipboard_import(ui.ctx());
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let import_file_response = ui.add_enabled(
                    can_import_from_file,
                    ToolbarButton::new(
                        file_import_icon_image(icon_tint),
                        file_import_tooltip(language),
                    ),
                );
                if import_file_response.clicked() {
                    app.request_points_file_import();
                }
            }
            let clear_response = ui.add_enabled(
                can_edit_points,
                ToolbarButton::new(clear_icon_image(icon_tint), clear_tooltip(language)),
            );
            if clear_response.clicked() {
                app.clear_points_text(true);
                app.clear_fit_outputs();
                app.status = Some(StatusMessage::Cleared);
            }
            ui.add_enabled_ui(can_edit_points, |ui| {
                ToolbarButton::new(actions_icon_image(icon_tint), actions_tooltip(language))
                    .show_menu(ui, |ui| {
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
                        if ui
                            .add_enabled(
                                can_move_points_to_positive_xy,
                                egui::Button::new(tr(
                                    language,
                                    "Move to positive x/y",
                                    "Перенести в +X/+Y",
                                )),
                            )
                            .clicked()
                        {
                            app.move_points_to_positive_xy();
                            ui.close();
                        }
                    });
            });
        });
    });

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
            // Подписи ошибок меняют порядок виджетов, поэтому фокус и выделение привязываем к слою.
            let editor_id = egui::Id::new(("points_text_editor", app.selected_layer().id));
            let points_editor = app.selected_points_editor_mut();
            // До первого изменения заимствуем текст, чтобы не копировать его на каждом кадре.
            let mut text = Cow::Borrowed(points_editor.text.as_str());
            let response = ui.add_enabled(
                can_edit_points,
                PointsTextEdit::new(&mut text, editor_id, language)
                    .error_line(parse_error_line)
                    .desired_rows(desired_rows),
            );
            let before_edit = if response.changed() {
                let updated_text = text.into_owned();
                Some(std::mem::replace(&mut points_editor.text, updated_text))
            } else {
                None
            };
            let _ = widgets::info_hover(response, points_input_hint(language));
            if let Some(before_edit) = before_edit {
                app.push_points_undo_snapshot(before_edit);
                app.selected_points_editor_mut().redo_stack.clear();
                app.invalidate_points_cache();
                app.clear_fit_outputs();
                app.status = Some(StatusMessage::Ready);
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
            let label = tr(
                language,
                "Normalize x/y before fit",
                "Нормализовать x/y перед фитингом",
            );
            let normalization_response =
                ui.add(ToggleSwitch::new(&mut app.normalize_parametric_data).label(label));
            let _ = widgets::info_hover(normalization_response, normalization_hint(language));
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
            "Navigation mode\n- Drag to pan the plot\n- Wheel/trackpad scroll pans the plot\n- Hold Ctrl and use wheel/trackpad to zoom\n- Double-click resets view bounds",
            "Режим навигации\n- Перетаскивание двигает график\n- Колесо/трекпад сдвигают график\n- Зажмите Ctrl и используйте колесо/трекпад для масштаба\n- Двойной клик сбрасывает вид в границы данных",
        ),
        PlotTool::SinglePoint => tr(
            language,
            "Single point tool\n- Press left mouse button on plot to place one sample immediately\n- Hold middle mouse button to pan the plot\n- Hold Ctrl and use wheel/trackpad to zoom\n- No extra points are added while button is held\n- Best for precise manual placement",
            "Инструмент одной точки\n- Нажмите левую кнопку на графике, чтобы сразу поставить одну точку\n- Зажмите среднюю кнопку мыши, чтобы двигать график\n- Зажмите Ctrl и используйте колесо/трекпад для масштаба\n- Пока кнопка зажата, новые точки не добавляются\n- Подходит для точного ручного ввода",
        ),
        PlotTool::Dotted => tr(
            language,
            "Dotted tool\n- Left click on plot to add one sample\n- Hold left mouse button and move cursor to place points along the path\n- Hold middle mouse button to pan the plot\n- Hold Ctrl and use wheel/trackpad to zoom",
            "Инструмент пунктира\n- Левый клик по графику добавляет одну точку\n- Зажмите левую кнопку и ведите курсор, чтобы ставить точки по траектории\n- Зажмите среднюю кнопку мыши, чтобы двигать график\n- Зажмите Ctrl и используйте колесо/трекпад для масштаба",
        ),
        PlotTool::Spray => tr(
            language,
            "Spray tool\n- Hold left mouse button to add a stream of points\n- Hold middle mouse button to pan the plot\n- Hold Ctrl and use wheel/trackpad to zoom\n- Rate controls points per second\n- Radius controls spread around cursor",
            "Инструмент распыления\n- Зажмите левую кнопку, чтобы добавлять поток точек\n- Зажмите среднюю кнопку мыши, чтобы двигать график\n- Зажмите Ctrl и используйте колесо/трекпад для масштаба\n- Скорость задаёт число точек в секунду\n- Радиус задаёт разброс вокруг курсора",
        ),
        PlotTool::Eraser => tr(
            language,
            "Eraser tool\n- Hold left mouse button to remove points\n- Hold middle mouse button to pan the plot\n- Hold Ctrl and use wheel/trackpad to zoom\n- Radius controls erase area around cursor",
            "Ластик\n- Зажмите левую кнопку, чтобы удалять точки\n- Зажмите среднюю кнопку мыши, чтобы двигать график\n- Зажмите Ctrl и используйте колесо/трекпад для масштаба\n- Радиус задаёт область стирания вокруг курсора",
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

fn spray_brush_mode_hint(language: UiLanguage, brush: SprayBrush) -> &'static str {
    match brush {
        SprayBrush::Uniform => tr(
            language,
            "Uniform brush\n- Equal probability inside circle",
            "Равномерная кисть\n- Одинаковая вероятность внутри круга",
        ),
        SprayBrush::Gaussian => tr(
            language,
            "Gaussian brush\n- Denser near center, softer edges",
            "Гауссова кисть\n- Выше плотность в центре, мягче по краям",
        ),
    }
}

fn normalization_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Parametric normalization\n- Fit runs on normalized x/y for better numerical conditioning\n- Displayed parameters and metrics remain in original units\n- Useful when x and y scales differ significantly",
        "Нормализация параметрических данных\n- Фитинг выполняется на нормализованных x/y для лучшей численной устойчивости\n- Параметры и метрики в интерфейсе остаются в исходных единицах\n- Полезно при сильно разных масштабах x и y",
    )
}

fn undo_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Undo\n- Revert last points edit",
        "Отменить\n- Вернуть последнее изменение точек",
    )
}

fn redo_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Redo\n- Reapply reverted points edit",
        "Повторить\n- Повторно применить отменённое изменение",
    )
}

fn clear_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Clear points\n- Remove all points from input",
        "Очистить точки\n- Удалить все точки из ввода",
    )
}

fn actions_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Actions\n- Open extra operations for points",
        "Действия\n- Открыть дополнительные операции с точками",
    )
}

fn layer_new_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "New layer\n- Add an empty point layer and select it",
        "Новый слой\n- Добавить пустой слой точек и выбрать его",
    )
}

fn layer_duplicate_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Duplicate layer\n- Copy the selected layer with its points, color, and visibility",
        "Дублировать слой\n- Скопировать выбранный слой с точками, цветом и видимостью",
    )
}

fn layer_clipboard_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "New layer from clipboard\n- Paste clipboard points into a new selected layer",
        "Новый слой из буфера\n- Вставить точки из буфера в новый выбранный слой",
    )
}

fn layer_clear_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Clear layer\n- Remove all points from the selected layer",
        "Очистить слой\n- Удалить все точки из выбранного слоя",
    )
}

fn layer_delete_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Delete layer\n- Delete the selected layer\n- If it is the last layer, reset it to an empty default layer",
        "Удалить слой\n- Удалить выбранный слой\n- Если это последний слой, сбросить его в пустой слой по умолчанию",
    )
}

fn clipboard_import_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Paste from clipboard\n- Creates a new selected layer\n- Supports decimal dot/comma and scientific notation\n- Skips non-data lines without numeric values\n- Fails if any data line has 1 or 3+ numeric values",
        "Вставить из буфера обмена\n- Создаёт новый выбранный слой\n- Поддерживает десятичную точку/запятую и научный формат\n- Пропускает служебные строки без чисел\n- Возвращает ошибку, если в строке данных 1 или 3+ чисел",
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn file_import_tooltip(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Import from file\n- Replaces current input points\n- Supports .csv and .xlsx files\n- Uses robust two-numeric-values-per-row parsing",
        "Импорт из файла\n- Полностью заменяет текущие входные точки\n- Поддерживает файлы .csv и .xlsx\n- Использует робастный парсинг с двумя числовыми значениями в строке",
    )
}
