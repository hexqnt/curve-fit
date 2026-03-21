use super::*;

impl CurveFitApp {
    pub(super) fn next_unit_random(&mut self) -> f64 {
        self.spray_seed = self
            .spray_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let value = self.spray_seed >> 11;
        value as f64 / ((1_u64 << 53) as f64)
    }

    pub(super) fn next_uniform_unit_disk_offset(&mut self) -> [f64; 2] {
        let radial = self.next_unit_random().sqrt();
        let angle = TAU * self.next_unit_random();
        [radial * angle.cos(), radial * angle.sin()]
    }

    pub(super) fn next_gaussian_unit_disk_offset(&mut self) -> [f64; 2] {
        loop {
            let u = self.next_unit_random();
            let radial = SPRAY_GAUSSIAN_SIGMA * (-2.0 * (1.0 - u).ln()).sqrt();
            if radial <= 1.0 {
                let angle = TAU * self.next_unit_random();
                return [radial * angle.cos(), radial * angle.sin()];
            }
        }
    }

    pub(super) fn next_spray_unit_disk_offset(&mut self) -> [f64; 2] {
        match self.spray_brush {
            SprayBrush::Uniform => self.next_uniform_unit_disk_offset(),
            SprayBrush::Gaussian => self.next_gaussian_unit_disk_offset(),
        }
    }

    pub(super) fn add_point_from_plot(&mut self, x: f64, y: f64) {
        let mut points = match self.parse_points_for_edit() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        match Point::try_new(x, y) {
            Ok(point) => {
                points.push(point);
                self.write_points_text(&points, true);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error.to_string()));
            }
        }
    }

    pub(super) fn spray_points_from_plot(
        &mut self,
        center_x: f64,
        center_y: f64,
        radius_x: f64,
        radius_y: f64,
    ) {
        let mut points = match self.parse_points_for_edit() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        for _ in 0..self.spray_density {
            let [offset_x, offset_y] = self.next_spray_unit_disk_offset();
            let x = center_x + offset_x * radius_x;
            let y = center_y + offset_y * radius_y;
            if let Ok(point) = Point::try_new(x, y) {
                points.push(point);
            }
        }

        self.write_points_text(&points, false);
    }

    pub(super) fn erase_points_from_plot(
        &mut self,
        center_x: f64,
        center_y: f64,
        radius_x: f64,
        radius_y: f64,
    ) {
        let mut points = match self.parse_points_for_edit() {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
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

        self.write_points_text(&points, false);
    }

    pub(super) fn plot_position_from_screen(
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

    pub(super) fn handle_plot_tools(&mut self, plot_response: &PlotResponse<()>) {
        if self.fit_in_progress {
            return;
        }

        let response = &plot_response.response;
        let is_continuous_tool = matches!(self.plot_tool, PlotTool::Spray | PlotTool::Eraser);
        let primary_down_on_plot = response.is_pointer_button_down_on();
        if is_continuous_tool && primary_down_on_plot {
            if self.active_tool_bounds.is_none() {
                self.push_points_undo_snapshot(self.points_text.clone());
            }
            self.active_tool_bounds
                .get_or_insert(*plot_response.transform.bounds());
        } else {
            self.active_tool_bounds = None;
        }
        let bounds = plot_response.transform.bounds();
        let radius_x_scale = bounds.width().abs().max(1e-6);
        let radius_y_scale = bounds.height().abs().max(1e-6);

        match self.plot_tool {
            PlotTool::None => {}
            PlotTool::SinglePoint => {
                if response.clicked_by(egui::PointerButton::Primary)
                    && let Some(screen_pos) = response.interact_pointer_pos()
                    && let Some(plot_pos) =
                        Self::plot_position_from_screen(plot_response, screen_pos)
                {
                    self.add_point_from_plot(plot_pos.x, plot_pos.y);
                }
            }
            PlotTool::Spray => {
                if primary_down_on_plot
                    && let Some(screen_pos) = response.interact_pointer_pos()
                    && let Some(plot_pos) =
                        Self::plot_position_from_screen(plot_response, screen_pos)
                {
                    let radius_x = self.spray_radius_rel * radius_x_scale;
                    let radius_y = self.spray_radius_rel * radius_y_scale;
                    self.spray_points_from_plot(plot_pos.x, plot_pos.y, radius_x, radius_y);
                }
            }
            PlotTool::Eraser => {
                if primary_down_on_plot
                    && let Some(screen_pos) = response.interact_pointer_pos()
                    && let Some(plot_pos) =
                        Self::plot_position_from_screen(plot_response, screen_pos)
                {
                    let radius_x = self.eraser_radius_rel * radius_x_scale;
                    let radius_y = self.eraser_radius_rel * radius_y_scale;
                    self.erase_points_from_plot(plot_pos.x, plot_pos.y, radius_x, radius_y);
                }
            }
        }
    }

    pub(super) fn ui_header(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();

        egui::ScrollArea::horizontal()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let delay_supported = !self.fit_in_progress;
                    let delay_slider =
                        egui::Slider::new(&mut self.iteration_delay_seconds, 0.0..=3.0)
                            .step_by(0.01)
                            .text(tr(
                                language,
                                "Iteration delay, sec",
                                "Задержка итерации, сек",
                            ));
                    let response = ui.add_enabled(delay_supported, delay_slider);
                    if !delay_supported {
                        let hint = tr(
                            language,
                            "Delay changes are locked while fitting is running.",
                            "Изменение задержки недоступно во время подгонки.",
                        );
                        response.on_disabled_hover_text(hint);
                    }

                    ui.separator();
                    ui.menu_button(tr(language, "View", "Вид"), |ui| {
                        if ui
                            .add(egui::Button::image_and_text(
                                fit_to_content_icon_image(icon_tint),
                                tr(language, "Fit to content", "Подогнать по содержимому"),
                            ))
                            .clicked()
                        {
                            self.fit_to_content_requested = true;
                            ui.close();
                        }
                        if ui
                            .add(egui::Button::image_and_text(
                                center_origin_icon_image(icon_tint),
                                tr(language, "Center to 0,0", "Центр к 0,0"),
                            ))
                            .clicked()
                        {
                            self.center_origin_requested = true;
                            self.origin_bottom_left_requested = false;
                            ui.close();
                        }
                        if ui
                            .add(egui::Button::new(tr(
                                language,
                                "Set 0,0 to bottom-left",
                                "0,0 в левый нижний угол",
                            )))
                            .clicked()
                        {
                            self.origin_bottom_left_requested = true;
                            self.center_origin_requested = false;
                            ui.close();
                        }
                    });

                    ui.separator();
                    ui.menu_button(tr(language, "Panels", "Панели"), |ui| {
                        ui.toggle_value(
                            &mut self.show_left_panel,
                            tr(language, "Left panel", "Левая панель"),
                        );
                        ui.toggle_value(
                            &mut self.show_right_panel,
                            tr(language, "Right panel", "Правая панель"),
                        );
                        ui.toggle_value(
                            &mut self.show_diagnostics_panel,
                            tr(language, "Diagnostics", "Диагностика"),
                        );
                    });
                });
            });
    }

    pub(super) fn ui_status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            self.ui_status(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(APP_VERSION_LABEL);
                ui.separator();
                let github_response =
                    ui.add(egui::Button::image_and_text(github_mark_image(), "GitHub"));
                if github_response.clicked() {
                    ui.ctx()
                        .open_url(egui::OpenUrl::new_tab(APP_REPOSITORY_URL));
                }

                ui.separator();
                egui::widgets::global_theme_preference_buttons(ui);

                ui.separator();
                ui.menu_image_text_button(
                    language_flag_image(self.ui_language),
                    self.ui_language.native_name(),
                    |ui| {
                        for candidate in UiLanguage::ALL {
                            let selected = self.ui_language == candidate;
                            if ui
                                .add(
                                    egui::Button::image_and_text(
                                        language_flag_image(candidate),
                                        candidate.native_name(),
                                    )
                                    .selected(selected),
                                )
                                .clicked()
                            {
                                self.ui_language = candidate;
                                ui.close();
                            }
                        }
                    },
                );
            });
        });
    }

    pub(super) fn ui_tools(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();
        ui.heading(tr(language, "Tools", "Инструменты"));

        ui.horizontal_wrapped(|ui| {
            for tool in [
                PlotTool::None,
                PlotTool::SinglePoint,
                PlotTool::Spray,
                PlotTool::Eraser,
            ] {
                let selected = self.plot_tool == tool;
                let button = egui::Button::image_and_text(
                    tool_icon_image(tool, icon_tint),
                    tool_label(language, tool),
                )
                .selected(selected);
                if ui.add(button).clicked() {
                    self.plot_tool = tool;
                }
            }
        });

        match self.plot_tool {
            PlotTool::None => {
                ui.label(tr(
                    language,
                    "Navigation mode: drag, zoom, and scroll the plot.",
                    "Режим навигации: перемещение, зум и прокрутка графика.",
                ));
            }
            PlotTool::SinglePoint => {}
            PlotTool::Spray => {
                ui.add(egui::Slider::new(&mut self.spray_density, 1..=30).text(tr(
                    language,
                    "Density",
                    "Плотность",
                )));
                ui.add(
                    egui::Slider::new(&mut self.spray_radius_rel, 0.002..=0.2)
                        .logarithmic(true)
                        .text(tr(language, "Radius", "Радиус")),
                );
                ui.horizontal_wrapped(|ui| {
                    ui.label(tr(language, "Brush", "Кисть"));
                    ui.selectable_value(
                        &mut self.spray_brush,
                        SprayBrush::Uniform,
                        spray_brush_label(language, SprayBrush::Uniform),
                    );
                    ui.selectable_value(
                        &mut self.spray_brush,
                        SprayBrush::Gaussian,
                        spray_brush_label(language, SprayBrush::Gaussian),
                    );
                });
            }
            PlotTool::Eraser => {
                ui.add(
                    egui::Slider::new(&mut self.eraser_radius_rel, 0.002..=0.2)
                        .logarithmic(true)
                        .text(tr(language, "Radius", "Радиус")),
                );
            }
        }
    }

    pub(super) fn ui_points_editor(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();
        let can_edit_points = !self.fit_in_progress;
        ui.heading(tr(language, "Input Points", "Точки"));
        ui.label(tr(
            language,
            "One point per line: x and y separated by space, tab, or ';'",
            "Одна точка на строку: x и y через пробел, табуляцию или ';'",
        ));

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    can_edit_points && !self.points_undo_stack.is_empty(),
                    egui::Button::image_and_text(
                        undo_icon_image(icon_tint),
                        tr(language, "Undo", "Отменить"),
                    ),
                )
                .clicked()
            {
                self.undo_points_edit();
            }
            if ui
                .add_enabled(
                    can_edit_points && !self.points_redo_stack.is_empty(),
                    egui::Button::image_and_text(
                        redo_icon_image(icon_tint),
                        tr(language, "Redo", "Повторить"),
                    ),
                )
                .clicked()
            {
                self.redo_points_edit();
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
                self.clear_points_text(true);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Cleared);
            }
        });

        let hint = tr(
            language,
            "Example:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
            "Пример:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
        );
        let text_height = ui.available_height();
        let row_height = ui.text_style_height(&egui::TextStyle::Monospace).max(1.0);
        let desired_rows = (text_height / row_height).floor().max(1.0) as usize;
        egui::ScrollArea::vertical()
            .id_salt("points_text_scroll")
            .max_height(text_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let text_width = ui.available_width();
                let before_edit = self.points_text.clone();
                let parse_error_line = self.points_cache().parse_error_line;
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
                    egui::TextEdit::multiline(&mut self.points_text)
                        .desired_width(text_width)
                        .desired_rows(desired_rows)
                        .font(egui::TextStyle::Monospace)
                        .hint_text(hint)
                        .layouter(&mut layouter)
                        .interactive(can_edit_points),
                );
                if response.changed() {
                    self.push_points_undo_snapshot(before_edit);
                    self.points_redo_stack.clear();
                    self.invalidate_points_cache();
                }
            });

        if let Err(error) = &self.points_cache().parsed_points {
            ui.colored_label(
                egui::Color32::from_rgb(200, 64, 64),
                format!("{POINTS_PARSE_ERROR_PREFIX}{error}"),
            );
        }

        if self.fit_in_progress {
            ui.label(tr(
                language,
                "Point editing is disabled while fitting is running.",
                "Редактирование точек отключено во время подгонки.",
            ));
        }
    }

    pub(super) fn ui_family_and_params(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let can_edit_params = !self.fit_in_progress;
        ui.heading(tr(language, "Model", "Модель"));

        let previous_model = self.selected_model;
        ui.add_enabled_ui(can_edit_params, |ui| {
            egui::ComboBox::from_label(tr(language, "Model type", "Тип модели"))
                .selected_text(model_choice_label(language, self.selected_model))
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
                                ui.selectable_label(self.selected_model == model, model_label);
                            if response.clicked() {
                                self.selected_model = model;
                            }
                        }
                    }
                });
        });

        let mut params_need_sync = false;
        if previous_model != self.selected_model {
            params_need_sync = true;
        }

        if self.selected_model.is_polynomial() {
            let previous_degree = self.polynomial_degree;
            ui.add_enabled(
                can_edit_params,
                egui::Slider::new(&mut self.polynomial_degree, 1..=9).text(tr(
                    language,
                    "Polynomial degree",
                    "Степень полинома",
                )),
            );
            if previous_degree != self.polynomial_degree {
                params_need_sync = true;
            }
        }

        if params_need_sync {
            self.sync_parameter_inputs();
            self.clear_fit_outputs();
        }

        let formula_info =
            model_formula_info(language, self.selected_model, self.polynomial_degree);
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new(tr(language, "Model Formula", "Формула модели")).strong());
            #[cfg(not(target_arch = "wasm32"))]
            {
                let dark_mode = ui.visuals().dark_mode;
                let (svg_uri, svg_bytes) =
                    self.cached_formula_svg(&formula_info.full_formula, dark_mode);
                ui.add(
                    egui::Image::from_bytes(svg_uri, svg_bytes)
                        .max_width(ui.available_width())
                        .fit_to_original_size(1.0),
                );
            }
            #[cfg(target_arch = "wasm32")]
            {
                let plain_formula = formula_plain_text(&formula_info.full_formula);
                let formula_label = egui::RichText::new(plain_formula).monospace();
                ui.label(formula_label);
            }
            ui.label(egui::RichText::new(formula_info.notes).small());
        });

        if let Some(family) = self.resolved_model().parametric_family() {
            let mut method_to_apply = None;
            ui.horizontal_wrapped(|ui| {
                ui.label(tr(language, "Initial parameters", "Начальные параметры"));
                ui.add_enabled_ui(can_edit_params, |ui| {
                    ui.menu_button(
                        tr(language, "+ Initialize", "+ Инициализация"),
                        |ui| {
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

            if let Some(method) = method_to_apply {
                self.apply_param_init_method(method);
            }

            for (index, parameter_name) in family.parameter_names().iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(*parameter_name);
                    ui.add_enabled(
                        can_edit_params,
                        egui::TextEdit::singleline(&mut self.parameter_inputs[index])
                            .desired_width(120.0),
                    );
                });
            }
        } else {
            ui.label(tr(
                language,
                "Spline models are non-parametric, but they optimize knot y-values as parameters.",
                "Сплайны непараметрические, но оптимизируют knot y как параметры.",
            ));
            ui.add_space(4.0);
            let min_knots = self
                .resolved_model()
                .spline_min_knots()
                .expect("non-parametric branch guarantees spline model");
            self.spline_knots = self.spline_knots.max(min_knots);
            self.sync_spline_initial_knot_y_inputs(self.spline_knots);
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
                self.apply_spline_param_init_method(method);
            }

            ui.add_enabled_ui(can_edit_params, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.spline_knots, min_knots..=40).text(tr(
                        language,
                        "Spline knot count",
                        "Число узлов сплайна",
                    )),
                );
                egui::ComboBox::from_label(tr(language, "Knot reduction", "Редукция узлов"))
                    .selected_text(spline_knot_strategy_label(
                        language,
                        self.spline_knot_strategy,
                    ))
                    .show_ui(ui, |ui| {
                        for strategy in SplineKnotStrategy::ALL {
                            ui.selectable_value(
                                &mut self.spline_knot_strategy,
                                strategy,
                                spline_knot_strategy_label(language, strategy),
                            );
                        }
                    });
                egui::ComboBox::from_label(tr(language, "Extrapolation", "Экстраполяция"))
                    .selected_text(spline_extrapolation_label(
                        language,
                        self.spline_extrapolation,
                    ))
                    .show_ui(ui, |ui| {
                        for extrapolation in SplineExtrapolation::ALL {
                            ui.selectable_value(
                                &mut self.spline_extrapolation,
                                extrapolation,
                                spline_extrapolation_label(language, extrapolation),
                            );
                        }
                    });
                egui::ComboBox::from_label(tr(language, "Duplicate x", "Дубли x"))
                    .selected_text(spline_duplicate_policy_label(
                        language,
                        self.spline_duplicate_x_policy,
                    ))
                    .show_ui(ui, |ui| {
                        for policy in SplineDuplicateXPolicy::ALL {
                            ui.selectable_value(
                                &mut self.spline_duplicate_x_policy,
                                policy,
                                spline_duplicate_policy_label(language, policy),
                            );
                        }
                    });
            });
            self.sync_spline_initial_knot_y_inputs(self.spline_knots);
            ui.label(format!(
                "{}: {}",
                tr(
                    language,
                    "Target spline parameter count",
                    "Целевое число параметров сплайна"
                ),
                self.spline_knots
            ));
            ui.label(tr(
                language,
                "Initial knot y values",
                "Начальные значения knot y",
            ));
            for (index, value) in self.spline_initial_knot_y_inputs.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("knot_y[{index}]"));
                    ui.add_enabled(
                        can_edit_params,
                        egui::TextEdit::singleline(value).desired_width(120.0),
                    );
                });
            }
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
            ui.label(egui::RichText::new(tr(
                language,
                "Sample density is selected automatically from knot count and data size.",
                "Плотность сэмплирования выбирается автоматически по числу узлов и размеру данных.",
            )).small());
        }
    }

    pub(super) fn ui_optimizer(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        let icon_tint = ui.visuals().text_color();
        ui.separator();
        ui.heading(tr(language, "Optimizer", "Оптимизатор"));
        egui::ComboBox::from_label(tr(language, "Method", "Метод"))
            .selected_text(optimizer_method_label(language, self.optimizer_method))
            .show_ui(ui, |ui| {
                for method in OptimizerMethod::ALL {
                    ui.selectable_value(
                        &mut self.optimizer_method,
                        method,
                        optimizer_method_label(language, method),
                    );
                }
            });
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.optimizer_mode,
                OptimizerUiMode::Basic,
                tr(language, "Basic", "Базовый"),
            );
            ui.selectable_value(
                &mut self.optimizer_mode,
                OptimizerUiMode::Advanced,
                tr(language, "Advanced", "Продвинутый"),
            );
        });

        if self.optimizer_mode == OptimizerUiMode::Basic {
            ui.label(tr(
                language,
                "Use presets to quickly control convergence speed and stability.",
                "Используйте пресеты для быстрого выбора баланса скорости и устойчивости.",
            ));

            let previous_preset = self.selected_optimizer_preset();
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
                    self.set_selected_optimizer_preset(OptimizerPreset::Custom);
                } else {
                    self.apply_selected_optimizer_preset(selected_preset);
                }
            }
            match self.optimizer_method {
                OptimizerMethod::Lbfgs => {
                    ui.label(format!("history_size = {}", self.lbfgs_inputs.history_size));
                    ui.label(format!("max_iters = {}", self.lbfgs_inputs.max_iters));
                    ui.label(format!("tol_grad = {:.2e}", self.lbfgs_inputs.tol_grad));
                    ui.label(format!("tol_cost = {:.2e}", self.lbfgs_inputs.tol_cost));
                }
                OptimizerMethod::NelderMead => {
                    ui.label(format!("max_iters = {}", self.nelder_mead_inputs.max_iters));
                    ui.label(format!(
                        "simplex_scale = {:.3}",
                        self.nelder_mead_inputs.simplex_scale
                    ));
                    ui.label(format!(
                        "sd_tolerance = {:.2e}",
                        self.nelder_mead_inputs.sd_tolerance
                    ));
                }
                OptimizerMethod::SteepestDescent => {
                    ui.label(format!(
                        "max_iters = {}",
                        self.steepest_descent_inputs.max_iters
                    ));
                    ui.label(format!("c1 = {:.2e}", self.steepest_descent_inputs.c1));
                    ui.label(format!("c2 = {:.3}", self.steepest_descent_inputs.c2));
                    ui.label(format!(
                        "width_tolerance = {:.2e}",
                        self.steepest_descent_inputs.width_tolerance
                    ));
                }
            }
        } else {
            ui.label(tr(
                language,
                "Use sliders to tune optimizer parameters.",
                "Используйте бегунки для настройки оптимизатора.",
            ));
            match self.optimizer_method {
                OptimizerMethod::Lbfgs => {
                    let before = self.lbfgs_inputs.clone();
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.history_size, 1..=50)
                            .text("history_size"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.max_iters, 10..=10_000)
                            .text("max_iters"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.tol_grad, 1e-12..=1e-2)
                            .logarithmic(true)
                            .text("tol_grad"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.tol_cost, 1e-14..=1e-4)
                            .logarithmic(true)
                            .text("tol_cost"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.c1, C1_MIN..=0.2)
                            .logarithmic(true)
                            .text("c1"),
                    );
                    ui.add(egui::Slider::new(&mut self.lbfgs_inputs.c2, 0.1..=C2_MAX).text("c2"));
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.step_min, STEP_MIN_MIN..=1.0)
                            .logarithmic(true)
                            .text("step_min"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.step_max, 1e-6..=STEP_MAX_MAX)
                            .logarithmic(true)
                            .text("step_max"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lbfgs_inputs.width_tolerance, 1e-14..=1e-3)
                            .logarithmic(true)
                            .text("width_tolerance"),
                    );

                    self.lbfgs_inputs.normalize_after_ui();
                    if self.lbfgs_inputs != before {
                        self.lbfgs_preset = OptimizerPreset::Custom;
                    }
                }
                OptimizerMethod::NelderMead => {
                    let before = self.nelder_mead_inputs.clone();
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.max_iters, 10..=10_000)
                            .text("max_iters"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.simplex_scale, 1e-4..=1.0)
                            .logarithmic(true)
                            .text("simplex_scale"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.sd_tolerance, 1e-14..=1e-2)
                            .logarithmic(true)
                            .text("sd_tolerance"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.alpha, 1e-3..=5.0)
                            .logarithmic(true)
                            .text("alpha"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.gamma, 1.0001..=5.0)
                            .logarithmic(true)
                            .text("gamma"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.rho, 1e-4..=0.5)
                            .logarithmic(true)
                            .text("rho"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.nelder_mead_inputs.sigma, 1e-4..=1.0)
                            .logarithmic(true)
                            .text("sigma"),
                    );

                    self.nelder_mead_inputs.normalize_after_ui();
                    if self.nelder_mead_inputs != before {
                        self.nelder_mead_preset = OptimizerPreset::Custom;
                    }
                }
                OptimizerMethod::SteepestDescent => {
                    let before = self.steepest_descent_inputs.clone();
                    ui.add(
                        egui::Slider::new(&mut self.steepest_descent_inputs.max_iters, 10..=10_000)
                            .text("max_iters"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.steepest_descent_inputs.c1, C1_MIN..=0.2)
                            .logarithmic(true)
                            .text("c1"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.steepest_descent_inputs.c2, 0.1..=C2_MAX)
                            .text("c2"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.steepest_descent_inputs.step_min,
                            STEP_MIN_MIN..=1.0,
                        )
                        .logarithmic(true)
                        .text("step_min"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.steepest_descent_inputs.step_max,
                            1e-6..=STEP_MAX_MAX,
                        )
                        .logarithmic(true)
                        .text("step_max"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.steepest_descent_inputs.width_tolerance,
                            1e-14..=1e-3,
                        )
                        .logarithmic(true)
                        .text("width_tolerance"),
                    );

                    self.steepest_descent_inputs.normalize_after_ui();
                    if self.steepest_descent_inputs != before {
                        self.steepest_descent_preset = OptimizerPreset::Custom;
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
            self.apply_selected_optimizer_preset(OptimizerPreset::Balanced);
        }
    }

    pub(super) fn ui_status(&self, ui: &mut egui::Ui) {
        if let Some(status) = &self.status {
            let color = if status.is_error() {
                egui::Color32::from_rgb(200, 64, 64)
            } else {
                egui::Color32::from_rgb(64, 160, 96)
            };
            ui.colored_label(color, status.text(self.ui_language));
        }
    }

    pub(super) fn ui_plot(&mut self, ui: &mut egui::Ui, height: f32) {
        let language = self.ui_language;
        let points = self.points_cache().plot_points.clone();
        let points_slice = points.as_slice();
        let (x_min, x_max) = plot_domain(points_slice);
        let navigation_mode = matches!(self.plot_tool, PlotTool::None);
        let spline_curve = self.spline_plot_curve.clone();
        let spline_curve_slice = spline_curve.as_deref();
        let sampled_curve = if spline_curve_slice.is_none() {
            let active_params = self
                .fit_preview_params
                .clone()
                .or_else(|| self.fit_result.as_ref().map(|result| result.params.clone()));
            active_params.map(|params| {
                self.cached_sampled_curve(&params, x_min, x_max, PARAMETRIC_PLOT_SAMPLES)
            })
        } else {
            None
        };
        let fitted_curve_points = spline_curve_slice.or(sampled_curve.as_deref());
        let fitted_line_name = if spline_curve_slice.is_some() {
            model_choice_label(language, self.selected_model).to_string()
        } else if self.fit_in_progress {
            if let Some(iteration) = self.fit_preview_iteration {
                format!(
                    "{} ({})",
                    tr(language, "Fitted", "Фитинг"),
                    format_args!("{} {iteration}", tr(language, "iter", "итер."))
                )
            } else {
                tr(language, "Fitted", "Фитинг").to_string()
            }
        } else {
            tr(language, "Fitted", "Фитинг").to_string()
        };
        let content_bounds = fit_bounds_for_content(points_slice, fitted_curve_points);
        let fit_bounds = if self.fit_to_content_requested {
            content_bounds
        } else {
            None
        };
        let center_bounds = if self.center_origin_requested {
            let (span_x, span_y) = self
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
        let origin_bottom_left_bounds = if self.origin_bottom_left_requested {
            let (max_x, max_y) = self
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
        let locked_tool_bounds = self.active_tool_bounds;

        let plot_response = Plot::new("fit_plot")
            .height(height)
            .legend(Legend::default())
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
                            .radius(3.0),
                    );
                }
                if let Some(fitted) = spline_curve_slice {
                    plot_ui.line(Line::new(fitted_line_name.clone(), fitted));
                } else if let Some(fitted) = sampled_curve.as_deref() {
                    plot_ui.line(Line::new(fitted_line_name.clone(), fitted));
                }
            });

        let bounds = plot_response.transform.bounds();
        self.last_plot_bounds = Some(*bounds);

        if self.fit_to_content_requested {
            self.fit_to_content_requested = false;
        }
        if self.center_origin_requested {
            self.center_origin_requested = false;
        }
        if self.origin_bottom_left_requested {
            self.origin_bottom_left_requested = false;
        }

        self.handle_plot_tools(&plot_response);
    }

    pub(super) fn ui_iteration_diagnostics(&mut self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        ui.heading(tr(
            language,
            "Iteration diagnostics",
            "Диагностика итераций",
        ));

        if self.iteration_diagnostics.loss_mse_points.is_empty() {
            ui.label(tr(
                language,
                "Run Fit to collect iteration history.",
                "Запустите фитинг, чтобы получить историю итераций.",
            ));
            self.diagnostics_loss_axis_width = 0.0;
            self.diagnostics_residual_axis_width = 0.0;
            self.diagnostics_params_axis_width = 0.0;
            return;
        }

        let has_residual_plot = !self.residual_plot_points.is_empty();
        let available_height = ui.available_height().max(2.0);
        let spacing = ui.spacing().item_spacing.y;
        let plot_count = if has_residual_plot { 3.0 } else { 2.0 };
        let total_spacing = spacing * (plot_count - 1.0);
        let plot_height = ((available_height - total_spacing).max(2.0)) / plot_count;
        let shared_axis_width = self
            .diagnostics_loss_axis_width
            .max(self.diagnostics_residual_axis_width)
            .max(self.diagnostics_params_axis_width);
        let loss_extra_padding = (shared_axis_width - self.diagnostics_loss_axis_width).max(0.0);
        let residual_extra_padding =
            (shared_axis_width - self.diagnostics_residual_axis_width).max(0.0);
        let params_extra_padding =
            (shared_axis_width - self.diagnostics_params_axis_width).max(0.0);
        let mut measured_loss_axis_width = 0.0;
        let mut measured_residual_axis_width = 0.0;
        let mut measured_params_axis_width = 0.0;

        {
            let loss_points = &self.iteration_diagnostics.loss_mse_points;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height),
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    if loss_extra_padding > 0.0 {
                        ui.add_space(loss_extra_padding);
                    }
                    let plot_response = Plot::new("loss_mse_plot")
                        .height(plot_height)
                        .legend(Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .allow_boxed_zoom(false)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(
                                tr(language, "Loss (MSE)", "Лосс (MSE)"),
                                PlotPoints::from_iter(loss_points.iter().copied()),
                            ));
                        });
                    measured_loss_axis_width = diagnostics_plot_y_axis_width(&plot_response);
                },
            );
        }

        if has_residual_plot {
            let residual_points = &self.residual_plot_points;
            let x_min = residual_points
                .iter()
                .map(|point| point.x)
                .fold(f64::INFINITY, f64::min);
            let x_max = residual_points
                .iter()
                .map(|point| point.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let zero_line = [[x_min, 0.0], [x_max, 0.0]];

            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height),
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    if residual_extra_padding > 0.0 {
                        ui.add_space(residual_extra_padding);
                    }
                    let plot_response = Plot::new("residuals_diagnostics_plot")
                        .height(plot_height)
                        .legend(Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .allow_boxed_zoom(false)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(
                                tr(language, "Zero", "Ноль"),
                                PlotPoints::from_iter(zero_line),
                            ));
                            plot_ui.points(
                                PlotPointsItem::new(
                                    tr(language, "Residuals", "Остатки"),
                                    residual_points.as_slice(),
                                )
                                .radius(2.5),
                            );
                        });
                    measured_residual_axis_width = diagnostics_plot_y_axis_width(&plot_response);
                },
            );
        }

        {
            let parameter_names = &self.iteration_diagnostics.parameter_names;
            let parameter_series = &self.iteration_diagnostics.parameter_series;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height),
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    if params_extra_padding > 0.0 {
                        ui.add_space(params_extra_padding);
                    }
                    let plot_response = Plot::new("parameter_iteration_plot")
                        .height(plot_height)
                        .legend(Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_double_click_reset(false)
                        .allow_boxed_zoom(false)
                        .show(ui, |plot_ui| {
                            for (name, series) in
                                parameter_names.iter().zip(parameter_series.iter())
                            {
                                plot_ui.line(Line::new(
                                    name.clone(),
                                    PlotPoints::from_iter(series.iter().copied()),
                                ));
                            }
                        });
                    measured_params_axis_width = diagnostics_plot_y_axis_width(&plot_response);
                },
            );
        }

        self.diagnostics_loss_axis_width = measured_loss_axis_width;
        self.diagnostics_residual_axis_width = measured_residual_axis_width;
        self.diagnostics_params_axis_width = measured_params_axis_width;
    }

    pub(super) fn ui_result(&self, ui: &mut egui::Ui) {
        let language = self.ui_language;
        ui.heading(tr(language, "Result", "Результат"));
        if self.fit_in_progress {
            ui.label(tr(
                language,
                "Fitting in progress. Curve updates after each iteration.",
                "Подгонка в процессе. Кривая обновляется после каждой итерации.",
            ));
            if let Some(iteration) = self.fit_preview_iteration {
                ui.label(format!(
                    "{}: {iteration}",
                    tr(language, "Iteration", "Итерация")
                ));
            }
            if let Some(params) = &self.fit_preview_params {
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

        let metrics = self.result_metrics.unwrap_or_else(|| {
            if let Some(result) = &self.fit_result {
                ExtendedMetrics {
                    mse: result.mse,
                    rmse: result.rmse,
                    ..ExtendedMetrics::default()
                }
            } else if let Some(result) = &self.spline_result {
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

        if let Some(result) = &self.fit_result {
            ui.label(format!(
                "{}: {}",
                tr(language, "Family", "Семейство"),
                family_label(language, result.family)
            ));
            ui.label(format!("MSE: {:.8}", metrics.mse));
            ui.label(format!("RMSE: {:.8}", metrics.rmse));
            ui.label(format!("MAE: {:.8}", metrics.mae));
            ui.label(format!("R²: {:.8}", metrics.r2));
            ui.label(format!(
                "{}: {:.8}",
                tr(language, "Max |error|", "Макс |ошибка|"),
                metrics.max_abs_error
            ));
            ui.label(format!(
                "{}: {}",
                tr(language, "Iterations", "Итерации"),
                result.iterations
            ));
            ui.separator();
            ui.label(tr(language, "Parameters", "Параметры"));
            for (name, value) in result
                .family
                .parameter_names()
                .iter()
                .zip(result.params.values())
            {
                ui.label(format!("{name} = {value:.8}"));
            }
        } else if let Some(result) = &self.spline_result {
            ui.label(format!(
                "{}: {}",
                tr(language, "Family", "Семейство"),
                model_choice_label(language, self.selected_model)
            ));
            ui.label(format!("MSE: {:.8}", metrics.mse));
            ui.label(format!("RMSE: {:.8}", metrics.rmse));
            ui.label(format!("MAE: {:.8}", metrics.mae));
            ui.label(format!("R²: {:.8}", metrics.r2));
            ui.label(format!(
                "{}: {:.8}",
                tr(language, "Max |error|", "Макс |ошибка|"),
                metrics.max_abs_error
            ));
            ui.label(format!(
                "{}: {}",
                tr(language, "Iterations", "Итерации"),
                result.iterations
            ));
            ui.separator();
            ui.label(format!(
                "{}: {}",
                tr(language, "Parameters", "Параметры"),
                result.knots.len()
            ));
            for (index, knot) in result.knots.iter().enumerate() {
                ui.label(format!(
                    "knot_y[{index}] @ x={:.8}: {:.8}",
                    knot[0], knot[1]
                ));
            }
        } else {
            ui.label(tr(
                language,
                "Run Fit to see optimization results.",
                "Нажмите Fit, чтобы увидеть результат оптимизации.",
            ));
        }
    }
}
