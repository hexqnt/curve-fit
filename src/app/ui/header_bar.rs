use super::*;

pub(super) fn ui_header(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    let icon_tint = ui.visuals().text_color();

    egui::ScrollArea::horizontal()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.menu_image_text_button(
                    view_icon_image(icon_tint),
                    tr(language, "View", "Вид"),
                    |ui| {
                        if ui
                            .add(egui::Button::image_and_text(
                                fit_to_content_icon_image(icon_tint),
                                tr(language, "Fit to content", "Подогнать по содержимому"),
                            ))
                            .clicked()
                        {
                            app.fit_to_content_requested = true;
                            ui.close();
                        }
                        if ui
                            .add(egui::Button::image_and_text(
                                center_origin_icon_image(icon_tint),
                                tr(language, "Center to 0,0", "Центр к 0,0"),
                            ))
                            .clicked()
                        {
                            app.center_origin_requested = true;
                            app.origin_bottom_left_requested = false;
                            ui.close();
                        }
                        if ui
                            .add(egui::Button::image_and_text(
                                origin_bottom_left_icon_image(icon_tint),
                                tr(
                                    language,
                                    "Set 0,0 to bottom-left",
                                    "0,0 в левый нижний угол",
                                ),
                            ))
                            .clicked()
                        {
                            app.origin_bottom_left_requested = true;
                            app.center_origin_requested = false;
                            ui.close();
                        }
                    },
                );

                ui.separator();
                ui.menu_image_text_button(
                    panels_icon_image(icon_tint),
                    tr(language, "Panels", "Панели"),
                    |ui| {
                        CurveFitApp::toggle_switch_labeled(
                            ui,
                            &mut app.panel.show_left,
                            tr(language, "Left panel", "Левая панель"),
                        );
                        CurveFitApp::toggle_switch_labeled(
                            ui,
                            &mut app.panel.show_right,
                            tr(language, "Right panel", "Правая панель"),
                        );
                        CurveFitApp::toggle_switch_labeled(
                            ui,
                            &mut app.panel.show_diagnostics,
                            tr(language, "Diagnostics", "Диагностика"),
                        );
                    },
                );

                ui.separator();
                let (min_iteration, max_iteration) =
                    app.replay_iteration_bounds().unwrap_or((0, 0));
                let mut selected_iteration =
                    app.replay_selected_iteration().unwrap_or(min_iteration);
                let replay_slider_enabled = !app.fit_in_progress && !app.replay.frames.is_empty();
                let response = ui.add_enabled(
                    replay_slider_enabled,
                    egui::Slider::new(&mut selected_iteration, min_iteration..=max_iteration)
                        .text(tr(language, "Displayed iteration", "Показываемая итерация")),
                );
                if replay_slider_enabled && response.changed() {
                    app.pause_replay();
                    app.select_nearest_replay_iteration(selected_iteration);
                }
                CurveFitApp::toggle_switch_labeled(
                    ui,
                    &mut app.replay.autoplay_on_fit,
                    tr(language, "Auto-play", "Автопромотка"),
                );
                let (play_icon, play_label) = if app.replay.autoplay {
                    (
                        replay_pause_icon_image(icon_tint),
                        tr(language, "Pause", "Пауза"),
                    )
                } else {
                    (
                        replay_play_icon_image(icon_tint),
                        tr(language, "Play", "Пуск"),
                    )
                };
                let can_toggle_play = !app.fit_in_progress && app.replay.frames.len() > 1;
                if ui
                    .add_enabled(
                        can_toggle_play,
                        egui::Button::image_and_text(play_icon, play_label),
                    )
                    .clicked()
                {
                    app.toggle_replay_autoplay();
                }
                CurveFitApp::info_tooltip(ui, replay_controls_hint(language));
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut app.replay.iteration_delay_seconds, 0.0..=3.0)
                        .step_by(0.01)
                        .text(tr(language, "Replay step, sec", "Шаг промотки, сек")),
                );
            });
        });
}

fn replay_controls_hint(language: UiLanguage) -> &'static str {
    tr(
        language,
        "Replay controls\n- Displayed iteration selects the frame shown on plot/diagnostics\n- Auto-play starts replay automatically after fit\n- Play/Pause controls manual playback\n- Replay step sets delay between frames in seconds",
        "Управление промоткой\n- Показываемая итерация выбирает кадр на графике и в диагностике\n- Автопромотка автоматически запускается после фитинга\n- Пуск/Пауза управляют ручным воспроизведением\n- Шаг промотки задаёт задержку между кадрами в секундах",
    )
}

pub(super) fn ui_status_bar(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        app.ui_status(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.weak(APP_VERSION_LABEL);
            ui.separator();
            let github_response = ui.add(egui::Button::image_and_text(
                github_mark_image(ui.visuals().dark_mode),
                "GitHub",
            ));
            if github_response.clicked() {
                ui.ctx()
                    .open_url(egui::OpenUrl::new_tab(APP_REPOSITORY_URL));
            }

            ui.separator();
            egui::widgets::global_theme_preference_buttons(ui);

            ui.separator();
            ui.menu_image_text_button(
                language_flag_image(app.ui_language),
                app.ui_language.native_name(),
                |ui| {
                    for candidate in UiLanguage::ALL {
                        let selected = app.ui_language == candidate;
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
                            app.ui_language = candidate;
                            ui.close();
                        }
                    }
                },
            );
        });
    });
}
