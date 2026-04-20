//! Отдельное окно с полной формулой модели и сопроводительными примечаниями.

use super::*;

pub(super) fn ui_formula_window(app: &mut CurveFitApp, ctx: &egui::Context) {
    if !app.panel.show_formula_window {
        return;
    }

    let language = app.ui_language;
    let formula_info = model_formula_info(
        language,
        app.selected_model,
        app.polynomial_degree,
        app.optimization_loss_metric,
    );
    let mut is_open = app.panel.show_formula_window;
    egui::Window::new(tr(language, "Model Formula", "Формула модели"))
        .open(&mut is_open)
        .default_size(egui::vec2(860.0, 520.0))
        .min_width(560.0)
        .min_height(320.0)
        .resizable(true)
        .show(ctx, |ui| {
            let formula_window_hint = tr(
                language,
                "Formula reference window\n- Every section uses LaTeX rendering and a plain-text fallback\n- You can copy only the model equation or the full reference",
                "Окно справки по формуле\n- Каждый раздел рендерится через LaTeX и имеет текстовый fallback\n- Можно копировать только модель или всю справку целиком",
            );
            ui.horizontal_wrapped(|ui| {
                let copy_model_response = CurveFitApp::info_hover(
                    ui.button(tr(language, "Copy model formula", "Скопировать формулу модели")),
                    formula_window_hint,
                );
                if copy_model_response.clicked() {
                    app.copy_text_to_clipboard(ui.ctx(), formula_info.model_plain_text.clone());
                }

                let copy_reference_response = ui.button(tr(
                    language,
                    "Copy full reference",
                    "Скопировать всю справку",
                ));
                if copy_reference_response.clicked() {
                    app.copy_text_to_clipboard(ui.ctx(), formula_info.reference_plain_text.clone());
                }
            });
            ui.add_space(4.0);
            egui::ScrollArea::vertical()
                .id_salt("formula_window_sections_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let dark_mode = ui.visuals().dark_mode;
                    for (index, section) in formula_info.sections.iter().enumerate() {
                        if index > 0 {
                            ui.add_space(6.0);
                            ui.separator();
                            ui.add_space(6.0);
                        }
                        egui::CollapsingHeader::new(section.title.as_str())
                            .id_salt(("formula_window_section", index))
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_space(2.0);
                                ui.push_id(index, |ui| {
                                    let svg_result =
                                        app.cached_formula_svg(&section.render_latex, dark_mode);
                                    egui::ScrollArea::horizontal()
                                        .id_salt("formula_window_section_formula_scroll")
                                        .auto_shrink([false, true])
                                        .show(ui, |ui| match svg_result {
                                            Ok((svg_uri, svg_bytes)) => {
                                                ui.add(
                                                    egui::Image::from_bytes(svg_uri, svg_bytes)
                                                        .fit_to_original_size(1.0),
                                                );
                                            }
                                            Err(_) => {
                                                ui.monospace(section.plain_text.as_str());
                                            }
                                        });
                                });
                                ui.add_space(4.0);
                                ui.label(egui::RichText::new(section.description.as_str()).small());
                            });
                    }
                });
        });
    app.panel.show_formula_window = is_open;
}
