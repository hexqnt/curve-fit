//! Отдельное окно с полной формулой модели и сопроводительными примечаниями.

use super::*;

pub(super) fn ui_formula_window(app: &mut CurveFitApp, ctx: &egui::Context) {
    if !app.panel.show_formula_window {
        return;
    }

    let language = app.ui_language;
    let formula_info = model_formula_info(language, app.selected_model, app.polynomial_degree);
    let plain_formula = formula_info.plain_text.clone();
    let mut is_open = app.panel.show_formula_window;
    egui::Window::new(tr(language, "Model Formula", "Формула модели"))
        .open(&mut is_open)
        .default_size(egui::vec2(760.0, 220.0))
        .min_width(420.0)
        .min_height(140.0)
        .resizable(true)
        .show(ctx, |ui| {
            let formula_window_hint = tr(
                language,
                "Formula window\n- Use horizontal scroll for very long formulas\n- Copy formula exports plain-text representation",
                "Окно формулы\n- Для очень длинных формул используйте горизонтальный скролл\n- Копирование формулы экспортирует текстовое представление",
            );
            ui.horizontal_wrapped(|ui| {
                let copy_response = CurveFitApp::info_hover(
                    ui.button(tr(language, "Copy formula", "Скопировать формулу")),
                    formula_window_hint,
                );
                if copy_response.clicked() {
                    app.copy_text_to_clipboard(ui.ctx(), plain_formula.clone());
                }
            });
            ui.add_space(4.0);
            let dark_mode = ui.visuals().dark_mode;
            let svg_result = app.cached_formula_svg(&formula_info.render_latex, dark_mode);
            egui::ScrollArea::horizontal()
                .id_salt("formula_window_scroll")
                .auto_shrink([false, true])
                .show(ui, |ui| match svg_result {
                    Ok((svg_uri, svg_bytes)) => {
                        ui.add(egui::Image::from_bytes(svg_uri, svg_bytes).fit_to_original_size(1.0));
                    }
                    Err(_) => {
                        ui.monospace(plain_formula.as_str());
                    }
                });
            ui.add_space(4.0);
            ui.label(egui::RichText::new(formula_info.notes).small());
        });
    app.panel.show_formula_window = is_open;
}
