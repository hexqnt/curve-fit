//! Отдельное окно с полной формулой модели и сопроводительными примечаниями.

use super::*;

const FORMULA_WINDOW_DEFAULT_HEIGHT: f32 = 520.0;
const FORMULA_WINDOW_MIN_WIDTH: f32 = 450.0;
const FORMULA_WINDOW_MIN_HEIGHT: f32 = 320.0;
const FORMULA_WINDOW_SCREEN_MARGIN: f32 = 24.0;
const FORMULA_WINDOW_CONTENT_MARGIN_X: f32 = 60.0;
const FORMULA_WINDOW_FALLBACK_CHAR_WIDTH: f32 = 8.0;

pub(super) fn ui_formula_window(app: &mut CurveFitApp, ctx: &egui::Context) {
    if !app.panel.show_formula_window {
        return;
    }

    let language = app.ui_language;
    let saturating_trend_tau_grid = app.parsed_saturating_trend_tau_grid().ok().flatten();
    let formula_info = model_formula_info(
        language,
        app.selected_model,
        app.polynomial_degree,
        app.rational_degree,
        app.saturating_trend_tau_count,
        saturating_trend_tau_grid
            .as_ref()
            .map(SaturatingTrendTauGrid::as_slice),
        app.optimization_loss_metric,
    );
    let mut is_open = app.panel.show_formula_window;
    let dark_mode = ctx.global_style().visuals.dark_mode;
    let window_width = formula_window_width(app, ctx, &formula_info, dark_mode);
    egui::Window::new(tr(language, "Model Formula", "Формула модели"))
        .open(&mut is_open)
        .default_size(egui::vec2(window_width, FORMULA_WINDOW_DEFAULT_HEIGHT))
        .min_width(window_width)
        .max_width(window_width)
        .min_height(FORMULA_WINDOW_MIN_HEIGHT)
        .resizable([false, true])
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

fn formula_window_width(
    app: &mut CurveFitApp,
    ctx: &egui::Context,
    formula_info: &ModelFormulaInfo,
    dark_mode: bool,
) -> f32 {
    let content_width = formula_info
        .sections
        .iter()
        .map(|section| formula_section_width(app, ctx, section, dark_mode))
        .fold(0.0_f32, f32::max);
    let target_width = content_width + FORMULA_WINDOW_CONTENT_MARGIN_X;
    let max_width = (ctx.content_rect().width() - 2.0 * FORMULA_WINDOW_SCREEN_MARGIN).max(320.0);
    let min_width = FORMULA_WINDOW_MIN_WIDTH.min(max_width);
    target_width.clamp(min_width, max_width)
}

fn formula_section_width(
    app: &mut CurveFitApp,
    ctx: &egui::Context,
    section: &FormulaReferenceSection,
    dark_mode: bool,
) -> f32 {
    let formula_width = app
        .cached_formula_svg(&section.render_latex, dark_mode)
        .ok()
        .and_then(|(_, bytes)| svg_width(&bytes))
        .unwrap_or_else(|| plain_text_width(section.plain_text.as_str()));

    let text_width = text_lines_width(
        ctx,
        egui::TextStyle::Small,
        [section.title.as_str(), section.description.as_str()].into_iter(),
    );

    formula_width.max(text_width)
}

fn text_lines_width<'a>(
    ctx: &egui::Context,
    text_style: egui::TextStyle,
    texts: impl Iterator<Item = &'a str>,
) -> f32 {
    let font_id = text_style.resolve(&ctx.global_style());
    ctx.fonts_mut(|fonts| {
        texts
            .flat_map(str::lines)
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| {
                fonts
                    .layout_no_wrap(line.to_owned(), font_id.clone(), egui::Color32::WHITE)
                    .size()
                    .x
            })
            .fold(0.0_f32, f32::max)
    })
}

fn plain_text_width(text: &str) -> f32 {
    text.lines()
        .map(|line| line.chars().count() as f32 * FORMULA_WINDOW_FALLBACK_CHAR_WIDTH)
        .fold(0.0_f32, f32::max)
}

fn svg_width(bytes: &[u8]) -> Option<f32> {
    let svg = std::str::from_utf8(bytes).ok()?;
    let after_attr = svg.split_once(r#"width=""#)?.1;
    let raw_width = after_attr.split_once('"')?.0;
    raw_width.parse().ok()
}
