//! Единая визуальная тема приложения поверх стандартной палитры `egui`.

use super::*;

#[inline]
fn rgb(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}

#[inline]
fn stroke(color: egui::Color32) -> egui::Stroke {
    egui::Stroke::new(1.0_f32, color)
}

#[derive(Debug, Clone, Copy)]
struct VisualPalette {
    panel_fill: egui::Color32,
    window_fill: egui::Color32,
    faint_bg_color: egui::Color32,
    extreme_bg_color: egui::Color32,
    code_bg_color: egui::Color32,
    window_stroke: egui::Color32,
    selection_bg_fill: egui::Color32,
    selection_stroke: egui::Color32,
    hyperlink_color: egui::Color32,
    inactive_bg_fill: egui::Color32,
    inactive_bg_stroke: egui::Color32,
    hovered_bg_fill: egui::Color32,
    hovered_bg_stroke: egui::Color32,
    active_bg_fill: egui::Color32,
    active_bg_stroke: egui::Color32,
    open_bg_fill: egui::Color32,
    open_bg_stroke: egui::Color32,
}

impl VisualPalette {
    fn dark() -> Self {
        Self {
            panel_fill: rgb(14, 17, 22),
            window_fill: rgb(17, 20, 26),
            faint_bg_color: rgb(24, 30, 38),
            extreme_bg_color: rgb(8, 11, 16),
            code_bg_color: rgb(10, 20, 28),
            window_stroke: rgb(52, 70, 85),
            selection_bg_fill: rgb(22, 88, 120),
            selection_stroke: rgb(152, 226, 255),
            hyperlink_color: rgb(94, 204, 255),
            inactive_bg_fill: rgb(28, 35, 44),
            inactive_bg_stroke: rgb(52, 70, 85),
            hovered_bg_fill: rgb(34, 49, 61),
            hovered_bg_stroke: rgb(70, 113, 138),
            active_bg_fill: rgb(27, 84, 108),
            active_bg_stroke: rgb(86, 171, 211),
            open_bg_fill: rgb(33, 57, 73),
            open_bg_stroke: rgb(72, 122, 150),
        }
    }

    fn light() -> Self {
        Self {
            panel_fill: rgb(239, 245, 249),
            window_fill: rgb(246, 250, 252),
            faint_bg_color: rgb(225, 236, 242),
            extreme_bg_color: rgb(251, 253, 255),
            code_bg_color: rgb(235, 245, 250),
            window_stroke: rgb(165, 188, 201),
            selection_bg_fill: rgb(150, 214, 235),
            selection_stroke: rgb(20, 76, 96),
            hyperlink_color: rgb(0, 118, 163),
            inactive_bg_fill: rgb(220, 234, 241),
            inactive_bg_stroke: rgb(163, 189, 203),
            hovered_bg_fill: rgb(208, 227, 237),
            hovered_bg_stroke: rgb(128, 170, 192),
            active_bg_fill: rgb(183, 220, 236),
            active_bg_stroke: rgb(87, 151, 182),
            open_bg_fill: rgb(198, 224, 236),
            open_bg_stroke: rgb(103, 160, 188),
        }
    }
}

#[inline]
fn apply_visual_palette(visuals: &mut egui::Visuals, palette: VisualPalette) {
    visuals.panel_fill = palette.panel_fill;
    visuals.window_fill = palette.window_fill;
    visuals.faint_bg_color = palette.faint_bg_color;
    visuals.extreme_bg_color = palette.extreme_bg_color;
    visuals.code_bg_color = palette.code_bg_color;
    visuals.window_stroke = stroke(palette.window_stroke);
    visuals.selection.bg_fill = palette.selection_bg_fill;
    visuals.selection.stroke = stroke(palette.selection_stroke);
    visuals.hyperlink_color = palette.hyperlink_color;
    visuals.widgets.inactive.weak_bg_fill = palette.inactive_bg_fill;
    visuals.widgets.inactive.bg_stroke = stroke(palette.inactive_bg_stroke);
    visuals.widgets.hovered.weak_bg_fill = palette.hovered_bg_fill;
    visuals.widgets.hovered.bg_stroke = stroke(palette.hovered_bg_stroke);
    visuals.widgets.active.weak_bg_fill = palette.active_bg_fill;
    visuals.widgets.active.bg_stroke = stroke(palette.active_bg_stroke);
    visuals.widgets.open.weak_bg_fill = palette.open_bg_fill;
    visuals.widgets.open.bg_stroke = stroke(palette.open_bg_stroke);
}

#[inline]
fn apply_widget_corner_radius(visuals: &mut egui::Visuals, radius: egui::CornerRadius) {
    visuals.widgets.noninteractive.corner_radius = radius;
    visuals.widgets.inactive.corner_radius = radius;
    visuals.widgets.hovered.corner_radius = radius;
    visuals.widgets.active.corner_radius = radius;
    visuals.widgets.open.corner_radius = radius;
}

impl CurveFitApp {
    /// Применяет единый визуальный стиль приложения.
    pub(super) fn apply_visual_style(ctx: &egui::Context) {
        ctx.global_style_mut(|style| {
            style.spacing.item_spacing = egui::vec2(10.0, 8.0);
            style.spacing.button_padding = egui::vec2(8.0, 5.0);
            style.spacing.interact_size = egui::vec2(44.0, 26.0);
            style.spacing.slider_width = 170.0;
            style.spacing.combo_width = 180.0;
            style.spacing.indent = 14.0;

            style.text_styles.insert(
                egui::TextStyle::Heading,
                egui::FontId::new(21.0, egui::FontFamily::Proportional),
            );
            style.text_styles.insert(
                egui::TextStyle::Body,
                egui::FontId::new(14.0, egui::FontFamily::Proportional),
            );
            style.text_styles.insert(
                egui::TextStyle::Button,
                egui::FontId::new(14.0, egui::FontFamily::Proportional),
            );
            style.text_styles.insert(
                egui::TextStyle::Monospace,
                egui::FontId::new(13.0, egui::FontFamily::Monospace),
            );
            style.text_styles.insert(
                egui::TextStyle::Small,
                egui::FontId::new(12.0, egui::FontFamily::Proportional),
            );

            let visuals = &mut style.visuals;
            apply_widget_corner_radius(visuals, egui::CornerRadius::same(UI_CORNER_RADIUS));
            let palette = if visuals.dark_mode {
                VisualPalette::dark()
            } else {
                VisualPalette::light()
            };
            apply_visual_palette(visuals, palette);
        });
    }

    /// Каркас боковых панелей с унифицированным отступом и рамкой.
    pub(super) fn side_panel_frame(style: &egui::Style) -> egui::Frame {
        egui::Frame::side_top_panel(style)
            .inner_margin(egui::Margin::symmetric(
                PANEL_INNER_MARGIN_X,
                PANEL_INNER_MARGIN_Y,
            ))
            .fill(style.visuals.panel_fill)
            .stroke(egui::Stroke::new(
                1.0_f32,
                style.visuals.widgets.noninteractive.bg_stroke.color,
            ))
    }

    /// Каркас верхней/нижней панели с компактным вертикальным отступом.
    pub(super) fn top_bottom_panel_frame(style: &egui::Style) -> egui::Frame {
        egui::Frame::side_top_panel(style)
            .inner_margin(egui::Margin::symmetric(PANEL_INNER_MARGIN_X, 6))
            .fill(style.visuals.panel_fill)
            .stroke(egui::Stroke::new(
                1.0_f32,
                style.visuals.widgets.noninteractive.bg_stroke.color,
            ))
    }
}
