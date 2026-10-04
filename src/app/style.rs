//! Единая визуальная тема приложения поверх стандартной палитры `egui`.

use eframe::egui;

use super::CurveFitApp;

const UI_CORNER_RADIUS: u8 = 6;
const PANEL_INNER_MARGIN_X: i8 = 10;
const PANEL_INNER_MARGIN_Y: i8 = 8;
const PANEL_CARD_CORNER_RADIUS: u8 = 7;
const PANEL_CARD_OUTER_MARGIN_Y: i8 = 4;
pub(super) const COLLAPSING_ICON_SCALE: f32 = 1.5;
pub(super) const COLLAPSING_HEADER_TEXT_OFFSET_X: f32 = 4.0;

pub(super) const ITEM_SPACING: egui::Vec2 = egui::vec2(10.0, 8.0);
pub(super) const PARAMETER_GRID_SPACING: egui::Vec2 = egui::vec2(8.0, 6.0);
pub(super) const VALUE_GRID_SPACING: egui::Vec2 = egui::vec2(8.0, 4.0);
pub(super) const PARAMETER_INPUT_WIDTH: f32 = 120.0;
pub(super) const SECTION_GAP: f32 = 2.0;
pub(super) const CONTENT_GAP: f32 = 4.0;
pub(super) const SEPARATOR_GAP: f32 = 6.0;
pub(super) const TOOLBAR_BUTTON_SIZE: egui::Vec2 = egui::vec2(32.0, 28.0);
pub(super) const TOOLBAR_BUTTON_SPACING: f32 = 6.0;
pub(super) const ACTION_BUTTON_CORNER_RADIUS: egui::CornerRadius =
    egui::CornerRadius::same(UI_CORNER_RADIUS + 1);
pub(super) const COMPACT_FIT_BUTTON_SIZE: egui::Vec2 = egui::vec2(118.0, 30.0);
pub(super) const FULL_FIT_BUTTON_HEIGHT: f32 = 34.0;
const TOOLTIP_LINE_SPACING: f32 = 3.0;
pub(super) const INFO_TOOLTIP_WIDTH: f32 = 380.0;
pub(super) const TOOLBAR_TOOLTIP_WIDTH: f32 = 360.0;

#[derive(Clone, Copy)]
pub(super) struct ButtonColors {
    pub fill: egui::Color32,
    pub border: egui::Color32,
    pub text: egui::Color32,
}

pub(super) struct UiColors {
    pub fit: ButtonColors,
    pub stop: ButtonColors,
    pub success: egui::Color32,
    pub input_error_bg: egui::Color32,
    pub fitted_curve: egui::Color32,
    pub loss: egui::Color32,
    pub residual: egui::Color32,
    pub zero_line: egui::Color32,
    toolbar_hover_fill: egui::Color32,
    toolbar_hover_border: egui::Color32,
}

impl UiColors {
    const DARK: Self = Self {
        fit: ButtonColors {
            fill: rgb(20, 94, 128),
            border: rgb(98, 199, 232),
            text: rgb(227, 247, 255),
        },
        stop: ButtonColors {
            fill: rgb(120, 58, 49),
            border: rgb(199, 99, 82),
            text: rgb(255, 238, 232),
        },
        success: rgb(112, 211, 202),
        input_error_bg: rgb(70, 26, 26),
        fitted_curve: rgb(96, 204, 238),
        loss: rgb(245, 126, 95),
        residual: rgb(106, 198, 230),
        zero_line: rgb(131, 147, 160),
        toolbar_hover_fill: rgb(44, 64, 79),
        toolbar_hover_border: rgb(96, 148, 177),
    };

    const LIGHT: Self = Self {
        fit: ButtonColors {
            fill: rgb(182, 224, 241),
            border: rgb(68, 146, 178),
            text: rgb(13, 67, 86),
        },
        stop: ButtonColors {
            fill: rgb(235, 208, 198),
            border: rgb(194, 106, 85),
            text: rgb(94, 37, 23),
        },
        success: rgb(24, 131, 141),
        input_error_bg: rgb(255, 230, 230),
        fitted_curve: rgb(24, 126, 165),
        loss: rgb(181, 93, 67),
        residual: rgb(40, 131, 165),
        zero_line: rgb(139, 151, 160),
        toolbar_hover_fill: rgb(196, 220, 232),
        toolbar_hover_border: rgb(105, 160, 186),
    };

    pub(super) fn for_visuals(visuals: &egui::Visuals) -> &'static Self {
        if visuals.dark_mode {
            &Self::DARK
        } else {
            &Self::LIGHT
        }
    }
}

pub(super) fn apply_toolbar_hover_style(style: &mut egui::Style) {
    let colors = UiColors::for_visuals(&style.visuals);
    let hovered = &mut style.visuals.widgets.hovered;
    hovered.expansion = 1.5;
    hovered.weak_bg_fill = colors.toolbar_hover_fill;
    hovered.bg_stroke = stroke(colors.toolbar_hover_border);
}

pub(super) fn setup_tooltip(ui: &mut egui::Ui, width: f32) {
    ui.set_max_width(width);
    ui.spacing_mut().item_spacing.y = TOOLTIP_LINE_SPACING;
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

impl CurveFitApp {
    pub(super) fn panel_card_frame(ui: &egui::Ui) -> egui::Frame {
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::symmetric(
                PANEL_INNER_MARGIN_X,
                PANEL_INNER_MARGIN_Y,
            ))
            .outer_margin(egui::Margin::symmetric(0, PANEL_CARD_OUTER_MARGIN_Y))
            .corner_radius(egui::CornerRadius::same(PANEL_CARD_CORNER_RADIUS))
            .fill(ui.visuals().faint_bg_color)
            .stroke(stroke(ui.visuals().widgets.noninteractive.bg_stroke.color))
    }

    /// Применяет единый визуальный стиль приложения.
    pub(super) fn apply_visual_style(ctx: &egui::Context) {
        ctx.global_style_mut(|style| {
            style.spacing.item_spacing = ITEM_SPACING;
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
        panel_frame(
            style,
            egui::Margin::symmetric(PANEL_INNER_MARGIN_X, PANEL_INNER_MARGIN_Y),
        )
    }

    /// Каркас верхней/нижней панели с компактным вертикальным отступом.
    pub(super) fn top_bottom_panel_frame(style: &egui::Style) -> egui::Frame {
        panel_frame(style, egui::Margin::symmetric(PANEL_INNER_MARGIN_X, 6))
    }
}

fn panel_frame(style: &egui::Style, margin: egui::Margin) -> egui::Frame {
    egui::Frame::side_top_panel(style)
        .inner_margin(margin)
        .fill(style.visuals.panel_fill)
        .stroke(stroke(style.visuals.widgets.noninteractive.bg_stroke.color))
}

#[inline]
const fn rgb(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}

#[inline]
fn stroke(color: egui::Color32) -> egui::Stroke {
    egui::Stroke::new(1.0_f32, color)
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
