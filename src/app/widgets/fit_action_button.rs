//! Кнопка запуска и остановки подгонки с оформлением по текущей теме.

use eframe::egui;

use crate::app::{
    i18n::{fit_icon_image, stop_icon_image, tr},
    style,
    types::UiLanguage,
};

#[derive(Clone, Copy)]
pub(in crate::app) enum FitAction {
    Fit,
    Stop,
}

#[must_use = "Add this widget to a UI with ui.add(widget)"]
pub(in crate::app) struct FitActionButton {
    language: UiLanguage,
    action: FitAction,
    min_size: egui::Vec2,
}

impl FitActionButton {
    pub(in crate::app) fn new(language: UiLanguage, action: FitAction) -> Self {
        Self {
            language,
            action,
            min_size: style::COMPACT_FIT_BUTTON_SIZE,
        }
    }

    pub(in crate::app) fn min_size(mut self, size: egui::Vec2) -> Self {
        self.min_size = size;
        self
    }
}

impl egui::Widget for FitActionButton {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let colors = style::UiColors::for_visuals(ui.visuals());
        let (colors, icon, text) = match self.action {
            FitAction::Stop => (
                colors.stop,
                stop_icon_image(colors.stop.text),
                tr(self.language, "Stop", "Стоп"),
            ),
            FitAction::Fit => (
                colors.fit,
                fit_icon_image(colors.fit.text),
                tr(self.language, "Fit", "Фитинг"),
            ),
        };
        ui.add(
            egui::Button::image_and_text(
                icon,
                egui::RichText::new(text).strong().color(colors.text),
            )
            .min_size(self.min_size)
            .fill(colors.fill)
            .stroke(egui::Stroke::new(1.0, colors.border))
            .corner_radius(style::ACTION_BUTTON_CORNER_RADIUS),
        )
    }
}
