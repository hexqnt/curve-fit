//! Кнопка тулбара объединяет оформление, подсказку и доступную подпись.

use eframe::egui;

use crate::app::style;

#[must_use = "Add this widget to a UI with ui.add(widget)"]
pub(in crate::app) struct ToolbarButton<'a> {
    button: egui::Button<'a>,
    hint: &'a str,
}

impl<'a> ToolbarButton<'a> {
    pub(in crate::app) fn new(icon: egui::Image<'a>, hint: &'a str) -> Self {
        Self {
            button: egui::Button::image(icon).min_size(style::TOOLBAR_BUTTON_SIZE),
            hint,
        }
    }

    pub(in crate::app) fn selected(mut self, selected: bool) -> Self {
        self.button = self.button.selected(selected);
        self
    }

    pub(in crate::app) fn frame(mut self, frame: bool) -> Self {
        self.button = self.button.frame(frame);
        self
    }

    pub(in crate::app) fn show_menu<R>(
        self,
        ui: &mut egui::Ui,
        contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::InnerResponse<Option<R>> {
        let (response, inner) =
            egui::containers::menu::MenuButton::from_button(self.button).ui(ui, contents);
        egui::InnerResponse {
            response: super::toolbar_hover_tooltip(response, self.hint),
            inner: inner.map(|inner| inner.inner),
        }
    }
}

impl egui::Widget for ToolbarButton<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        super::toolbar_hover_tooltip(ui.add(self.button), self.hint)
    }
}
