//! Типизированный выбор заимствует варианты и мономорфизирует функцию подписи.

use eframe::egui;

#[must_use = "Add this widget to a UI with ui.add(widget)"]
pub(in crate::app) struct Choice<'a, T, Label> {
    selector: egui::ComboBox,
    selected: &'a mut T,
    values: &'a [T],
    label: Label,
}

impl<'a, T, Label> Choice<'a, T, Label> {
    pub(in crate::app) fn new(
        selector: egui::ComboBox,
        selected: &'a mut T,
        values: &'a [T],
        label: Label,
    ) -> Self {
        Self {
            selector,
            selected,
            values,
            label,
        }
    }
}

impl<'a, T: Copy + PartialEq, Label: Fn(T) -> &'a str> egui::Widget for Choice<'a, T, Label> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let before = *self.selected;
        let mut response = self
            .selector
            .selected_text((self.label)(before))
            .show_ui(ui, |ui| {
                for &value in self.values {
                    ui.selectable_value(self.selected, value, (self.label)(value));
                }
            })
            .response;
        if *self.selected != before {
            response.mark_changed();
        }
        response
    }
}
