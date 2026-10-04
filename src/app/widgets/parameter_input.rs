//! Подпись и числовой черновик с локальной ошибкой после завершения ввода.

use eframe::egui;

use crate::app::{NumberDraft, UiLanguage, i18n::tr, number_draft::NumberEdit, style};

#[must_use = "Add this widget to a UI with ui.add(widget)"]
pub(in crate::app) struct ParameterInput<'a> {
    label: egui::WidgetText,
    draft: &'a mut NumberDraft,
    language: UiLanguage,
}

impl<'a> ParameterInput<'a> {
    pub(in crate::app) fn new(
        label: impl Into<egui::WidgetText>,
        draft: &'a mut NumberDraft,
        language: UiLanguage,
    ) -> Self {
        Self {
            label: label.into(),
            draft,
            language,
        }
    }
}

impl egui::Widget for ParameterInput<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let id = ui.make_persistent_id(("parameter_input", self.label.text()));
        let _label = ui.label(self.label);
        let response = self.draft.edit(|text| {
            let response = ui.add(
                egui::TextEdit::singleline(text)
                    .id(id)
                    .desired_width(style::PARAMETER_INPUT_WIDTH),
            );
            NumberEdit {
                changed: response.changed(),
                finished: response.lost_focus(),
                output: response,
            }
        });
        #[cfg(feature = "testing")]
        let response = response.labelled_by(_label.id);
        if self.draft.has_visible_error() {
            ui.end_row();
            ui.label("");
            ui.colored_label(
                ui.visuals().error_fg_color,
                tr(self.language, "Enter a number", "Введите число"),
            );
        }
        response
    }
}
