//! Моноширинный редактор с подсветкой строки ошибки и устойчивым идентификатором.

use eframe::egui;

use crate::app::{i18n::tr, style, types::UiLanguage};

#[must_use = "Add this widget to a UI with ui.add(widget)"]
pub(in crate::app) struct PointsTextEdit<'a, Buffer> {
    text: &'a mut Buffer,
    id: egui::Id,
    error_line: Option<usize>,
    rows: usize,
    language: UiLanguage,
}

impl<'a, Buffer: egui::TextBuffer> PointsTextEdit<'a, Buffer> {
    pub(in crate::app) fn new(text: &'a mut Buffer, id: egui::Id, language: UiLanguage) -> Self {
        Self {
            text,
            id,
            error_line: None,
            rows: 6,
            language,
        }
    }

    pub(in crate::app) fn error_line(mut self, line: Option<usize>) -> Self {
        self.error_line = line;
        self
    }

    pub(in crate::app) fn desired_rows(mut self, rows: usize) -> Self {
        self.rows = rows;
        self
    }
}

impl<Buffer: egui::TextBuffer> egui::Widget for PointsTextEdit<'_, Buffer> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let mut layouter = |ui: &egui::Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
            let mut job = egui::text::LayoutJob::default();
            job.wrap.max_width = wrap_width;
            let format = egui::TextFormat {
                font_id: egui::TextStyle::Monospace.resolve(ui.style()),
                color: ui.visuals().text_color(),
                ..Default::default()
            };
            let error_bg = style::UiColors::for_visuals(ui.visuals()).input_error_bg;
            for (index, line) in text.as_str().split_inclusive('\n').enumerate() {
                let mut line_format = format.clone();
                if self.error_line == Some(index + 1) {
                    line_format.background = error_bg;
                }
                job.append(line, 0.0, line_format);
            }
            if text.as_str().is_empty() {
                job.append("", 0.0, format);
            }
            ui.fonts_mut(|fonts| fonts.layout_job(job))
        };
        let response = ui.add(
            egui::TextEdit::multiline(self.text)
                .id(self.id)
                .desired_width(ui.available_width())
                .desired_rows(self.rows)
                .font(egui::TextStyle::Monospace)
                .hint_text(tr(
                    self.language,
                    "Example:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
                    "Пример:\n0.0 1.5\n0.5\t2.0\n1.0;2.8",
                ))
                .layouter(&mut layouter),
        );
        #[cfg(feature = "testing")]
        ui.ctx().accesskit_node_builder(response.id, |node| {
            node.set_label(tr(self.language, "Input points", "Ввод точек"));
        });
        response
    }
}
