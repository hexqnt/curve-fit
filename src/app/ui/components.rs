//! Небольшие общие виджеты и оформление повторяющихся элементов интерфейса.

use eframe::egui;

use crate::app::style;

mod fit_action_button;
mod toggle_switch;

pub(super) use fit_action_button::{FitAction, FitActionButton};
pub(super) use toggle_switch::ToggleSwitch;

pub(super) fn parameter_grid(id: impl egui::AsIdSalt) -> egui::Grid {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing(style::PARAMETER_GRID_SPACING)
}

pub(super) fn value_grid(id: impl egui::AsIdSalt) -> egui::Grid {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing(style::VALUE_GRID_SPACING)
}

pub(super) fn value_row(ui: &mut egui::Ui, label: &str, value: impl Into<egui::RichText>) {
    ui.label(label);
    ui.monospace(value);
    ui.end_row();
}

pub(super) fn parameter_input(
    ui: &mut egui::Ui,
    label: impl Into<egui::WidgetText>,
    value: &mut dyn egui::TextBuffer,
    enabled: bool,
) -> egui::Response {
    let _label = ui.label(label);
    let response = ui.add_enabled(
        enabled,
        egui::TextEdit::singleline(value).desired_width(style::PARAMETER_INPUT_WIDTH),
    );
    #[cfg(feature = "testing")]
    let response = response.labelled_by(_label.id);
    response
}

pub(super) fn toolbar_icon_button(icon: egui::Image<'static>) -> egui::Button<'static> {
    egui::Button::image(icon).min_size(style::TOOLBAR_BUTTON_SIZE)
}

pub(super) fn with_toolbar_hover_style(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    ui.scope(|ui| {
        style::apply_toolbar_hover_style(ui.style_mut());
        add_contents(ui);
    });
}

pub(super) fn info_hover(response: egui::Response, text: impl AsRef<str>) -> egui::Response {
    help_tooltip(response, text, TooltipKind::Info)
}

pub(super) fn toolbar_hover_tooltip(
    response: egui::Response,
    text: &'static str,
) -> egui::Response {
    #[cfg(feature = "testing")]
    response.ctx.accesskit_node_builder(response.id, |node| {
        node.set_label(text.lines().next().unwrap_or(text));
    });
    help_tooltip(response, text, TooltipKind::Toolbar)
}

#[derive(Clone, Copy)]
enum TooltipKind {
    Info,
    Toolbar,
}

fn help_tooltip(
    response: egui::Response,
    text: impl AsRef<str>,
    kind: TooltipKind,
) -> egui::Response {
    response.on_hover_ui(|ui| {
        let width = match kind {
            TooltipKind::Info => style::INFO_TOOLTIP_WIDTH,
            TooltipKind::Toolbar => style::TOOLBAR_TOOLTIP_WIDTH,
        };
        style::setup_tooltip(ui, width);

        let mut lines = text
            .as_ref()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .peekable();
        if let Some(title) = lines.next() {
            if matches!(kind, TooltipKind::Toolbar) || lines.peek().is_some() {
                ui.strong(title);
            } else {
                ui.label(title);
            }
        }
        for line in lines {
            ui.small(line);
        }
    })
}
