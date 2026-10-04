//! Строка слоя заимствует его оформление и возвращает команды для списка слоёв.

use eframe::egui;

use crate::app::{
    PointLayer, UiLanguage,
    i18n::{layer_hidden_icon_image, layer_visible_icon_image, tr},
};

#[derive(Debug, Clone, Copy)]
pub(in crate::app) enum LayerAction {
    New,
    Duplicate,
    Paste,
    ToggleVisibility,
    ShowOnly,
    Clear,
    Delete,
}

pub(in crate::app) struct LayerRowResponse {
    pub(in crate::app) select: bool,
    pub(in crate::app) action: Option<LayerAction>,
}

pub(in crate::app) struct PointLayerRow<'a> {
    layer: &'a mut PointLayer,
    language: UiLanguage,
    selected: bool,
    editable: bool,
    clipboard_available: bool,
    point_count: usize,
    has_error: bool,
}

impl<'a> PointLayerRow<'a> {
    pub(in crate::app) const HEIGHT: f32 = 32.0;

    pub(in crate::app) fn new(layer: &'a mut PointLayer, language: UiLanguage) -> Self {
        Self {
            layer,
            language,
            selected: false,
            editable: true,
            clipboard_available: true,
            point_count: 0,
            has_error: false,
        }
    }

    pub(in crate::app) fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
    pub(in crate::app) fn editable(mut self, editable: bool) -> Self {
        self.editable = editable;
        self
    }
    pub(in crate::app) fn clipboard_available(mut self, available: bool) -> Self {
        self.clipboard_available = available;
        self
    }
    pub(in crate::app) fn point_count(mut self, count: usize, has_error: bool) -> Self {
        self.point_count = count;
        self.has_error = has_error;
        self
    }

    pub(in crate::app) fn show(self, mut row: egui_extras::TableRow<'_, '_>) -> LayerRowResponse {
        let mut result = LayerRowResponse {
            select: false,
            action: None,
        };
        // Четыре ячейки имеют фиксированное число ответов; покадровый Vec не нужен.
        let mut children = [None, None, None, None];
        row.set_selected(self.selected);
        row.col(|ui| {
            let icon = if self.layer.visible {
                layer_visible_icon_image(ui.visuals().text_color())
            } else {
                layer_hidden_icon_image(ui.visuals().weak_text_color())
            };
            let hint = tr(
                self.language,
                "Layer visibility\n- Show or hide this layer in the plot and fitting input",
                "Видимость слоя\n- Показать или скрыть слой на графике и во входе фитинга",
            );
            let response = ui.add_enabled(
                self.editable,
                super::ToolbarButton::new(icon, hint).frame(false),
            );
            if response.double_clicked() {
                result.action = Some(LayerAction::ShowOnly);
            } else if response.clicked() {
                result.action = Some(LayerAction::ToggleVisibility);
            }
            children[0] = Some((response, None));
        });
        row.col(|ui| {
            let response = ui.color_edit_button_srgba(&mut self.layer.color);
            let popup_id = response.id.with("layer_context_menu");
            children[1] = Some((response, Some(popup_id)));
        });
        row.col(|ui| {
            let size = [ui.available_width().max(40.0), Self::HEIGHT - 8.0];
            let response = if self.selected {
                let response = ui.add_sized(
                    size,
                    egui::TextEdit::singleline(&mut self.layer.name)
                        .id(egui::Id::new(("point_layer_name", self.layer.id))),
                );
                #[cfg(feature = "testing")]
                ui.ctx().accesskit_node_builder(response.id, |node| {
                    node.set_label(tr(self.language, "Layer name", "Название слоя"));
                });
                response
            } else {
                ui.add_sized(
                    size,
                    egui::Label::new(self.layer.name.as_str()).sense(egui::Sense::click()),
                )
            };
            result.select |= response.clicked();
            if !self.selected {
                children[2] = Some((response, None));
            }
        });
        row.col(|ui| {
            let text = if self.has_error {
                format!("{} !", self.point_count)
            } else {
                self.point_count.to_string()
            };
            let response = ui.add_sized(
                [ui.available_width().max(1.0), Self::HEIGHT - 8.0],
                egui::Label::new(egui::RichText::new(text).small()),
            );
            children[3] = Some((response, None));
        });
        let response = row.response();
        result.select |= response.clicked();
        let pointer_over_child = children
            .iter()
            .flatten()
            .any(|(response, _)| response.contains_pointer());
        let row_context = (!pointer_over_child).then_some((response, None));
        for (response, popup_id) in row_context
            .into_iter()
            .chain(children.into_iter().flatten())
        {
            let show_contents = |ui: &mut egui::Ui| {
                layer_menu(
                    ui,
                    self.language,
                    self.layer.visible,
                    self.editable,
                    self.clipboard_available,
                )
            };
            let mut popup = egui::Popup::context_menu(&response);
            if let Some(popup_id) = popup_id {
                popup = popup.id(popup_id);
            }
            let menu = popup.show(show_contents);
            if let Some(menu) = menu {
                result.select = true;
                result.action = menu.inner.or(result.action);
            }
        }
        result
    }
}

fn layer_menu(
    ui: &mut egui::Ui,
    language: UiLanguage,
    visible: bool,
    editable: bool,
    clipboard_available: bool,
) -> Option<LayerAction> {
    let mut selected = None;
    for (action, enabled, label) in [
        (
            LayerAction::New,
            editable,
            tr(language, "New empty layer", "Новый пустой слой"),
        ),
        (
            LayerAction::Duplicate,
            editable,
            tr(language, "Duplicate layer", "Дублировать слой"),
        ),
        (
            LayerAction::Paste,
            editable && clipboard_available,
            tr(language, "New layer from clipboard", "Новый слой из буфера"),
        ),
        (
            LayerAction::ToggleVisibility,
            editable,
            if visible {
                tr(language, "Hide layer", "Скрыть слой")
            } else {
                tr(language, "Show layer", "Показать слой")
            },
        ),
        (
            LayerAction::Clear,
            editable,
            tr(language, "Clear layer", "Очистить слой"),
        ),
        (
            LayerAction::Delete,
            editable,
            tr(language, "Delete layer", "Удалить слой"),
        ),
    ] {
        if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
            selected = Some(action);
            ui.close();
        }
    }
    selected
}
