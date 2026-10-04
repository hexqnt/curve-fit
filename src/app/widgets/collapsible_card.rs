//! Сворачиваемая карточка с необязательными элементами в свёрнутом заголовке.

use eframe::egui;

use crate::app::style::{self, COLLAPSING_HEADER_TEXT_OFFSET_X, COLLAPSING_ICON_SCALE};

pub(in crate::app) struct CollapsibleCard<Id> {
    id_salt: Id,
    title: egui::WidgetText,
}

impl<Id: egui::AsIdSalt> CollapsibleCard<Id> {
    pub(in crate::app) fn new(id_salt: Id, title: impl Into<egui::WidgetText>) -> Self {
        Self {
            id_salt,
            title: title.into(),
        }
    }

    pub(in crate::app) fn show(self, ui: &mut egui::Ui, body: impl FnOnce(&mut egui::Ui)) {
        self.show_with_state(&mut (), ui, |_, ui| body(ui), |_, _| {});
    }

    pub(in crate::app) fn show_with_state<State>(
        self,
        state: &mut State,
        ui: &mut egui::Ui,
        add_body: impl FnOnce(&mut State, &mut egui::Ui),
        add_collapsed_trailing: impl FnOnce(&mut State, &mut egui::Ui),
    ) {
        let title = self.title.heading();
        style::panel_card_frame(ui).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.scope(|ui| {
                ui.spacing_mut().indent += COLLAPSING_HEADER_TEXT_OFFSET_X;

                let collapsing_id = ui.make_persistent_id(self.id_salt);
                let mut collapsing_state =
                    egui::containers::collapsing_header::CollapsingState::load_with_default_open(
                        ui.ctx(),
                        collapsing_id,
                        true,
                    );
                ui.horizontal(|ui| {
                    let previous_item_spacing = ui.spacing().item_spacing;
                    ui.spacing_mut().item_spacing.x = 0.0;
                    collapsing_state.show_toggle_button(ui, paint_enlarged_collapsing_icon);
                    ui.spacing_mut().item_spacing = previous_item_spacing;

                    let title_response =
                        ui.add(egui::Label::new(title).sense(egui::Sense::click()));
                    if title_response.clicked() {
                        collapsing_state.toggle(ui);
                    }

                    if !collapsing_state.is_open() {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            add_collapsed_trailing(state, ui);
                        });
                    }
                });

                let _ = collapsing_state.show_body_unindented(ui, |ui| {
                    add_body(state, ui);
                });
            });
        });
    }
}

fn paint_enlarged_collapsing_icon(ui: &mut egui::Ui, openness: f32, response: &egui::Response) {
    let enlarged_rect = egui::Rect::from_center_size(
        response.rect.center(),
        response.rect.size() * COLLAPSING_ICON_SCALE,
    )
    .translate(egui::vec2(-0.5 * COLLAPSING_HEADER_TEXT_OFFSET_X, 0.0));
    let enlarged_response = response.clone().with_new_rect(enlarged_rect);
    egui::containers::collapsing_header::paint_default_icon(ui, openness, &enlarged_response);
}
