//! Переключатель с заимствованным состоянием и необязательной подписью.

use eframe::egui;

/// Заимствует состояние переключателя на время построения кадра.
#[must_use = "Add this widget to a UI with ui.add(widget)"]
pub(in crate::app::ui) struct ToggleSwitch<'a> {
    on: &'a mut bool,
    label: Option<egui::WidgetText>,
}

impl<'a> ToggleSwitch<'a> {
    pub(in crate::app::ui) fn new(on: &'a mut bool) -> Self {
        Self { on, label: None }
    }

    pub(in crate::app::ui) fn label(mut self, label: impl Into<egui::WidgetText>) -> Self {
        self.label = Some(label.into());
        self
    }

    fn show_switch(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
        let desired_size = ui.spacing().interact_size.y * egui::vec2(1.50, 0.8);
        let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        if response.clicked() {
            *on = !*on;
            response.mark_changed();
        }

        response.widget_info(|| {
            egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, "")
        });

        if ui.is_rect_visible(rect) {
            let how_on = ui.ctx().animate_bool_responsive(response.id, *on);
            let visuals = ui.style().interact_selectable(&response, *on);
            let rect = rect.expand(visuals.expansion);
            let radius = 0.5 * rect.height();

            ui.painter().rect(
                rect,
                radius,
                visuals.bg_fill,
                visuals.bg_stroke,
                egui::StrokeKind::Inside,
            );

            let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
            let center = egui::pos2(circle_x, rect.center().y);
            ui.painter()
                .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
        }

        response
    }
}

impl egui::Widget for ToggleSwitch<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        if let Some(label) = self.label {
            ui.horizontal(|ui| {
                let switch_response = Self::show_switch(ui, self.on);
                #[cfg(feature = "testing")]
                switch_response
                    .ctx
                    .accesskit_node_builder(switch_response.id, |node| {
                        node.set_label(label.text());
                    });
                let _label_response = ui.label(label);
                #[cfg(feature = "testing")]
                let switch_response = switch_response.labelled_by(_label_response.id);
                switch_response
            })
            .inner
        } else {
            Self::show_switch(ui, self.on)
        }
    }
}
