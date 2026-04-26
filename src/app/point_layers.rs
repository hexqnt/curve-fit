//! Модель слоёв точек и операции выбора/создания/удаления.

use super::*;

const DEFAULT_LAYER_NAME_PREFIX: &str = "Layer";
const LAYER_COLOR_PALETTE: [egui::Color32; 8] = [
    egui::Color32::from_rgb(232, 132, 116),
    egui::Color32::from_rgb(103, 176, 226),
    egui::Color32::from_rgb(119, 195, 155),
    egui::Color32::from_rgb(234, 196, 111),
    egui::Color32::from_rgb(172, 148, 235),
    egui::Color32::from_rgb(102, 202, 198),
    egui::Color32::from_rgb(222, 128, 168),
    egui::Color32::from_rgb(158, 202, 112),
];

/// Стабильный идентификатор слоя точек.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PointLayerId(u64);

impl PointLayerId {
    fn first() -> Self {
        Self(1)
    }
}

/// Один слой исходных точек.
#[derive(Debug, Clone)]
pub(super) struct PointLayer {
    pub(super) id: PointLayerId,
    pub(super) name: String,
    pub(super) visible: bool,
    pub(super) color: egui::Color32,
    pub(super) points: PointsEditorState,
}

impl PointLayer {
    fn new(id: PointLayerId, name: String, color: egui::Color32) -> Self {
        Self {
            id,
            name,
            visible: true,
            color,
            points: PointsEditorState::default(),
        }
    }

    fn reset_to_default(&mut self) {
        self.name = format!("{DEFAULT_LAYER_NAME_PREFIX} 1");
        self.visible = true;
        self.color = LAYER_COLOR_PALETTE[0];
        self.points = PointsEditorState::default();
    }

    pub(super) fn display_name(&self) -> &str {
        let name = self.name.trim();
        if name.is_empty() {
            "Unnamed layer"
        } else {
            name
        }
    }
}

/// Список слоёв и текущий выбор.
#[derive(Debug, Clone)]
pub(super) struct PointLayersState {
    pub(super) layers: Vec<PointLayer>,
    pub(super) selected_id: PointLayerId,
    next_id: u64,
    next_name_index: usize,
    next_color_index: usize,
}

impl Default for PointLayersState {
    fn default() -> Self {
        let selected_id = PointLayerId::first();
        Self {
            layers: vec![PointLayer::new(
                selected_id,
                format!("{DEFAULT_LAYER_NAME_PREFIX} 1"),
                LAYER_COLOR_PALETTE[0],
            )],
            selected_id,
            next_id: 2,
            next_name_index: 2,
            next_color_index: 1,
        }
    }
}

impl PointLayersState {
    pub(super) fn selected_index(&self) -> usize {
        self.layers
            .iter()
            .position(|layer| layer.id == self.selected_id)
            .unwrap_or(0)
    }

    pub(super) fn selected(&self) -> &PointLayer {
        &self.layers[self.selected_index()]
    }

    pub(super) fn selected_mut(&mut self) -> &mut PointLayer {
        let selected_index = self.selected_index();
        &mut self.layers[selected_index]
    }

    pub(super) fn select(&mut self, id: PointLayerId) {
        if self.layers.iter().any(|layer| layer.id == id) {
            self.selected_id = id;
        }
    }

    pub(super) fn show_only(&mut self, id: PointLayerId) -> bool {
        if !self.layers.iter().any(|layer| layer.id == id) {
            return false;
        }

        let mut changed = false;
        for layer in &mut self.layers {
            let visible = layer.id == id;
            changed |= layer.visible != visible;
            layer.visible = visible;
        }
        changed
    }

    pub(super) fn create_empty_layer(&mut self) -> PointLayerId {
        let id = PointLayerId(self.next_id);
        self.next_id += 1;
        let name = format!("{DEFAULT_LAYER_NAME_PREFIX} {}", self.next_name_index);
        self.next_name_index += 1;
        let color = LAYER_COLOR_PALETTE[self.next_color_index % LAYER_COLOR_PALETTE.len()];
        self.next_color_index += 1;
        self.layers.push(PointLayer::new(id, name, color));
        self.selected_id = id;
        id
    }

    pub(super) fn create_layer_from_points(&mut self, points: &[Point]) -> PointLayerId {
        let id = self.create_empty_layer();
        let layer = self.selected_mut();
        layer.points.text = points_to_text(points);
        set_points_editor_cache_from_valid_points(&mut layer.points, points);
        id
    }

    pub(super) fn duplicate_selected_layer(&mut self) -> PointLayerId {
        let selected_index = self.selected_index();
        let id = PointLayerId(self.next_id);
        self.next_id += 1;

        let mut layer = self.layers[selected_index].clone();
        layer.id = id;
        layer.name = format!("{} copy", layer.name);

        let insert_index = selected_index + 1;
        self.layers.insert(insert_index, layer);
        self.selected_id = id;
        id
    }

    pub(super) fn delete_selected_layer(&mut self) {
        if self.layers.len() <= 1 {
            self.layers[0].reset_to_default();
            self.selected_id = self.layers[0].id;
            self.next_name_index = self.next_name_index.max(2);
            return;
        }

        let selected_index = self.selected_index();
        self.layers.remove(selected_index);
        let next_index = selected_index.min(self.layers.len().saturating_sub(1));
        self.selected_id = self.layers[next_index].id;
    }
}
