//! Наблюдение состояния и воспроизводимая подготовка UI-тестов; доступны только с feature `testing`.

use super::{CurveFitApp, PointLayer, StatusMessage, UiLanguage};
use crate::{FitResult, OptimizerConfig, OptimizerMethod};

pub use super::point_layers::PointLayerId;

#[derive(Debug, Clone, Copy)]
pub struct LayerState<'a> {
    pub id: PointLayerId,
    pub name: &'a str,
    pub visible: bool,
    pub points_text: &'a str,
    pub undo: &'a [String],
    pub redo: &'a [String],
}

impl<'a> From<&'a PointLayer> for LayerState<'a> {
    fn from(layer: &'a PointLayer) -> Self {
        Self {
            id: layer.id,
            name: &layer.name,
            visible: layer.visible,
            points_text: &layer.points.text,
            undo: &layer.points.undo_stack,
            redo: &layer.points.redo_stack,
        }
    }
}

#[derive(Debug)]
pub struct AppState<'a> {
    layers: &'a [PointLayer],
    pub selected_layer_index: usize,
    pub selected_layer: LayerState<'a>,
    pub parameter_inputs: &'a [String],
    pub optimizer_method: OptimizerMethod,
    pub error: Option<&'a str>,
    pub fit_in_progress: bool,
    pub fit_result: Option<&'a FitResult>,
    pub replay_iteration: Option<u64>,
    pub replay_frames: usize,
    pub diagnostic_iterations: usize,
    pub auto_play_on_fit: bool,
    pub show_formula_window: bool,
    pub show_left_panel: bool,
    pub show_right_panel: bool,
    pub show_diagnostics: bool,
}

impl AppState<'_> {
    pub fn layers(&self) -> impl ExactSizeIterator<Item = LayerState<'_>> {
        self.layers.iter().map(LayerState::from)
    }
}

impl CurveFitApp {
    /// Читает состояние без парсинга настроек и копирования данных.
    #[must_use]
    pub fn inspect(&self) -> AppState<'_> {
        AppState {
            layers: &self.point_layers.layers,
            selected_layer_index: self.point_layers.selected_index(),
            selected_layer: LayerState::from(self.selected_layer()),
            parameter_inputs: &self.parameter_inputs,
            optimizer_method: self.optimizer_method,
            error: match &self.status {
                Some(StatusMessage::Error(error)) => Some(error),
                _ => None,
            },
            fit_in_progress: self.fit_in_progress,
            fit_result: self.fit_result.as_ref(),
            replay_iteration: self.replay_selected_iteration(),
            replay_frames: self.replay.frames.len(),
            diagnostic_iterations: self.iteration_diagnostics.loss_points.len(),
            auto_play_on_fit: self.replay.autoplay_on_fit,
            show_formula_window: self.panel.show_formula_window,
            show_left_panel: self.panel.show_left,
            show_right_panel: self.panel.show_right,
            show_diagnostics: self.panel.show_diagnostics,
        }
    }
}

/// Создаёт приложение тем же путём, что и `CurveFitApp::new`, с фиксированным языком.
#[must_use]
pub fn from_context(ctx: &eframe::egui::Context) -> CurveFitApp {
    CurveFitApp::with_language(ctx, UiLanguage::English)
}

/// Преобразует настройки только для сценариев, которым нужна конфигурация оптимизатора.
pub fn optimizer_config(app: &CurveFitApp) -> Result<OptimizerConfig, String> {
    app.optimizer_config()
}
