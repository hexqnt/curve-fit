//! Состояние текстового редактора точек, кэш парсинга и история undo/redo.

use super::*;

/// Кэш результата последнего парсинга текста точек и подготовленных данных для графика.
#[derive(Debug, Clone)]
pub(super) struct ParsedPointsCache {
    pub(super) parsed_points: Result<Vec<Point>, String>,
    pub(super) parse_error_line: Option<usize>,
    pub(super) plot_points: Arc<[PlotPoint]>,
}

#[derive(Debug, Clone)]
pub(super) struct VisiblePointLayerPlotData {
    /// Имя слоя для легенды графика.
    pub(super) name: String,
    pub(super) color: egui::Color32,
    pub(super) plot_points: Arc<[PlotPoint]>,
}

/// Локальное состояние текстового редактора точек, включая debounce и историю изменений.
#[derive(Debug, Clone)]
pub(super) struct PointsEditorState {
    pub(super) text: String,
    pub(super) cache: Option<ParsedPointsCache>,
    pub(super) cache_dirty: bool,
    pub(super) text_sync_pending: bool,
    pub(super) parse_debounce_deadline: Option<Instant>,
    pub(super) undo_stack: Vec<String>,
    pub(super) redo_stack: Vec<String>,
}

impl Default for PointsEditorState {
    fn default() -> Self {
        Self {
            text: String::new(),
            cache: None,
            cache_dirty: true,
            text_sync_pending: false,
            parse_debounce_deadline: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }
}

pub(super) fn invalidate_points_editor_cache(points: &mut PointsEditorState) {
    points.cache_dirty = true;
    points.text_sync_pending = false;
    // Небольшой debounce уменьшает число парсингов во время быстрого ввода текста.
    points.parse_debounce_deadline =
        Some(Instant::now() + Duration::from_millis(POINTS_PARSE_DEBOUNCE_MS));
}

pub(super) fn points_editor_cache_with_policy(
    points: &mut PointsEditorState,
    force: bool,
) -> &ParsedPointsCache {
    // Политика пересчета кэша:
    // - сразу, если кэша нет;
    // - по force;
    // - или после окончания debounce.
    let should_parse = if points.cache.is_none() {
        true
    } else if !points.cache_dirty {
        false
    } else if force {
        true
    } else {
        points
            .parse_debounce_deadline
            .map(|deadline| Instant::now() >= deadline)
            .unwrap_or(true)
    };

    if should_parse || points.cache.is_none() {
        // Текст парсим ровно один раз на пакет правок, а дальше работаем из кэша.
        points.cache = Some(parse_points_text_cache(&points.text));
        points.cache_dirty = false;
        points.text_sync_pending = false;
        points.parse_debounce_deadline = None;
    }
    points.cache.get_or_insert_with(|| ParsedPointsCache {
        parsed_points: Err("Internal error: points cache is unavailable".to_string()),
        parse_error_line: None,
        plot_points: Vec::<PlotPoint>::new().into(),
    })
}

pub(super) fn flush_points_editor_text_from_cache_if_pending(points: &mut PointsEditorState) {
    if !points.text_sync_pending {
        return;
    }

    // Лениво сериализуем точки обратно в текст только тогда, когда это реально нужно UI.
    let synced_text = points.cache.as_ref().and_then(|cache| {
        cache
            .parsed_points
            .as_ref()
            .ok()
            .map(|points| points_to_text(points))
    });
    if let Some(synced_text) = synced_text {
        points.text = synced_text;
    }
    points.text_sync_pending = false;
}

pub(super) fn set_points_editor_cache_from_valid_points(
    points: &mut PointsEditorState,
    valid_points: &[Point],
) {
    let parsed_points = valid_points.to_vec();
    let plot_points: Arc<[PlotPoint]> = parsed_points
        .iter()
        .map(|point| PlotPoint::new(point.x(), point.y()))
        .collect::<Vec<_>>()
        .into();
    points.cache = Some(ParsedPointsCache {
        parsed_points: Ok(parsed_points),
        parse_error_line: None,
        plot_points,
    });
    points.cache_dirty = false;
    points.text_sync_pending = false;
    points.parse_debounce_deadline = None;
}

impl CurveFitApp {
    pub(super) fn selected_layer(&self) -> &PointLayer {
        self.point_layers.selected()
    }

    pub(super) fn selected_layer_mut(&mut self) -> &mut PointLayer {
        self.point_layers.selected_mut()
    }

    pub(super) fn selected_points_editor(&self) -> &PointsEditorState {
        &self.selected_layer().points
    }

    pub(super) fn selected_points_editor_mut(&mut self) -> &mut PointsEditorState {
        &mut self.selected_layer_mut().points
    }

    pub(super) fn invalidate_points_cache(&mut self) {
        invalidate_points_editor_cache(self.selected_points_editor_mut());
    }

    pub(super) fn points_cache_with_policy(&mut self, force: bool) -> &ParsedPointsCache {
        points_editor_cache_with_policy(self.selected_points_editor_mut(), force)
    }

    pub(super) fn points_cache(&mut self) -> &ParsedPointsCache {
        self.points_cache_with_policy(false)
    }

    pub(super) fn maybe_refresh_points_cache_after_debounce(&mut self) {
        let now = Instant::now();
        let mut refreshed_any = false;
        for layer in &mut self.point_layers.layers {
            if layer.points.cache_dirty
                && layer
                    .points
                    .parse_debounce_deadline
                    .map(|deadline| now >= deadline)
                    .unwrap_or(true)
            {
                points_editor_cache_with_policy(&mut layer.points, true);
                refreshed_any = true;
            }
        }
        if refreshed_any {
            self.refresh_status_after_points_edit();
        }
    }

    pub(super) fn idle_status_after_points_edit(&self) -> StatusMessage {
        if self.fit_result.is_some() || self.spline_result.is_some() {
            StatusMessage::FitCompleted
        } else {
            StatusMessage::Ready
        }
    }

    pub(super) fn refresh_status_after_points_edit(&mut self) {
        if let Some(error) = self.first_visible_points_parse_error() {
            self.status = Some(StatusMessage::Error(format!(
                "{POINTS_PARSE_ERROR_PREFIX}{error}"
            )));
            return;
        }

        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
    }

    fn finish_valid_points_change(&mut self) {
        self.clear_fit_outputs();
        self.refresh_status_after_points_edit();
        if !matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
    }

    fn first_visible_points_parse_error(&mut self) -> Option<String> {
        for layer in &mut self.point_layers.layers {
            if !layer.visible {
                continue;
            }

            let display_name = layer.display_name().to_owned();
            let cache = points_editor_cache_with_policy(&mut layer.points, true);
            if let Err(error) = &cache.parsed_points {
                return Some(format!("Layer '{display_name}': {error}"));
            }
        }

        None
    }

    pub(super) fn push_points_undo_snapshot(&mut self, snapshot: String) {
        let points = self.selected_points_editor_mut();
        if points
            .undo_stack
            .last()
            .is_some_and(|last| *last == snapshot)
        {
            return;
        }
        points.undo_stack.push(snapshot);
        // Ограничиваем историю фиксированным размером, чтобы не раздувать память.
        if points.undo_stack.len() > POINTS_HISTORY_LIMIT {
            let overflow = points.undo_stack.len() - POINTS_HISTORY_LIMIT;
            points.undo_stack.drain(0..overflow);
        }
    }

    pub(super) fn flush_points_text_from_cache_if_pending(&mut self) {
        flush_points_editor_text_from_cache_if_pending(self.selected_points_editor_mut());
    }

    pub(super) fn push_current_points_undo_snapshot(&mut self) {
        self.flush_points_text_from_cache_if_pending();
        self.push_points_undo_snapshot(self.selected_points_editor().text.clone());
    }

    pub(super) fn edit_valid_points_in_cache<F>(
        &mut self,
        record_undo: bool,
        sync_text_immediately: bool,
        edit: F,
    ) -> Result<(), String>
    where
        F: FnOnce(&mut Vec<Point>) -> bool,
    {
        let parse_error = match &self.points_cache_with_policy(true).parsed_points {
            Ok(_) => None,
            Err(error) => Some(error.clone()),
        };
        if let Some(error) = parse_error {
            return Err(error);
        }

        if record_undo {
            self.push_current_points_undo_snapshot();
        }
        self.selected_points_editor_mut().redo_stack.clear();

        let maybe_synced_text = {
            let points_state = self.selected_points_editor_mut();
            let Some(cache) = points_state.cache.as_mut() else {
                return Err("Internal error: points cache is unavailable".to_string());
            };
            // Мутируем уже проверенные точки прямо в кэше, чтобы не гонять повторный parse/format.
            let points = match cache.parsed_points.as_mut() {
                Ok(points) => points,
                Err(error) => return Err(error.clone()),
            };
            if !edit(points) {
                return Ok(());
            }
            cache.parse_error_line = None;
            cache.plot_points = points
                .iter()
                .map(|point| PlotPoint::new(point.x(), point.y()))
                .collect::<Vec<_>>()
                .into();
            sync_text_immediately.then(|| points_to_text(points))
        };

        if let Some(synced_text) = maybe_synced_text {
            let points_state = self.selected_points_editor_mut();
            points_state.text = synced_text;
            points_state.text_sync_pending = false;
        } else {
            self.selected_points_editor_mut().text_sync_pending = true;
        }

        self.finish_valid_points_change();

        Ok(())
    }

    pub(super) fn apply_points_text_change(&mut self, new_text: String, keep_redo: bool) {
        if self.selected_points_editor().text == new_text {
            return;
        }
        self.selected_points_editor_mut().text = new_text;
        self.invalidate_points_cache();
        self.clear_fit_outputs();
        if !keep_redo {
            self.selected_points_editor_mut().redo_stack.clear();
        }
    }

    pub(super) fn undo_points_edit(&mut self) {
        if self.fit_in_progress {
            return;
        }
        let Some(previous) = self.selected_points_editor_mut().undo_stack.pop() else {
            return;
        };
        let current_text = self.selected_points_editor().text.clone();
        self.selected_points_editor_mut()
            .redo_stack
            .push(current_text);
        self.apply_points_text_change(previous, true);
        self.refresh_status_after_points_edit();
    }

    pub(super) fn redo_points_edit(&mut self) {
        if self.fit_in_progress {
            return;
        }
        let Some(next) = self.selected_points_editor_mut().redo_stack.pop() else {
            return;
        };
        self.push_current_points_undo_snapshot();
        self.apply_points_text_change(next, true);
        self.refresh_status_after_points_edit();
    }

    pub(super) fn parse_visible_points_strict(&mut self) -> Result<Points, String> {
        let mut parsed_points = Vec::new();
        for layer in &mut self.point_layers.layers {
            if !layer.visible {
                continue;
            }
            let display_name = layer.display_name().to_owned();
            let cache = points_editor_cache_with_policy(&mut layer.points, true);
            match &cache.parsed_points {
                Ok(points) => parsed_points.extend(points.iter().copied()),
                Err(error) => return Err(format!("Layer '{display_name}': {error}")),
            }
        }
        Points::try_from(parsed_points).map_err(|error| error.to_string())
    }

    pub(super) fn create_empty_point_layer(&mut self) -> PointLayerId {
        self.point_layers.create_empty_layer()
    }

    pub(super) fn create_point_layer_from_points(&mut self, points: &[Point]) -> PointLayerId {
        let id = self.point_layers.create_layer_from_points(points);
        self.finish_valid_points_change();
        id
    }

    pub(super) fn duplicate_selected_point_layer(&mut self) -> PointLayerId {
        let id = self.point_layers.duplicate_selected_layer();
        self.finish_valid_points_change();
        id
    }

    pub(super) fn delete_selected_point_layer(&mut self) {
        self.point_layers.delete_selected_layer();
        self.finish_valid_points_change();
    }

    pub(super) fn visible_point_layer_plot_data(&mut self) -> Vec<VisiblePointLayerPlotData> {
        self.point_layers
            .layers
            .iter_mut()
            .filter(|layer| layer.visible)
            .filter_map(|layer| {
                let name = layer.display_name().to_owned();
                let color = layer.color;
                let cache = points_editor_cache_with_policy(&mut layer.points, false);
                if cache.plot_points.is_empty() {
                    return None;
                }
                Some(VisiblePointLayerPlotData {
                    name,
                    color,
                    plot_points: Arc::clone(&cache.plot_points),
                })
            })
            .collect()
    }

    pub(super) fn set_points_cache_from_valid_points(&mut self, points: &[Point]) {
        set_points_editor_cache_from_valid_points(self.selected_points_editor_mut(), points);
    }

    pub(super) fn clear_points_text(&mut self, record_undo: bool) {
        self.flush_points_text_from_cache_if_pending();
        if self.selected_points_editor().text.is_empty() {
            return;
        }
        let previous = std::mem::take(&mut self.selected_points_editor_mut().text);
        if record_undo {
            self.push_points_undo_snapshot(previous);
        }
        self.selected_points_editor_mut().redo_stack.clear();
        self.set_points_cache_from_valid_points(&[]);
        self.finish_valid_points_change();
    }

    pub(super) fn write_points_text(&mut self, points: &[Point], record_undo: bool) {
        self.flush_points_text_from_cache_if_pending();
        let new_text = points_to_text(points);
        if self.selected_points_editor().text == new_text {
            return;
        }
        if record_undo {
            self.push_current_points_undo_snapshot();
        }
        let points_state = self.selected_points_editor_mut();
        points_state.text = new_text;
        points_state.redo_stack.clear();
        self.set_points_cache_from_valid_points(points);
        self.finish_valid_points_change();
    }

    pub(super) fn can_move_points_to_positive_xy(&mut self) -> bool {
        matches!(
            &self.points_cache_with_policy(false).parsed_points,
            Ok(points) if !points.is_empty()
        )
    }

    pub(super) fn move_points_to_positive_xy(&mut self) {
        let points = match &self.points_cache_with_policy(true).parsed_points {
            Ok(points) if !points.is_empty() => points,
            Ok(_) => return,
            Err(error) => {
                self.status = Some(StatusMessage::Error(format!(
                    "{POINTS_PARSE_ERROR_PREFIX}{error}"
                )));
                return;
            }
        };

        let mut min_x = points[0].x();
        let mut min_y = points[0].y();
        for point in points.iter().skip(1) {
            min_x = min_x.min(point.x());
            min_y = min_y.min(point.y());
        }

        let dx = (POINTS_POSITIVE_AXIS_EPS - min_x).max(0.0);
        let dy = (POINTS_POSITIVE_AXIS_EPS - min_y).max(0.0);

        let shifted = match points
            .iter()
            .map(|point| Point::try_new(point.x() + dx, point.y() + dy))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(format!(
                    "Failed to move points to positive x/y: {error}"
                )));
                return;
            }
        };

        self.write_points_text(&shifted, true);
    }

    pub(super) fn fill_points_with_residuals(&mut self) {
        if self.residual_plot_points.is_empty() {
            return;
        }

        let points = match self
            .residual_plot_points
            .iter()
            .map(|point| Point::try_new(point.x, point.y))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(points) => points,
            Err(error) => {
                self.status = Some(StatusMessage::Error(format!(
                    "Failed to convert residual into point: {error}"
                )));
                return;
            }
        };

        self.write_points_text(&points, true);
    }
}
