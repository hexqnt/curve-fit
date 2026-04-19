//! Состояние текстового редактора точек, кэш парсинга и история undo/redo.

use super::*;

/// Кэш результата последнего парсинга текста точек и подготовленных данных для графика.
#[derive(Debug, Clone)]
pub(super) struct ParsedPointsCache {
    pub(super) parsed_points: Result<Vec<Point>, String>,
    pub(super) parse_error_line: Option<usize>,
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

impl CurveFitApp {
    pub(super) fn invalidate_points_cache(&mut self) {
        self.points.cache_dirty = true;
        self.points.text_sync_pending = false;
        // Небольшой debounce уменьшает число парсингов во время быстрого ввода текста.
        self.points.parse_debounce_deadline =
            Some(Instant::now() + Duration::from_millis(POINTS_PARSE_DEBOUNCE_MS));
    }

    pub(super) fn points_cache_with_policy(&mut self, force: bool) -> &ParsedPointsCache {
        // Политика пересчета кэша:
        // - сразу, если кэша нет;
        // - по force;
        // - или после окончания debounce.
        let should_parse = if self.points.cache.is_none() {
            true
        } else if !self.points.cache_dirty {
            false
        } else if force {
            true
        } else {
            self.points
                .parse_debounce_deadline
                .map(|deadline| Instant::now() >= deadline)
                .unwrap_or(true)
        };

        if should_parse || self.points.cache.is_none() {
            // Текст парсим ровно один раз на пакет правок, а дальше работаем из кэша.
            self.points.cache = Some(parse_points_text_cache(&self.points.text));
            self.points.cache_dirty = false;
            self.points.text_sync_pending = false;
            self.points.parse_debounce_deadline = None;
        }
        self.points.cache.get_or_insert_with(|| ParsedPointsCache {
            parsed_points: Err("Internal error: points cache is unavailable".to_string()),
            parse_error_line: None,
            plot_points: Vec::<PlotPoint>::new().into(),
        })
    }

    pub(super) fn points_cache(&mut self) -> &ParsedPointsCache {
        self.points_cache_with_policy(false)
    }

    pub(super) fn maybe_refresh_points_cache_after_debounce(&mut self) {
        if self.points.cache_dirty
            && self
                .points
                .parse_debounce_deadline
                .map(|deadline| Instant::now() >= deadline)
                .unwrap_or(true)
        {
            self.points_cache_with_policy(true);
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
        let parse_error = match &self.points_cache_with_policy(true).parsed_points {
            Ok(_) => None,
            Err(error) => Some(error.clone()),
        };

        if let Some(error) = parse_error {
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

    pub(super) fn push_points_undo_snapshot(&mut self, snapshot: String) {
        if self
            .points
            .undo_stack
            .last()
            .is_some_and(|last| *last == snapshot)
        {
            return;
        }
        self.points.undo_stack.push(snapshot);
        // Ограничиваем историю фиксированным размером, чтобы не раздувать память.
        if self.points.undo_stack.len() > POINTS_HISTORY_LIMIT {
            let overflow = self.points.undo_stack.len() - POINTS_HISTORY_LIMIT;
            self.points.undo_stack.drain(0..overflow);
        }
    }

    pub(super) fn flush_points_text_from_cache_if_pending(&mut self) {
        if !self.points.text_sync_pending {
            return;
        }

        // Лениво сериализуем точки обратно в текст только тогда, когда это реально нужно UI.
        let synced_text = self.points.cache.as_ref().and_then(|cache| {
            cache
                .parsed_points
                .as_ref()
                .ok()
                .map(|points| points_to_text(points))
        });
        if let Some(synced_text) = synced_text {
            self.points.text = synced_text;
        }
        self.points.text_sync_pending = false;
    }

    pub(super) fn push_current_points_undo_snapshot(&mut self) {
        self.flush_points_text_from_cache_if_pending();
        self.push_points_undo_snapshot(self.points.text.clone());
    }

    pub(super) fn edit_valid_points_in_cache<F>(
        &mut self,
        record_undo: bool,
        sync_text_immediately: bool,
        edit: F,
    ) -> Result<(), String>
    where
        F: FnOnce(&mut Vec<Point>),
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
        self.points.redo_stack.clear();

        let maybe_synced_text = {
            let Some(cache) = self.points.cache.as_mut() else {
                return Err("Internal error: points cache is unavailable".to_string());
            };
            // Мутируем уже проверенные точки прямо в кэше, чтобы не гонять повторный parse/format.
            let points = match cache.parsed_points.as_mut() {
                Ok(points) => points,
                Err(error) => return Err(error.clone()),
            };
            edit(points);
            cache.parse_error_line = None;
            cache.plot_points = points
                .iter()
                .map(|point| PlotPoint::new(point.x(), point.y()))
                .collect::<Vec<_>>()
                .into();
            sync_text_immediately.then(|| points_to_text(points))
        };

        if let Some(synced_text) = maybe_synced_text {
            self.points.text = synced_text;
            self.points.text_sync_pending = false;
        } else {
            self.points.text_sync_pending = true;
        }

        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }

        Ok(())
    }

    pub(super) fn apply_points_text_change(&mut self, new_text: String, keep_redo: bool) {
        if self.points.text == new_text {
            return;
        }
        self.points.text = new_text;
        self.invalidate_points_cache();
        if !keep_redo {
            self.points.redo_stack.clear();
        }
    }

    pub(super) fn undo_points_edit(&mut self) {
        if self.fit_in_progress {
            return;
        }
        let Some(previous) = self.points.undo_stack.pop() else {
            return;
        };
        self.points.redo_stack.push(self.points.text.clone());
        self.apply_points_text_change(previous, true);
        self.refresh_status_after_points_edit();
    }

    pub(super) fn redo_points_edit(&mut self) {
        if self.fit_in_progress {
            return;
        }
        let Some(next) = self.points.redo_stack.pop() else {
            return;
        };
        self.push_current_points_undo_snapshot();
        self.apply_points_text_change(next, true);
        self.refresh_status_after_points_edit();
    }

    pub(super) fn parse_points_strict(&mut self) -> Result<Points, String> {
        let parsed_points = match &self.points_cache_with_policy(true).parsed_points {
            Ok(points) => points.clone(),
            Err(error) => return Err(error.clone()),
        };
        Points::try_from(parsed_points).map_err(|error| error.to_string())
    }

    pub(super) fn set_points_cache_from_valid_points(&mut self, points: &[Point]) {
        let parsed_points = points.to_vec();
        let plot_points: Arc<[PlotPoint]> = parsed_points
            .iter()
            .map(|point| PlotPoint::new(point.x(), point.y()))
            .collect::<Vec<_>>()
            .into();
        self.points.cache = Some(ParsedPointsCache {
            parsed_points: Ok(parsed_points),
            parse_error_line: None,
            plot_points,
        });
        self.points.cache_dirty = false;
        self.points.text_sync_pending = false;
        self.points.parse_debounce_deadline = None;
    }

    pub(super) fn clear_points_text(&mut self, record_undo: bool) {
        self.flush_points_text_from_cache_if_pending();
        if self.points.text.is_empty() {
            return;
        }
        let previous = std::mem::take(&mut self.points.text);
        if record_undo {
            self.push_points_undo_snapshot(previous);
        }
        self.points.redo_stack.clear();
        self.set_points_cache_from_valid_points(&[]);
        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
    }

    pub(super) fn write_points_text(&mut self, points: &[Point], record_undo: bool) {
        self.flush_points_text_from_cache_if_pending();
        let new_text = points_to_text(points);
        if self.points.text == new_text {
            return;
        }
        if record_undo {
            self.push_current_points_undo_snapshot();
        }
        self.points.text = new_text;
        self.points.redo_stack.clear();
        self.set_points_cache_from_valid_points(points);
        if matches!(
            self.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(POINTS_PARSE_ERROR_PREFIX)
        ) {
            self.status = Some(self.idle_status_after_points_edit());
        }
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
