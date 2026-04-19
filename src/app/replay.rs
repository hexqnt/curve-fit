//! Replay состояния оптимизации по сохраненным кадрам параметров или сплайновой кривой.

use super::*;

/// Полезная нагрузка кадра replay: параметры модели или уже сэмплированная кривая сплайна.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ReplayFramePayload {
    Parametric { params: CurveParams },
    Spline { curve: Arc<[PlotPoint]> },
}

/// Один кадр replay для конкретной итерации оптимизации.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ReplayFrame {
    pub(super) iteration: u64,
    pub(super) payload: ReplayFramePayload,
}

pub(super) fn upsert_replay_frame_in(frames: &mut Vec<ReplayFrame>, frame: ReplayFrame) {
    if let Some(last) = frames.last_mut()
        && last.iteration == frame.iteration
    {
        // Одна и та же итерация может прийти повторно как более свежий снимок.
        *last = frame;
        return;
    }

    frames.push(frame);
}

/// Состояние управления replay в UI: список кадров, выбор и автовоспроизведение.
#[derive(Debug, Clone)]
pub(super) struct ReplayState {
    pub(super) iteration_delay_seconds: f64,
    pub(super) frames: Vec<ReplayFrame>,
    pub(super) selected_index: Option<usize>,
    pub(super) autoplay_on_fit: bool,
    pub(super) autoplay: bool,
    pub(super) last_step_at: Option<Instant>,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            iteration_delay_seconds: 0.0,
            frames: Vec::new(),
            selected_index: None,
            autoplay_on_fit: true,
            autoplay: false,
            last_step_at: None,
        }
    }
}

impl CurveFitApp {
    pub(super) fn clear_replay_state(&mut self) {
        self.replay.frames.clear();
        self.replay.selected_index = None;
        self.replay.autoplay = false;
        self.replay.last_step_at = None;
    }

    #[cfg(test)]
    pub(super) fn upsert_parametric_replay_frame(&mut self, iteration: u64, params: CurveParams) {
        self.upsert_replay_frame(ReplayFrame {
            iteration,
            payload: ReplayFramePayload::Parametric { params },
        });
    }

    #[cfg(test)]
    pub(super) fn upsert_spline_replay_frame(&mut self, iteration: u64, curve: Vec<PlotPoint>) {
        self.upsert_replay_frame(ReplayFrame {
            iteration,
            payload: ReplayFramePayload::Spline {
                curve: curve.into(),
            },
        });
    }

    #[cfg(test)]
    pub(super) fn upsert_replay_frame(&mut self, frame: ReplayFrame) {
        upsert_replay_frame_in(&mut self.replay.frames, frame);
    }

    pub(super) fn replay_iteration_bounds(&self) -> Option<(u64, u64)> {
        let first = self.replay.frames.first()?;
        let last = self.replay.frames.last()?;
        Some((first.iteration, last.iteration))
    }

    pub(super) fn replay_selected_iteration(&self) -> Option<u64> {
        let index = self.replay.selected_index?;
        self.replay.frames.get(index).map(|frame| frame.iteration)
    }

    pub(super) fn set_replay_selected_index(&mut self, index: usize) {
        let Some(frame) = self.replay.frames.get(index) else {
            return;
        };

        self.replay.selected_index = Some(index);
        self.fit_preview_iteration = Some(frame.iteration);

        match &frame.payload {
            ReplayFramePayload::Parametric { params } => {
                self.fit_preview_params = Some(params.clone());
                self.spline_plot_curve = None;
            }
            ReplayFramePayload::Spline { curve } => {
                self.fit_preview_params = None;
                self.spline_plot_curve = Some(Arc::clone(curve));
            }
        }
    }

    pub(super) fn select_nearest_replay_iteration(&mut self, iteration: u64) {
        let Some(index) = self.nearest_replay_frame_index(iteration) else {
            return;
        };
        self.set_replay_selected_index(index);
    }

    pub(super) fn nearest_replay_frame_index(&self, iteration: u64) -> Option<usize> {
        let frames = self.replay.frames.as_slice();
        if frames.is_empty() {
            return None;
        }

        match frames.binary_search_by_key(&iteration, |frame| frame.iteration) {
            Ok(index) => Some(index),
            Err(insert) => {
                // `binary_search` возвращает позицию вставки; дальше выбираем ближайшего соседа.
                if insert == 0 {
                    Some(0)
                } else if insert >= frames.len() {
                    Some(frames.len() - 1)
                } else {
                    let prev = insert - 1;
                    let prev_distance = iteration.saturating_sub(frames[prev].iteration);
                    let next_distance = frames[insert].iteration.saturating_sub(iteration);
                    if next_distance < prev_distance {
                        Some(insert)
                    } else {
                        Some(prev)
                    }
                }
            }
        }
    }

    pub(super) fn start_replay_from_beginning(&mut self) {
        if self.replay.frames.is_empty() {
            self.replay.autoplay = false;
            self.replay.last_step_at = None;
            return;
        }

        self.set_replay_selected_index(0);
        self.replay.autoplay = self.replay.autoplay_on_fit && self.replay.frames.len() > 1;
        self.replay.last_step_at = None;
    }

    pub(super) fn select_replay_last_frame(&mut self) {
        if let Some(last_index) = self.replay.frames.len().checked_sub(1) {
            self.set_replay_selected_index(last_index);
        }
    }

    pub(super) fn finalize_replay_after_fit_completion(&mut self) {
        if self.replay.autoplay_on_fit && self.replay.frames.len() > 1 {
            self.start_replay_from_beginning();
            return;
        }

        self.pause_replay();
        self.select_replay_last_frame();
    }

    #[cfg(test)]
    pub(super) fn finalize_replay_after_fit_stopped(&mut self) {
        self.pause_replay();
        if !self.replay.autoplay_on_fit {
            self.select_replay_last_frame();
        }
    }

    pub(super) fn toggle_replay_autoplay(&mut self) {
        if self.replay.autoplay {
            self.replay.autoplay = false;
            self.replay.last_step_at = None;
            return;
        }

        if self.replay.frames.len() < 2 {
            return;
        }

        let at_end = self
            .replay
            .selected_index
            .is_none_or(|index| index + 1 >= self.replay.frames.len());
        if at_end {
            self.set_replay_selected_index(0);
        }

        self.replay.autoplay = true;
        self.replay.last_step_at = None;
    }

    pub(super) fn pause_replay(&mut self) {
        self.replay.autoplay = false;
        self.replay.last_step_at = None;
    }

    pub(super) fn tick_replay(&mut self, ctx: &egui::Context) {
        if self.fit_in_progress || !self.replay.autoplay {
            return;
        }

        let step_interval = if self.replay.iteration_delay_seconds > 0.0 {
            Duration::from_secs_f64(self.replay.iteration_delay_seconds)
        } else {
            Duration::from_millis(REPLAY_FAST_REPAINT_INTERVAL_MS)
        };

        let Some(current_index) = self.replay.selected_index else {
            self.pause_replay();
            return;
        };
        if current_index + 1 >= self.replay.frames.len() {
            self.pause_replay();
            return;
        }

        let now = Instant::now();
        let should_step = self.replay.last_step_at.is_none_or(|last_step_at| {
            now.saturating_duration_since(last_step_at) >= step_interval
        });
        if should_step {
            self.set_replay_selected_index(current_index + 1);
            self.replay.last_step_at = Some(now);
            if current_index + 2 < self.replay.frames.len() {
                ctx.request_repaint_after(step_interval);
            } else {
                self.pause_replay();
            }
            return;
        }

        let elapsed = self
            .replay
            .last_step_at
            .map(|last_step_at| now.saturating_duration_since(last_step_at))
            .unwrap_or_default();
        let remaining = step_interval.saturating_sub(elapsed);
        ctx.request_repaint_after(remaining);
    }
}
