//! Состояние видимости панелей и активной вкладки диагностического интерфейса.

/// Вкладки нижней панели диагностики.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum DiagnosticsTab {
    #[default]
    Loss,
    Residuals,
}

/// Пользовательские флаги видимости панелей и служебное состояние их раскладки.
#[derive(Debug, Clone)]
pub(super) struct PanelState {
    pub(super) show_left: bool,
    pub(super) show_right: bool,
    pub(super) show_formula_window: bool,
    pub(super) show_diagnostics: bool,
    pub(super) diagnostics_tab: DiagnosticsTab,
    pub(super) diagnostics_hide_non_loss_by_default_pending: bool,
    pub(super) diagnostics_shared_axis_width: f32,
}

impl Default for PanelState {
    fn default() -> Self {
        Self {
            show_left: true,
            show_right: true,
            show_formula_window: false,
            show_diagnostics: true,
            diagnostics_tab: DiagnosticsTab::Loss,
            diagnostics_hide_non_loss_by_default_pending: true,
            diagnostics_shared_axis_width: 0.0,
        }
    }
}
