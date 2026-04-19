//! Статусы приложения, отображаемые в нижней панели и при ошибках операций.

use super::*;

/// Краткое состояние приложения, пригодное для вывода в статус-бар.
#[derive(Debug, Clone)]
pub(super) enum StatusMessage {
    Ready,
    Cleared,
    FittingInProgress,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    FittingStopping,
    FitStopped,
    FitCompleted,
    Error(String),
}

impl StatusMessage {
    /// Проверяет, содержит ли статус текст ошибки.
    pub(super) fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Возвращает локализованный текст статуса.
    pub(super) fn text(&self, language: UiLanguage) -> &str {
        match self {
            Self::Ready => tr(language, "Ready", "Готово"),
            Self::Cleared => tr(language, "Input cleared", "Поле ввода очищено"),
            Self::FittingInProgress => tr(language, "Fitting in progress", "Подгонка в процессе"),
            Self::FittingStopping => tr(language, "Stopping fit...", "Останавливаем подгонку..."),
            Self::FitStopped => tr(language, "Fit stopped", "Подгонка остановлена"),
            Self::FitCompleted => tr(language, "Fit completed", "Фитинг завершен"),
            Self::Error(message) => message.as_str(),
        }
    }
}
