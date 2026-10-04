//! Числовой черновик сохраняет незавершённый ввод и разбирается только при изменении текста.

use std::{
    borrow::Cow,
    fmt,
    hash::{Hash, Hasher},
    num::ParseFloatError,
};

#[derive(Debug)]
pub(super) enum NumberError {
    Empty,
    Invalid(ParseFloatError),
}

impl NumberError {
    pub(super) fn for_field(&self, field: impl fmt::Display) -> String {
        match self {
            Self::Empty => format!("Field '{field}' is empty"),
            Self::Invalid(error) => format!("Failed to parse '{field}' as f64: {error}"),
        }
    }
}

pub(super) fn parse_number(text: &str) -> Result<f64, NumberError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(NumberError::Empty);
    }
    let normalized = if text.contains(',') && !text.contains('.') {
        Cow::Owned(text.replace(',', "."))
    } else {
        Cow::Borrowed(text)
    };
    normalized.parse().map_err(NumberError::Invalid)
}

/// Результат редактирования отделяет изменение текста от завершения ввода.
pub(super) struct NumberEdit<R> {
    pub(super) output: R,
    pub(super) changed: bool,
    pub(super) finished: bool,
}

#[derive(Debug)]
pub(super) struct NumberDraft {
    text: String,
    value: Result<f64, NumberError>,
    committed: bool,
}

impl From<String> for NumberDraft {
    fn from(text: String) -> Self {
        let value = parse_number(&text);
        Self {
            text,
            value,
            committed: false,
        }
    }
}

impl From<&str> for NumberDraft {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl From<f64> for NumberDraft {
    fn from(value: f64) -> Self {
        Self {
            text: value.to_string(),
            value: Ok(value),
            committed: false,
        }
    }
}

impl Hash for NumberDraft {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.text().hash(state);
    }
}

impl NumberDraft {
    pub(super) fn text(&self) -> &str {
        &self.text
    }

    pub(super) fn value(&self, field: impl fmt::Display) -> Result<f64, String> {
        self.value
            .as_ref()
            .copied()
            .map_err(|error| error.for_field(field))
    }

    pub(super) fn has_visible_error(&self) -> bool {
        self.committed && self.value.is_err()
    }

    /// Разбор обновляется после редактирования; чтение поля не меняет кэш.
    pub(super) fn edit<R>(&mut self, edit: impl FnOnce(&mut String) -> NumberEdit<R>) -> R {
        let edit = edit(&mut self.text);
        if edit.changed {
            self.value = parse_number(&self.text);
            self.committed = false;
        }
        if edit.finished {
            self.committed = true;
        }
        edit.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_input_and_decimal_comma_preserve_the_draft() {
        let mut draft = NumberDraft::from(1.0);
        draft.edit(|text| {
            *text = "-".into();
            NumberEdit {
                output: (),
                changed: true,
                finished: false,
            }
        });
        assert_eq!(draft.text(), "-");
        assert!(draft.value("a").is_err());
        assert!(!draft.has_visible_error());
        draft.edit(|_| NumberEdit {
            output: (),
            changed: false,
            finished: true,
        });
        assert!(draft.has_visible_error());
        draft.edit(|text| {
            *text = " -1,25e2 ".into();
            NumberEdit {
                output: (),
                changed: true,
                finished: false,
            }
        });
        assert_eq!(draft.value("a").unwrap(), -125.0);
        assert_eq!(draft.text(), " -1,25e2 ");
        assert!(!draft.has_visible_error());
    }

    #[test]
    fn cached_errors_keep_field_names_at_the_boundary() {
        let draft = NumberDraft::from(" ");
        assert_eq!(
            draft.value("tau[0]").unwrap_err(),
            "Field 'tau[0]' is empty"
        );
        let draft = NumberDraft::from("abc");
        assert!(
            draft
                .value("parameter[1]")
                .unwrap_err()
                .starts_with("Failed to parse 'parameter[1]' as f64:")
        );
    }
}
