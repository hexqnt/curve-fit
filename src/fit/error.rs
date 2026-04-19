//! Ошибки пайплайна оптимизации и построения сплайнов.

use super::*;

#[derive(Debug, Clone, PartialEq)]
/// Ошибки, возникающие при подгонке моделей и сплайнов.
pub enum FitError {
    InvalidInput(InputError),
    InvalidSplineInput(String),
    Cancelled,
    Optimizer(String),
    MissingBestParameters,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(error) => write!(f, "{error}"),
            Self::InvalidSplineInput(message) => write!(f, "{message}"),
            Self::Cancelled => f.write_str("Optimization cancelled by user"),
            Self::Optimizer(error) => write!(f, "Optimization failed: {error}"),
            Self::MissingBestParameters => f.write_str("Optimizer did not return best parameters"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<InputError> for FitError {
    fn from(value: InputError) -> Self {
        Self::InvalidInput(value)
    }
}

pub(super) fn optimizer_error(error: impl fmt::Display) -> FitError {
    FitError::Optimizer(error.to_string())
}
