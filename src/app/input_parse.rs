//! Парсинг пользовательских строк в типизированные начальные параметры и узлы сплайна.

use super::*;

fn parse_indexed_f64_inputs(inputs: &[String], field_prefix: &str) -> Result<Vec<f64>, String> {
    inputs
        .iter()
        .enumerate()
        .map(|(index, raw_value)| parse_f64(&format!("{field_prefix}[{index}]"), raw_value))
        .collect()
}

/// Уже распарсенные и типизированные начальные параметры параметрической модели.
#[derive(Debug, Clone)]
pub(super) struct ParsedInitialParams(CurveParams);

impl ParsedInitialParams {
    pub(super) fn parse(
        family: CurveFamily,
        inputs: &[String],
        saturating_trend_tau_grid: Option<&SaturatingTrendTauGrid>,
    ) -> Result<Self, String> {
        let expected_count = family.parameter_count();
        if inputs.len() != expected_count {
            return Err(format!(
                "Initial parameter count mismatch: expected {expected_count}, got {}",
                inputs.len()
            ));
        }

        let values = parse_indexed_f64_inputs(inputs, "parameter")?;

        let params =
            CurveParams::try_from_slice_with_tau_grid(family, &values, saturating_trend_tau_grid)
                .map_err(|error| error.to_string())?;
        Ok(Self(params))
    }

    pub(super) fn into_curve_params(self) -> CurveParams {
        self.0
    }
}

/// Уже распарсенная и провалидированная сетка `τ` для saturating trend basis.
#[derive(Debug, Clone)]
pub(super) struct ParsedSaturatingTrendTauGrid(SaturatingTrendTauGrid);

impl ParsedSaturatingTrendTauGrid {
    pub(super) fn parse(inputs: &[String], expected_count: usize) -> Result<Self, String> {
        if inputs.len() < expected_count {
            return Err(format!(
                "Saturating-trend tau grid expects at least {expected_count} values, got {}",
                inputs.len()
            ));
        }

        let values = parse_indexed_f64_inputs(&inputs[..expected_count], "tau")?;

        let grid =
            SaturatingTrendTauGrid::from_values(&values).map_err(|error| error.to_string())?;
        Ok(Self(grid))
    }

    pub(super) fn into_tau_grid(self) -> SaturatingTrendTauGrid {
        self.0
    }
}

/// Провалидированные начальные значения `y` для внутренних узлов сплайна.
#[derive(Debug, Clone)]
pub(super) struct ParsedSplineInitialKnotY {
    values: Vec<f64>,
}

impl ParsedSplineInitialKnotY {
    pub(super) fn parse(inputs: &[String], expected_count: usize) -> Result<Self, String> {
        if inputs.len() != expected_count {
            return Err(format!(
                "Spline initialization expects {expected_count} values, got {}",
                inputs.len()
            ));
        }

        Ok(Self {
            values: parse_indexed_f64_inputs(inputs, "spline_knot_y")?,
        })
    }

    pub(super) fn as_slice(&self) -> &[f64] {
        self.values.as_slice()
    }

    pub(super) fn into_vec(self) -> Vec<f64> {
        self.values
    }
}

impl CurveFitApp {
    pub(super) fn parse_initial_params(&self) -> Result<ParsedInitialParams, String> {
        let family = self.resolved_model().parametric_family().ok_or_else(|| {
            "Current model is non-parametric and has no initial parameters".to_string()
        })?;

        let tau_grid = self.parsed_saturating_trend_tau_grid()?;
        ParsedInitialParams::parse(family, &self.parameter_inputs, tau_grid.as_ref())
    }

    pub(super) fn parsed_saturating_trend_tau_grid(
        &self,
    ) -> Result<Option<SaturatingTrendTauGrid>, String> {
        let Some(expected_count) = self
            .resolved_model()
            .parametric_family()
            .and_then(CurveFamily::saturating_trend_tau_count)
        else {
            return Ok(None);
        };
        ParsedSaturatingTrendTauGrid::parse(&self.saturating_trend_tau_inputs, expected_count)
            .map(ParsedSaturatingTrendTauGrid::into_tau_grid)
            .map(Some)
    }

    pub(super) fn parse_spline_initial_knot_y(
        &self,
        expected_count: usize,
    ) -> Result<ParsedSplineInitialKnotY, String> {
        ParsedSplineInitialKnotY::parse(&self.spline_initial_knot_y_inputs, expected_count)
    }
}
