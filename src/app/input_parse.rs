use super::*;

#[derive(Debug, Clone)]
pub(super) struct ParsedInitialParams(CurveParams);

impl ParsedInitialParams {
    pub(super) fn parse(family: CurveFamily, inputs: &[String]) -> Result<Self, String> {
        let expected_count = family.parameter_count();
        if inputs.len() != expected_count {
            return Err(format!(
                "Initial parameter count mismatch: expected {expected_count}, got {}",
                inputs.len()
            ));
        }

        let mut values = Vec::with_capacity(expected_count);
        for (index, raw_value) in inputs.iter().enumerate() {
            let field = format!("parameter[{index}]");
            values.push(parse_f64(&field, raw_value)?);
        }

        let params =
            CurveParams::try_from_values(family, values).map_err(|error| error.to_string())?;
        Ok(Self(params))
    }

    pub(super) fn into_curve_params(self) -> CurveParams {
        self.0
    }
}

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

        let mut values = Vec::with_capacity(expected_count);
        for (index, raw_value) in inputs.iter().enumerate() {
            let field = format!("spline_knot_y[{index}]");
            values.push(parse_f64(&field, raw_value)?);
        }

        Ok(Self { values })
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

        ParsedInitialParams::parse(family, &self.parameter_inputs)
    }

    pub(super) fn parse_spline_initial_knot_y(
        &self,
        expected_count: usize,
    ) -> Result<ParsedSplineInitialKnotY, String> {
        ParsedSplineInitialKnotY::parse(&self.spline_initial_knot_y_inputs, expected_count)
    }
}
