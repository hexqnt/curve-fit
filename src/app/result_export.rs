//! Сериализация результата фитинга в JSON и интеграция с экспортом из интерфейса.

use super::*;
#[cfg(not(target_arch = "wasm32"))]
use chrono::{SecondsFormat, Utc};
#[cfg(not(target_arch = "wasm32"))]
use egui_file_dialog::DialogState;
use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(target_arch = "wasm32")]
use web_sys::js_sys::Date;

/// Сериализуемая сводка результата фитинга для буфера обмена и сохранения в файл.
#[derive(Debug, Clone, Serialize)]
pub(super) struct FitExportRecord {
    fitted_at: String,
    point_count: usize,
    model: FitExportModel,
    optimizer: FitExportOptimizer,
    convergence: FitExportConvergence,
    metrics: FitExportMetrics,
    result: FitExportResult,
}

#[derive(Debug, Clone, Serialize)]
struct FitExportModel {
    selected: FitExportNamedId,
    fitted: FitExportNamedId,
}

#[derive(Debug, Clone, Serialize)]
struct FitExportOptimizer {
    method: FitExportNamedId,
    loss_metric: FitExportNamedId,
    #[serde(skip_serializing_if = "Option::is_none")]
    metric_quantization_decimal_places: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct FitExportConvergence {
    iterations: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize)]
struct FitExportMetrics {
    mse: f64,
    rmse: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    mae: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    r2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_abs_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FitExportResult {
    Parametric {
        parameter_count: usize,
        parameters: Vec<FitExportParameter>,
    },
    Spline {
        knot_count: usize,
        knots: Vec<FitExportKnot>,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
struct FitExportNamedId {
    id: &'static str,
    name: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct FitExportParameter {
    name: &'static str,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct FitExportKnot {
    x: f64,
    y: f64,
}

impl CurveFitApp {
    pub(super) fn has_fit_export_record(&self) -> bool {
        self.fit_export_record.is_some()
    }

    pub(super) fn clear_fit_export_state(&mut self) {
        self.fit_export_record = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.fit_export_pending_json = None;
        }
    }

    pub(super) fn build_fit_export_json_pretty(&self) -> Result<String, String> {
        let record = self
            .fit_export_record
            .as_ref()
            .ok_or_else(|| "No fit export data is available".to_string())?;
        serde_json::to_string_pretty(record)
            .map_err(|error| format!("Failed to serialize fit export JSON: {error}"))
    }

    pub(super) fn copy_fit_export_json(&mut self, ctx: &egui::Context) {
        match self.build_fit_export_json_pretty() {
            Ok(json) => {
                self.copy_text_to_clipboard(ctx, json);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn request_fit_export_save_json(&mut self) {
        let json = match self.build_fit_export_json_pretty() {
            Ok(json) => json,
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
                return;
            }
        };

        self.fit_export_pending_json = Some(json);
        if let Some(directory) = self
            .fit_export_last_directory
            .clone()
            .filter(|directory| directory.is_dir())
        {
            self.fit_export_file_dialog.config_mut().initial_directory = directory;
        }
        self.fit_export_file_dialog.config_mut().default_file_name = default_fit_export_file_name();
        self.fit_export_file_dialog.save_file();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_fit_export_save_dialog(&mut self, ctx: &egui::Context) {
        self.fit_export_file_dialog.update(ctx);
        if matches!(self.fit_export_file_dialog.state(), DialogState::Cancelled) {
            self.fit_export_pending_json = None;
        }

        let Some(path) = self.fit_export_file_dialog.take_picked() else {
            return;
        };
        let Some(json) = self.fit_export_pending_json.take() else {
            return;
        };

        let path = ensure_json_extension(path);
        if let Err(error) = std::fs::write(&path, json) {
            self.status = Some(StatusMessage::Error(format!(
                "Failed to save fit JSON to '{}': {error}",
                path.display()
            )));
        } else {
            self.fit_export_last_directory = dialog_directory_from_path(&path);
        }
    }

    pub(super) fn store_parametric_fit_export_record(
        &mut self,
        result: &FitResult,
        point_count: usize,
    ) {
        let parameters: Vec<FitExportParameter> = result
            .family
            .parameter_names()
            .iter()
            .copied()
            .zip(result.params.values())
            .map(|(name, value)| FitExportParameter { name, value })
            .collect();

        let metrics = if let Some(metrics) = self.result_metrics {
            FitExportMetrics::from_complete_metrics(metrics)
        } else {
            FitExportMetrics::from_basic_metrics(result.mse, result.rmse)
        };
        self.fit_export_record = Some(self.build_fit_export_record(
            result.iterations,
            point_count,
            metrics,
            FitExportResult::Parametric {
                parameter_count: parameters.len(),
                parameters,
            },
        ));
    }

    pub(super) fn store_spline_fit_export_record(
        &mut self,
        result: &SplineResult,
        point_count: usize,
    ) {
        let knots: Vec<FitExportKnot> = result
            .knots
            .iter()
            .map(|knot| FitExportKnot {
                x: knot[0],
                y: knot[1],
            })
            .collect();

        self.fit_export_record = Some(self.build_fit_export_record(
            result.iterations,
            point_count,
            FitExportMetrics::from_complete_metrics(ExtendedMetrics {
                mse: result.mse,
                rmse: result.rmse,
                mae: result.mae,
                r2: result.r2,
                max_abs_error: result.max_abs_error,
            }),
            FitExportResult::Spline {
                knot_count: knots.len(),
                knots,
            },
        ));
    }

    fn build_fit_export_record(
        &self,
        iterations: u64,
        point_count: usize,
        metrics: FitExportMetrics,
        result: FitExportResult,
    ) -> FitExportRecord {
        FitExportRecord {
            fitted_at: now_utc_rfc3339_millis(),
            point_count,
            model: FitExportModel {
                selected: model_choice_ref(self.selected_model),
                fitted: resolved_model_ref(self.resolved_model()),
            },
            optimizer: FitExportOptimizer {
                method: optimizer_method_ref(self.fit_optimizer_method),
                loss_metric: objective_metric_ref(self.fit_loss_metric),
                metric_quantization_decimal_places: metric_quantization_decimal_places(
                    self.fit_metric_quantization,
                ),
            },
            convergence: FitExportConvergence {
                iterations,
                duration_ms: self.last_fit_duration.map(|duration| duration.as_millis()),
            },
            metrics,
            result,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn now_utc_rfc3339_millis() -> String {
    chrono::DateTime::<Utc>::from(std::time::SystemTime::now())
        .to_rfc3339_opts(SecondsFormat::Millis, true)
}

#[cfg(target_arch = "wasm32")]
fn now_utc_rfc3339_millis() -> String {
    Date::new_0().to_iso_string().into()
}

fn metric_quantization_decimal_places(metric_quantization: MetricQuantization) -> Option<u8> {
    match metric_quantization {
        MetricQuantization::Disabled => None,
        MetricQuantization::Enabled(decimal_places) => Some(decimal_places.get()),
    }
}

impl FitExportMetrics {
    fn from_basic_metrics(mse: f64, rmse: f64) -> Self {
        Self {
            mse,
            rmse,
            mae: None,
            r2: None,
            max_abs_error: None,
        }
    }

    fn from_complete_metrics(metrics: ExtendedMetrics) -> Self {
        Self {
            mse: metrics.mse,
            rmse: metrics.rmse,
            mae: Some(metrics.mae),
            r2: Some(metrics.r2),
            max_abs_error: Some(metrics.max_abs_error),
        }
    }
}

fn model_choice_ref(model: ModelChoice) -> FitExportNamedId {
    FitExportNamedId {
        id: model_choice_id(model),
        name: model_choice_label(UiLanguage::English, model),
    }
}

fn resolved_model_ref(model: ResolvedModel) -> FitExportNamedId {
    match model {
        ResolvedModel::Parametric(family) => curve_family_ref(family),
        ResolvedModel::LinearSpline => spline_family_ref(SplineFamilyKind::Linear),
        ResolvedModel::MonotoneCubicSpline => spline_family_ref(SplineFamilyKind::MonotoneCubic),
        ResolvedModel::NaturalCubicSpline => spline_family_ref(SplineFamilyKind::NaturalCubic),
        ResolvedModel::AkimaSpline => spline_family_ref(SplineFamilyKind::Akima),
    }
}

fn optimizer_method_ref(method: OptimizerMethod) -> FitExportNamedId {
    FitExportNamedId {
        id: optimizer_method_id(method),
        name: optimizer_method_label(UiLanguage::English, method),
    }
}

fn objective_metric_ref(metric: OptimizationLossMetric) -> FitExportNamedId {
    FitExportNamedId {
        id: metric.id(),
        name: objective_metric_name(metric),
    }
}

fn curve_family_id(family: CurveFamily) -> &'static str {
    match family {
        CurveFamily::Linear => "linear",
        CurveFamily::Quadratic => "quadratic",
        CurveFamily::Cubic => "cubic",
        CurveFamily::Quartic => "quartic",
        CurveFamily::Quintic => "quintic",
        CurveFamily::Sextic => "sextic",
        CurveFamily::Septic => "septic",
        CurveFamily::Octic => "octic",
        CurveFamily::Nonic => "nonic",
        CurveFamily::Arrhenius => "arrhenius",
        CurveFamily::Inverse => "inverse",
        CurveFamily::Logistic => "logistic",
        CurveFamily::Gompertz => "gompertz",
        CurveFamily::BiExponential => "bi_exponential",
        CurveFamily::DampedSinusoid => "damped_sinusoid",
        CurveFamily::Lorentzian => "lorentzian",
        CurveFamily::NaturalLog => "natural_log",
        CurveFamily::FourPl => "four_pl",
        CurveFamily::FivePl => "five_pl",
        CurveFamily::MichaelisMenten => "michaelis_menten",
        CurveFamily::ExponentialBasic => "exponential_basic",
        CurveFamily::ExponentialLinear => "exponential_linear",
        CurveFamily::ExponentialHalfLife => "exponential_half_life",
        CurveFamily::FallingExponential => "falling_exponential",
        CurveFamily::HyperbolicTangent => "hyperbolic_tangent",
        CurveFamily::ArctangentStep => "arctangent_step",
        CurveFamily::Softplus => "softplus",
        CurveFamily::Power => "power",
        CurveFamily::Gaussian => "gaussian",
        CurveFamily::Rational11 => "rational_11",
        CurveFamily::Rational22 => "rational_22",
        CurveFamily::Rational33 => "rational_33",
        CurveFamily::Rational44 => "rational_44",
        CurveFamily::Rational55 => "rational_55",
        CurveFamily::Emg => "emg",
        CurveFamily::PseudoVoigt => "pseudo_voigt",
    }
}

fn curve_family_ref(family: CurveFamily) -> FitExportNamedId {
    FitExportNamedId {
        id: curve_family_id(family),
        name: family.label(),
    }
}

fn spline_family_ref(family: SplineFamilyKind) -> FitExportNamedId {
    FitExportNamedId {
        id: spline_family_id(family),
        name: spline_family_name(family),
    }
}

fn model_choice_id(model: ModelChoice) -> &'static str {
    match model {
        ModelChoice::Polynomial => "polynomial",
        ModelChoice::Arrhenius => "arrhenius",
        ModelChoice::Inverse => "inverse",
        ModelChoice::Logistic => "logistic",
        ModelChoice::Gompertz => "gompertz",
        ModelChoice::BiExponential => "bi_exponential",
        ModelChoice::DampedSinusoid => "damped_sinusoid",
        ModelChoice::Lorentzian => "lorentzian",
        ModelChoice::NaturalLog => "natural_log",
        ModelChoice::FourPl => "four_pl",
        ModelChoice::FivePl => "five_pl",
        ModelChoice::MichaelisMenten => "michaelis_menten",
        ModelChoice::ExponentialBasic => "exponential_basic",
        ModelChoice::ExponentialLinear => "exponential_linear",
        ModelChoice::ExponentialHalfLife => "exponential_half_life",
        ModelChoice::FallingExponential => "falling_exponential",
        ModelChoice::HyperbolicTangent => "hyperbolic_tangent",
        ModelChoice::ArctangentStep => "arctangent_step",
        ModelChoice::Softplus => "softplus",
        ModelChoice::Power => "power",
        ModelChoice::Gaussian => "gaussian",
        ModelChoice::Rational => "rational",
        ModelChoice::Emg => "emg",
        ModelChoice::PseudoVoigt => "pseudo_voigt",
        ModelChoice::LinearSpline => "linear_spline",
        ModelChoice::MonotoneCubicSpline => "monotone_cubic_spline",
        ModelChoice::NaturalCubicSpline => "natural_cubic_spline",
        ModelChoice::AkimaSpline => "akima_spline",
    }
}

fn optimizer_method_id(method: OptimizerMethod) -> &'static str {
    match method {
        OptimizerMethod::Lbfgs => "lbfgs",
        OptimizerMethod::NelderMead => "nelder_mead",
        OptimizerMethod::SteepestDescent => "steepest_descent",
        OptimizerMethod::NewtonCg => "newton_cg",
        OptimizerMethod::Sgd => "sgd",
        OptimizerMethod::Adam => "adam",
    }
}

fn objective_metric_name(metric: OptimizationLossMetric) -> &'static str {
    match metric {
        OptimizationLossMetric::Mse => "Mean Squared Error (L2)",
        OptimizationLossMetric::Mae => "Mean Absolute Error (L1)",
        OptimizationLossMetric::SoftL1 => "Soft L1",
        OptimizationLossMetric::Chebyshev => "Chebyshev Distance (Linf)",
        OptimizationLossMetric::Msle => "Mean Squared Logarithmic Error",
    }
}

fn spline_family_id(family: SplineFamilyKind) -> &'static str {
    match family {
        SplineFamilyKind::Linear => "linear_spline",
        SplineFamilyKind::MonotoneCubic => "monotone_cubic_spline",
        SplineFamilyKind::NaturalCubic => "natural_cubic_spline",
        SplineFamilyKind::Akima => "akima_spline",
    }
}

fn spline_family_name(family: SplineFamilyKind) -> &'static str {
    match family {
        SplineFamilyKind::Linear => "Linear Spline",
        SplineFamilyKind::MonotoneCubic => "Monotone Cubic Spline",
        SplineFamilyKind::NaturalCubic => "Natural Cubic Spline",
        SplineFamilyKind::Akima => "Akima Spline",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn default_fit_export_file_name() -> String {
    let timestamp = now_utc_rfc3339_millis().replace(':', "-");
    format!("fit-result-{timestamp}.json")
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_json_extension(mut path: PathBuf) -> PathBuf {
    let has_extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| !extension.trim().is_empty());
    if !has_extension {
        path.set_extension("json");
    }
    path
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::ensure_json_extension;
    use std::path::PathBuf;

    #[test]
    fn ensure_json_extension_adds_extension_for_path_without_it() {
        let path = PathBuf::from("/tmp/fit-result");
        let with_extension = ensure_json_extension(path);
        assert_eq!(with_extension, PathBuf::from("/tmp/fit-result.json"));
    }
}
