//! Создание `CurveFitApp` и инициализация значений по умолчанию для UI и рантайма.

use super::*;

impl CurveFitApp {
    /// Создает приложение и настраивает загрузчики изображений для иконок/формул.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        Self {
            ui_language: UiLanguage::from_system_locale(),
            ..Self::default()
        }
    }
}

impl Default for CurveFitApp {
    fn default() -> Self {
        let selected_model = ModelChoice::Polynomial;
        let polynomial_degree = 1;
        let rational_degree = MIN_RATIONAL_DEGREE;
        let saturating_trend_tau_count = MAX_SATURATING_TREND_TAU_COUNT;
        let selected_family = polynomial_family(polynomial_degree);
        let default_lbfgs = LbfgsConfig::default();
        let default_nelder_mead = NelderMeadConfig::default();
        let default_steepest_descent = SteepestDescentConfig::default();
        let default_newton_cg = NewtonCgConfig::default();
        let default_sgd = SgdConfig::default();
        let default_adam = AdamConfig::default();

        Self {
            points: PointsEditorState::default(),
            #[cfg(not(target_arch = "wasm32"))]
            points_file_import_dialog: FileDialog::new()
                .title("Import points from file")
                .add_file_filter_extensions("Point files", vec!["csv", "CSV", "xlsx", "XLSX"])
                .add_file_filter_extensions("CSV files", vec!["csv", "CSV"])
                .add_file_filter_extensions("Excel files", vec!["xlsx", "XLSX"])
                .default_file_filter("Point files"),
            #[cfg(not(target_arch = "wasm32"))]
            points_file_import_last_directory: None,
            #[cfg(not(target_arch = "wasm32"))]
            clipboard_import_request_pending: false,
            #[cfg(not(target_arch = "wasm32"))]
            clipboard_import_requested_at: None,
            #[cfg(target_arch = "wasm32")]
            clipboard_import_web_in_flight: false,
            #[cfg(target_arch = "wasm32")]
            clipboard_import_web_result: Rc::new(RefCell::new(None)),
            #[cfg(target_arch = "wasm32")]
            clipboard_copy_web_in_flight: false,
            #[cfg(target_arch = "wasm32")]
            clipboard_copy_web_result: Rc::new(RefCell::new(None)),
            selected_model,
            polynomial_degree,
            rational_degree,
            saturating_trend_tau_count,
            saturating_trend_tau_inputs: tau_grid_to_input_strings(
                &DEFAULT_SATURATING_TREND_TAUS_YEARS,
            ),
            parameter_inputs: params_to_input_strings(&selected_family.default_params()),
            optimizer_method: OptimizerMethod::Lbfgs,
            optimizer_mode: OptimizerUiMode::Basic,
            optimization_loss_metric: OptimizationLossMetric::default(),
            metric_quantization_enabled: false,
            metric_quantization_decimal_places: DEFAULT_METRIC_QUANTIZATION_DECIMAL_PLACES,
            normalize_parametric_data: false,
            lbfgs_inputs: LbfgsInputState::from_config(&default_lbfgs),
            lbfgs_preset: infer_lbfgs_preset(&default_lbfgs),
            nelder_mead_inputs: NelderMeadInputState::from_config(&default_nelder_mead),
            nelder_mead_preset: infer_nelder_mead_preset(&default_nelder_mead),
            steepest_descent_inputs: SteepestDescentInputState::from_config(
                &default_steepest_descent,
            ),
            steepest_descent_preset: infer_steepest_descent_preset(&default_steepest_descent),
            newton_cg_inputs: NewtonCgInputState::from_config(&default_newton_cg),
            newton_cg_preset: infer_newton_cg_preset(&default_newton_cg),
            sgd_inputs: SgdInputState::from_config(&default_sgd),
            sgd_preset: infer_sgd_preset(&default_sgd),
            adam_inputs: AdamInputState::from_config(&default_adam),
            adam_preset: infer_adam_preset(&default_adam),
            ui_language: UiLanguage::English,
            plot_tool: PlotTool::SinglePoint,
            spray_points_per_second: 140,
            spray_radius_rel: 0.02,
            spray_brush: SprayBrush::Uniform,
            eraser_radius_rel: 0.03,
            spray_seed: 0xDEADBEEFCAFEBABE,
            spray_points_budget: 0.0,
            spray_last_emit_at: None,
            fit_to_content_requested: false,
            center_origin_requested: false,
            origin_bottom_left_requested: true,
            last_plot_bounds: None,
            active_tool_bounds: None,
            panel: PanelState::default(),
            spline_knots: crate::fit::DEFAULT_SPLINE_KNOTS,
            spline_knot_strategy: SplineKnotStrategy::default(),
            spline_extrapolation: SplineExtrapolation::default(),
            spline_duplicate_x_policy: SplineDuplicateXPolicy::default(),
            spline_initial_knot_y_inputs: Vec::new(),
            auto_refit_enabled: false,
            auto_refit_pending_rerun: false,
            last_right_panel_fit_snapshot: None,
            replay: ReplayState::default(),
            fit_in_progress: false,
            fit_loss_metric: OptimizationLossMetric::default(),
            fit_metric_quantization: MetricQuantization::Disabled,
            fit_optimizer_method: OptimizerMethod::default(),
            fit_export_record: None,
            #[cfg(not(target_arch = "wasm32"))]
            fit_export_file_dialog: FileDialog::new()
                .title("Save fit result JSON")
                .add_save_extension("JSON files", "json")
                .default_save_extension("JSON files")
                .default_file_name("fit-result.json"),
            #[cfg(not(target_arch = "wasm32"))]
            fit_export_last_directory: None,
            #[cfg(not(target_arch = "wasm32"))]
            fit_export_pending_json: None,
            fit_preview_params: None,
            fit_preview_iteration: None,
            fit_started_at: None,
            last_fit_duration: None,
            fit_result: None,
            spline_result: None,
            active_fit_points: None,
            fit_run_ui_seed: None,
            result_metrics: None,
            residual_plot_points: Vec::new(),
            spline_plot_curve: None,
            formula_svg_cache: Vec::new(),
            sampled_curve_cache: None,
            iteration_diagnostics: IterationDiagnostics::default(),
            status: Some(StatusMessage::Ready),
            #[cfg(not(target_arch = "wasm32"))]
            fit_worker_rx: None,
            #[cfg(not(target_arch = "wasm32"))]
            fit_cancel_flag: None,
            #[cfg(not(target_arch = "wasm32"))]
            discard_fit_worker_updates: false,
            #[cfg(target_arch = "wasm32")]
            wasm_fit_job: None,
        }
    }
}
