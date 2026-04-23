//! Центральные типы состояния приложения, снимки UI и сообщения рантайма фитинга.

use super::*;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Неизменяемое представление выбранного оптимизатора и его input-состояния.
pub(super) enum ActiveOptimizerView<'a> {
    Lbfgs {
        inputs: &'a LbfgsInputState,
        preset: OptimizerPreset,
    },
    NelderMead {
        inputs: &'a NelderMeadInputState,
        preset: OptimizerPreset,
    },
    SteepestDescent {
        inputs: &'a SteepestDescentInputState,
        preset: OptimizerPreset,
    },
    NewtonCg {
        inputs: &'a NewtonCgInputState,
        preset: OptimizerPreset,
    },
    Sgd {
        inputs: &'a SgdInputState,
        preset: OptimizerPreset,
    },
    Adam {
        inputs: &'a AdamInputState,
        preset: OptimizerPreset,
    },
}

impl ActiveOptimizerView<'_> {
    fn preset(self) -> OptimizerPreset {
        match self {
            Self::Lbfgs { preset, .. }
            | Self::NelderMead { preset, .. }
            | Self::SteepestDescent { preset, .. }
            | Self::NewtonCg { preset, .. }
            | Self::Sgd { preset, .. }
            | Self::Adam { preset, .. } => preset,
        }
    }

    fn config(self) -> Result<OptimizerConfig, String> {
        match self {
            Self::Lbfgs { inputs, .. } => inputs.to_config().map(OptimizerConfig::Lbfgs),
            Self::NelderMead { inputs, .. } => inputs.to_config().map(OptimizerConfig::NelderMead),
            Self::SteepestDescent { inputs, .. } => {
                inputs.to_config().map(OptimizerConfig::SteepestDescent)
            }
            Self::NewtonCg { inputs, .. } => inputs.to_config().map(OptimizerConfig::NewtonCg),
            Self::Sgd { inputs, .. } => inputs.to_config().map(OptimizerConfig::Sgd),
            Self::Adam { inputs, .. } => inputs.to_config().map(OptimizerConfig::Adam),
        }
    }
}

/// Изменяемое представление выбранного оптимизатора для применения preset-ов.
pub(super) enum ActiveOptimizerViewMut<'a> {
    Lbfgs {
        inputs: &'a mut LbfgsInputState,
        preset: &'a mut OptimizerPreset,
    },
    NelderMead {
        inputs: &'a mut NelderMeadInputState,
        preset: &'a mut OptimizerPreset,
    },
    SteepestDescent {
        inputs: &'a mut SteepestDescentInputState,
        preset: &'a mut OptimizerPreset,
    },
    NewtonCg {
        inputs: &'a mut NewtonCgInputState,
        preset: &'a mut OptimizerPreset,
    },
    Sgd {
        inputs: &'a mut SgdInputState,
        preset: &'a mut OptimizerPreset,
    },
    Adam {
        inputs: &'a mut AdamInputState,
        preset: &'a mut OptimizerPreset,
    },
}

impl ActiveOptimizerViewMut<'_> {
    fn set_preset(self, value: OptimizerPreset) {
        match self {
            Self::Lbfgs { preset, .. }
            | Self::NelderMead { preset, .. }
            | Self::SteepestDescent { preset, .. }
            | Self::NewtonCg { preset, .. }
            | Self::Sgd { preset, .. }
            | Self::Adam { preset, .. } => *preset = value,
        }
    }

    fn apply_preset(self, value: OptimizerPreset) {
        match self {
            Self::Lbfgs { inputs, preset } => {
                *inputs = LbfgsInputState::from_config(&lbfgs_config_from_preset(value));
                *preset = value;
            }
            Self::NelderMead { inputs, preset } => {
                *inputs = NelderMeadInputState::from_config(&nelder_mead_config_from_preset(value));
                *preset = value;
            }
            Self::SteepestDescent { inputs, preset } => {
                *inputs = SteepestDescentInputState::from_config(
                    &steepest_descent_config_from_preset(value),
                );
                *preset = value;
            }
            Self::NewtonCg { inputs, preset } => {
                *inputs = NewtonCgInputState::from_config(&newton_cg_config_from_preset(value));
                *preset = value;
            }
            Self::Sgd { inputs, preset } => {
                *inputs = SgdInputState::from_config(&sgd_config_from_preset(value));
                *preset = value;
            }
            Self::Adam { inputs, preset } => {
                *inputs = AdamInputState::from_config(&adam_config_from_preset(value));
                *preset = value;
            }
        }
    }
}

/// Легковесный отпечаток состояния правой панели для детекта изменений без аллокаций.
type RightPanelFitFingerprint = u64;

#[derive(Debug)]
/// Трасса одной итерации параметрического фитинга для replay/диагностики.
pub(super) struct ParametricIterationTraceEntry {
    pub(super) iteration: u64,
    pub(super) metrics: IterationMetricSnapshot,
    pub(super) params: CurveParams,
}

#[derive(Debug)]
/// Трасса одной итерации сплайнового фитинга для replay/диагностики.
pub(super) struct SplineIterationTraceEntry {
    pub(super) iteration: u64,
    pub(super) metrics: IterationMetricSnapshot,
    pub(super) knot_y: Vec<f64>,
    pub(super) curve: Vec<[f64; 2]>,
}

#[derive(Debug)]
/// Исходные данные для заполнения UI перед стартом worker-а.
pub(super) enum FitRunUiSeed {
    Parametric { initial_params: CurveParams },
    Spline { initial_curve: Vec<PlotPoint> },
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
/// Сообщения от фонового worker-а фитинга в UI-поток.
pub(super) enum FitWorkerMessage {
    Stopped,
    Finished {
        result: FitResult,
        trace: Vec<ParametricIterationTraceEntry>,
    },
    SplineFinished {
        result: SplineResult,
        metrics: IterationMetricSnapshot,
        trace: Vec<SplineIterationTraceEntry>,
    },
    Failed(String),
}

#[cfg(target_arch = "wasm32")]
/// WASM-вариант раннера без фонового потока.
pub(super) enum WasmFitRunner {
    Parametric {
        runner: IncrementalFitRunner,
        normalization: Option<ParametricNormalization>,
    },
    Spline(IncrementalSplineFitRunner),
}

#[cfg(target_arch = "wasm32")]
/// Состояние выполнения WASM-раннера между deferred/running фазами.
pub(super) enum WasmFitJob {
    Deferred(WasmFitRunner),
    Running(WasmFitRunner),
}

/// Состояние и UI-логика интерактивного приложения для подгонки кривых.
pub struct CurveFitApp {
    pub(super) points: PointsEditorState,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) points_file_import_dialog: FileDialog,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) points_file_import_last_directory: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) clipboard_import_request_pending: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) clipboard_import_requested_at: Option<Instant>,
    #[cfg(target_arch = "wasm32")]
    pub(super) clipboard_import_web_in_flight: bool,
    #[cfg(target_arch = "wasm32")]
    pub(super) clipboard_import_web_result: Rc<RefCell<Option<Result<String, String>>>>,
    #[cfg(target_arch = "wasm32")]
    pub(super) clipboard_copy_web_in_flight: bool,
    #[cfg(target_arch = "wasm32")]
    pub(super) clipboard_copy_web_result: Rc<RefCell<Option<Result<(), String>>>>,
    pub(super) selected_model: ModelChoice,
    pub(super) polynomial_degree: usize,
    pub(super) rational_degree: usize,
    pub(super) saturating_trend_tau_count: usize,
    pub(super) saturating_trend_tau_inputs: Vec<String>,
    pub(super) parameter_inputs: Vec<String>,
    pub(super) optimizer_method: OptimizerMethod,
    pub(super) optimizer_mode: OptimizerUiMode,
    pub(super) optimization_loss_metric: OptimizationLossMetric,
    pub(super) metric_quantization_enabled: bool,
    pub(super) metric_quantization_decimal_places: u8,
    pub(super) normalize_parametric_data: bool,
    pub(super) lbfgs_inputs: LbfgsInputState,
    pub(super) lbfgs_preset: OptimizerPreset,
    pub(super) nelder_mead_inputs: NelderMeadInputState,
    pub(super) nelder_mead_preset: OptimizerPreset,
    pub(super) steepest_descent_inputs: SteepestDescentInputState,
    pub(super) steepest_descent_preset: OptimizerPreset,
    pub(super) newton_cg_inputs: NewtonCgInputState,
    pub(super) newton_cg_preset: OptimizerPreset,
    pub(super) sgd_inputs: SgdInputState,
    pub(super) sgd_preset: OptimizerPreset,
    pub(super) adam_inputs: AdamInputState,
    pub(super) adam_preset: OptimizerPreset,
    pub(super) ui_language: UiLanguage,
    pub(super) plot_tool: PlotTool,
    pub(super) spray_points_per_second: usize,
    pub(super) spray_radius_rel: f64,
    pub(super) spray_brush: SprayBrush,
    pub(super) eraser_radius_rel: f64,
    pub(super) spray_seed: u64,
    pub(super) spray_points_budget: f64,
    pub(super) spray_last_emit_at: Option<Instant>,
    pub(super) fit_to_content_requested: bool,
    pub(super) center_origin_requested: bool,
    pub(super) origin_bottom_left_requested: bool,
    pub(super) last_plot_bounds: Option<PlotBounds>,
    pub(super) active_tool_bounds: Option<PlotBounds>,
    pub(super) panel: PanelState,
    pub(super) replay: ReplayState,
    pub(super) spline_knots: usize,
    pub(super) spline_knot_strategy: SplineKnotStrategy,
    pub(super) spline_extrapolation: SplineExtrapolation,
    pub(super) spline_duplicate_x_policy: SplineDuplicateXPolicy,
    pub(super) spline_initial_knot_y_inputs: Vec<String>,
    pub(super) auto_refit_enabled: bool,
    pub(super) auto_refit_pending_rerun: bool,
    pub(super) last_right_panel_fit_snapshot: Option<RightPanelFitFingerprint>,
    pub(super) fit_in_progress: bool,
    pub(super) fit_loss_metric: OptimizationLossMetric,
    pub(super) fit_metric_quantization: MetricQuantization,
    pub(super) fit_optimizer_method: OptimizerMethod,
    pub(super) fit_export_record: Option<FitExportRecord>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fit_export_file_dialog: FileDialog,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fit_export_last_directory: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fit_export_pending_json: Option<String>,
    pub(super) fit_preview_params: Option<CurveParams>,
    pub(super) fit_preview_iteration: Option<u64>,
    pub(super) fit_started_at: Option<Instant>,
    pub(super) last_fit_duration: Option<Duration>,
    pub(super) fit_result: Option<FitResult>,
    pub(super) spline_result: Option<SplineResult>,
    pub(super) active_fit_points: Option<Points>,
    pub(super) fit_run_ui_seed: Option<FitRunUiSeed>,
    pub(super) result_metrics: Option<ExtendedMetrics>,
    pub(super) residual_plot_points: Vec<PlotPoint>,
    pub(super) spline_plot_curve: Option<Arc<[PlotPoint]>>,
    pub(super) formula_svg_cache: Vec<FormulaSvgCache>,
    pub(super) sampled_curve_cache: Option<SampledCurveCache>,
    pub(super) iteration_diagnostics: IterationDiagnostics,
    pub(super) status: Option<StatusMessage>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fit_worker_rx: Option<Receiver<FitWorkerMessage>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fit_cancel_flag: Option<Arc<AtomicBool>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) discard_fit_worker_updates: bool,
    #[cfg(target_arch = "wasm32")]
    pub(super) wasm_fit_job: Option<WasmFitJob>,
}

impl CurveFitApp {
    fn randomized_init_values(&mut self, count: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            let random = self.next_unit_random();
            let value =
                PARAM_INIT_RANDOM_MIN + (PARAM_INIT_RANDOM_MAX - PARAM_INIT_RANDOM_MIN) * random;
            values.push(value);
        }
        values
    }

    fn resolved_parametric_family_for_init(&mut self) -> Option<CurveFamily> {
        let family = self.resolved_model().parametric_family();
        if family.is_none() {
            self.status = Some(StatusMessage::Error(
                "Current model is non-parametric and has no initial parameters".to_string(),
            ));
        }
        family
    }

    fn resolved_spline_family_and_init_config_for_init(
        &mut self,
    ) -> Option<(SplineFamilyKind, SplineConfig)> {
        let family_and_config = self.spline_family_and_init_config();
        if family_and_config.is_none() {
            self.status = Some(StatusMessage::Error(
                "Current model is parametric and has no spline parameters".to_string(),
            ));
        }
        family_and_config
    }

    fn apply_param_init_result(&mut self, params_result: Result<CurveParams, String>) {
        match params_result {
            Ok(params) => {
                if let Some(taus) = params.saturating_trend_taus() {
                    self.set_saturating_trend_tau_inputs(taus);
                }
                self.set_parameter_inputs_from_params(&params);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Ready);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
            }
        }
    }

    fn apply_spline_param_init_result(&mut self, values_result: Result<Vec<f64>, String>) {
        match values_result {
            Ok(values) => {
                self.set_spline_initial_knot_y_inputs(&values);
                self.clear_fit_outputs();
                self.status = Some(StatusMessage::Ready);
            }
            Err(error) => {
                self.status = Some(StatusMessage::Error(error));
            }
        }
    }

    pub(super) fn resolved_model(&self) -> ResolvedModel {
        ResolvedModel::from_choice(
            self.selected_model,
            self.polynomial_degree,
            self.rational_degree,
            self.saturating_trend_tau_count,
        )
    }

    pub(super) fn active_optimizer_view(&self) -> ActiveOptimizerView<'_> {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => ActiveOptimizerView::Lbfgs {
                inputs: &self.lbfgs_inputs,
                preset: self.lbfgs_preset,
            },
            OptimizerMethod::NelderMead => ActiveOptimizerView::NelderMead {
                inputs: &self.nelder_mead_inputs,
                preset: self.nelder_mead_preset,
            },
            OptimizerMethod::SteepestDescent => ActiveOptimizerView::SteepestDescent {
                inputs: &self.steepest_descent_inputs,
                preset: self.steepest_descent_preset,
            },
            OptimizerMethod::NewtonCg => ActiveOptimizerView::NewtonCg {
                inputs: &self.newton_cg_inputs,
                preset: self.newton_cg_preset,
            },
            OptimizerMethod::Sgd => ActiveOptimizerView::Sgd {
                inputs: &self.sgd_inputs,
                preset: self.sgd_preset,
            },
            OptimizerMethod::Adam => ActiveOptimizerView::Adam {
                inputs: &self.adam_inputs,
                preset: self.adam_preset,
            },
        }
    }

    pub(super) fn active_optimizer_view_mut(&mut self) -> ActiveOptimizerViewMut<'_> {
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => ActiveOptimizerViewMut::Lbfgs {
                inputs: &mut self.lbfgs_inputs,
                preset: &mut self.lbfgs_preset,
            },
            OptimizerMethod::NelderMead => ActiveOptimizerViewMut::NelderMead {
                inputs: &mut self.nelder_mead_inputs,
                preset: &mut self.nelder_mead_preset,
            },
            OptimizerMethod::SteepestDescent => ActiveOptimizerViewMut::SteepestDescent {
                inputs: &mut self.steepest_descent_inputs,
                preset: &mut self.steepest_descent_preset,
            },
            OptimizerMethod::NewtonCg => ActiveOptimizerViewMut::NewtonCg {
                inputs: &mut self.newton_cg_inputs,
                preset: &mut self.newton_cg_preset,
            },
            OptimizerMethod::Sgd => ActiveOptimizerViewMut::Sgd {
                inputs: &mut self.sgd_inputs,
                preset: &mut self.sgd_preset,
            },
            OptimizerMethod::Adam => ActiveOptimizerViewMut::Adam {
                inputs: &mut self.adam_inputs,
                preset: &mut self.adam_preset,
            },
        }
    }

    pub(super) fn selected_optimizer_preset(&self) -> OptimizerPreset {
        self.active_optimizer_view().preset()
    }

    pub(super) fn set_selected_optimizer_preset(&mut self, preset: OptimizerPreset) {
        self.active_optimizer_view_mut().set_preset(preset);
    }

    pub(super) fn apply_selected_optimizer_preset(&mut self, preset: OptimizerPreset) {
        self.active_optimizer_view_mut().apply_preset(preset);
    }

    pub(super) fn optimizer_config(&self) -> Result<OptimizerConfig, String> {
        self.active_optimizer_view().config()
    }

    fn hash_f64<H: Hasher>(hasher: &mut H, value: f64) {
        let normalized_bits = if value == 0.0 {
            0.0f64.to_bits()
        } else {
            value.to_bits()
        };
        normalized_bits.hash(hasher);
    }

    fn hash_active_optimizer_inputs<H: Hasher>(&self, hasher: &mut H) {
        std::mem::discriminant(&self.optimizer_method).hash(hasher);
        match self.optimizer_method {
            OptimizerMethod::Lbfgs => {
                self.lbfgs_inputs.history_size.hash(hasher);
                self.lbfgs_inputs.max_iters.hash(hasher);
                Self::hash_f64(hasher, self.lbfgs_inputs.tol_grad);
                Self::hash_f64(hasher, self.lbfgs_inputs.tol_cost);
                Self::hash_f64(hasher, self.lbfgs_inputs.c1);
                Self::hash_f64(hasher, self.lbfgs_inputs.c2);
                Self::hash_f64(hasher, self.lbfgs_inputs.step_min);
                Self::hash_f64(hasher, self.lbfgs_inputs.step_max);
                Self::hash_f64(hasher, self.lbfgs_inputs.width_tolerance);
            }
            OptimizerMethod::NelderMead => {
                self.nelder_mead_inputs.max_iters.hash(hasher);
                Self::hash_f64(hasher, self.nelder_mead_inputs.simplex_scale);
                Self::hash_f64(hasher, self.nelder_mead_inputs.sd_tolerance);
                Self::hash_f64(hasher, self.nelder_mead_inputs.alpha);
                Self::hash_f64(hasher, self.nelder_mead_inputs.gamma);
                Self::hash_f64(hasher, self.nelder_mead_inputs.rho);
                Self::hash_f64(hasher, self.nelder_mead_inputs.sigma);
            }
            OptimizerMethod::SteepestDescent => {
                self.steepest_descent_inputs.max_iters.hash(hasher);
                Self::hash_f64(hasher, self.steepest_descent_inputs.c1);
                Self::hash_f64(hasher, self.steepest_descent_inputs.c2);
                Self::hash_f64(hasher, self.steepest_descent_inputs.step_min);
                Self::hash_f64(hasher, self.steepest_descent_inputs.step_max);
                Self::hash_f64(hasher, self.steepest_descent_inputs.width_tolerance);
            }
            OptimizerMethod::NewtonCg => {
                self.newton_cg_inputs.max_iters.hash(hasher);
                Self::hash_f64(hasher, self.newton_cg_inputs.tol);
                Self::hash_f64(hasher, self.newton_cg_inputs.curvature_threshold);
                Self::hash_f64(hasher, self.newton_cg_inputs.c1);
                Self::hash_f64(hasher, self.newton_cg_inputs.c2);
                Self::hash_f64(hasher, self.newton_cg_inputs.step_min);
                Self::hash_f64(hasher, self.newton_cg_inputs.step_max);
                Self::hash_f64(hasher, self.newton_cg_inputs.width_tolerance);
            }
            OptimizerMethod::Sgd => {
                self.sgd_inputs.max_iters.hash(hasher);
                Self::hash_f64(hasher, self.sgd_inputs.learning_rate);
            }
            OptimizerMethod::Adam => {
                self.adam_inputs.max_iters.hash(hasher);
                Self::hash_f64(hasher, self.adam_inputs.learning_rate);
            }
        }
    }

    pub(super) fn capture_right_panel_fit_snapshot(&self) -> RightPanelFitFingerprint {
        let mut hasher = DefaultHasher::new();
        std::mem::discriminant(&self.selected_model).hash(&mut hasher);
        self.polynomial_degree.hash(&mut hasher);
        self.rational_degree.hash(&mut hasher);
        self.saturating_trend_tau_count.hash(&mut hasher);
        self.saturating_trend_tau_inputs.hash(&mut hasher);
        self.parameter_inputs.hash(&mut hasher);
        self.spline_knots.hash(&mut hasher);
        std::mem::discriminant(&self.spline_knot_strategy).hash(&mut hasher);
        std::mem::discriminant(&self.spline_extrapolation).hash(&mut hasher);
        std::mem::discriminant(&self.spline_duplicate_x_policy).hash(&mut hasher);
        self.spline_initial_knot_y_inputs.hash(&mut hasher);
        std::mem::discriminant(&self.optimization_loss_metric).hash(&mut hasher);
        self.metric_quantization_enabled.hash(&mut hasher);
        self.metric_quantization_decimal_places.hash(&mut hasher);
        self.hash_active_optimizer_inputs(&mut hasher);
        hasher.finish()
    }

    pub(super) fn track_right_panel_fit_changes_and_maybe_refit(&mut self) {
        let snapshot = self.capture_right_panel_fit_snapshot();
        let Some(last_snapshot) = self.last_right_panel_fit_snapshot.as_mut() else {
            self.last_right_panel_fit_snapshot = Some(snapshot);
            return;
        };

        if *last_snapshot == snapshot {
            return;
        }

        *last_snapshot = snapshot;

        if !self.auto_refit_enabled {
            return;
        }

        if self.fit_in_progress {
            self.auto_refit_pending_rerun = true;
            return;
        }

        self.run_fit();
    }

    pub(super) fn maybe_run_pending_auto_refit(&mut self) {
        if !self.auto_refit_enabled {
            self.auto_refit_pending_rerun = false;
            return;
        }

        if !self.auto_refit_pending_rerun || self.fit_in_progress {
            return;
        }

        self.auto_refit_pending_rerun = false;
        self.run_fit();
    }

    pub(super) fn auto_spline_samples(points_len: usize, knots: usize) -> usize {
        // На больших датасетах используем более плотную дискретизацию,
        // но ограничиваем верхний порог ради отзывчивости UI.
        let by_knots = knots.saturating_mul(SPLINE_AUTO_SAMPLES_PER_KNOT);
        let by_points = points_len.saturating_mul(SPLINE_AUTO_SAMPLES_PER_POINT);
        by_knots
            .max(by_points)
            .clamp(SPLINE_AUTO_SAMPLES_MIN, SPLINE_AUTO_SAMPLES_MAX)
    }

    pub(super) fn spline_config_for_model(
        &self,
        model: ResolvedModel,
        points_len: usize,
    ) -> Option<SplineConfig> {
        let min_knots = model.spline_min_knots()?;
        let knots = self.spline_knots.max(min_knots);
        Some(
            SplineConfig {
                knots,
                samples: Self::auto_spline_samples(points_len, knots),
                knot_strategy: self.spline_knot_strategy,
                extrapolation: self.spline_extrapolation,
                duplicate_x_policy: self.spline_duplicate_x_policy,
            }
            .normalized(),
        )
    }

    pub(super) fn cached_formula_svg(
        &mut self,
        formula: &str,
        dark_mode: bool,
    ) -> Result<(String, Arc<[u8]>), String> {
        if let Some(cache) = self
            .formula_svg_cache
            .iter()
            .find(|cache| cache.formula == formula && cache.dark_mode == dark_mode)
        {
            return cache
                .render_result
                .as_ref()
                .map(|bytes| (cache.uri.clone(), Arc::clone(bytes)))
                .map_err(|error| error.clone());
        }

        let uri = formula_svg_uri(formula, dark_mode);
        let render_result = match formula_svg_bytes(formula, dark_mode) {
            Ok(bytes) => Ok(Arc::<[u8]>::from(bytes)),
            Err(error) => {
                eprintln!("Failed to render formula SVG: {error}");
                Err(error)
            }
        };
        self.formula_svg_cache.push(FormulaSvgCache {
            formula: formula.to_string(),
            dark_mode,
            uri: uri.clone(),
            render_result: render_result.clone(),
        });
        render_result.map(|bytes| (uri, bytes))
    }

    pub(super) fn cached_sampled_curve(
        &mut self,
        params: &CurveParams,
        x_min: f64,
        x_max: f64,
        samples: usize,
    ) -> Arc<[PlotPoint]> {
        let x_min_bits = x_min.to_bits();
        let x_max_bits = x_max.to_bits();
        if let Some(cache) = &self.sampled_curve_cache
            && cache.params == *params
            && cache.x_min_bits == x_min_bits
            && cache.x_max_bits == x_max_bits
            && cache.samples == samples
        {
            return Arc::clone(&cache.curve);
        }

        let curve: Arc<[PlotPoint]> = sample_curve(params, x_min, x_max, samples)
            .into_iter()
            .map(|[x, y]| PlotPoint::new(x, y))
            .collect::<Vec<_>>()
            .into();
        self.sampled_curve_cache = Some(SampledCurveCache {
            params: params.clone(),
            x_min_bits,
            x_max_bits,
            samples,
            curve: Arc::clone(&curve),
        });
        curve
    }

    pub(super) fn sync_parameter_inputs(&mut self) {
        if let Some(family) = self.resolved_model().parametric_family() {
            let default_params = family.default_params();
            self.set_parameter_inputs_from_params(&default_params);
        } else {
            self.parameter_inputs.clear();
        }
    }

    pub(super) fn set_parameter_inputs_from_params(&mut self, params: &CurveParams) {
        self.parameter_inputs = params_to_input_strings(params);
    }

    pub(super) fn set_saturating_trend_tau_inputs(&mut self, values: &[f64]) {
        self.saturating_trend_tau_inputs = tau_grid_to_input_strings(values);
    }

    pub(super) fn sync_spline_initial_knot_y_inputs(&mut self, knot_count: usize) {
        if self.spline_initial_knot_y_inputs.len() < knot_count {
            self.spline_initial_knot_y_inputs
                .resize_with(knot_count, || "0.0".to_string());
        } else {
            self.spline_initial_knot_y_inputs.truncate(knot_count);
        }
    }

    pub(super) fn set_spline_initial_knot_y_inputs(&mut self, values: &[f64]) {
        self.spline_initial_knot_y_inputs = values.iter().map(|value| value.to_string()).collect();
    }

    pub(super) fn selected_metric_quantization(&self) -> Result<MetricQuantization, String> {
        MetricQuantization::from_ui_state(
            self.metric_quantization_enabled,
            self.metric_quantization_decimal_places,
        )
    }

    // Таймер относится только к одному запуску фита:
    // новый старт сбрасывает прошлый замер, а при успехе фиксируется длительность.
    pub(super) fn start_fit_timer(&mut self) {
        self.fit_started_at = Some(Instant::now());
        self.last_fit_duration = None;
    }

    pub(super) fn complete_fit_timer_successfully(&mut self) {
        self.last_fit_duration = self
            .fit_started_at
            .take()
            .map(|started_at| Instant::now().saturating_duration_since(started_at));
    }

    pub(super) fn reset_fit_timer(&mut self) {
        self.fit_started_at = None;
        self.last_fit_duration = None;
    }

    pub(super) fn format_fit_duration(duration: Duration) -> String {
        if duration < Duration::from_secs(1) {
            format!("{} ms", duration.as_millis())
        } else {
            format!("{:.2} s", duration.as_secs_f64())
        }
    }

    pub(super) fn clear_fit_outputs(&mut self) {
        self.cancel_fit_and_discard_updates();
        self.reset_fit_timer();
        self.fit_result = None;
        self.spline_result = None;
        self.clear_fit_export_state();
        self.active_fit_points = None;
        self.fit_run_ui_seed = None;
        self.result_metrics = None;
        self.residual_plot_points.clear();
        self.spline_plot_curve = None;
        self.sampled_curve_cache = None;
        self.iteration_diagnostics.clear();
        self.panel.diagnostics_hide_non_loss_by_default_pending = true;
        self.clear_fit_preview();
        self.clear_replay_state();
    }

    pub(super) fn spline_family_and_init_config(&self) -> Option<(SplineFamilyKind, SplineConfig)> {
        let model = self.resolved_model();
        let family = model.spline_family()?;
        let config = self.spline_config_for_model(model, 2)?;
        Some((family, config))
    }

    pub(super) fn build_randomized_spline_initial_knot_y(&mut self, knot_count: usize) -> Vec<f64> {
        self.randomized_init_values(knot_count)
    }

    pub(super) fn build_data_based_initial_params(
        &mut self,
        family: CurveFamily,
    ) -> Result<CurveParams, String> {
        if !ParamInitMethod::DataBased.is_supported_for_family(family) {
            return Err(format!(
                "Data-based initialization is not supported for family {family}"
            ));
        }

        let points = self.parse_points_strict()?;
        family
            .validate_points(&points)
            .map_err(|error| error.to_string())?;
        let tau_grid = self.parsed_saturating_trend_tau_grid()?;
        data_based_params_for_family(family, &points, tau_grid.as_ref())
    }

    pub(super) fn build_randomized_initial_params(
        &mut self,
        family: CurveFamily,
    ) -> Result<CurveParams, String> {
        if !ParamInitMethod::Randomized.is_supported_for_family(family) {
            return Err(format!(
                "Randomized initialization is not supported for family {family}"
            ));
        }

        let values = self.randomized_init_values(family.parameter_count());
        let tau_grid = self.parsed_saturating_trend_tau_grid()?;
        CurveParams::try_from_slice_with_tau_grid(family, &values, tau_grid.as_ref())
            .map_err(|error| error.to_string())
    }

    pub(super) fn has_fitted_params_for_family(&self, family: CurveFamily) -> bool {
        self.fit_result
            .as_ref()
            .is_some_and(|result| result.family == family)
    }

    pub(super) fn build_fitted_initial_params(
        &self,
        family: CurveFamily,
    ) -> Result<CurveParams, String> {
        let Some(result) = &self.fit_result else {
            return Err("No fitted model parameters are available for initialization".to_string());
        };

        if result.family != family {
            return Err(format!(
                "Fitted model family mismatch: expected {family}, got {}",
                result.family
            ));
        }

        Ok(result.params.clone())
    }

    pub(super) fn apply_fitted_param_init(&mut self) {
        let Some(family) = self.resolved_parametric_family_for_init() else {
            return;
        };
        self.apply_param_init_result(self.build_fitted_initial_params(family));
    }

    pub(super) fn apply_param_init_method(&mut self, method: ParamInitMethod) {
        let Some(family) = self.resolved_parametric_family_for_init() else {
            return;
        };

        if !method.is_supported_for_family(family) {
            self.status = Some(StatusMessage::Error(format!(
                "Initialization method '{}' is not supported for family {family}",
                param_init_method_name_en(method),
            )));
            return;
        }

        let params_result = match method {
            ParamInitMethod::Default => {
                self.parsed_saturating_trend_tau_grid()
                    .and_then(|tau_grid| {
                        CurveParams::try_from_slice_with_tau_grid(
                            family,
                            &family.default_params().values(),
                            tau_grid.as_ref(),
                        )
                        .map_err(|error| error.to_string())
                    })
            }
            ParamInitMethod::DataBased => self.build_data_based_initial_params(family),
            ParamInitMethod::Randomized => self.build_randomized_initial_params(family),
        };

        self.apply_param_init_result(params_result);
    }

    pub(super) fn apply_spline_param_init_method(&mut self, method: ParamInitMethod) {
        let Some((family, config)) = self.resolved_spline_family_and_init_config_for_init() else {
            return;
        };

        self.sync_spline_initial_knot_y_inputs(config.knots);
        let values_result = match method {
            ParamInitMethod::Default => Ok(vec![0.0; config.knots]),
            ParamInitMethod::DataBased => {
                let points = match self.parse_points_strict() {
                    Ok(points) => points,
                    Err(error) => {
                        self.status = Some(StatusMessage::Error(error));
                        return;
                    }
                };
                default_spline_initial_knot_y(&points, family, config)
                    .map_err(|error| error.to_string())
            }
            ParamInitMethod::Randomized => {
                Ok(self.build_randomized_spline_initial_knot_y(config.knots))
            }
        };

        self.apply_spline_param_init_result(values_result);
    }
}
