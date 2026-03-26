use crate::domain::{
    AdamConfig, LbfgsConfig, NelderMeadConfig, OptimizerMethod, SgdConfig, SteepestDescentConfig,
};

use super::{C1_MIN, C2_MAX, STEP_MAX_MAX, STEP_MIN_MIN, UiLanguage};

pub(super) fn optimizer_method_label(
    language: UiLanguage,
    method: OptimizerMethod,
) -> &'static str {
    match (language, method) {
        (UiLanguage::English, OptimizerMethod::Lbfgs) => "LBFGS",
        (UiLanguage::English, OptimizerMethod::NelderMead) => "Nelder-Mead",
        (UiLanguage::English, OptimizerMethod::SteepestDescent) => "Steepest Descent",
        (UiLanguage::English, OptimizerMethod::Sgd) => "SGD",
        (UiLanguage::English, OptimizerMethod::Adam) => "Adam",
        (UiLanguage::Russian, OptimizerMethod::Lbfgs) => "LBFGS",
        (UiLanguage::Russian, OptimizerMethod::NelderMead) => "Нелдер-Мид",
        (UiLanguage::Russian, OptimizerMethod::SteepestDescent) => "Наискорейший спуск",
        (UiLanguage::Russian, OptimizerMethod::Sgd) => "SGD",
        (UiLanguage::Russian, OptimizerMethod::Adam) => "Adam",
    }
}

pub(super) fn optimizer_preset_label(
    language: UiLanguage,
    preset: OptimizerPreset,
) -> &'static str {
    match (language, preset) {
        (UiLanguage::English, OptimizerPreset::Fast) => "Fast",
        (UiLanguage::English, OptimizerPreset::Balanced) => "Balanced",
        (UiLanguage::English, OptimizerPreset::Precise) => "Precise",
        (UiLanguage::English, OptimizerPreset::Custom) => "Custom",
        (UiLanguage::Russian, OptimizerPreset::Fast) => "Быстрый",
        (UiLanguage::Russian, OptimizerPreset::Balanced) => "Сбалансированный",
        (UiLanguage::Russian, OptimizerPreset::Precise) => "Точный",
        (UiLanguage::Russian, OptimizerPreset::Custom) => "Произвольный",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum OptimizerUiMode {
    #[default]
    Basic,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum OptimizerPreset {
    Fast,
    #[default]
    Balanced,
    Precise,
    Custom,
}

impl OptimizerPreset {
    pub(super) const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::Precise];
}

pub(super) fn lbfgs_config_from_preset(preset: OptimizerPreset) -> LbfgsConfig {
    match preset {
        OptimizerPreset::Fast => {
            LbfgsConfig::try_new(5, 80, 1e-6, 1e-9, 1e-4, 0.9, 1e-10, 1.0, 1e-8)
                .expect("fast LBFGS preset must be valid")
        }
        OptimizerPreset::Balanced => LbfgsConfig::default(),
        OptimizerPreset::Precise => {
            LbfgsConfig::try_new(10, 500, 1e-10, 1e-14, 1e-4, 0.95, 1e-12, 10.0, 1e-12)
                .expect("precise LBFGS preset must be valid")
        }
        OptimizerPreset::Custom => LbfgsConfig::default(),
    }
}

pub(super) fn infer_lbfgs_preset(config: &LbfgsConfig) -> OptimizerPreset {
    for preset in OptimizerPreset::ALL {
        if &lbfgs_config_from_preset(preset) == config {
            return preset;
        }
    }
    OptimizerPreset::Custom
}

pub(super) fn nelder_mead_config_from_preset(preset: OptimizerPreset) -> NelderMeadConfig {
    match preset {
        OptimizerPreset::Fast => NelderMeadConfig::try_new(150, 0.10, 1e-6, 1.0, 2.0, 0.5, 0.5)
            .expect("fast Nelder-Mead preset must be valid"),
        OptimizerPreset::Balanced => NelderMeadConfig::default(),
        OptimizerPreset::Precise => {
            NelderMeadConfig::try_new(1_200, 0.02, 1e-10, 1.0, 2.0, 0.5, 0.5)
                .expect("precise Nelder-Mead preset must be valid")
        }
        OptimizerPreset::Custom => NelderMeadConfig::default(),
    }
}

pub(super) fn infer_nelder_mead_preset(config: &NelderMeadConfig) -> OptimizerPreset {
    for preset in OptimizerPreset::ALL {
        if &nelder_mead_config_from_preset(preset) == config {
            return preset;
        }
    }
    OptimizerPreset::Custom
}

pub(super) fn steepest_descent_config_from_preset(
    preset: OptimizerPreset,
) -> SteepestDescentConfig {
    match preset {
        OptimizerPreset::Fast => SteepestDescentConfig::try_new(120, 1e-4, 0.9, 1e-10, 1.0, 1e-8)
            .expect("fast steepest descent preset must be valid"),
        OptimizerPreset::Balanced => SteepestDescentConfig::default(),
        OptimizerPreset::Precise => {
            SteepestDescentConfig::try_new(1_000, 1e-4, 0.95, 1e-12, 10.0, 1e-12)
                .expect("precise steepest descent preset must be valid")
        }
        OptimizerPreset::Custom => SteepestDescentConfig::default(),
    }
}

pub(super) fn infer_steepest_descent_preset(config: &SteepestDescentConfig) -> OptimizerPreset {
    for preset in OptimizerPreset::ALL {
        if &steepest_descent_config_from_preset(preset) == config {
            return preset;
        }
    }
    OptimizerPreset::Custom
}

pub(super) fn sgd_config_from_preset(preset: OptimizerPreset) -> SgdConfig {
    match preset {
        OptimizerPreset::Fast => {
            SgdConfig::try_new(250, 3e-2).expect("fast SGD preset must be valid")
        }
        OptimizerPreset::Balanced => SgdConfig::default(),
        OptimizerPreset::Precise => {
            SgdConfig::try_new(4_000, 3e-3).expect("precise SGD preset must be valid")
        }
        OptimizerPreset::Custom => SgdConfig::default(),
    }
}

pub(super) fn infer_sgd_preset(config: &SgdConfig) -> OptimizerPreset {
    for preset in OptimizerPreset::ALL {
        if &sgd_config_from_preset(preset) == config {
            return preset;
        }
    }
    OptimizerPreset::Custom
}

pub(super) fn adam_config_from_preset(preset: OptimizerPreset) -> AdamConfig {
    match preset {
        OptimizerPreset::Fast => {
            AdamConfig::try_new(200, 2e-2).expect("fast Adam preset must be valid")
        }
        OptimizerPreset::Balanced => AdamConfig::default(),
        OptimizerPreset::Precise => {
            AdamConfig::try_new(3_000, 1e-3).expect("precise Adam preset must be valid")
        }
        OptimizerPreset::Custom => AdamConfig::default(),
    }
}

pub(super) fn infer_adam_preset(config: &AdamConfig) -> OptimizerPreset {
    for preset in OptimizerPreset::ALL {
        if &adam_config_from_preset(preset) == config {
            return preset;
        }
    }
    OptimizerPreset::Custom
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LbfgsInputState {
    pub(super) history_size: usize,
    pub(super) max_iters: u64,
    pub(super) tol_grad: f64,
    pub(super) tol_cost: f64,
    pub(super) c1: f64,
    pub(super) c2: f64,
    pub(super) step_min: f64,
    pub(super) step_max: f64,
    pub(super) width_tolerance: f64,
}

impl LbfgsInputState {
    pub(super) fn from_config(config: &LbfgsConfig) -> Self {
        Self {
            history_size: config.history_size,
            max_iters: config.max_iters,
            tol_grad: config.tol_grad,
            tol_cost: config.tol_cost,
            c1: config.c1,
            c2: config.c2,
            step_min: config.step_min,
            step_max: config.step_max,
            width_tolerance: config.width_tolerance,
        }
    }

    pub(super) fn normalize_after_ui(&mut self) {
        self.c1 = self.c1.clamp(C1_MIN, C2_MAX - 1e-4);
        self.c2 = self.c2.clamp(self.c1 + 1e-4, C2_MAX);

        self.step_min = self.step_min.clamp(STEP_MIN_MIN, STEP_MAX_MAX - 1e-6);
        self.step_max = self.step_max.clamp(self.step_min + 1e-6, STEP_MAX_MAX);
    }

    pub(super) fn to_config(&self) -> Result<LbfgsConfig, String> {
        LbfgsConfig::try_new(
            self.history_size,
            self.max_iters,
            self.tol_grad,
            self.tol_cost,
            self.c1,
            self.c2,
            self.step_min,
            self.step_max,
            self.width_tolerance,
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NelderMeadInputState {
    pub(super) max_iters: u64,
    pub(super) simplex_scale: f64,
    pub(super) sd_tolerance: f64,
    pub(super) alpha: f64,
    pub(super) gamma: f64,
    pub(super) rho: f64,
    pub(super) sigma: f64,
}

impl NelderMeadInputState {
    pub(super) fn from_config(config: &NelderMeadConfig) -> Self {
        Self {
            max_iters: config.max_iters,
            simplex_scale: config.simplex_scale,
            sd_tolerance: config.sd_tolerance,
            alpha: config.alpha,
            gamma: config.gamma,
            rho: config.rho,
            sigma: config.sigma,
        }
    }

    pub(super) fn normalize_after_ui(&mut self) {
        self.simplex_scale = self.simplex_scale.clamp(1e-4, 2.0);
        self.sd_tolerance = self.sd_tolerance.clamp(1e-14, 1e-2);
        self.alpha = self.alpha.clamp(1e-3, 5.0);
        self.gamma = self.gamma.clamp(1.0001, 5.0);
        self.rho = self.rho.clamp(1e-4, 0.5);
        self.sigma = self.sigma.clamp(1e-4, 1.0);
    }

    pub(super) fn to_config(&self) -> Result<NelderMeadConfig, String> {
        NelderMeadConfig::try_new(
            self.max_iters,
            self.simplex_scale,
            self.sd_tolerance,
            self.alpha,
            self.gamma,
            self.rho,
            self.sigma,
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SteepestDescentInputState {
    pub(super) max_iters: u64,
    pub(super) c1: f64,
    pub(super) c2: f64,
    pub(super) step_min: f64,
    pub(super) step_max: f64,
    pub(super) width_tolerance: f64,
}

impl SteepestDescentInputState {
    pub(super) fn from_config(config: &SteepestDescentConfig) -> Self {
        Self {
            max_iters: config.max_iters,
            c1: config.c1,
            c2: config.c2,
            step_min: config.step_min,
            step_max: config.step_max,
            width_tolerance: config.width_tolerance,
        }
    }

    pub(super) fn normalize_after_ui(&mut self) {
        self.c1 = self.c1.clamp(C1_MIN, C2_MAX - 1e-4);
        self.c2 = self.c2.clamp(self.c1 + 1e-4, C2_MAX);
        self.step_min = self.step_min.clamp(STEP_MIN_MIN, STEP_MAX_MAX - 1e-6);
        self.step_max = self.step_max.clamp(self.step_min + 1e-6, STEP_MAX_MAX);
    }

    pub(super) fn to_config(&self) -> Result<SteepestDescentConfig, String> {
        SteepestDescentConfig::try_new(
            self.max_iters,
            self.c1,
            self.c2,
            self.step_min,
            self.step_max,
            self.width_tolerance,
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SgdInputState {
    pub(super) max_iters: u64,
    pub(super) learning_rate: f64,
}

impl SgdInputState {
    pub(super) fn from_config(config: &SgdConfig) -> Self {
        Self {
            max_iters: config.max_iters,
            learning_rate: config.learning_rate,
        }
    }

    pub(super) fn normalize_after_ui(&mut self) {
        self.learning_rate = self.learning_rate.clamp(1e-6, 1.0);
    }

    pub(super) fn to_config(&self) -> Result<SgdConfig, String> {
        SgdConfig::try_new(self.max_iters, self.learning_rate).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct AdamInputState {
    pub(super) max_iters: u64,
    pub(super) learning_rate: f64,
}

impl AdamInputState {
    pub(super) fn from_config(config: &AdamConfig) -> Self {
        Self {
            max_iters: config.max_iters,
            learning_rate: config.learning_rate,
        }
    }

    pub(super) fn normalize_after_ui(&mut self) {
        self.learning_rate = self.learning_rate.clamp(1e-6, 1.0);
    }

    pub(super) fn to_config(&self) -> Result<AdamConfig, String> {
        AdamConfig::try_new(self.max_iters, self.learning_rate).map_err(|error| error.to_string())
    }
}
