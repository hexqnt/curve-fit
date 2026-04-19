//! Типобезопасные конфигурации оптимизаторов с локальной проверкой инвариантов.

use super::InputError;

#[derive(Debug, Clone, PartialEq)]
/// Параметры L-BFGS и line-search с проверяемыми инвариантами.
pub struct LbfgsConfig {
    pub history_size: usize,
    pub max_iters: u64,
    pub tol_grad: f64,
    pub tol_cost: f64,
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    pub width_tolerance: f64,
}

impl LbfgsConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        history_size: usize,
        max_iters: u64,
        tol_grad: f64,
        tol_cost: f64,
        c1: f64,
        c2: f64,
        step_min: f64,
        step_max: f64,
        width_tolerance: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            history_size,
            max_iters,
            tol_grad,
            tol_cost,
            c1,
            c2,
            step_min,
            step_max,
            width_tolerance,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.history_size == 0 {
            return Err(InputError::InvalidLbfgsConfig(
                "history_size must be greater than 0",
            ));
        }
        if self.max_iters == 0 {
            return Err(InputError::InvalidLbfgsConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.tol_grad.is_finite() || self.tol_grad < 0.0 {
            return Err(InputError::InvalidLbfgsConfig(
                "tol_grad must be finite and >= 0",
            ));
        }
        if !self.tol_cost.is_finite() || self.tol_cost < 0.0 {
            return Err(InputError::InvalidLbfgsConfig(
                "tol_cost must be finite and >= 0",
            ));
        }
        if !self.c1.is_finite()
            || !self.c2.is_finite()
            || self.c1 <= 0.0
            || self.c1 >= self.c2
            || self.c2 >= 1.0
        {
            return Err(InputError::InvalidLbfgsConfig(
                "c1 and c2 must satisfy 0 < c1 < c2 < 1",
            ));
        }
        if !self.step_min.is_finite()
            || !self.step_max.is_finite()
            || self.step_min < 0.0
            || self.step_max <= self.step_min
        {
            return Err(InputError::InvalidLbfgsConfig(
                "step bounds must satisfy 0 <= step_min < step_max",
            ));
        }
        if !self.width_tolerance.is_finite() || self.width_tolerance < 0.0 {
            return Err(InputError::InvalidLbfgsConfig(
                "width_tolerance must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            history_size: 7,
            max_iters: 200,
            tol_grad: 1e-8,
            tol_cost: 1e-12,
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-12,
            step_max: 10.0,
            width_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры метода Nelder-Mead с проверяемыми инвариантами.
pub struct NelderMeadConfig {
    pub max_iters: u64,
    pub simplex_scale: f64,
    pub sd_tolerance: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub rho: f64,
    pub sigma: f64,
}

impl NelderMeadConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        max_iters: u64,
        simplex_scale: f64,
        sd_tolerance: f64,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            simplex_scale,
            sd_tolerance,
            alpha,
            gamma,
            rho,
            sigma,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.simplex_scale.is_finite() || self.simplex_scale <= 0.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "simplex_scale must be finite and > 0",
            ));
        }
        if !self.sd_tolerance.is_finite() || self.sd_tolerance < 0.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "sd_tolerance must be finite and >= 0",
            ));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "alpha must be finite and > 0",
            ));
        }
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "gamma must be finite and > 1",
            ));
        }
        if !self.rho.is_finite() || self.rho <= 0.0 || self.rho > 0.5 {
            return Err(InputError::InvalidNelderMeadConfig(
                "rho must be finite and in (0, 0.5]",
            ));
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 || self.sigma > 1.0 {
            return Err(InputError::InvalidNelderMeadConfig(
                "sigma must be finite and in (0, 1]",
            ));
        }
        Ok(())
    }
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iters: 400,
            simplex_scale: 0.05,
            sd_tolerance: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры steepest descent с line-search и проверяемыми инвариантами.
pub struct SteepestDescentConfig {
    pub max_iters: u64,
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    pub width_tolerance: f64,
}

impl SteepestDescentConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        max_iters: u64,
        c1: f64,
        c2: f64,
        step_min: f64,
        step_max: f64,
        width_tolerance: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            c1,
            c2,
            step_min,
            step_max,
            width_tolerance,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidSteepestDescentConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.c1.is_finite()
            || !self.c2.is_finite()
            || self.c1 <= 0.0
            || self.c1 >= self.c2
            || self.c2 >= 1.0
        {
            return Err(InputError::InvalidSteepestDescentConfig(
                "c1 and c2 must satisfy 0 < c1 < c2 < 1",
            ));
        }
        if !self.step_min.is_finite()
            || !self.step_max.is_finite()
            || self.step_min < 0.0
            || self.step_max <= self.step_min
        {
            return Err(InputError::InvalidSteepestDescentConfig(
                "step bounds must satisfy 0 <= step_min < step_max",
            ));
        }
        if !self.width_tolerance.is_finite() || self.width_tolerance < 0.0 {
            return Err(InputError::InvalidSteepestDescentConfig(
                "width_tolerance must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl Default for SteepestDescentConfig {
    fn default() -> Self {
        Self {
            max_iters: 300,
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-12,
            step_max: 10.0,
            width_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры Newton-CG с line-search и проверяемыми инвариантами.
pub struct NewtonCgConfig {
    pub max_iters: u64,
    pub tol: f64,
    pub curvature_threshold: f64,
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
    pub width_tolerance: f64,
}

impl NewtonCgConfig {
    #[allow(clippy::too_many_arguments)]
    /// Создает конфигурацию и валидирует все ограничения аргументов.
    pub fn try_new(
        max_iters: u64,
        tol: f64,
        curvature_threshold: f64,
        c1: f64,
        c2: f64,
        step_min: f64,
        step_max: f64,
        width_tolerance: f64,
    ) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            tol,
            curvature_threshold,
            c1,
            c2,
            step_min,
            step_max,
            width_tolerance,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "tol must be finite and > 0",
            ));
        }
        if !self.curvature_threshold.is_finite() || self.curvature_threshold < 0.0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "curvature_threshold must be finite and >= 0",
            ));
        }
        if !self.c1.is_finite()
            || !self.c2.is_finite()
            || self.c1 <= 0.0
            || self.c1 >= self.c2
            || self.c2 >= 1.0
        {
            return Err(InputError::InvalidNewtonCgConfig(
                "c1 and c2 must satisfy 0 < c1 < c2 < 1",
            ));
        }
        if !self.step_min.is_finite()
            || !self.step_max.is_finite()
            || self.step_min < 0.0
            || self.step_max <= self.step_min
        {
            return Err(InputError::InvalidNewtonCgConfig(
                "step bounds must satisfy 0 <= step_min < step_max",
            ));
        }
        if !self.width_tolerance.is_finite() || self.width_tolerance < 0.0 {
            return Err(InputError::InvalidNewtonCgConfig(
                "width_tolerance must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl Default for NewtonCgConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol: 1e-10,
            curvature_threshold: 0.0,
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-12,
            step_max: 10.0,
            width_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры стохастического градиентного спуска (SGD).
pub struct SgdConfig {
    pub max_iters: u64,
    pub learning_rate: f64,
}

impl SgdConfig {
    /// Создает конфигурацию и валидирует ограничения аргументов.
    pub fn try_new(max_iters: u64, learning_rate: f64) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            learning_rate,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidSgdConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(InputError::InvalidSgdConfig(
                "learning_rate must be finite and > 0",
            ));
        }
        Ok(())
    }
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            max_iters: 1_000,
            learning_rate: 1e-2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Параметры оптимизатора Adam.
pub struct AdamConfig {
    pub max_iters: u64,
    pub learning_rate: f64,
}

impl AdamConfig {
    /// Создает конфигурацию и валидирует ограничения аргументов.
    pub fn try_new(max_iters: u64, learning_rate: f64) -> Result<Self, InputError> {
        let config = Self {
            max_iters,
            learning_rate,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), InputError> {
        if self.max_iters == 0 {
            return Err(InputError::InvalidAdamConfig(
                "max_iters must be greater than 0",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(InputError::InvalidAdamConfig(
                "learning_rate must be finite and > 0",
            ));
        }
        Ok(())
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            max_iters: 800,
            learning_rate: 5e-3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Метод оптимизации для подгонки параметрических моделей и сплайнов.
pub enum OptimizerMethod {
    #[default]
    Lbfgs,
    NelderMead,
    SteepestDescent,
    NewtonCg,
    Sgd,
    Adam,
}

impl OptimizerMethod {
    /// Полный список методов для UI и переборов.
    pub const ALL: [Self; 6] = [
        Self::Lbfgs,
        Self::NelderMead,
        Self::SteepestDescent,
        Self::NewtonCg,
        Self::Sgd,
        Self::Adam,
    ];
}

#[derive(Debug, Clone, PartialEq)]
/// Объединенная конфигурация оптимизатора.
pub enum OptimizerConfig {
    Lbfgs(LbfgsConfig),
    NelderMead(NelderMeadConfig),
    SteepestDescent(SteepestDescentConfig),
    NewtonCg(NewtonCgConfig),
    Sgd(SgdConfig),
    Adam(AdamConfig),
}

impl OptimizerConfig {
    /// Возвращает выбранный метод оптимизации.
    pub fn method(&self) -> OptimizerMethod {
        match self {
            Self::Lbfgs(_) => OptimizerMethod::Lbfgs,
            Self::NelderMead(_) => OptimizerMethod::NelderMead,
            Self::SteepestDescent(_) => OptimizerMethod::SteepestDescent,
            Self::NewtonCg(_) => OptimizerMethod::NewtonCg,
            Self::Sgd(_) => OptimizerMethod::Sgd,
            Self::Adam(_) => OptimizerMethod::Adam,
        }
    }

    /// Возвращает ограничение на число итераций для выбранного метода.
    pub fn max_iters(&self) -> u64 {
        match self {
            Self::Lbfgs(config) => config.max_iters,
            Self::NelderMead(config) => config.max_iters,
            Self::SteepestDescent(config) => config.max_iters,
            Self::NewtonCg(config) => config.max_iters,
            Self::Sgd(config) => config.max_iters,
            Self::Adam(config) => config.max_iters,
        }
    }
}

macro_rules! impl_optimizer_config_from {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for OptimizerConfig {
                fn from(value: $ty) -> Self {
                    Self::$variant(value)
                }
            }

            impl From<&$ty> for OptimizerConfig {
                fn from(value: &$ty) -> Self {
                    Self::$variant(value.clone())
                }
            }
        )+
    };
}

impl_optimizer_config_from! {
    Lbfgs => LbfgsConfig,
    NelderMead => NelderMeadConfig,
    SteepestDescent => SteepestDescentConfig,
    NewtonCg => NewtonCgConfig,
    Sgd => SgdConfig,
    Adam => AdamConfig,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::Lbfgs(LbfgsConfig::default())
    }
}
