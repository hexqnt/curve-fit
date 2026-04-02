use super::*;

fn fit_spline_family_with_optimizer_config(
    points: &Points,
    config: SplineConfig,
    family: SplineFamilyKind,
    optimizer_config: &OptimizerConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_optimizer_config_and_loss_metric(
        points,
        config,
        family,
        optimizer_config,
        OptimizationLossMetric::Mse,
    )
}

pub(crate) fn fit_spline_family_with_optimizer_config_and_loss_metric(
    points: &Points,
    config: SplineConfig,
    family: SplineFamilyKind,
    optimizer_config: &OptimizerConfig,
    loss_metric: OptimizationLossMetric,
) -> Result<SplineResult, FitError> {
    let mut runner = if loss_metric == OptimizationLossMetric::Mse {
        IncrementalSplineFitRunner::new_with_optimizer_config(
            points,
            family,
            config,
            optimizer_config,
        )?
    } else {
        IncrementalSplineFitRunner::new_with_optimizer_config_and_loss_metric(
            points,
            family,
            config,
            optimizer_config,
            loss_metric,
        )?
    };
    loop {
        match runner.step()? {
            IncrementalSplineFitStep::Iteration { .. } => {}
            IncrementalSplineFitStep::Finished { result, .. } => return Ok(result),
            IncrementalSplineFitStep::Cancelled => return Err(FitError::Cancelled),
        }
    }
}

pub(crate) fn default_spline_initial_knot_y(
    points: &Points,
    family: SplineFamilyKind,
    config: SplineConfig,
) -> Result<Vec<f64>, FitError> {
    let prepared = prepare_spline_inputs(points, config, family, None)?;
    Ok(prepared.initial_y)
}

/// Подгоняет линейный сплайн с явными `samples` и `knots`.
pub fn fit_linear_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_linear_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет линейный сплайн с полной конфигурацией.
pub fn fit_linear_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    let optimizer_config = OptimizerConfig::Lbfgs(spline_lbfgs_config());
    fit_linear_spline_with_optimizer_config(points, config, &optimizer_config)
}

/// Подгоняет линейный сплайн с полной конфигурацией и выбранным оптимизатором.
pub fn fit_linear_spline_with_optimizer_config(
    points: &Points,
    config: SplineConfig,
    optimizer_config: &OptimizerConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_optimizer_config(
        points,
        config,
        SplineFamilyKind::Linear,
        optimizer_config,
    )
}

/// Подгоняет монотонный кубический сплайн с явными `samples` и `knots`.
pub fn fit_monotone_cubic_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_monotone_cubic_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет монотонный кубический сплайн с полной конфигурацией.
pub fn fit_monotone_cubic_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    let optimizer_config = OptimizerConfig::Lbfgs(spline_lbfgs_config());
    fit_monotone_cubic_spline_with_optimizer_config(points, config, &optimizer_config)
}

/// Подгоняет монотонный кубический сплайн с полной конфигурацией и выбранным оптимизатором.
pub fn fit_monotone_cubic_spline_with_optimizer_config(
    points: &Points,
    config: SplineConfig,
    optimizer_config: &OptimizerConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_optimizer_config(
        points,
        config,
        SplineFamilyKind::MonotoneCubic,
        optimizer_config,
    )
}

/// Подгоняет натуральный кубический сплайн с явными `samples` и `knots`.
pub fn fit_natural_cubic_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_natural_cubic_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет натуральный кубический сплайн с полной конфигурацией.
pub fn fit_natural_cubic_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    let optimizer_config = OptimizerConfig::Lbfgs(spline_lbfgs_config());
    fit_natural_cubic_spline_with_optimizer_config(points, config, &optimizer_config)
}

/// Подгоняет натуральный кубический сплайн с полной конфигурацией и выбранным оптимизатором.
pub fn fit_natural_cubic_spline_with_optimizer_config(
    points: &Points,
    config: SplineConfig,
    optimizer_config: &OptimizerConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_optimizer_config(
        points,
        config,
        SplineFamilyKind::NaturalCubic,
        optimizer_config,
    )
}

/// Подгоняет сплайн Акимы с явными `samples` и `knots`.
pub fn fit_akima_spline(
    points: &Points,
    samples: usize,
    knots: usize,
) -> Result<SplineResult, FitError> {
    fit_akima_spline_with_config(
        points,
        SplineConfig {
            knots,
            samples,
            ..SplineConfig::default()
        },
    )
}

/// Подгоняет сплайн Акимы с полной конфигурацией.
pub fn fit_akima_spline_with_config(
    points: &Points,
    config: SplineConfig,
) -> Result<SplineResult, FitError> {
    let optimizer_config = OptimizerConfig::Lbfgs(spline_lbfgs_config());
    fit_akima_spline_with_optimizer_config(points, config, &optimizer_config)
}

/// Подгоняет сплайн Акимы с полной конфигурацией и выбранным оптимизатором.
pub fn fit_akima_spline_with_optimizer_config(
    points: &Points,
    config: SplineConfig,
    optimizer_config: &OptimizerConfig,
) -> Result<SplineResult, FitError> {
    fit_spline_family_with_optimizer_config(
        points,
        config,
        SplineFamilyKind::Akima,
        optimizer_config,
    )
}
