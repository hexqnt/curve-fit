//! Публичные обертки для параметрической подгонки и API с колбэком прогресса.

use super::*;

/// Подгоняет параметрическую модель без колбэка прогресса.
pub fn fit_curve(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    config: &LbfgsConfig,
) -> Result<FitResult, FitError> {
    let optimizer_config = OptimizerConfig::from(config);
    fit_curve_with_optimizer_config(points, family, initial_params, &optimizer_config)
}

/// Подгоняет параметрическую модель без колбэка прогресса и с выбранным оптимизатором.
pub fn fit_curve_with_optimizer_config(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    optimizer_config: &OptimizerConfig,
) -> Result<FitResult, FitError> {
    fit_curve_with_progress_and_optimizer_config_and_loss_metric(
        points,
        family,
        initial_params,
        optimizer_config,
        OptimizationLossMetric::Mse,
        |_iteration, _params| true,
    )
}

/// Подгоняет параметрическую модель с колбэком на каждой итерации.
///
/// Возврат `false` из `on_iteration` запрашивает досрочную остановку.
pub fn fit_curve_with_progress<F>(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    config: &LbfgsConfig,
    on_iteration: F,
) -> Result<FitResult, FitError>
where
    F: FnMut(u64, Option<CurveParams>) -> bool + 'static,
{
    let optimizer_config = OptimizerConfig::from(config);
    fit_curve_with_progress_and_optimizer_config_and_loss_metric(
        points,
        family,
        initial_params,
        &optimizer_config,
        OptimizationLossMetric::Mse,
        on_iteration,
    )
}

/// Подгоняет параметрическую модель с колбэком и выбранным оптимизатором.
pub fn fit_curve_with_progress_and_optimizer_config<F>(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    optimizer_config: &OptimizerConfig,
    on_iteration: F,
) -> Result<FitResult, FitError>
where
    F: FnMut(u64, Option<CurveParams>) -> bool + 'static,
{
    fit_curve_with_progress_and_optimizer_config_and_loss_metric(
        points,
        family,
        initial_params,
        optimizer_config,
        OptimizationLossMetric::Mse,
        on_iteration,
    )
}

pub(crate) fn fit_curve_with_progress_and_optimizer_config_and_loss_metric<F>(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    optimizer_config: &OptimizerConfig,
    loss_metric: OptimizationLossMetric,
    on_iteration: F,
) -> Result<FitResult, FitError>
where
    F: FnMut(u64, Option<CurveParams>) -> bool + 'static,
{
    fit_curve_with_progress_and_optimizer_config_and_loss_metric_and_metric_quantization(
        points,
        family,
        initial_params,
        optimizer_config,
        loss_metric,
        MetricQuantization::Disabled,
        on_iteration,
    )
}

pub(crate) fn fit_curve_with_progress_and_optimizer_config_and_loss_metric_and_metric_quantization<
    F,
>(
    points: &Points,
    family: CurveFamily,
    initial_params: CurveParams,
    optimizer_config: &OptimizerConfig,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    on_iteration: F,
) -> Result<FitResult, FitError>
where
    F: FnMut(u64, Option<CurveParams>) -> bool + 'static,
{
    let mut on_iteration = on_iteration;
    let mut runner =
        IncrementalFitRunner::new_with_optimizer_config_and_loss_metric_and_metric_quantization(
            points,
            family,
            initial_params,
            optimizer_config,
            loss_metric,
            metric_quantization,
        )?;
    loop {
        match runner.step()? {
            IncrementalFitStep::Iteration {
                iteration, params, ..
            } => {
                if !on_iteration(iteration, Some(params)) {
                    return Err(FitError::Cancelled);
                }
            }
            IncrementalFitStep::Finished(result) => return Ok(result),
            IncrementalFitStep::Cancelled => return Err(FitError::Cancelled),
        }
    }
}
