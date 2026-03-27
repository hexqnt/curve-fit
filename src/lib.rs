#![cfg_attr(feature = "portable-simd", feature(portable_simd))]
#![forbid(unsafe_code)]
//! Публичный API библиотеки для подгонки параметрических кривых и сплайнов.

pub mod app;
pub mod domain;
pub mod fit;

pub use app::CurveFitApp;
pub use domain::{
    AdamConfig, CurveFamily, CurveParams, FitResult, InputError, LbfgsConfig, NelderMeadConfig,
    NewtonCgConfig, OptimizerConfig, OptimizerMethod, Point, Points, SgdConfig,
    SteepestDescentConfig,
};
pub use fit::{
    DEFAULT_SPLINE_KNOTS, DEFAULT_SPLINE_SAMPLES, FitError, IncrementalFitRunner,
    IncrementalFitStep, IterationMetricSnapshot, OptimizationLossMetric, SplineConfig,
    SplineDuplicateXPolicy, SplineExtrapolation, SplineKnotStrategy, SplineResult,
    calculate_metrics, fit_akima_spline, fit_akima_spline_with_config,
    fit_akima_spline_with_optimizer_config, fit_curve, fit_curve_with_optimizer_config,
    fit_curve_with_progress, fit_curve_with_progress_and_optimizer_config, fit_linear_spline,
    fit_linear_spline_with_config, fit_linear_spline_with_optimizer_config,
    fit_monotone_cubic_spline, fit_monotone_cubic_spline_with_config,
    fit_monotone_cubic_spline_with_optimizer_config, fit_natural_cubic_spline,
    fit_natural_cubic_spline_with_config, fit_natural_cubic_spline_with_optimizer_config,
    sample_curve,
};
