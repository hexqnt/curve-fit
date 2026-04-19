//! Objective-слой параметрической подгонки, включая быстрый SIMD-путь для части семейств.

use super::*;

pub(super) struct CurveProblem {
    family: CurveFamily,
    point_x: Box<[f64]>,
    point_y: Box<[f64]>,
    loss_metric: OptimizationLossMetric,
    metric_quantization: MetricQuantization,
    residual_quantizer: ResidualQuantizer,
}

#[derive(Clone, Copy)]
pub(super) struct CurveProblemPredictionLoss<'a> {
    pub(super) problem: &'a CurveProblem,
}

impl PredictionLoss for CurveProblemPredictionLoss<'_> {
    fn value(&self, prediction: f64, target: f64) -> f64 {
        self.problem.loss_value_from_prediction(prediction, target)
    }

    fn d_prediction(&self, prediction: f64, target: f64) -> f64 {
        self.problem
            .loss_derivative_from_prediction(prediction, target)
    }

    fn d2_prediction(&self, prediction: f64, target: f64) -> f64 {
        self.problem
            .loss_second_derivative_from_prediction(prediction, target)
    }
}

struct CurveProblemObjective<'a> {
    problem: &'a CurveProblem,
}

#[derive(Clone, Copy)]
struct CurveProblemTerm<'a> {
    problem: &'a CurveProblem,
}

impl CurveProblemTerm<'_> {
    fn simd_enabled(&self) -> bool {
        matches!(
            self.problem.metric_quantization,
            MetricQuantization::Disabled
        )
    }

    fn fallback_term(&self) -> models::DataTerm<'_, CurveProblemPredictionLoss<'_>> {
        let loss = CurveProblemPredictionLoss {
            problem: self.problem,
        };
        models::DataTerm::new(
            self.problem.family,
            self.problem.point_x.as_ref(),
            self.problem.point_y.as_ref(),
            loss,
        )
    }
}

impl TermValue for CurveProblemTerm<'_> {
    fn add_value(&self, param: &[f64], value: &mut f64) {
        if self.simd_enabled() && self.problem.family.is_polynomial() {
            *value += simd::polynomial_cost(
                param,
                self.problem.point_x.as_ref(),
                self.problem.point_y.as_ref(),
                self.problem.loss_metric,
            );
            return;
        }
        if self.simd_enabled() && self.problem.family == CurveFamily::Inverse {
            *value += simd::inverse_cost(
                param,
                self.problem.point_x.as_ref(),
                self.problem.point_y.as_ref(),
                self.problem.loss_metric,
            );
            return;
        }
        self.fallback_term().add_value(param, value);
    }
}

impl TermGrad for CurveProblemTerm<'_> {
    fn add_value_grad(&self, param: &[f64], value: &mut f64, gradient: &mut [f64]) {
        if self.simd_enabled() && self.problem.family.is_polynomial() {
            let parameter_count = self.problem.family.parameter_count();
            debug_assert!(parameter_count <= MAX_POLYNOMIAL_PARAMS);
            let mut local_gradient = [0.0; MAX_POLYNOMIAL_PARAMS];
            let local_gradient = &mut local_gradient[..parameter_count];
            simd::accumulate_polynomial_gradient(
                self.problem.point_x.as_ref(),
                self.problem.point_y.as_ref(),
                param,
                self.problem.loss_metric,
                local_gradient,
            );
            let sample_scale = 1.0 / self.problem.point_x.len() as f64;
            *value += simd::polynomial_cost(
                param,
                self.problem.point_x.as_ref(),
                self.problem.point_y.as_ref(),
                self.problem.loss_metric,
            );
            for (gradient_value, local_value) in
                gradient.iter_mut().zip(local_gradient.iter().copied())
            {
                *gradient_value += local_value * sample_scale;
            }
            return;
        }
        if self.simd_enabled() && self.problem.family == CurveFamily::Inverse {
            let mut local_gradient = [0.0; 2];
            simd::accumulate_inverse_gradient(
                self.problem.point_x.as_ref(),
                self.problem.point_y.as_ref(),
                param,
                self.problem.loss_metric,
                &mut local_gradient,
            );
            let sample_scale = 1.0 / self.problem.point_x.len() as f64;
            for local_value in &mut local_gradient {
                *local_value *= sample_scale;
            }
            *value += simd::inverse_cost(
                param,
                self.problem.point_x.as_ref(),
                self.problem.point_y.as_ref(),
                self.problem.loss_metric,
            );
            for (gradient_value, local_value) in gradient.iter_mut().zip(local_gradient) {
                *gradient_value += local_value;
            }
            return;
        }
        self.fallback_term().add_value_grad(param, value, gradient);
    }
}

impl TermHessian for CurveProblemTerm<'_> {
    fn add_value_grad_hessian(
        &self,
        param: &[f64],
        value: &mut f64,
        gradient: &mut [f64],
        hessian: &mut Array2<f64>,
    ) {
        self.fallback_term()
            .add_value_grad_hessian(param, value, gradient, hessian);
    }
}

impl CurveProblemObjective<'_> {
    fn objective(&self) -> models::CurveObjective<CurveProblemTerm<'_>> {
        models::CurveObjective::new(
            self.problem.family.parameter_count(),
            CurveProblemTerm {
                problem: self.problem,
            },
        )
    }

    fn value(&self, param: &[f64]) -> f64 {
        self.objective().value(param)
    }

    fn value_grad(&self, param: &[f64]) -> (f64, Vec<f64>) {
        self.objective().value_grad(param)
    }

    fn value_grad_hessian(&self, param: &[f64]) -> (f64, Vec<f64>, Array2<f64>) {
        self.objective().value_grad_hessian(param)
    }
}

impl CurveProblem {
    pub(super) fn new_with_metric_quantization(
        family: CurveFamily,
        points: &Points,
        loss_metric: OptimizationLossMetric,
        metric_quantization: MetricQuantization,
    ) -> Self {
        let mut point_x = Vec::with_capacity(points.len());
        let mut point_y = Vec::with_capacity(points.len());
        for point in points.as_slice() {
            point_x.push(point.x());
            point_y.push(point.y());
        }
        Self {
            family,
            point_x: point_x.into_boxed_slice(),
            point_y: point_y.into_boxed_slice(),
            loss_metric,
            metric_quantization,
            residual_quantizer: ResidualQuantizer::new(metric_quantization),
        }
    }

    #[inline]
    fn residual(&self, predicted: f64, observed: f64) -> f64 {
        self.residual_quantizer.residual(predicted, observed)
    }

    #[inline]
    fn loss_value_from_prediction(&self, predicted: f64, observed: f64) -> f64 {
        self.loss_metric
            .value_from_residual(self.residual(predicted, observed))
    }

    #[inline]
    fn loss_derivative_from_prediction(&self, predicted: f64, observed: f64) -> f64 {
        self.loss_metric
            .residual_derivative(self.residual(predicted, observed))
    }

    #[inline]
    fn loss_second_derivative_from_prediction(&self, predicted: f64, observed: f64) -> f64 {
        self.loss_metric
            .residual_second_derivative(self.residual(predicted, observed))
    }

    fn objective(&self) -> CurveProblemObjective<'_> {
        CurveProblemObjective { problem: self }
    }
}

impl CostFunction for CurveProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let param = array1_as_slice(param);
        let value = self.objective().value(param);
        if value.is_finite() {
            Ok(value)
        } else {
            Ok(LARGE_COST)
        }
    }
}

impl Gradient for CurveProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let param = array1_as_slice(param);
        let (_, mut gradient) = self.objective().value_grad(param);
        for value in &mut gradient {
            if !value.is_finite() {
                *value = LARGE_COST;
            }
        }
        Ok(vec_to_array1(&gradient))
    }
}

impl Hessian for CurveProblem {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        if !matches!(self.loss_metric, OptimizationLossMetric::Mae) {
            let (_, _, hessian) = self.objective().value_grad_hessian(array1_as_slice(param));
            return Ok(hessian);
        }
        numerical_hessian_from_gradient(self, param)
    }
}

/// Равномерно дискретизирует параметрическую кривую на отрезке `x_min..=x_max`.
pub fn sample_curve(params: &CurveParams, x_min: f64, x_max: f64, samples: usize) -> Vec<[f64; 2]> {
    let sample_count = samples.max(2);
    let family = params.family();
    let mut sample_x_min = x_min;
    let mut sample_x_max = x_max;

    if family.requires_positive_x() {
        sample_x_min = positive_x(sample_x_min);
        sample_x_max = sample_x_max.max(sample_x_min + PARAM_EPS);
    }

    let span = sample_x_max - sample_x_min;
    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count - 1) as f64;
            let x = sample_x_min + span * t;
            [x, params.evaluate(x)]
        })
        .collect()
}
