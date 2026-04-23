use crate::domain::{CurveFamily, SaturatingTrendTauGrid};

use super::{
    CentralDiffGradient, CentralDiffHessian, ObjectiveGrad, ObjectiveHessian, ObjectiveValue,
    Param, PredictionLoss, TermGrad, TermHessian, TermValue, dispatch,
};

const OBJECTIVE_GRADIENT_FD_REL_STEP: f64 = 1e-5;
const OBJECTIVE_GRADIENT_FD_MIN_STEP: f64 = 1e-7;
const OBJECTIVE_HESSIAN_FD_REL_STEP: f64 = 1e-4;
const OBJECTIVE_HESSIAN_FD_MIN_STEP: f64 = 1e-6;

/// Data-term для подгонки параметрической кривой по набору точек.
pub(crate) struct DataTerm<'a, L> {
    family: CurveFamily,
    saturating_trend_tau_grid: Option<SaturatingTrendTauGrid>,
    x_values: &'a [f64],
    y_values: &'a [f64],
    loss: L,
}

impl<'a, L> DataTerm<'a, L> {
    pub(crate) fn new(
        family: CurveFamily,
        x_values: &'a [f64],
        y_values: &'a [f64],
        saturating_trend_tau_grid: Option<&SaturatingTrendTauGrid>,
        loss: L,
    ) -> Self {
        Self {
            family,
            saturating_trend_tau_grid: saturating_trend_tau_grid.cloned(),
            x_values,
            y_values,
            loss,
        }
    }
}

struct DataValueObjective<'a, L> {
    family: CurveFamily,
    saturating_trend_tau_grid: Option<&'a SaturatingTrendTauGrid>,
    x_values: &'a [f64],
    y_values: &'a [f64],
    loss: &'a L,
}

impl<L> ObjectiveValue for DataValueObjective<'_, L>
where
    L: PredictionLoss,
{
    fn value(&self, param: &Param) -> f64 {
        dispatch::objective_value(
            self.family,
            self.x_values,
            self.y_values,
            param,
            self.saturating_trend_tau_grid
                .map(SaturatingTrendTauGrid::as_slice),
            self.loss,
        )
    }
}

struct DataValueGradObjective<'a, L> {
    family: CurveFamily,
    saturating_trend_tau_grid: Option<&'a SaturatingTrendTauGrid>,
    x_values: &'a [f64],
    y_values: &'a [f64],
    loss: &'a L,
}

impl<L> ObjectiveValue for DataValueGradObjective<'_, L>
where
    L: PredictionLoss,
{
    fn value(&self, param: &Param) -> f64 {
        dispatch::objective_value(
            self.family,
            self.x_values,
            self.y_values,
            param,
            self.saturating_trend_tau_grid
                .map(SaturatingTrendTauGrid::as_slice),
            self.loss,
        )
    }
}

impl<L> ObjectiveGrad for DataValueGradObjective<'_, L>
where
    L: PredictionLoss,
{
    fn value_grad(&self, param: &Param) -> (f64, Vec<f64>) {
        if let Some((value, gradient)) = dispatch::objective_value_grad_analytic(
            self.family,
            self.x_values,
            self.y_values,
            param,
            self.saturating_trend_tau_grid
                .map(SaturatingTrendTauGrid::as_slice),
            self.loss,
        ) {
            return (value, gradient);
        }

        let objective = DataValueObjective {
            family: self.family,
            saturating_trend_tau_grid: self.saturating_trend_tau_grid,
            x_values: self.x_values,
            y_values: self.y_values,
            loss: self.loss,
        };
        let numerical = CentralDiffGradient::new(
            objective,
            OBJECTIVE_GRADIENT_FD_REL_STEP,
            OBJECTIVE_GRADIENT_FD_MIN_STEP,
        );
        numerical.value_grad(param)
    }
}

fn add_gradient(dst: &mut [f64], src: &[f64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter().copied()) {
        *dst_value += src_value;
    }
}

impl<L> TermValue for DataTerm<'_, L>
where
    L: PredictionLoss,
{
    fn add_value(&self, param: &Param, value: &mut f64) {
        *value += dispatch::objective_value(
            self.family,
            self.x_values,
            self.y_values,
            param,
            self.saturating_trend_tau_grid
                .as_ref()
                .map(SaturatingTrendTauGrid::as_slice),
            &self.loss,
        );
    }
}

impl<L> TermGrad for DataTerm<'_, L>
where
    L: PredictionLoss,
{
    fn add_value_grad(&self, param: &Param, value: &mut f64, gradient: &mut [f64]) {
        if let Some((local_value, local_gradient)) = dispatch::objective_value_grad_analytic(
            self.family,
            self.x_values,
            self.y_values,
            param,
            self.saturating_trend_tau_grid
                .as_ref()
                .map(SaturatingTrendTauGrid::as_slice),
            &self.loss,
        ) {
            *value += local_value;
            add_gradient(gradient, &local_gradient);
            return;
        }

        let objective = DataValueObjective {
            family: self.family,
            saturating_trend_tau_grid: self.saturating_trend_tau_grid.as_ref(),
            x_values: self.x_values,
            y_values: self.y_values,
            loss: &self.loss,
        };
        let numerical = CentralDiffGradient::new(
            objective,
            OBJECTIVE_GRADIENT_FD_REL_STEP,
            OBJECTIVE_GRADIENT_FD_MIN_STEP,
        );
        let (local_value, local_gradient) = numerical.value_grad(param);
        *value += local_value;
        add_gradient(gradient, &local_gradient);
    }
}

impl<L> TermHessian for DataTerm<'_, L>
where
    L: PredictionLoss,
{
    fn add_value_grad_hessian(
        &self,
        param: &Param,
        value: &mut f64,
        gradient: &mut [f64],
        hessian: &mut super::Hessian,
    ) {
        if let Some((local_value, local_gradient, local_hessian)) =
            dispatch::objective_value_grad_raw_hessian_analytic(
                self.family,
                self.x_values,
                self.y_values,
                param,
                self.saturating_trend_tau_grid
                    .as_ref()
                    .map(SaturatingTrendTauGrid::as_slice),
                &self.loss,
            )
        {
            *value += local_value;
            add_gradient(gradient, &local_gradient);
            *hessian += &local_hessian;
            return;
        }

        let objective = DataValueGradObjective {
            family: self.family,
            saturating_trend_tau_grid: self.saturating_trend_tau_grid.as_ref(),
            x_values: self.x_values,
            y_values: self.y_values,
            loss: &self.loss,
        };
        let numerical = CentralDiffHessian::new(
            objective,
            OBJECTIVE_HESSIAN_FD_REL_STEP,
            OBJECTIVE_HESSIAN_FD_MIN_STEP,
        );
        let (local_value, local_gradient, local_hessian) = numerical.value_grad_raw_hessian(param);

        *value += local_value;
        add_gradient(gradient, &local_gradient);
        *hessian += &local_hessian;
    }
}
