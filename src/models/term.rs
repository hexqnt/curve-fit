use super::{Hessian, Param};

/// Вклад отдельного терма в значение objective.
pub(crate) trait TermValue {
    fn add_value(&self, param: &Param, value: &mut f64);
}

/// Вклад отдельного терма в значение и градиент objective.
pub(crate) trait TermGrad: TermValue {
    fn add_value_grad(&self, param: &Param, value: &mut f64, gradient: &mut [f64]);
}

/// Вклад отдельного терма в значение, градиент и raw-гессиан objective.
pub(crate) trait TermHessian: TermGrad {
    fn add_value_grad_hessian(
        &self,
        param: &Param,
        value: &mut f64,
        gradient: &mut [f64],
        hessian: &mut Hessian,
    );
}
