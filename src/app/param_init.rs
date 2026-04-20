//! Эвристики и служебные функции для построения стартовых параметров моделей.

use crate::domain::{CurveFamily, CurveParams, Points};

const PARAM_INIT_SPAN_EPS: f64 = 1e-9;

pub(super) fn polynomial_family(degree: usize) -> CurveFamily {
    match degree.clamp(1, 9) {
        1 => CurveFamily::Linear,
        2 => CurveFamily::Quadratic,
        3 => CurveFamily::Cubic,
        4 => CurveFamily::Quartic,
        5 => CurveFamily::Quintic,
        6 => CurveFamily::Sextic,
        7 => CurveFamily::Septic,
        8 => CurveFamily::Octic,
        _ => CurveFamily::Nonic,
    }
}

pub(super) fn rational_family(degree: usize) -> CurveFamily {
    CurveFamily::from_rational_degree(degree)
}

pub(super) fn is_advanced_param_init_supported(family: CurveFamily) -> bool {
    family.is_polynomial()
        || family.is_rational()
        || matches!(
            family,
            CurveFamily::Logistic
                | CurveFamily::Gompertz
                | CurveFamily::BiExponential
                | CurveFamily::DampedSinusoid
                | CurveFamily::Gaussian
                | CurveFamily::ExponentialBasic
                | CurveFamily::Power
                | CurveFamily::Emg
                | CurveFamily::PseudoVoigt
        )
}

/// Строит начальные параметры модели на основе статистики входных точек.
///
/// Для неподдерживаемых семейств возвращает понятную ошибку для UI.
pub(super) fn data_based_params_for_family(
    family: CurveFamily,
    points: &Points,
) -> Result<CurveParams, String> {
    if family.is_polynomial() {
        return data_based_polynomial_params(family, points);
    }
    if family.is_rational() {
        return data_based_rational_params(family, points);
    }

    match family {
        CurveFamily::Logistic => data_based_logistic_params(points),
        CurveFamily::Gompertz => data_based_gompertz_params(points),
        CurveFamily::BiExponential => data_based_bi_exponential_params(points),
        CurveFamily::DampedSinusoid => data_based_damped_sinusoid_params(points),
        CurveFamily::Gaussian => data_based_gaussian_params(points),
        CurveFamily::ExponentialBasic => data_based_exponential_basic_params(points),
        CurveFamily::Power => data_based_power_params(points),
        CurveFamily::Emg => data_based_emg_params(points),
        CurveFamily::PseudoVoigt => data_based_pseudo_voigt_params(points),
        _ => Err(format!(
            "Data-based initialization is not supported for family {family}"
        )),
    }
}

fn build_curve_params(family: CurveFamily, values: Vec<f64>) -> Result<CurveParams, String> {
    CurveParams::try_from_values(family, values).map_err(|error| error.to_string())
}

fn data_based_polynomial_params(
    family: CurveFamily,
    points: &Points,
) -> Result<CurveParams, String> {
    let (slope, intercept) = linear_regression(points)?;
    let parameter_count = family.parameter_count();
    let mut values = vec![0.0; parameter_count];
    values[parameter_count - 2] = slope;
    values[parameter_count - 1] = intercept;
    build_curve_params(family, values)
}

fn data_based_logistic_params(points: &Points) -> Result<CurveParams, String> {
    let (a, b, c) = data_based_sigmoid_abc(points);
    build_curve_params(CurveFamily::Logistic, vec![a, b, c])
}

fn data_based_gompertz_params(points: &Points) -> Result<CurveParams, String> {
    let (a, b, c) = data_based_sigmoid_abc(points);
    build_curve_params(CurveFamily::Gompertz, vec![a, b, c])
}

fn data_based_bi_exponential_params(points: &Points) -> Result<CurveParams, String> {
    let x_span = {
        let (x_min, x_max, _, _, _) = point_extrema(points);
        (x_max - x_min).max(PARAM_INIT_SPAN_EPS)
    };
    let (y_at_min_x, y_at_max_x) = y_at_x_bounds(points);
    let direction = if y_at_min_x >= y_at_max_x { 1.0 } else { -1.0 };
    let amplitude = (y_at_min_x - y_at_max_x).abs().max(PARAM_INIT_SPAN_EPS);
    let a1 = direction * 0.7 * amplitude;
    let a2 = direction * 0.3 * amplitude;
    let k1 = 3.0 / x_span;
    let k2 = 0.5 / x_span;
    let c = y_at_max_x;

    build_curve_params(CurveFamily::BiExponential, vec![a1, k1, a2, k2, c])
}

fn data_based_damped_sinusoid_params(points: &Points) -> Result<CurveParams, String> {
    let sorted = sorted_by_x(points);
    let first = sorted[0];
    let last = sorted[sorted.len() - 1];
    let x_span = (last.x() - first.x()).abs().max(PARAM_INIT_SPAN_EPS);
    let (_, _, y_min, y_max, _) = point_extrema(points);
    let center = mean_y(points);
    let amplitude = ((y_max - y_min) * 0.5).abs().max(PARAM_INIT_SPAN_EPS);
    let k = 0.5 / x_span;
    let zero_crossings = count_centered_sign_changes(&sorted, center);
    let omega = if zero_crossings > 0 {
        std::f64::consts::PI * zero_crossings as f64 / x_span
    } else {
        std::f64::consts::TAU / x_span
    };
    let denom = (amplitude * (-k * first.x()).exp())
        .abs()
        .max(PARAM_INIT_SPAN_EPS);
    let ratio = ((first.y() - center) / denom).clamp(-1.0, 1.0);
    let phi = ratio.asin() - omega * first.x();

    build_curve_params(
        CurveFamily::DampedSinusoid,
        vec![amplitude, k, omega, phi, center],
    )
}

fn data_based_gaussian_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, _, y_max, x_at_y_max) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    let sigma = (x_span / 6.0).max(PARAM_INIT_SPAN_EPS);
    build_curve_params(CurveFamily::Gaussian, vec![y_max, x_at_y_max, sigma])
}

fn data_based_exponential_basic_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, y_min, y_max, _) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    let mut amplitude = y_max - y_min;
    if amplitude.abs() < PARAM_INIT_SPAN_EPS {
        amplitude = 1.0;
    }
    build_curve_params(
        CurveFamily::ExponentialBasic,
        vec![y_min, amplitude, 1.0 / x_span],
    )
}

fn data_based_power_params(points: &Points) -> Result<CurveParams, String> {
    let (slope, intercept) = linear_regression_by(points, |point| {
        if point.x() <= 0.0 {
            return Err("Data-based initialization for family Power requires x > 0".to_string());
        }
        if point.y() <= 0.0 {
            return Err("Data-based initialization for family Power requires y > 0".to_string());
        }
        Ok((point.x().ln(), point.y().ln()))
    })?;
    build_curve_params(CurveFamily::Power, vec![intercept.exp(), slope])
}

fn data_based_rational_params(family: CurveFamily, points: &Points) -> Result<CurveParams, String> {
    let Some(degree) = family.rational_degree() else {
        return Err(format!("Family {family} is not rational"));
    };

    let (slope, intercept) = linear_regression(points)?;
    if degree == 1 {
        return build_curve_params(CurveFamily::Rational11, vec![slope, intercept, 0.0, 0.0]);
    }

    let mut values = vec![0.0; family.parameter_count()];
    values[degree - 1] = slope;
    values[degree] = intercept;
    build_curve_params(family, values)
}

fn data_based_emg_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, y_min, y_max, x_at_y_max) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    let y_span = (y_max - y_min).abs().max(PARAM_INIT_SPAN_EPS);
    let left_span = (x_at_y_max - x_min).abs();
    let right_span = (x_max - x_at_y_max).abs();
    let tau_sign = if right_span >= left_span { 1.0 } else { -1.0 };
    let tau = tau_sign * (x_span / 6.0).max(PARAM_INIT_SPAN_EPS);
    let a = y_span * tau.abs();
    build_curve_params(
        CurveFamily::Emg,
        vec![a, x_at_y_max, x_span / 6.0, tau, y_min],
    )
}

fn data_based_pseudo_voigt_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, y_min, y_max, x_at_y_max) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    let y_span = (y_max - y_min).abs().max(PARAM_INIT_SPAN_EPS);
    let width = (x_span / 6.0).max(PARAM_INIT_SPAN_EPS);
    build_curve_params(
        CurveFamily::PseudoVoigt,
        vec![y_span, x_at_y_max, width, width, 0.0, y_min],
    )
}

fn data_based_sigmoid_abc(points: &Points) -> (f64, f64, f64) {
    let (x_min, x_max, _, y_max, _) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    (y_max, 4.0 / x_span, (x_min + x_max) * 0.5)
}

fn linear_regression(points: &Points) -> Result<(f64, f64), String> {
    linear_regression_by(points, |point| Ok((point.x(), point.y())))
}

fn linear_regression_by<F>(points: &Points, mut map_point: F) -> Result<(f64, f64), String>
where
    F: FnMut(crate::domain::Point) -> Result<(f64, f64), String>,
{
    if points.len() < 2 {
        return Err("Linear regression requires at least two points".to_string());
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let sample_count = points.len() as f64;

    for point in points.as_slice().iter().copied() {
        let (x, y) = map_point(point)?;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    let denominator = sample_count * sum_xx - sum_x * sum_x;
    let slope = if denominator.abs() <= PARAM_INIT_SPAN_EPS {
        0.0
    } else {
        (sample_count * sum_xy - sum_x * sum_y) / denominator
    };
    let intercept = (sum_y - slope * sum_x) / sample_count;
    Ok((slope, intercept))
}

fn point_extrema(points: &Points) -> (f64, f64, f64, f64, f64) {
    let first = points.as_slice()[0];
    let mut x_min = first.x();
    let mut x_max = first.x();
    let mut y_min = first.y();
    let mut y_max = first.y();
    let mut x_at_y_max = first.x();

    for point in points.as_slice().iter().skip(1) {
        x_min = x_min.min(point.x());
        x_max = x_max.max(point.x());
        y_min = y_min.min(point.y());
        if point.y() > y_max {
            y_max = point.y();
            x_at_y_max = point.x();
        }
    }

    (x_min, x_max, y_min, y_max, x_at_y_max)
}

fn y_at_x_bounds(points: &Points) -> (f64, f64) {
    let first = points.as_slice()[0];
    let mut min_x = first.x();
    let mut max_x = first.x();
    let mut y_at_min_x = first.y();
    let mut y_at_max_x = first.y();

    for point in points.as_slice().iter().skip(1) {
        if point.x() < min_x {
            min_x = point.x();
            y_at_min_x = point.y();
        }
        if point.x() > max_x {
            max_x = point.x();
            y_at_max_x = point.y();
        }
    }

    (y_at_min_x, y_at_max_x)
}

fn mean_y(points: &Points) -> f64 {
    points.as_slice().iter().map(|point| point.y()).sum::<f64>() / points.len() as f64
}

fn sorted_by_x(points: &Points) -> Vec<crate::domain::Point> {
    let mut sorted = points.as_slice().to_vec();
    sorted.sort_by(|left, right| left.x().total_cmp(&right.x()));
    sorted
}

fn count_centered_sign_changes(sorted: &[crate::domain::Point], center: f64) -> usize {
    let mut sign_changes = 0_usize;
    let mut previous = sorted[0].y() - center;
    for point in sorted.iter().skip(1) {
        let current = point.y() - center;
        if previous * current < 0.0 {
            sign_changes += 1;
        }
        if current.abs() > PARAM_INIT_SPAN_EPS {
            previous = current;
        }
    }
    sign_changes
}
