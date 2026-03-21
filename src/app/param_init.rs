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

fn is_polynomial_family(family: CurveFamily) -> bool {
    matches!(
        family,
        CurveFamily::Linear
            | CurveFamily::Quadratic
            | CurveFamily::Cubic
            | CurveFamily::Quartic
            | CurveFamily::Quintic
            | CurveFamily::Sextic
            | CurveFamily::Septic
            | CurveFamily::Octic
            | CurveFamily::Nonic
    )
}

pub(super) fn is_advanced_param_init_supported(family: CurveFamily) -> bool {
    is_polynomial_family(family)
        || matches!(
            family,
            CurveFamily::Logistic
                | CurveFamily::Gaussian
                | CurveFamily::ExponentialBasic
                | CurveFamily::Power
        )
}

/// Строит начальные параметры модели на основе статистики входных точек.
///
/// Для неподдерживаемых семейств возвращает понятную ошибку для UI.
pub(super) fn data_based_params_for_family(
    family: CurveFamily,
    points: &Points,
) -> Result<CurveParams, String> {
    if is_polynomial_family(family) {
        return data_based_polynomial_params(family, points);
    }

    match family {
        CurveFamily::Logistic => data_based_logistic_params(points),
        CurveFamily::Gaussian => data_based_gaussian_params(points),
        CurveFamily::ExponentialBasic => data_based_exponential_basic_params(points),
        CurveFamily::Power => data_based_power_params(points),
        _ => Err(format!(
            "Data-based initialization is not supported for family {family}"
        )),
    }
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
    CurveParams::try_from_values(family, values).map_err(|error| error.to_string())
}

fn data_based_logistic_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, _, y_max, _) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    CurveParams::try_from_values(
        CurveFamily::Logistic,
        vec![y_max, 4.0 / x_span, (x_min + x_max) * 0.5],
    )
    .map_err(|error| error.to_string())
}

fn data_based_gaussian_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, _, y_max, x_at_y_max) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    let sigma = (x_span / 6.0).max(PARAM_INIT_SPAN_EPS);
    CurveParams::try_from_values(CurveFamily::Gaussian, vec![y_max, x_at_y_max, sigma])
        .map_err(|error| error.to_string())
}

fn data_based_exponential_basic_params(points: &Points) -> Result<CurveParams, String> {
    let (x_min, x_max, y_min, y_max, _) = point_extrema(points);
    let x_span = (x_max - x_min).max(PARAM_INIT_SPAN_EPS);
    let mut amplitude = y_max - y_min;
    if amplitude.abs() < PARAM_INIT_SPAN_EPS {
        amplitude = 1.0;
    }
    CurveParams::try_from_values(
        CurveFamily::ExponentialBasic,
        vec![y_min, amplitude, 1.0 / x_span],
    )
    .map_err(|error| error.to_string())
}

fn data_based_power_params(points: &Points) -> Result<CurveParams, String> {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let sample_count = points.len() as f64;

    for point in points.as_slice() {
        if point.x() <= 0.0 {
            return Err("Data-based initialization for family Power requires x > 0".to_string());
        }
        if point.y() <= 0.0 {
            return Err("Data-based initialization for family Power requires y > 0".to_string());
        }
        let log_x = point.x().ln();
        let log_y = point.y().ln();
        sum_x += log_x;
        sum_y += log_y;
        sum_xx += log_x * log_x;
        sum_xy += log_x * log_y;
    }

    let denominator = sample_count * sum_xx - sum_x * sum_x;
    let slope = if denominator.abs() <= PARAM_INIT_SPAN_EPS {
        0.0
    } else {
        (sample_count * sum_xy - sum_x * sum_y) / denominator
    };
    let intercept = (sum_y - slope * sum_x) / sample_count;
    CurveParams::try_from_values(CurveFamily::Power, vec![intercept.exp(), slope])
        .map_err(|error| error.to_string())
}

fn linear_regression(points: &Points) -> Result<(f64, f64), String> {
    if points.len() < 2 {
        return Err("Linear regression requires at least two points".to_string());
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let sample_count = points.len() as f64;

    for point in points.as_slice() {
        sum_x += point.x();
        sum_y += point.y();
        sum_xx += point.x() * point.x();
        sum_xy += point.x() * point.y();
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
