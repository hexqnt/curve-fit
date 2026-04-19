//! Вспомогательные вычисления для диапазонов графика и автоподгонки вида.

use egui_plot::{PlotBounds, PlotPoint};

pub(super) fn plot_domain(points: &[PlotPoint]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 1.0);
    }

    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    for point in points.iter().skip(1) {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
    }

    if (max_x - min_x).abs() < 1e-9 {
        (min_x - 1.0, max_x + 1.0)
    } else {
        let padding = (max_x - min_x) * 0.1;
        (min_x - padding, max_x + padding)
    }
}

/// Рассчитывает границы графика так, чтобы уместить и исходные точки, и fitted-кривую.
pub(super) fn fit_bounds_for_content(
    points: &[PlotPoint],
    fitted_curve: Option<&[PlotPoint]>,
) -> Option<PlotBounds> {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for point in points {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }

    if let Some(fitted) = fitted_curve {
        for point in fitted {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }
    }

    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }

    let span_x = (max_x - min_x).abs();
    let span_y = (max_y - min_y).abs();
    let pad_x = span_x.max(1e-3) * 0.08;
    let pad_y = span_y.max(1e-3) * 0.08;

    Some(PlotBounds::from_min_max(
        [min_x - pad_x, min_y - pad_y],
        [max_x + pad_x, max_y + pad_y],
    ))
}
