//! Нижняя панель статуса и компактный выбор метрики оптимизации.

use std::borrow::Cow;

use super::*;
use crate::app::widgets::{Choice, ToggleSwitch};

const COLLAPSED_METRIC_SELECTOR_WIDTH: f32 = 150.0;

pub(super) fn ui_optimization_metric(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    ui.horizontal_wrapped(|ui| {
        let metric_response = ui_optimization_metric_selector(
            app,
            ui,
            egui::ComboBox::from_label(tr(language, "Metric", "Метрика")),
        );
        let _ = widgets::info_hover(
            metric_response,
            tr(
                language,
                "Optimization metric\n- This metric is minimized during fitting\n- Diagnostics shows it as loss(metric)\n- MSE (L2): smooth gradients\n- MAE (L1): more robust to outliers\n- soft_l1: compromise\n- Chebyshev: minimizes max absolute residual\n- MSLE: emphasizes relative/log-scaled deviations",
                "Метрика оптимизации\n- Эта метрика минимизируется во время фитинга\n- В диагностике она отображается как loss(metric)\n- MSE (L2): более гладкие градиенты\n- MAE (L1): устойчивее к выбросам\n- soft_l1: компромисс\n- Chebyshev: минимизирует максимальный модуль остатка\n- MSLE: акцент на относительных/логарифмических отклонениях",
            ),
        );
    });
    ui.add_space(style::SECTION_GAP);
    ui.horizontal_wrapped(|ui| {
        let label = tr(
            language,
            "Quantize objective/metrics before residual",
            "Квантизовать objective/метрики перед residual",
        );
        let quantization_response =
            ui.add(ToggleSwitch::new(&mut app.metric_quantization_enabled).label(label));
        let _ = widgets::info_hover(
            quantization_response,
            tr(
                language,
                "Quantization before residual\n- Pipeline: Q(y_pred) - Q(y_true)\n- Affects optimization objective and all reported metrics\n- Useful when measurements are coarse/discrete\n- Too aggressive rounding can slow or destabilize convergence",
                "Квантизация перед residual\n- Пайплайн: Q(y_pred) - Q(y_true)\n- Влияет на objective оптимизации и все отображаемые метрики\n- Полезна при грубых/дискретных измерениях\n- Слишком сильное округление может замедлить или ухудшить сходимость",
            ),
        );
    });
    app.metric_quantization_decimal_places = app.metric_quantization_decimal_places.clamp(
        MetricQuantizationDecimalPlaces::MIN,
        MetricQuantizationDecimalPlaces::MAX,
    );
    ui.add_enabled_ui(app.metric_quantization_enabled, |ui| {
        ui.add(
            egui::Slider::new(
                &mut app.metric_quantization_decimal_places,
                MetricQuantizationDecimalPlaces::MIN..=MetricQuantizationDecimalPlaces::MAX,
            )
            .text(tr(language, "Decimal places", "Знаков после запятой")),
        );
    });
}

pub(super) fn ui_optimization_metric_selector_compact(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    ui_optimization_metric_selector(
        app,
        ui,
        egui::ComboBox::from_id_salt("collapsed_header_optimization_metric_selector")
            .width(COLLAPSED_METRIC_SELECTOR_WIDTH),
    );
}

fn ui_optimization_metric_selector(
    app: &mut CurveFitApp,
    ui: &mut egui::Ui,
    selector: egui::ComboBox,
) -> egui::Response {
    let language = app.ui_language;
    ui.add(Choice::new(
        selector,
        &mut app.optimization_loss_metric,
        &OptimizationLossMetric::ALL,
        |metric| optimization_loss_metric_label(language, metric),
    ))
}

pub(super) fn ui_status(app: &CurveFitApp, ui: &mut egui::Ui) {
    if let Some(status) = &app.status {
        let color = if status.is_error() {
            ui.visuals().error_fg_color
        } else {
            style::UiColors::for_visuals(ui.visuals()).success
        };
        ui.horizontal(|ui| {
            ui.colored_label(color, "●");
            let status_text = match (status, app.last_fit_duration) {
                (StatusMessage::FitCompleted, Some(duration)) => Cow::Owned(format!(
                    "{} ({})",
                    status.text(app.ui_language),
                    CurveFitApp::format_fit_duration(duration)
                )),
                _ => Cow::Borrowed(status.text(app.ui_language)),
            };
            ui.label(status_text.as_ref());
        });
    }
}
