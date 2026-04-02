use super::*;

pub(super) fn ui_optimization_metric(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
    ui.horizontal_wrapped(|ui| {
        egui::ComboBox::from_label(tr(language, "Metric", "Метрика"))
            .selected_text(optimization_loss_metric_label(
                language,
                app.optimization_loss_metric,
            ))
            .show_ui(ui, |ui| {
                for metric in OptimizationLossMetric::ALL {
                    ui.selectable_value(
                        &mut app.optimization_loss_metric,
                        metric,
                        optimization_loss_metric_label(language, metric),
                    );
                }
            });
        CurveFitApp::info_tooltip(
            ui,
            tr(
                language,
                "Optimization metric\n- This metric is minimized during fitting\n- Diagnostics shows it as loss(metric)\n- MSE: smooth gradients, MAE: more robust to outliers, soft_l1: compromise",
                "Метрика оптимизации\n- Эта метрика минимизируется во время фитинга\n- В диагностике она отображается как loss(metric)\n- MSE: более гладкие градиенты, MAE: устойчивее к выбросам, soft_l1: компромисс",
            ),
        );
    });
    ui.add_space(2.0);
    ui.horizontal_wrapped(|ui| {
        CurveFitApp::toggle_switch_labeled(
            ui,
            &mut app.metric_quantization_enabled,
            tr(
                language,
                "Quantize objective/metrics before residual",
                "Квантизовать objective/метрики перед residual",
            ),
        );
        CurveFitApp::info_tooltip(
            ui,
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

pub(super) fn ui_status(app: &CurveFitApp, ui: &mut egui::Ui) {
    if let Some(status) = &app.status {
        let color = if status.is_error() {
            ui.visuals().error_fg_color
        } else {
            if ui.visuals().dark_mode {
                egui::Color32::from_rgb(112, 211, 202)
            } else {
                egui::Color32::from_rgb(24, 131, 141)
            }
        };
        ui.horizontal(|ui| {
            ui.colored_label(color, "●");
            ui.label(status.text(app.ui_language));
        });
    }
}
