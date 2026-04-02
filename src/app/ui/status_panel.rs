use super::*;

pub(super) fn ui_optimization_metric(app: &mut CurveFitApp, ui: &mut egui::Ui) {
    let language = app.ui_language;
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
    ui.label(
        egui::RichText::new(tr(
            language,
            "The selected metric is minimized during fitting and shown as loss(metric) in diagnostics.",
            "Выбранная метрика минимизируется при фитинге и отображается как loss(metric) в диагностике.",
        ))
        .small(),
    );
    ui.add_space(2.0);
    CurveFitApp::toggle_switch_labeled(
        ui,
        &mut app.metric_quantization_enabled,
        tr(
            language,
            "Quantize objective/metrics before residual",
            "Квантизовать objective/метрики перед residual",
        ),
    );
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
    ui.label(
        egui::RichText::new(tr(
            language,
            "Residual pipeline: Q(y_pred) - Q(y_true). Quantization affects optimization objective and reported metrics.",
            "Пайплайн residual: Q(y_pred) - Q(y_true). Квантизация влияет на objective оптимизации и отображаемые метрики.",
        ))
        .small(),
    );
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
