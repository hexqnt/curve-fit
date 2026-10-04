//! Выбор способа инициализации возвращает команду, не изменяя состояние приложения.

use eframe::egui;

use crate::app::{
    CurveFamily, ParamInitMethod, UiLanguage,
    i18n::{param_init_method_disabled_label, param_init_method_label, tr},
};

#[derive(Debug, Clone, Copy)]
pub(in crate::app) enum InitAction {
    Method(ParamInitMethod),
    FromFitted,
}

/// Политика мономорфизируется: у сплайна нет пункта переноса обученных параметров.
pub(in crate::app) trait InitializationPolicy {
    fn hint(&self, language: UiLanguage) -> &'static str;
    fn supports(&self, method: ParamInitMethod) -> bool;
    fn fitted_available(&self) -> Option<bool>;
}

pub(in crate::app) struct ParametricInitialization {
    family: CurveFamily,
    fitted_available: bool,
}

pub(in crate::app) struct SplineInitialization;

impl InitializationPolicy for ParametricInitialization {
    fn hint(&self, language: UiLanguage) -> &'static str {
        tr(
            language,
            "Initial parameters (parametric models)\n- These values are the optimizer starting point\n- +Initialize can fill defaults/data-based/randomized values\n- \"From fitted model\" reuses parameters from the latest fit of the same family",
            "Начальные параметры (параметрические модели)\n- Эти значения являются стартовой точкой оптимизатора\n- +Инициализация может подставить значения по умолчанию/по данным/случайно\n- \"Из обученной модели\" берёт параметры из последнего фитинга того же семейства",
        )
    }
    fn supports(&self, method: ParamInitMethod) -> bool {
        method.is_supported_for_family(self.family)
    }
    fn fitted_available(&self) -> Option<bool> {
        Some(self.fitted_available)
    }
}

impl InitializationPolicy for SplineInitialization {
    fn hint(&self, language: UiLanguage) -> &'static str {
        tr(
            language,
            "Initial parameters (spline models)\n- Spline is non-parametric, but optimizer still tunes knot y-values\n- +Initialize sets starting knot_y values\n- Better initialization usually reduces iteration count",
            "Начальные параметры (сплайны)\n- Сплайн непараметрический, но оптимизатор всё равно настраивает knot y\n- +Инициализация задаёт стартовые значения knot_y\n- Более удачная инициализация обычно снижает число итераций",
        )
    }
    fn supports(&self, _: ParamInitMethod) -> bool {
        true
    }
    fn fitted_available(&self) -> Option<bool> {
        None
    }
}

pub(in crate::app) struct InitializationMenu<Policy> {
    language: UiLanguage,
    policy: Policy,
}

impl InitializationMenu<ParametricInitialization> {
    pub(in crate::app) fn parametric(
        language: UiLanguage,
        family: CurveFamily,
        fitted_available: bool,
    ) -> Self {
        Self {
            language,
            policy: ParametricInitialization {
                family,
                fitted_available,
            },
        }
    }
}

impl InitializationMenu<SplineInitialization> {
    pub(in crate::app) fn spline(language: UiLanguage) -> Self {
        Self {
            language,
            policy: SplineInitialization,
        }
    }
}

impl<Policy: InitializationPolicy> InitializationMenu<Policy> {
    pub(in crate::app) fn show(self, ui: &mut egui::Ui, enabled: bool) -> Option<InitAction> {
        let language = self.language;
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            let label = ui.label(tr(language, "Initial parameters", "Начальные параметры"));
            let _ = super::info_hover(label, self.policy.hint(language));
            ui.add_enabled_ui(enabled, |ui| {
                ui.menu_button(
                    tr(language, "+ Initialize", "+ Инициализация"),
                    |ui| {
                        if let Some(fitted_available) = self.policy.fitted_available() {
                            let label = if fitted_available {
                                tr(language, "From fitted model", "Из обученной модели")
                            } else {
                                tr(
                                    language,
                                    "From fitted model (fit this model first)",
                                    "Из обученной модели (сначала обучите эту модель)",
                                )
                            };
                            if ui
                                .add_enabled(fitted_available, egui::Button::new(label))
                                .clicked()
                            {
                                action = Some(InitAction::FromFitted);
                                ui.close();
                            }
                            ui.separator();
                        }
                        for method in ParamInitMethod::ALL {
                            let supported = self.policy.supports(method);
                            if supported {
                                if ui
                                    .button(param_init_method_label(language, method))
                                    .clicked()
                                {
                                    action = Some(InitAction::Method(method));
                                    ui.close();
                                }
                            } else {
                                let label = format!(
                                    "{} ({})",
                                    param_init_method_label(language, method),
                                    tr(language, "not available", "недоступно")
                                );
                                let response = ui.add_enabled(false, egui::Button::new(label));
                                let _ = super::info_hover(
                                    response,
                                    param_init_method_disabled_label(language, method),
                                );
                            }
                        }
                    },
                );
            });
        });
        action
    }
}
