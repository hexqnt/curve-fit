//! Общий harness полного приложения с фиксированными языком, ОС и размером окна.

use curve_fit::{
    CurveFitApp,
    testing::{self, AppState},
};
use eframe::egui;
use egui_kittest::{Harness, HarnessBuilder};

pub type AppHarness = Harness<'static, CurveFitApp>;

fn builder() -> HarnessBuilder<CurveFitApp> {
    Harness::builder()
        .with_size(egui::vec2(1440.0, 1200.0))
        .with_step_dt(1.0 / 60.0)
        .with_os(egui::os::OperatingSystem::Nix)
}

#[must_use]
pub fn harness() -> AppHarness {
    harness_with_size(egui::vec2(1440.0, 1200.0))
}

#[must_use]
pub fn harness_with_size(size: egui::Vec2) -> AppHarness {
    builder()
        .with_size(size)
        .build_eframe(|cc| testing::from_context(&cc.egui_ctx))
}

#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn harness_with_paused_fit() -> (AppHarness, testing::FitWorkerPause) {
    let mut pause = None;
    let harness = builder().build_eframe(|cc| {
        let (app, first_fit_pause) = testing::from_context_with_paused_fit(&cc.egui_ctx);
        pause = Some(first_fit_pause);
        app
    });
    (
        harness,
        pause.expect("Harness must initialize the fit pause"),
    )
}

#[must_use]
pub fn state(harness: &AppHarness) -> AppState<'_> {
    harness.state().inspect()
}
