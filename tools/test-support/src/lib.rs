//! Общий harness полного приложения с фиксированными языком, ОС и размером окна.

use curve_fit::{
    CurveFitApp,
    testing::{self, AppState},
};
use eframe::egui;
use egui_kittest::Harness;

pub type AppHarness = Harness<'static, CurveFitApp>;

#[must_use]
pub fn harness() -> AppHarness {
    Harness::builder()
        .with_size(egui::vec2(1440.0, 1200.0))
        .with_step_dt(1.0 / 60.0)
        .with_os(egui::os::OperatingSystem::Nix)
        .build_eframe(|cc| testing::from_context(&cc.egui_ctx))
}

#[must_use]
pub fn state(harness: &AppHarness) -> AppState<'_> {
    harness.state().inspect()
}
