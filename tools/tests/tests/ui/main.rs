#![cfg(not(target_arch = "wasm32"))]

use std::time::{Duration, Instant};

use curve_fit::{OptimizerConfig, OptimizerMethod, testing};
use curve_fit_test_support::{AppHarness, harness, state};
use eframe::egui::{self, Key, Modifiers, accesskit::Role};
use egui_kittest::kittest::{NodeT as _, Queryable as _};

mod initialization;
mod layers;
mod optimizer;
mod points;
mod windows;

fn click(harness: &mut AppHarness, label: &str) {
    harness.get_by_label(label).click();
    harness.run();
}

fn replace_text(harness: &mut AppHarness, role: Role, label: &str, text: &str) {
    harness.get_by_role_and_label(role, label).click();
    harness.run();
    assert!(harness.get_by_role_and_label(role, label).is_focused());
    harness.key_press_modifiers(Modifiers::COMMAND, Key::A);
    harness.key_press(Key::Backspace);
    harness.run();
    harness.get_by_role_and_label(role, label).type_text(text);
    harness.run();
}

fn points(harness: &mut AppHarness, text: &str) {
    replace_text(harness, Role::MultilineTextInput, "Input points", text);
}

fn choose(harness: &mut AppHarness, label: &str, choice: &str) {
    harness.get_by_role_and_label(Role::ComboBox, label).click();
    harness.run();
    harness.get_by_label(choice).scroll_to_me();
    harness.run();
    click(harness, choice);
    assert_eq!(
        harness
            .get_by_role_and_label(Role::ComboBox, label)
            .value()
            .as_deref(),
        Some(choice)
    );
}

fn clear_and_undo(harness: &mut AppHarness) {
    click(harness, "Clear points");
    click(harness, "Undo");
    assert!(!state(harness).selected_layer.redo.is_empty());
}

fn paste_layer(harness: &mut AppHarness, text: &str) {
    harness.get_by_label("New layer from clipboard").click();
    // Обрабатываем клик и проверяем запрос платформе до доставки Paste.
    harness.step();
    assert!(
        harness.output().viewport_output.values().any(|output| {
            output
                .commands
                .iter()
                .any(|command| matches!(command, egui::ViewportCommand::RequestPaste))
        }),
        "Clipboard import did not request paste"
    );
    harness
        .input_mut()
        .events
        .push(egui::Event::Paste(text.into()));
    harness.run();
}

fn fit_and_wait(harness: &mut AppHarness) {
    // Автопромотка запрашивает новые кадры даже после окончания фитинга.
    if state(harness).auto_play_on_fit {
        click(harness, "Auto-play");
        assert!(!state(harness).auto_play_on_fit);
    }
    harness.get_by_role_and_label(Role::Button, "Fit").click();
    harness.step();
    let deadline = Instant::now() + Duration::from_secs(10);
    while state(harness).fit_in_progress {
        assert!(Instant::now() < deadline, "Fit did not complete in time");
        std::thread::sleep(Duration::from_millis(1));
        harness.step();
    }
    let snapshot = state(harness);
    assert!(snapshot.error.is_none(), "Fit failed: {:?}", snapshot.error);
    assert!(snapshot.fit_result.is_some(), "Fit produced no result");
    harness.run();
}
