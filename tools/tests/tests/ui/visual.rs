use super::*;

#[test]
fn widget_states_match_theme_snapshots() {
    let mut snapshots = egui_kittest::SnapshotResults::new();
    for (theme, name) in [("☀ Light", "light"), ("🌙 Dark", "dark")] {
        let (mut harness, pause) = harness_with_paused_fit();
        click(&mut harness, theme);
        harness.remove_cursor();
        harness.run();
        harness.snapshot(format!("{name}_idle"));

        click(&mut harness, "Single point tool");
        harness.get_by_label("New layer").hover();
        // Фиксируем hover до задержки появления подсказки.
        harness.run_steps(2);
        harness.snapshot(format!("{name}_selected_hovered"));

        points(&mut harness, "0 1\n1 3\n2 5\n3 7\n");
        start_paused_fit(&mut harness);
        harness.remove_cursor();
        harness.run_steps(20);
        harness.snapshot(format!("{name}_fitting"));

        harness.get_by_role_and_label(Role::Button, "Stop").click();
        harness.step();
        pause.resume();
        wait_for_fit(&mut harness);
        snapshots.extend_harness(&mut harness);
    }
}
