use curve_fit::CurveFitApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "curve-fit",
        native_options,
        Box::new(|cc| Ok(Box::new(CurveFitApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast;

    wasm_bindgen_futures::spawn_local(async {
        let window = web_sys::window().expect("Window is not available");
        let document = window.document().expect("Document is not available");
        let canvas = document
            .get_element_by_id("curve_fit_canvas")
            .expect("Canvas element '#curve_fit_canvas' was not found")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Element '#curve_fit_canvas' is not an HtmlCanvasElement");

        let web_options = eframe::WebOptions::default();
        let web_runner = eframe::WebRunner::new();

        web_runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(CurveFitApp::new(cc)))),
            )
            .await
            .expect("Failed to start web application");
    });
}
