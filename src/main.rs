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
        let app_loader = document.get_element_by_id("app_loader");
        let app_loader_spinner = document.get_element_by_id("app_loader_spinner");
        let app_loader_status = document.get_element_by_id("app_loader_status");
        let canvas = document
            .get_element_by_id("curve_fit_canvas")
            .expect("Canvas element '#curve_fit_canvas' was not found")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Element '#curve_fit_canvas' is not an HtmlCanvasElement");

        let web_options = eframe::WebOptions::default();
        let web_runner = eframe::WebRunner::new();

        let start_result = web_runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(CurveFitApp::new(cc)))),
            )
            .await;

        match start_result {
            Ok(()) => remove_app_loader(app_loader.as_ref()),
            Err(error) => {
                eprintln!("Failed to start web application: {error:?}");
                show_app_loader_error(
                    app_loader.as_ref(),
                    app_loader_spinner.as_ref(),
                    app_loader_status.as_ref(),
                );
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn remove_app_loader(app_loader: Option<&web_sys::Element>) {
    if let Some(app_loader) = app_loader {
        app_loader.remove();
    }
}

#[cfg(target_arch = "wasm32")]
fn show_app_loader_error(
    app_loader: Option<&web_sys::Element>,
    app_loader_spinner: Option<&web_sys::Element>,
    app_loader_status: Option<&web_sys::Element>,
) {
    if let Some(app_loader) = app_loader {
        if let Err(error) = app_loader.set_attribute("class", "app-loader app-loader-error") {
            eprintln!("Failed to update app loader state: {error:?}");
        }
    }

    if let Some(app_loader_spinner) = app_loader_spinner {
        if let Err(error) = app_loader_spinner.set_attribute("style", "display: none;") {
            eprintln!("Failed to hide app loader spinner: {error:?}");
        }
    }

    if let Some(app_loader_status) = app_loader_status {
        app_loader_status.set_inner_html("Failed to load application. Check console for details.");
    }
}
