//! Импорт точек из буфера обмена для native- и wasm-режимов.

use super::*;

impl CurveFitApp {
    pub(super) fn clipboard_import_in_progress(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.clipboard_import_request_pending
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.clipboard_import_web_in_flight
        }
    }

    pub(super) fn request_points_clipboard_import(&mut self, ctx: &egui::Context) {
        if self.fit_in_progress {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.clipboard_import_request_pending {
                return;
            }
            self.clipboard_import_request_pending = true;
            self.clipboard_import_requested_at = Some(Instant::now());
            ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaste);
            ctx.request_repaint();
        }

        #[cfg(target_arch = "wasm32")]
        {
            if self.clipboard_import_web_in_flight {
                return;
            }
            self.clipboard_import_web_in_flight = true;
            self.clipboard_import_web_result.borrow_mut().take();

            let ctx = ctx.clone();
            let result_slot = Rc::clone(&self.clipboard_import_web_result);
            wasm_bindgen_futures::spawn_local(async move {
                let result = read_text_from_web_clipboard().await;
                *result_slot.borrow_mut() = Some(result);
                ctx.request_repaint();
            });
        }
    }

    pub(super) fn poll_points_clipboard_import(&mut self, ctx: &egui::Context) {
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;

        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.clipboard_import_request_pending {
                if let Some(text) = take_requested_paste_event(ctx) {
                    self.clipboard_import_request_pending = false;
                    self.clipboard_import_requested_at = None;
                    self.handle_points_clipboard_import_result(Ok(text));
                    return;
                }

                let timed_out = self
                    .clipboard_import_requested_at
                    .is_some_and(|requested_at| {
                        Instant::now().saturating_duration_since(requested_at)
                            >= Duration::from_millis(CLIPBOARD_IMPORT_PASTE_TIMEOUT_MS)
                    });
                if timed_out {
                    self.clipboard_import_request_pending = false;
                    self.clipboard_import_requested_at = None;
                    self.handle_points_clipboard_import_result(Err(
                        "Clipboard is empty or unavailable".to_string(),
                    ));
                } else {
                    ctx.request_repaint();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if !self.clipboard_import_web_in_flight {
                return;
            }

            let maybe_result = self.clipboard_import_web_result.borrow_mut().take();
            if let Some(result) = maybe_result {
                self.clipboard_import_web_in_flight = false;
                self.handle_points_clipboard_import_result(result);
            }
        }
    }

    pub(super) fn copy_text_to_clipboard(&mut self, ctx: &egui::Context, text: String) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            ctx.copy_text(text);
        }

        #[cfg(target_arch = "wasm32")]
        {
            if self.clipboard_copy_web_in_flight {
                return;
            }

            let promise = match start_write_text_to_web_clipboard(&text) {
                Ok(promise) => promise,
                Err(error) => {
                    self.set_clipboard_copy_error(error);
                    return;
                }
            };

            let ctx = ctx.clone();
            let result_slot = Rc::clone(&self.clipboard_copy_web_result);
            self.clipboard_copy_web_in_flight = true;
            self.clipboard_copy_web_result.borrow_mut().take();
            wasm_bindgen_futures::spawn_local(async move {
                let result = wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map(|_| ())
                    .map_err(|error| {
                        format!(
                            "Failed to write clipboard text: {}",
                            describe_web_clipboard_js_error(&error)
                        )
                    });
                *result_slot.borrow_mut() = Some(result);
                ctx.request_repaint();
            });
        }
    }

    pub(super) fn poll_clipboard_copy(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = ctx;

        #[cfg(target_arch = "wasm32")]
        {
            if !self.clipboard_copy_web_in_flight {
                return;
            }

            let maybe_result = self.clipboard_copy_web_result.borrow_mut().take();
            if let Some(result) = maybe_result {
                self.clipboard_copy_web_in_flight = false;
                if let Err(error) = result {
                    self.set_clipboard_copy_error(error);
                }
            } else {
                ctx.request_repaint();
            }
        }
    }

    pub(super) fn handle_points_clipboard_import_result(&mut self, result: Result<String, String>) {
        let text = match result {
            Ok(text) => text,
            Err(error) => {
                self.set_clipboard_import_error(error);
                return;
            }
        };

        if text.trim().is_empty() {
            self.set_clipboard_import_error("Clipboard text is empty");
            return;
        }

        if let Err(error) = self.import_points_from_clipboard_text(&text) {
            self.set_clipboard_import_error(error);
            return;
        }

        self.status = Some(self.idle_status_after_points_edit());
    }

    pub(super) fn import_points_from_clipboard_text(
        &mut self,
        text: &str,
    ) -> Result<usize, String> {
        let points = parse_points_from_clipboard_text(text)?;
        let imported_count = points.len();
        self.create_point_layer_from_points(&points);
        Ok(imported_count)
    }

    fn set_clipboard_import_error(&mut self, message: impl AsRef<str>) {
        let message = message.as_ref();
        if message.starts_with(CLIPBOARD_IMPORT_ERROR_PREFIX) {
            self.status = Some(StatusMessage::Error(message.to_owned()));
        } else {
            self.status = Some(StatusMessage::Error(format!(
                "{CLIPBOARD_IMPORT_ERROR_PREFIX}{message}"
            )));
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn set_clipboard_copy_error(&mut self, message: impl AsRef<str>) {
        let message = message.as_ref();
        if message.starts_with(CLIPBOARD_COPY_ERROR_PREFIX) {
            self.status = Some(StatusMessage::Error(message.to_owned()));
        } else {
            self.status = Some(StatusMessage::Error(format!(
                "{CLIPBOARD_COPY_ERROR_PREFIX}{message}"
            )));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn take_requested_paste_event(ctx: &egui::Context) -> Option<String> {
    ctx.input_mut(|input| {
        let event_index = input
            .events
            .iter()
            .position(|event| matches!(event, egui::Event::Paste(_)))?;
        match input.events.remove(event_index) {
            egui::Event::Paste(text) => Some(text),
            _ => None,
        }
    })
}

#[cfg(target_arch = "wasm32")]
async fn read_text_from_web_clipboard() -> Result<String, String> {
    use wasm_bindgen_futures::JsFuture;

    let window = web_clipboard_window()?;
    let clipboard = window.navigator().clipboard();
    let text = JsFuture::from(clipboard.read_text())
        .await
        .map_err(|error| {
            format!(
                "Failed to read clipboard text: {}",
                describe_web_clipboard_js_error(&error)
            )
        })?
        .as_string()
        .ok_or_else(|| "Clipboard did not return text content".to_string())?;

    Ok(text)
}

#[cfg(target_arch = "wasm32")]
fn start_write_text_to_web_clipboard(text: &str) -> Result<web_sys::js_sys::Promise, String> {
    // Для web Clipboard API запись должна стартовать прямо в обработчике user gesture.
    let window = web_clipboard_window()?;
    let clipboard = window.navigator().clipboard();
    Ok(clipboard.write_text(text))
}

#[cfg(target_arch = "wasm32")]
fn web_clipboard_window() -> Result<web_sys::Window, String> {
    let window = web_sys::window().ok_or_else(|| "Window is unavailable".to_string())?;
    if !window.is_secure_context() {
        return Err(
            "Clipboard API is unavailable in non-secure context (HTTPS or localhost required)"
                .to_string(),
        );
    }
    Ok(window)
}

#[cfg(target_arch = "wasm32")]
fn describe_web_clipboard_js_error(error: &wasm_bindgen::JsValue) -> String {
    if let Some(message) = error.as_string() {
        return message;
    }

    let message = web_sys::js_sys::Reflect::get(error, &wasm_bindgen::JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string());
    let name = web_sys::js_sys::Reflect::get(error, &wasm_bindgen::JsValue::from_str("name"))
        .ok()
        .and_then(|value| value.as_string());

    match (name, message) {
        (Some(name), Some(message)) => format!("{name}: {message}"),
        (None, Some(message)) => message,
        (Some(name), None) => name,
        (None, None) => format!("{error:?}"),
    }
}
