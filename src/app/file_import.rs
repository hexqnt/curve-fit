//! Импорт точек из CSV/XLSX и интеграция с файловым диалогом пользовательского интерфейса.

#[cfg(not(target_arch = "wasm32"))]
use super::points_text::parse_point_from_clipboard_like_fragments;
use super::*;

#[cfg(not(target_arch = "wasm32"))]
use std::borrow::Cow;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use calamine::{Data, Reader, open_workbook_auto};
#[cfg(not(target_arch = "wasm32"))]
use egui_file_dialog::DialogState;

#[cfg(not(target_arch = "wasm32"))]
const CSV_DELIMITER_CANDIDATES: [u8; 4] = *b",;\t|";
#[cfg(not(target_arch = "wasm32"))]
const MAX_CSV_DELIMITER_DETECTION_LINES: usize = 64;
#[cfg(not(target_arch = "wasm32"))]
const UTF8_BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, Default)]
struct DelimiterScore {
    lines_with_hits: usize,
    total_hits: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl CurveFitApp {
    pub(super) fn points_file_import_in_progress(&self) -> bool {
        matches!(self.points_file_import_dialog.state(), DialogState::Open)
    }

    pub(super) fn request_points_file_import(&mut self) {
        if self.fit_in_progress || self.points_file_import_in_progress() {
            return;
        }
        if let Some(directory) = self
            .points_file_import_last_directory
            .clone()
            .filter(|directory| directory.is_dir())
        {
            self.points_file_import_dialog
                .config_mut()
                .initial_directory = directory;
        }
        self.points_file_import_dialog.pick_file();
    }

    pub(super) fn poll_points_file_import_dialog(&mut self, ctx: &egui::Context) {
        self.points_file_import_dialog.update(ctx);

        let Some(path) = self.points_file_import_dialog.take_picked() else {
            return;
        };
        self.handle_points_file_import_path(&path);
    }

    pub(super) fn handle_points_file_import_path(&mut self, path: &Path) {
        self.points_file_import_last_directory = dialog_directory_from_path(path);
        if let Err(error) = self.import_points_from_file(path) {
            self.set_file_import_error(error);
            return;
        }

        self.status = Some(self.idle_status_after_points_edit());
    }

    pub(super) fn import_points_from_file(&mut self, path: &Path) -> Result<usize, String> {
        let points = parse_points_from_data_file(path)?;
        let imported_count = points.len();
        self.write_points_text(&points, true);
        Ok(imported_count)
    }

    fn set_file_import_error(&mut self, message: impl AsRef<str>) {
        let message = message.as_ref();
        if message.starts_with(FILE_IMPORT_ERROR_PREFIX) {
            self.status = Some(StatusMessage::Error(message.to_owned()));
        } else {
            self.status = Some(StatusMessage::Error(format!(
                "{FILE_IMPORT_ERROR_PREFIX}{message}"
            )));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_points_from_data_file(path: &Path) -> Result<Vec<Point>, String> {
    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::trim)
        .filter(|extension| !extension.is_empty())
        .ok_or_else(|| {
            format!(
                "Unsupported file extension for '{}': expected .csv or .xlsx",
                path.display()
            )
        })?;

    if extension.eq_ignore_ascii_case("csv") {
        return parse_points_from_csv_file(path);
    }
    if extension.eq_ignore_ascii_case("xlsx") {
        return parse_points_from_xlsx_file(path);
    }

    Err(format!(
        "Unsupported file extension '.{extension}' for '{}': expected .csv or .xlsx",
        path.display()
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_points_from_csv_file(path: &Path) -> Result<Vec<Point>, String> {
    let file_bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read CSV file '{}': {error}", path.display()))?;
    parse_points_from_csv_bytes(&file_bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_points_from_csv_bytes(file_bytes: &[u8]) -> Result<Vec<Point>, String> {
    let normalized = strip_utf8_bom(file_bytes);
    let delimiter = detect_csv_delimiter(normalized);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .delimiter(delimiter)
        .from_reader(std::io::Cursor::new(normalized));

    let mut points = Vec::new();
    for (line_index, record_result) in reader.byte_records().enumerate() {
        let line_number = line_index + 1;
        let record = record_result
            .map_err(|error| format!("CSV parse error at line {line_number}: {error}"))?;

        if let Some(point) = parse_point_from_clipboard_like_fragments(
            line_number,
            record.iter().map(String::from_utf8_lossy),
        )? {
            points.push(point);
        }
    }

    if points.is_empty() {
        return Err("No valid points found in CSV file".to_string());
    }

    Ok(points)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_points_from_xlsx_file(path: &Path) -> Result<Vec<Point>, String> {
    let mut workbook = open_workbook_auto(path)
        .map_err(|error| format!("Failed to open Excel file '{}': {error}", path.display()))?;
    let sheet_name = workbook.sheet_names().into_iter().next().ok_or_else(|| {
        format!(
            "Excel file '{}' does not contain any worksheet",
            path.display()
        )
    })?;

    let range = workbook.worksheet_range(&sheet_name).map_err(|error| {
        format!(
            "Failed to read worksheet '{sheet_name}' in '{}': {error}",
            path.display()
        )
    })?;

    parse_points_from_xlsx_range(&range)
        .map_err(|error| format!("Worksheet '{sheet_name}': {error}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_points_from_xlsx_range(range: &calamine::Range<Data>) -> Result<Vec<Point>, String> {
    let mut points = Vec::new();
    let first_row_number = range
        .start()
        .map(|(row, _)| row as usize + 1)
        .unwrap_or(1usize);

    for (row_index, row) in range.rows().enumerate() {
        // Нумерация ошибок дается в координатах листа (1-based), чтобы ее было проще
        // сопоставлять с номером строки в табличном редакторе.
        let line_number = first_row_number + row_index;

        if let Some(point) = parse_point_from_clipboard_like_fragments(
            line_number,
            row.iter().map(spreadsheet_cell_to_text),
        )? {
            points.push(point);
        }
    }

    if points.is_empty() {
        return Err("No valid points found in Excel worksheet".to_string());
    }

    Ok(points)
}

#[cfg(not(target_arch = "wasm32"))]
fn spreadsheet_cell_to_text(cell: &Data) -> Cow<'_, str> {
    match cell {
        Data::Int(value) => Cow::Owned(value.to_string()),
        Data::Float(value) => Cow::Owned(value.to_string()),
        Data::String(value) | Data::DateTimeIso(value) | Data::DurationIso(value) => {
            Cow::Borrowed(value.as_str())
        }
        Data::Bool(value) => Cow::Borrowed(if *value { "true" } else { "false" }),
        Data::DateTime(value) => Cow::Owned(value.to_string()),
        Data::Error(error) => Cow::Owned(error.to_string()),
        Data::Empty => Cow::Borrowed(""),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn strip_utf8_bom(file_bytes: &[u8]) -> &[u8] {
    if file_bytes.starts_with(&UTF8_BOM) {
        &file_bytes[UTF8_BOM.len()..]
    } else {
        file_bytes
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_csv_delimiter(file_bytes: &[u8]) -> u8 {
    let mut scores = [DelimiterScore::default(); CSV_DELIMITER_CANDIDATES.len()];
    let mut analyzed_non_empty_lines = 0usize;

    for raw_line in file_bytes.split(|byte| *byte == b'\n') {
        let line = trim_ascii_whitespace(raw_line);
        if line.is_empty() {
            continue;
        }

        analyzed_non_empty_lines += 1;
        for (index, delimiter) in CSV_DELIMITER_CANDIDATES.iter().copied().enumerate() {
            let hits = count_unquoted_delimiter(line, delimiter);
            if hits > 0 {
                scores[index].lines_with_hits += 1;
                scores[index].total_hits += hits;
            }
        }

        if analyzed_non_empty_lines >= MAX_CSV_DELIMITER_DETECTION_LINES {
            break;
        }
    }

    let mut best_delimiter = CSV_DELIMITER_CANDIDATES[0];
    let mut best_score = scores[0];
    for (index, score) in scores.iter().copied().enumerate().skip(1) {
        if (score.lines_with_hits, score.total_hits)
            > (best_score.lines_with_hits, best_score.total_hits)
        {
            best_score = score;
            best_delimiter = CSV_DELIMITER_CANDIDATES[index];
        }
    }

    if best_score.lines_with_hits == 0 {
        CSV_DELIMITER_CANDIDATES[0]
    } else {
        best_delimiter
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn trim_ascii_whitespace(input: &[u8]) -> &[u8] {
    let start = input
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(input.len());
    let end = input
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|index| index + 1)
        .unwrap_or(start);
    &input[start..end]
}

#[cfg(not(target_arch = "wasm32"))]
fn count_unquoted_delimiter(line: &[u8], delimiter: u8) -> usize {
    let mut quote_open = false;
    let mut count = 0usize;
    let mut index = 0usize;

    while index < line.len() {
        match line[index] {
            b'"' => {
                if quote_open && line.get(index + 1).copied() == Some(b'"') {
                    index += 2;
                    continue;
                }
                quote_open = !quote_open;
            }
            byte if byte == delimiter && !quote_open => {
                count += 1;
            }
            _ => {}
        }
        index += 1;
    }

    count
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use calamine::{Cell, Data, Range};
    use std::path::{Path, PathBuf};

    fn assert_approx_eq(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }

    fn write_temp_file(extension: &str, contents: &[u8]) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};

        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time must be after UNIX_EPOCH")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "curve-fit-file-import-{}-{suffix}.{extension}",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("temporary import file must be writable");
        path
    }

    fn cleanup_temp_file(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn csv_parser_detects_supported_delimiters() {
        let samples = [
            ("1,2\n3,4\n", (3.0, 4.0)),
            ("1;2\n3;4\n", (3.0, 4.0)),
            ("1\t2\n3\t4\n", (3.0, 4.0)),
            ("1|2\n3|4\n", (3.0, 4.0)),
        ];

        for (text, expected_last) in samples {
            let points = parse_points_from_csv_bytes(text.as_bytes())
                .expect("csv payload with valid delimiter must parse");
            assert_eq!(points.len(), 2);
            assert_approx_eq(points[1].x(), expected_last.0, 1e-12);
            assert_approx_eq(points[1].y(), expected_last.1, 1e-12);
        }
    }

    #[test]
    fn csv_parser_supports_decimal_comma_scientific_and_mixed_rows() {
        let text = "sample;meta\nx;y\n1,23e-3;4.5E+1\nalpha;x=-2,5 y=6,0e2\n";
        let points = parse_points_from_csv_bytes(text.as_bytes()).expect("csv payload must parse");

        assert_eq!(points.len(), 2);
        assert_approx_eq(points[0].x(), 1.23e-3, 1e-15);
        assert_approx_eq(points[0].y(), 45.0, 1e-12);
        assert_approx_eq(points[1].x(), -2.5, 1e-12);
        assert_approx_eq(points[1].y(), 600.0, 1e-12);
    }

    #[test]
    fn csv_parser_fails_on_line_with_three_values() {
        let error =
            parse_points_from_csv_bytes(b"1,2,3\n").expect_err("row with three values must fail");
        assert!(
            error.contains("Line 1: expected exactly two numeric values, got 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn csv_parser_fails_when_no_points_found() {
        let error = parse_points_from_csv_bytes(b"header;value\nname;unit\n")
            .expect_err("csv payload without points must fail");
        assert!(
            error.contains("No valid points found in CSV file"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn xlsx_range_parser_supports_numeric_and_text_cells() {
        let range = Range::from_sparse(vec![
            Cell::new((0, 0), Data::String("name".to_string())),
            Cell::new((0, 1), Data::String("value".to_string())),
            Cell::new((1, 0), Data::Float(1.5)),
            Cell::new((1, 1), Data::String("2,75".to_string())),
            Cell::new((2, 0), Data::String("sample".to_string())),
            Cell::new((2, 1), Data::String("x=-3 y=4".to_string())),
        ]);

        let points =
            parse_points_from_xlsx_range(&range).expect("xlsx range with valid rows must parse");

        assert_eq!(points.len(), 2);
        assert_approx_eq(points[0].x(), 1.5, 1e-12);
        assert_approx_eq(points[0].y(), 2.75, 1e-12);
        assert_approx_eq(points[1].x(), -3.0, 1e-12);
        assert_approx_eq(points[1].y(), 4.0, 1e-12);
    }

    #[test]
    fn xlsx_range_parser_fails_on_line_with_three_values() {
        let range = Range::from_sparse(vec![
            Cell::new((0, 0), Data::Int(1)),
            Cell::new((0, 1), Data::Int(2)),
            Cell::new((0, 2), Data::Int(3)),
        ]);

        let error =
            parse_points_from_xlsx_range(&range).expect_err("xlsx row with three values must fail");
        assert!(
            error.contains("Line 1: expected exactly two numeric values, got 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn xlsx_range_parser_uses_sheet_row_number_in_error() {
        let range = Range::from_sparse(vec![
            Cell::new((9, 0), Data::Int(1)),
            Cell::new((9, 1), Data::Int(2)),
            Cell::new((9, 2), Data::Int(3)),
        ]);

        let error =
            parse_points_from_xlsx_range(&range).expect_err("xlsx row with three values must fail");
        assert!(
            error.contains("Line 10: expected exactly two numeric values, got 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn xlsx_range_parser_fails_when_no_points_found() {
        let range = Range::from_sparse(vec![
            Cell::new((0, 0), Data::String("name".to_string())),
            Cell::new((0, 1), Data::String("value".to_string())),
        ]);

        let error =
            parse_points_from_xlsx_range(&range).expect_err("empty xlsx worksheet must fail");
        assert!(
            error.contains("No valid points found in Excel worksheet"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn file_import_replaces_points_text_pushes_undo_and_clears_redo() {
        let previous_text = "0 1\n1 2\n";
        let mut app = CurveFitApp {
            points: super::PointsEditorState {
                text: previous_text.to_string(),
                redo_stack: vec!["stale redo entry".to_string()],
                ..Default::default()
            },
            status: Some(StatusMessage::Error(format!(
                "{}previous error",
                super::FILE_IMPORT_ERROR_PREFIX
            ))),
            ..Default::default()
        };
        let path = write_temp_file("csv", b"10;20\n30;40\n");

        app.handle_points_file_import_path(&path);
        cleanup_temp_file(&path);

        assert_eq!(
            app.points.text,
            "10.00000000 20.00000000\n30.00000000 40.00000000\n"
        );
        assert_eq!(app.points.undo_stack, vec![previous_text.to_string()]);
        assert!(app.points.redo_stack.is_empty());
        assert!(matches!(app.status, Some(StatusMessage::Ready)));
    }

    #[test]
    fn file_import_error_keeps_existing_points_text() {
        let previous_text = "0 1\n1 2\n";
        let mut app = CurveFitApp {
            points: super::PointsEditorState {
                text: previous_text.to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let path = write_temp_file("csv", b"1,2,3\n");

        app.handle_points_file_import_path(&path);
        cleanup_temp_file(&path);

        assert_eq!(app.points.text, previous_text);
        assert!(app.points.undo_stack.is_empty());
        assert!(matches!(
            app.status.as_ref(),
            Some(StatusMessage::Error(message)) if message.starts_with(super::FILE_IMPORT_ERROR_PREFIX)
        ));
    }
}
