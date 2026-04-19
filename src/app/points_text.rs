//! Парсинг и сериализация текстового и clipboard-представления точек.

use std::borrow::Cow;
use std::fmt::Write as _;

use egui_plot::PlotPoint;

use crate::domain::Point;

use super::ParsedPointsCache;

pub(super) fn parse_f64(field_name: &str, raw_value: &str) -> Result<f64, String> {
    let trimmed = raw_value.trim();
    if trimmed.is_empty() {
        return Err(format!("Field '{field_name}' is empty"));
    }

    let normalized = if trimmed.contains(',') && !trimmed.contains('.') {
        Cow::Owned(trimmed.replace(',', "."))
    } else {
        Cow::Borrowed(trimmed)
    };

    normalized
        .parse::<f64>()
        .map_err(|error| format!("Failed to parse '{field_name}' as f64: {error}"))
}

fn parse_point_line(line_number: usize, raw_line: &str) -> Result<Option<Point>, String> {
    let line = raw_line.trim();
    if line.is_empty() {
        return Ok(None);
    }

    let mut tokens = line
        .split(|symbol: char| symbol.is_whitespace() || symbol == ';')
        .filter(|token| !token.is_empty());

    let x_raw = tokens.next().ok_or_else(|| {
        format!("Line {line_number}: expected two values, got empty line after parsing")
    })?;
    let y_raw = tokens
        .next()
        .ok_or_else(|| format!("Line {line_number}: expected two values 'x y'"))?;
    if tokens.next().is_some() {
        return Err(format!(
            "Line {line_number}: expected exactly two values separated by space/tab/';'"
        ));
    }

    let x = parse_f64(&format!("line {line_number} x"), x_raw)?;
    let y = parse_f64(&format!("line {line_number} y"), y_raw)?;
    let point = Point::try_new(x, y).map_err(|error| format!("Line {line_number}: {error}"))?;
    Ok(Some(point))
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn is_number_start_byte(byte: u8) -> bool {
    byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.' | b',')
}

fn is_digit_after_disallowed_prefix(bytes: &[u8], index: usize) -> bool {
    if index == 0 || !bytes[index].is_ascii_digit() {
        return false;
    }

    matches!(bytes[index - 1], b'+' | b'-' | b'.' | b',')
}

fn parse_numeric_token_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start;
    if matches!(bytes[index], b'+' | b'-') {
        index += 1;
        if index >= bytes.len() {
            return None;
        }
    }

    let integer_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    let has_integer = index > integer_start;

    let mut has_fraction = false;
    if index < bytes.len() && matches!(bytes[index], b'.' | b',') {
        index += 1;
        let fraction_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        has_fraction = index > fraction_start;
    }

    if !has_integer && !has_fraction {
        return None;
    }

    if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
        let exponent_marker_index = index;
        index += 1;
        if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
            index += 1;
        }
        let exponent_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == exponent_start {
            // Экспонента без цифр не считается частью токена.
            index = exponent_marker_index;
        }
    }

    Some(index)
}

fn clipboard_numeric_tokens(line: &str) -> Vec<&str> {
    // Сканируем строку по байтам без regex, чтобы дешевле отсеивать числа внутри
    // идентификаторов вроде `row-1`, `v1.2` или `sample_3`.
    let bytes = line.as_bytes();
    let mut tokens = Vec::with_capacity(4);
    let mut index = 0;

    while index < bytes.len() {
        let current = bytes[index];
        if !is_number_start_byte(current) {
            index += 1;
            continue;
        }

        if is_digit_after_disallowed_prefix(bytes, index) {
            index += 1;
            continue;
        }

        let previous_is_word = index > 0 && is_word_byte(bytes[index - 1]);
        if previous_is_word {
            index += 1;
            continue;
        }

        let Some(end) = parse_numeric_token_end(bytes, index) else {
            index += 1;
            continue;
        };
        if end <= index {
            index += 1;
            continue;
        }

        tokens.push(&line[index..end]);
        index = end;
    }

    tokens
}

/// Парсит одну строку/запись в режиме `Clipboard-like`:
/// - `0` числовых токенов -> строка пропускается;
/// - `2` числовых токена -> создается точка `(x, y)`;
/// - `1` или `3+` токенов -> ошибка.
pub(super) fn parse_point_from_clipboard_like_fragments<I, S>(
    line_number: usize,
    fragments: I,
) -> Result<Option<Point>, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut numeric_count = 0usize;
    let mut first_token = None::<String>;
    let mut second_token = None::<String>;

    for fragment in fragments {
        for token in clipboard_numeric_tokens(fragment.as_ref()) {
            numeric_count += 1;
            if numeric_count == 1 {
                first_token = Some(token.to_string());
            } else if numeric_count == 2 {
                second_token = Some(token.to_string());
            }
        }
    }

    match numeric_count {
        0 => Ok(None),
        2 => {
            let x_raw = first_token
                .as_deref()
                .expect("first numeric token must exist for count=2");
            let y_raw = second_token
                .as_deref()
                .expect("second numeric token must exist for count=2");
            let x = parse_f64(&format!("line {line_number} x"), x_raw)?;
            let y = parse_f64(&format!("line {line_number} y"), y_raw)?;
            let point =
                Point::try_new(x, y).map_err(|error| format!("Line {line_number}: {error}"))?;
            Ok(Some(point))
        }
        count => Err(format!(
            "Line {line_number}: expected exactly two numeric values, got {count}"
        )),
    }
}

/// Парсит произвольный текст из буфера обмена в список точек `(x, y)`.
///
/// Правила:
/// - строки без числовых токенов пропускаются;
/// - строка с 1 числом или 3+ числами считается ошибкой;
/// - строка с ровно 2 числами конвертируется в точку.
pub(super) fn parse_points_from_clipboard_text(text: &str) -> Result<Vec<Point>, String> {
    let mut points = Vec::new();

    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(point) =
            parse_point_from_clipboard_like_fragments(line_number, std::iter::once(line))?
        {
            points.push(point);
        }
    }

    if points.is_empty() {
        return Err("No valid points found in clipboard text".to_string());
    }

    Ok(points)
}

/// Парсит многострочный ввод точек и подготавливает данные для графика.
///
/// При наличии ошибок возвращает первую ошибку парсинга,
/// но при этом сохраняет успешно распарсенные точки для визуального предпросмотра.
pub(super) fn parse_points_text_cache(text: &str) -> ParsedPointsCache {
    let mut parsed_points = Vec::new();
    let mut plot_points = Vec::new();
    let mut parse_error = None;
    let mut parse_error_line = None;

    for (line_index, line) in text.lines().enumerate() {
        match parse_point_line(line_index + 1, line) {
            Ok(Some(point)) => {
                plot_points.push(PlotPoint::new(point.x(), point.y()));
                parsed_points.push(point);
            }
            Ok(None) => {}
            Err(error) => {
                // Сохраняем первую ошибку, но продолжаем разбор ради частичного preview на графике.
                if parse_error.is_none() {
                    parse_error = Some(error);
                    parse_error_line = Some(line_index + 1);
                }
            }
        }
    }

    let parsed_points = match parse_error {
        Some(error) => Err(error),
        None => Ok(parsed_points),
    };

    ParsedPointsCache {
        parsed_points,
        parse_error_line,
        plot_points: plot_points.into(),
    }
}

/// Сериализует точки обратно в текстовый формат `x y` (по одной точке на строку).
pub(super) fn points_to_text(points: &[Point]) -> String {
    if points.is_empty() {
        return String::new();
    }

    let mut text = String::with_capacity(points.len() * 24);
    for point in points {
        writeln!(&mut text, "{:.8} {:.8}", point.x(), point.y())
            .expect("writing points to String must succeed");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }

    #[test]
    fn clipboard_like_row_parser_parses_two_values() {
        let point =
            parse_point_from_clipboard_like_fragments(5, ["meta", "x=1.5", "note", "y=2,75"])
                .expect("row must parse")
                .expect("row must produce a point");

        assert_approx_eq(point.x(), 1.5, 1e-12);
        assert_approx_eq(point.y(), 2.75, 1e-12);
    }

    #[test]
    fn clipboard_like_row_parser_skips_rows_without_numbers() {
        let point = parse_point_from_clipboard_like_fragments(3, ["header", "label", "unit"])
            .expect("row without numbers must be skipped");
        assert!(point.is_none());
    }

    #[test]
    fn clipboard_like_row_parser_fails_on_single_value() {
        let error = parse_point_from_clipboard_like_fragments(2, ["id=42"])
            .expect_err("row with one numeric value must fail");
        assert!(
            error.contains("Line 2: expected exactly two numeric values, got 1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn clipboard_like_row_parser_fails_on_three_values() {
        let error = parse_point_from_clipboard_like_fragments(7, ["1", "2", "3"])
            .expect_err("row with three numeric values must fail");
        assert!(
            error.contains("Line 7: expected exactly two numeric values, got 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn clipboard_parser_accepts_mixed_delimiters() {
        let text = "0 1\n2\t3\n4;5\n6|7\n8/9";
        let points = parse_points_from_clipboard_text(text).expect("clipboard text must parse");
        assert_eq!(points.len(), 5);
        assert_approx_eq(points[0].x(), 0.0, 1e-12);
        assert_approx_eq(points[0].y(), 1.0, 1e-12);
        assert_approx_eq(points[4].x(), 8.0, 1e-12);
        assert_approx_eq(points[4].y(), 9.0, 1e-12);
    }

    #[test]
    fn clipboard_parser_supports_decimal_comma_and_scientific_notation() {
        let text = "1,23e-3;4.5E+1\n-2,5\t6,0e2";
        let points = parse_points_from_clipboard_text(text).expect("clipboard text must parse");
        assert_eq!(points.len(), 2);
        assert_approx_eq(points[0].x(), 1.23e-3, 1e-15);
        assert_approx_eq(points[0].y(), 45.0, 1e-12);
        assert_approx_eq(points[1].x(), -2.5, 1e-12);
        assert_approx_eq(points[1].y(), 600.0, 1e-12);
    }

    #[test]
    fn clipboard_parser_allows_extra_non_numeric_columns() {
        let text = "sample_a alpha x=1.5 beta y=2,75 note";
        let points = parse_points_from_clipboard_text(text).expect("clipboard text must parse");
        assert_eq!(points.len(), 1);
        assert_approx_eq(points[0].x(), 1.5, 1e-12);
        assert_approx_eq(points[0].y(), 2.75, 1e-12);
    }

    #[test]
    fn clipboard_parser_skips_lines_without_numbers() {
        let text = "name;value\nheader row\n1;2";
        let points = parse_points_from_clipboard_text(text).expect("clipboard text must parse");
        assert_eq!(points.len(), 1);
        assert_approx_eq(points[0].x(), 1.0, 1e-12);
        assert_approx_eq(points[0].y(), 2.0, 1e-12);
    }

    #[test]
    fn clipboard_parser_ignores_embedded_numbers_inside_labels() {
        let text = "row-1 v1.2 label\nx=3 y=4";
        let points = parse_points_from_clipboard_text(text).expect("clipboard text must parse");
        assert_eq!(points.len(), 1);
        assert_approx_eq(points[0].x(), 3.0, 1e-12);
        assert_approx_eq(points[0].y(), 4.0, 1e-12);
    }

    #[test]
    fn clipboard_parser_fails_on_single_numeric_value_per_data_line() {
        let error = parse_points_from_clipboard_text("row id=42")
            .expect_err("line with one numeric value must fail");
        assert!(
            error.contains("Line 1: expected exactly two numeric values, got 1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn clipboard_parser_fails_on_three_or_more_numeric_values() {
        let error = parse_points_from_clipboard_text("1 2 3")
            .expect_err("line with three numeric values must fail");
        assert!(
            error.contains("Line 1: expected exactly two numeric values, got 3"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn clipboard_parser_fails_when_no_points_are_found() {
        let error = parse_points_from_clipboard_text(" \n\t\n")
            .expect_err("empty clipboard payload must fail");
        assert!(
            error.contains("No valid points found in clipboard text"),
            "unexpected error: {error}"
        );
    }
}
