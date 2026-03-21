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
        trimmed.replace(',', ".")
    } else {
        trimmed.to_string()
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
        plot_points,
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
