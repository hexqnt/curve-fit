#[cfg(not(target_arch = "wasm32"))]
use std::hash::{DefaultHasher, Hash, Hasher};

use super::i18n::tr;
use super::{ModelChoice, ModelFormulaInfo, ResolvedModel, UiLanguage};

fn model_formula_full(model: ModelChoice, polynomial_degree: usize) -> String {
    match model {
        ModelChoice::Polynomial => polynomial_formula_full(polynomial_degree),
        ModelChoice::Arrhenius => r"y = A·exp(\frac{B}{x})".to_string(),
        ModelChoice::Inverse => r"y = A + \frac{B}{x}".to_string(),
        ModelChoice::Logistic => r"y = \frac{A}{1 + exp(-B·(x - C))}".to_string(),
        ModelChoice::Gompertz => r"y = A·exp(-exp(-B·(x - C)))".to_string(),
        ModelChoice::Lorentzian => r"y = C + \frac{A}{1 + (\frac{x - x_0}{gamma})^{2}}".to_string(),
        ModelChoice::NaturalLog => r"y = A·ln(\frac{x}{B})".to_string(),
        ModelChoice::FourPl => r"y = d + \frac{a - d}{1 + (\frac{x}{c})^{b}}".to_string(),
        ModelChoice::FivePl => r"y = d + \frac{a - d}{(1 + (\frac{x}{c})^{b})^{m}}".to_string(),
        ModelChoice::MichaelisMenten => r"y = \frac{V_{max}·x}{K_{m} + x}".to_string(),
        ModelChoice::ExponentialBasic => r"y = a + b·exp(-c·x)".to_string(),
        ModelChoice::ExponentialLinear => r"y = a·exp(b·x) + c·x + d".to_string(),
        ModelChoice::ExponentialHalfLife => r"y = a + \frac{b}{2^{\frac{x}{c}}}".to_string(),
        ModelChoice::FallingExponential => {
            r"y = Y_{0} - \frac{V_{0}}{K}·(1 - exp(-K·x))".to_string()
        }
        ModelChoice::HyperbolicTangent => r"y = a·tanh(b·(x - c)) + d".to_string(),
        ModelChoice::ArctangentStep => r"y = a·atan(b·(x - c)) + d".to_string(),
        ModelChoice::Softplus => r"y = a·ln(1 + exp(b·(x - c))) + d".to_string(),
        ModelChoice::Power => r"y = a·x^{b}".to_string(),
        ModelChoice::Gaussian => r"y = a·exp(-\frac{(x - b)^{2}}{2·c^{2}})".to_string(),
        ModelChoice::LinearSpline => {
            r"y(x) = y_{i} + \frac{y_{i+1} - y_{i}}{x_{i+1} - x_{i}}·(x - x_{i})".to_string()
        }
        ModelChoice::MonotoneCubicSpline => {
            r"y(x) = Hermite(y_{i}, y_{i+1}, m_{i}, m_{i+1}), m_{i} by Fritsch-Carlson".to_string()
        }
        ModelChoice::NaturalCubicSpline => {
            r"y(x) = cubic spline, S''(x_{0}) = S''(x_{n}) = 0".to_string()
        }
        ModelChoice::AkimaSpline => {
            r"y(x) = Hermite(y_{i}, y_{i+1}, m_{i}, m_{i+1}), \quad m_i \text{by Akima weights}"
                .to_string()
        }
    }
}

pub(super) fn model_formula_info(
    language: UiLanguage,
    model: ModelChoice,
    polynomial_degree: usize,
) -> ModelFormulaInfo {
    let full_formula = model_formula_full(model, polynomial_degree);
    let min_points = model_min_points(model, polynomial_degree);
    let mut notes = format!(
        "{}: {min_points}",
        tr(language, "Minimum points", "Минимум точек")
    );

    if let Some(constraint) = model_constraint_note(language, model) {
        notes.push('\n');
        notes.push_str(constraint);
    }

    notes.push('\n');
    notes.push_str(model_ml_note(language, model));

    ModelFormulaInfo {
        full_formula,
        notes,
    }
}

fn model_min_points(model: ModelChoice, polynomial_degree: usize) -> usize {
    match ResolvedModel::from_choice(model, polynomial_degree) {
        ResolvedModel::Parametric(family) => family.min_points(),
        ResolvedModel::LinearSpline | ResolvedModel::MonotoneCubicSpline => 2,
        ResolvedModel::NaturalCubicSpline => 3,
        ResolvedModel::AkimaSpline => 5,
    }
}

fn model_constraint_note(language: UiLanguage, model: ModelChoice) -> Option<&'static str> {
    match model {
        ModelChoice::Arrhenius
        | ModelChoice::Inverse
        | ModelChoice::NaturalLog
        | ModelChoice::FourPl
        | ModelChoice::FivePl
        | ModelChoice::Power => Some(tr(language, "Constraint: x > 0", "Ограничение: x > 0")),
        _ => None,
    }
}

fn model_ml_note(language: UiLanguage, model: ModelChoice) -> &'static str {
    match (language, model) {
        (UiLanguage::English, ModelChoice::Polynomial) => {
            "Global basis model. Higher degrees can overfit."
        }
        (UiLanguage::English, ModelChoice::LinearSpline) => {
            "Simple piecewise interpolation baseline."
        }
        (UiLanguage::English, ModelChoice::Arrhenius) => {
            "Common exponential-in-inverse-x transform model."
        }
        (UiLanguage::English, ModelChoice::Inverse) => {
            "Hyperbolic decay model; often used as a baseline."
        }
        (UiLanguage::English, ModelChoice::Logistic) => {
            "Sigmoid response model for bounded transitions."
        }
        (UiLanguage::English, ModelChoice::Gompertz) => {
            "Asymmetric sigmoid growth model with a long lower tail."
        }
        (UiLanguage::English, ModelChoice::Lorentzian) => "Peak-shaped model with heavy tails.",
        (UiLanguage::English, ModelChoice::NaturalLog) => {
            "Log transform response, useful for diminishing returns."
        }
        (UiLanguage::English, ModelChoice::ExponentialLinear) => {
            "Exponential trend with linear drift background."
        }
        (UiLanguage::English, ModelChoice::HyperbolicTangent) => {
            "Smooth S-curve transition with bounded tails."
        }
        (UiLanguage::English, ModelChoice::ArctangentStep) => {
            "Soft threshold model with heavier tails than logistic."
        }
        (UiLanguage::English, ModelChoice::Softplus) => {
            "Smooth ReLU-like activation used in ML and calibration."
        }
        (UiLanguage::English, ModelChoice::MonotoneCubicSpline) => {
            "Useful for monotone response and calibration curves."
        }
        (UiLanguage::English, ModelChoice::NaturalCubicSpline) => {
            "Smooth interpolation with natural boundary conditions."
        }
        (UiLanguage::English, ModelChoice::AkimaSpline) => {
            "Robust piecewise cubic interpolation near sharp local changes."
        }
        (UiLanguage::English, _) => "Parametric nonlinear regression model.",
        (UiLanguage::Russian, ModelChoice::Polynomial) => {
            "Глобальная базисная модель. На высоких степенях может переобучаться."
        }
        (UiLanguage::Russian, ModelChoice::LinearSpline) => {
            "Простой базовый кусочно-линейный интерполятор."
        }
        (UiLanguage::Russian, ModelChoice::Arrhenius) => {
            "Экспонента от обратного x; полезна в кинетических зависимостях."
        }
        (UiLanguage::Russian, ModelChoice::Inverse) => {
            "Гиперболический спад, удобная базовая модель."
        }
        (UiLanguage::Russian, ModelChoice::Logistic) => {
            "Сигмоидальная модель ограниченного перехода."
        }
        (UiLanguage::Russian, ModelChoice::Gompertz) => {
            "Асимметричная сигмоида с длинным нижним хвостом."
        }
        (UiLanguage::Russian, ModelChoice::Lorentzian) => {
            "Пиковая модель с более тяжёлыми хвостами."
        }
        (UiLanguage::Russian, ModelChoice::NaturalLog) => {
            "Логарифмический отклик для эффекта убывающей отдачи."
        }
        (UiLanguage::Russian, ModelChoice::ExponentialLinear) => {
            "Экспоненциальный тренд с линейным дрейфом фона."
        }
        (UiLanguage::Russian, ModelChoice::HyperbolicTangent) => {
            "Гладкий S-переход с ограниченными хвостами."
        }
        (UiLanguage::Russian, ModelChoice::ArctangentStep) => {
            "Мягкий пороговый переход с более тяжёлыми хвостами."
        }
        (UiLanguage::Russian, ModelChoice::Softplus) => "Гладкая ReLU-подобная активация из ML.",
        (UiLanguage::Russian, ModelChoice::MonotoneCubicSpline) => {
            "Подходит для монотонных откликов и калибровочных кривых."
        }
        (UiLanguage::Russian, ModelChoice::NaturalCubicSpline) => {
            "Гладкая интерполяция с натуральными граничными условиями."
        }
        (UiLanguage::Russian, ModelChoice::AkimaSpline) => {
            "Устойчивый кубический интерполятор при локальных резких изменениях."
        }
        (UiLanguage::Russian, _) => "Параметрическая модель нелинейной регрессии.",
    }
}

fn polynomial_formula_full(degree: usize) -> String {
    let degree = degree.clamp(1, 9);
    let symbols = polynomial_parameter_symbols();
    let mut terms = Vec::with_capacity(degree + 1);
    for (index, symbol) in symbols.iter().copied().enumerate().take(degree + 1) {
        let power = degree - index;
        let term = match power {
            0 => symbol.to_string(),
            1 => format!("{symbol}·x"),
            _ => format!("{symbol}·x^{{{power}}}"),
        };
        terms.push(term);
    }
    format!("y = {}", terms.join(" + "))
}

fn polynomial_parameter_symbols() -> [char; 10] {
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn formula_svg_uri(formula: &str, dark_mode: bool) -> String {
    let mut hasher = DefaultHasher::new();
    formula.hash(&mut hasher);
    dark_mode.hash(&mut hasher);
    format!("bytes://formula_model_{}.svg", hasher.finish())
}

/// Возвращает человекочитаемое текстовое представление формулы.
///
/// Используется как fallback для платформ, где SVG-текст может рендериться нестабильно.
pub(super) fn formula_plain_text(formula: &str) -> String {
    latex_group_to_text(formula)
}

/// Строит SVG-рендер формулы для показа в UI.
///
/// Здесь intentionally используется небольшой ручной парсер,
/// чтобы не тянуть тяжёлые зависимости ради ограниченного подмножества LaTeX.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn formula_svg_bytes(formula: &str, dark_mode: bool) -> Vec<u8> {
    let spans = parse_formula_spans(formula);
    let visible_chars: usize = spans.iter().map(|span| span.text.chars().count()).sum();
    let width = ((visible_chars.max(24) as u32) * 14 + 48).clamp(380, 2100);
    let height = 68_u32;
    let rect_width = width - 2;
    let rect_height = height - 2;
    let (background, border, text) = if dark_mode {
        ("#0f172a", "#334155", "#f8fafc")
    } else {
        ("#ffffff", "#cbd5e1", "#111827")
    };
    let tspan_markup = formula_spans_to_svg(&spans);

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="1" y="1" width="{rect_width}" height="{rect_height}" rx="10" fill="{background}" stroke="{border}" stroke-width="1.4"/>
<text x="16" y="44" font-family="Cambria Math, STIX Two Math, DejaVu Serif, serif" font-size="24" fill="{text}">{tspan_markup}</text>
</svg>"#
    )
    .into_bytes()
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormulaSpanKind {
    Normal,
    Superscript,
    Subscript,
    FractionNumerator,
    FractionSlash,
    FractionDenominator,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
struct FormulaSpan {
    kind: FormulaSpanKind,
    text: String,
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_formula_spans(formula: &str) -> Vec<FormulaSpan> {
    let mut spans = Vec::new();
    let mut normal = String::new();
    let mut chars = formula.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if try_consume_keyword(&mut chars, "frac") && matches!(chars.next(), Some('{')) {
                if !normal.is_empty() {
                    spans.push(FormulaSpan {
                        kind: FormulaSpanKind::Normal,
                        text: std::mem::take(&mut normal),
                    });
                }
                let numerator = read_braced_group(&mut chars);
                if matches!(chars.next(), Some('{')) {
                    let denominator = read_braced_group(&mut chars);
                    spans.push(FormulaSpan {
                        kind: FormulaSpanKind::FractionNumerator,
                        text: latex_group_to_text(&numerator),
                    });
                    spans.push(FormulaSpan {
                        kind: FormulaSpanKind::FractionSlash,
                        text: "∕".to_string(),
                    });
                    spans.push(FormulaSpan {
                        kind: FormulaSpanKind::FractionDenominator,
                        text: latex_group_to_text(&denominator),
                    });
                    continue;
                }
                normal.push_str("\\frac{");
                normal.push_str(&numerator);
                continue;
            }
            normal.push(ch);
            continue;
        }
        if (ch == '^' || ch == '_') && matches!(chars.peek(), Some('{')) {
            if !normal.is_empty() {
                spans.push(FormulaSpan {
                    kind: FormulaSpanKind::Normal,
                    text: std::mem::take(&mut normal),
                });
            }
            chars.next();
            let content = read_braced_group(&mut chars);
            spans.push(FormulaSpan {
                kind: if ch == '^' {
                    FormulaSpanKind::Superscript
                } else {
                    FormulaSpanKind::Subscript
                },
                text: content,
            });
        } else {
            normal.push(ch);
        }
    }

    if !normal.is_empty() {
        spans.push(FormulaSpan {
            kind: FormulaSpanKind::Normal,
            text: normal,
        });
    }
    spans
}

fn try_consume_keyword(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    keyword: &str,
) -> bool {
    let mut probe = chars.clone();
    for expected in keyword.chars() {
        if probe.next() != Some(expected) {
            return false;
        }
    }
    *chars = probe;
    true
}

fn read_braced_group(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut depth = 1_u32;
    let mut content = String::new();
    for ch in chars.by_ref() {
        match ch {
            '{' => {
                depth += 1;
                content.push(ch);
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                content.push(ch);
            }
            _ => content.push(ch),
        }
    }
    content
}

fn latex_group_to_text(text: &str) -> String {
    let mut output = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if try_consume_keyword(&mut chars, "frac") && matches!(chars.next(), Some('{')) {
                let numerator = read_braced_group(&mut chars);
                if matches!(chars.next(), Some('{')) {
                    let denominator = read_braced_group(&mut chars);
                    output.push('(');
                    output.push_str(&latex_group_to_text(&numerator));
                    output.push(')');
                    output.push('∕');
                    output.push('(');
                    output.push_str(&latex_group_to_text(&denominator));
                    output.push(')');
                    continue;
                }
                output.push_str("\\frac{");
                output.push_str(&numerator);
                continue;
            }
            if try_consume_keyword(&mut chars, "quad") {
                output.push(' ');
                continue;
            }
            if try_consume_keyword(&mut chars, "text") && matches!(chars.next(), Some('{')) {
                let content = read_braced_group(&mut chars);
                output.push_str(&latex_group_to_text(&content));
                continue;
            }
            output.push(ch);
            continue;
        }
        if (ch == '^' || ch == '_') && matches!(chars.peek(), Some('{')) {
            chars.next();
            let group = read_braced_group(&mut chars);
            output.push(ch);
            output.push_str(&latex_group_to_text(&group));
            continue;
        }
        output.push(ch);
    }
    output
}

#[cfg(not(target_arch = "wasm32"))]
fn formula_spans_to_svg(spans: &[FormulaSpan]) -> String {
    let mut markup = String::new();
    for span in spans {
        let escaped = escape_svg_text(&span.text);
        let tspan = match span.kind {
            FormulaSpanKind::Normal => format!("<tspan>{escaped}</tspan>"),
            FormulaSpanKind::Superscript | FormulaSpanKind::FractionNumerator => {
                format!("<tspan baseline-shift=\"super\" font-size=\"66%\">{escaped}</tspan>")
            }
            FormulaSpanKind::FractionSlash => format!("<tspan font-size=\"88%\">{escaped}</tspan>"),
            FormulaSpanKind::Subscript | FormulaSpanKind::FractionDenominator => {
                format!("<tspan baseline-shift=\"sub\" font-size=\"66%\">{escaped}</tspan>")
            }
        };
        markup.push_str(&tspan);
    }
    markup
}

#[cfg(not(target_arch = "wasm32"))]
fn escape_svg_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\'' => output.push_str("&apos;"),
            _ => output.push(ch),
        }
    }
    output
}
