//! Формулы моделей и SVG-рендер через RaTeX для карточки и отдельного окна.

use std::hash::{DefaultHasher, Hash, Hasher};

use ratex_layout::{LayoutOptions, layout, to_display_list};
use ratex_parser::parse;
use ratex_svg::{SvgOptions, render_to_svg};
use ratex_types::color::Color as RatexColor;
use ratex_types::display_item::DisplayList;

use super::i18n::tr;
use super::{ModelChoice, ModelFormulaInfo, ResolvedModel, UiLanguage};

// Держим размер формулы на уровне основного текста интерфейса.
const FORMULA_FONT_SIZE: f64 = 22.0;
const FORMULA_INNER_PADDING: f64 = 10.0;
const FORMULA_STROKE_WIDTH: f64 = 1.5;
const FORMULA_FRAME_PADDING_X: f64 = 16.0;
const FORMULA_FRAME_PADDING_Y: f64 = 14.0;
const FORMULA_MIN_WIDTH: f64 = 380.0;
const FORMULA_MIN_HEIGHT: f64 = 68.0;
const FORMULA_BORDER_RADIUS: f64 = 10.0;

#[derive(Debug, Clone)]
struct FormulaSource {
    render_latex: String,
    plain_text: String,
}

impl FormulaSource {
    fn single(render_latex: &str) -> Self {
        Self {
            render_latex: render_latex.to_string(),
            plain_text: latex_to_plain_text(render_latex),
        }
    }

    fn explicit(render_latex: &str, plain_text: &str) -> Self {
        Self {
            render_latex: render_latex.to_string(),
            plain_text: plain_text.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FormulaSvgTheme {
    background: &'static str,
    border: &'static str,
    text: RatexColor,
}

impl FormulaSvgTheme {
    fn new(dark_mode: bool) -> Self {
        if dark_mode {
            Self {
                background: "#0f172a",
                border: "#334155",
                text: ratex_rgb(0xF8, 0xFA, 0xFC),
            }
        } else {
            Self {
                background: "#ffffff",
                border: "#cbd5e1",
                text: ratex_rgb(0x11, 0x18, 0x27),
            }
        }
    }
}

fn model_formula_source(model: ModelChoice, polynomial_degree: usize) -> FormulaSource {
    match model {
        ModelChoice::Polynomial => {
            let render_latex = polynomial_formula_full(polynomial_degree);
            FormulaSource {
                plain_text: latex_to_plain_text(&render_latex),
                render_latex,
            }
        }
        ModelChoice::Arrhenius => FormulaSource::single(r"y = A \cdot \exp(\frac{B}{x})"),
        ModelChoice::Inverse => FormulaSource::single(r"y = A + \frac{B}{x}"),
        ModelChoice::Logistic => FormulaSource::single(r"y = \frac{A}{1 + \exp(-B \cdot (x - C))}"),
        ModelChoice::Gompertz => {
            FormulaSource::single(r"y = A \cdot \exp(-\exp(-B \cdot (x - C)))")
        }
        ModelChoice::BiExponential => FormulaSource::single(
            r"y = a_{1} \cdot \exp(-k_{1} \cdot x) + a_{2} \cdot \exp(-k_{2} \cdot x) + c",
        ),
        ModelChoice::DampedSinusoid => FormulaSource::single(
            r"y = A \cdot \exp(-k \cdot x) \cdot \sin(\omega \cdot x + \phi) + c",
        ),
        ModelChoice::Lorentzian => {
            FormulaSource::single(r"y = C + \frac{A}{1 + (\frac{x - x_0}{\gamma})^{2}}")
        }
        ModelChoice::NaturalLog => FormulaSource::single(r"y = A \cdot \ln(\frac{x}{B})"),
        ModelChoice::FourPl => {
            FormulaSource::single(r"y = d + \frac{a - d}{1 + (\frac{x}{c})^{b}}")
        }
        ModelChoice::FivePl => {
            FormulaSource::single(r"y = d + \frac{a - d}{(1 + (\frac{x}{c})^{b})^{m}}")
        }
        ModelChoice::MichaelisMenten => {
            FormulaSource::single(r"y = \frac{V_{\text{max}} \cdot x}{K_{m} + x}")
        }
        ModelChoice::ExponentialBasic => FormulaSource::single(r"y = a + b \cdot \exp(-c \cdot x)"),
        ModelChoice::ExponentialLinear => {
            FormulaSource::single(r"y = a \cdot \exp(b \cdot x) + c \cdot x + d")
        }
        ModelChoice::ExponentialHalfLife => {
            FormulaSource::single(r"y = a + \frac{b}{2^{\frac{x}{c}}}")
        }
        ModelChoice::FallingExponential => {
            FormulaSource::single(r"y = Y_{0} - \frac{V_{0}}{K} \cdot (1 - \exp(-K \cdot x))")
        }
        ModelChoice::HyperbolicTangent => {
            FormulaSource::single(r"y = a \cdot \tanh(b \cdot (x - c)) + d")
        }
        ModelChoice::ArctangentStep => {
            FormulaSource::single(r"y = a \cdot \arctan(b \cdot (x - c)) + d")
        }
        ModelChoice::Softplus => {
            FormulaSource::single(r"y = a \cdot \ln(1 + \exp(b \cdot (x - c))) + d")
        }
        ModelChoice::Power => FormulaSource::single(r"y = a \cdot x^{b}"),
        ModelChoice::Gaussian => {
            FormulaSource::single(r"y = a \cdot \exp(-\frac{(x - b)^{2}}{2 \cdot c^{2}})")
        }
        ModelChoice::Rational11 => {
            FormulaSource::single(r"y = d + \frac{a \cdot x + b}{1 + c \cdot x}")
        }
        ModelChoice::Rational22 => FormulaSource::single(
            r"y = \frac{a \cdot x^{2} + b \cdot x + c}{1 + d \cdot x + e \cdot x^{2}}",
        ),
        ModelChoice::Emg => FormulaSource::single(
            r"y = c + \frac{a}{2 \cdot \tau} \cdot \exp(\frac{\sigma^{2}}{2 \cdot \tau^{2}} - \frac{x - \mu}{\tau}) \cdot \text{erfc}(\frac{1}{\sqrt{2}}(\frac{\sigma}{|\tau|} - \frac{x - \mu}{\sigma}))",
        ),
        ModelChoice::PseudoVoigt => FormulaSource::explicit(
            r"\begin{aligned}
y &= c + a \cdot (\eta \cdot G(x; x_0, \sigma) + (1-\eta) \cdot L(x; x_0, \gamma)) \\ 
\eta &= \text{sigmoid}(\eta_{\text{raw}}) \\
G(x; x_0, \sigma) &= \exp(-\frac{(x - x_0)^{2}}{2 \cdot \sigma^{2}}) \\
L(x; x_0, \gamma) &= \frac{1}{1 + (\frac{x - x_0}{\gamma})^{2}}
\end{aligned}",
            "y = c + a·(η·G(x; x_0, σ) + (1-η)·L(x; x_0, γ)), η = sigmoid(η_raw)\n\
G(x; x_0, σ) = exp(-((x - x_0)^2)∕(2·σ^2))\n\
L(x; x_0, γ) = 1∕(1 + ((x - x_0)∕γ)^2)",
        ),
        ModelChoice::LinearSpline => FormulaSource::single(
            r"y(x) = y_{i} + \frac{y_{i+1} - y_{i}}{x_{i+1} - x_{i}} \cdot (x - x_{i})",
        ),
        ModelChoice::MonotoneCubicSpline => FormulaSource::single(
            r"y(x) = \text{Hermite}(y_{i}, y_{i+1}, m_{i}, m_{i+1}), m_{i} \text{ by Fritsch-Carlson}",
        ),
        ModelChoice::NaturalCubicSpline => FormulaSource::single(
            r"\begin{aligned}
            y(x) &= \text{cubic spline}, \\
            S''(x_{0}) &= S''(x_{n}) = 0
            \end{aligned}",
        ),
        ModelChoice::AkimaSpline => FormulaSource::single(
            r"y(x) = \text{Hermite}(y_{i}, y_{i+1}, m_{i}, m_{i+1}), m_{i} \text{ by Akima weights}",
        ),
    }
}

pub(super) fn model_formula_info(
    language: UiLanguage,
    model: ModelChoice,
    polynomial_degree: usize,
) -> ModelFormulaInfo {
    let formula = model_formula_source(model, polynomial_degree);
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
        render_latex: formula.render_latex,
        plain_text: formula.plain_text,
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
        (UiLanguage::English, ModelChoice::BiExponential) => {
            "Two-timescale exponential model with strong parameter coupling."
        }
        (UiLanguage::English, ModelChoice::DampedSinusoid) => {
            "Oscillatory model with damping; often has many local minima."
        }
        (UiLanguage::English, ModelChoice::Lorentzian) => "Peak-shaped model with heavy tails.",
        (UiLanguage::English, ModelChoice::NaturalLog) => {
            "Log transform response, useful for diminishing returns."
        }
        (UiLanguage::English, ModelChoice::ExponentialLinear) => {
            "Exponential trend with linear drift background."
        }
        (UiLanguage::English, ModelChoice::Rational11) => {
            "Compact rational model with one linear pole term."
        }
        (UiLanguage::English, ModelChoice::Rational22) => {
            "Flexible rational model with quadratic numerator/denominator."
        }
        (UiLanguage::English, ModelChoice::Emg) => {
            "Asymmetric peak (EMG); signed tau controls left/right tail."
        }
        (UiLanguage::English, ModelChoice::PseudoVoigt) => {
            "Mixture of Gaussian and Lorentzian peaks with learnable blend."
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
        (UiLanguage::Russian, ModelChoice::BiExponential) => {
            "Двухэкспоненциальная модель с сильной связью параметров."
        }
        (UiLanguage::Russian, ModelChoice::DampedSinusoid) => {
            "Осциллирующая модель с затуханием и множеством локальных минимумов."
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
        (UiLanguage::Russian, ModelChoice::Rational11) => {
            "Компактная рациональная модель с линейным полюсом."
        }
        (UiLanguage::Russian, ModelChoice::Rational22) => {
            "Гибкая рациональная модель с квадратами в числителе и знаменателе."
        }
        (UiLanguage::Russian, ModelChoice::Emg) => {
            "Асимметричный пик EMG; знак tau задаёт левый или правый хвост."
        }
        (UiLanguage::Russian, ModelChoice::PseudoVoigt) => {
            "Смесь гауссового и лоренцевого пиков с обучаемой долей."
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
            1 => format!(r"{symbol} \cdot x"),
            _ => format!(r"{symbol} \cdot x^{{{power}}}"),
        };
        terms.push(term);
    }
    format!("y = {}", terms.join(" + "))
}

fn polynomial_parameter_symbols() -> [char; 10] {
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
}

pub(super) fn formula_svg_uri(formula: &str, dark_mode: bool) -> String {
    let mut hasher = DefaultHasher::new();
    formula.hash(&mut hasher);
    dark_mode.hash(&mut hasher);
    format!("bytes://formula_model_{}.svg", hasher.finish())
}

/// Строит SVG-рендер формулы с самодостаточными контурами глифов.
pub(super) fn formula_svg_bytes(formula: &str, dark_mode: bool) -> Result<Vec<u8>, String> {
    let theme = FormulaSvgTheme::new(dark_mode);
    let display_list = render_formula_display_list(formula, theme.text)?;
    let svg_options = SvgOptions {
        font_size: FORMULA_FONT_SIZE,
        padding: FORMULA_INNER_PADDING,
        stroke_width: FORMULA_STROKE_WIDTH,
        embed_glyphs: true,
        font_dir: String::new(),
    };
    let inner_svg = render_to_svg(&display_list, &svg_options);
    let inner_body = extract_svg_body(&inner_svg)
        .ok_or_else(|| "Failed to extract inner SVG body from RaTeX output".to_string())?;

    let inner_width = display_list.width * svg_options.font_size + 2.0 * svg_options.padding;
    let inner_height = (display_list.height + display_list.depth) * svg_options.font_size
        + 2.0 * svg_options.padding;
    let width = (inner_width + 2.0 * FORMULA_FRAME_PADDING_X).max(FORMULA_MIN_WIDTH);
    let height = (inner_height + 2.0 * FORMULA_FRAME_PADDING_Y).max(FORMULA_MIN_HEIGHT);
    let rect_width = width - 2.0;
    let rect_height = height - 2.0;

    Ok(format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="1" y="1" width="{rect_width}" height="{rect_height}" rx="{border_radius}" fill="{background}" stroke="{border}" stroke-width="1.4"/>
<g transform="translate({content_x} {content_y})">
{inner_body}
</g>
</svg>"#,
        width = svg_number(width),
        height = svg_number(height),
        rect_width = svg_number(rect_width),
        rect_height = svg_number(rect_height),
        border_radius = svg_number(FORMULA_BORDER_RADIUS),
        background = theme.background,
        border = theme.border,
        content_x = svg_number(FORMULA_FRAME_PADDING_X),
        content_y = svg_number(FORMULA_FRAME_PADDING_Y),
    )
    .into_bytes())
}

fn render_formula_display_list(
    formula: &str,
    text_color: RatexColor,
) -> Result<DisplayList, String> {
    let ast = parse(formula).map_err(|error| format!("Failed to parse formula: {error}"))?;
    let layout_options = LayoutOptions::default().with_color(text_color);
    let layout_box = layout(&ast, &layout_options);
    Ok(to_display_list(&layout_box))
}

fn extract_svg_body(svg: &str) -> Option<&str> {
    let start = svg.find('>')? + 1;
    let end = svg.rfind("</svg>")?;
    Some(&svg[start..end])
}

fn svg_number(value: f64) -> String {
    let formatted = format!("{value:.6}");
    let formatted = formatted.trim_end_matches('0');
    let formatted = formatted.trim_end_matches('.');
    if formatted.is_empty() || formatted == "-" {
        "0".to_string()
    } else {
        formatted.to_string()
    }
}

fn ratex_rgb(r: u8, g: u8, b: u8) -> RatexColor {
    RatexColor::rgb(
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
    )
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

fn read_latex_command(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut command = String::new();
    while let Some(ch) = chars.peek().copied() {
        if ch.is_ascii_alphabetic() {
            command.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    command
}

fn latex_symbol(command: &str) -> Option<&'static str> {
    match command {
        "alpha" => Some("α"),
        "beta" => Some("β"),
        "gamma" => Some("γ"),
        "delta" => Some("δ"),
        "epsilon" => Some("ε"),
        "zeta" => Some("ζ"),
        "eta" => Some("η"),
        "theta" => Some("θ"),
        "iota" => Some("ι"),
        "kappa" => Some("κ"),
        "lambda" => Some("λ"),
        "mu" => Some("μ"),
        "nu" => Some("ν"),
        "xi" => Some("ξ"),
        "pi" => Some("π"),
        "rho" => Some("ρ"),
        "sigma" => Some("σ"),
        "tau" => Some("τ"),
        "upsilon" => Some("υ"),
        "phi" => Some("φ"),
        "chi" => Some("χ"),
        "psi" => Some("ψ"),
        "omega" => Some("ω"),
        "Gamma" => Some("Γ"),
        "Delta" => Some("Δ"),
        "Theta" => Some("Θ"),
        "Lambda" => Some("Λ"),
        "Xi" => Some("Ξ"),
        "Pi" => Some("Π"),
        "Sigma" => Some("Σ"),
        "Upsilon" => Some("Υ"),
        "Phi" => Some("Φ"),
        "Psi" => Some("Ψ"),
        "Omega" => Some("Ω"),
        "cdot" => Some("·"),
        "times" => Some("×"),
        "pm" => Some("±"),
        "leq" => Some("≤"),
        "geq" => Some("≥"),
        "infty" => Some("∞"),
        "exp" => Some("exp"),
        "ln" => Some("ln"),
        "sin" => Some("sin"),
        "tanh" => Some("tanh"),
        "arctan" => Some("arctan"),
        _ => None,
    }
}

fn latex_to_plain_text(text: &str) -> String {
    let mut output = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let command = read_latex_command(&mut chars);
            if command.is_empty() {
                if chars.peek() == Some(&'\\') {
                    chars.next();
                    output.push('\n');
                } else {
                    output.push(ch);
                }
                continue;
            }
            match command.as_str() {
                "frac" => {
                    if matches!(chars.next(), Some('{')) {
                        let numerator = read_braced_group(&mut chars);
                        if matches!(chars.next(), Some('{')) {
                            let denominator = read_braced_group(&mut chars);
                            output.push('(');
                            output.push_str(&latex_to_plain_text(&numerator));
                            output.push(')');
                            output.push('∕');
                            output.push('(');
                            output.push_str(&latex_to_plain_text(&denominator));
                            output.push(')');
                            continue;
                        }
                        output.push_str("\\frac{");
                        output.push_str(&numerator);
                        continue;
                    }
                    output.push_str("\\frac");
                }
                "quad" => output.push(' '),
                "text" => {
                    if matches!(chars.next(), Some('{')) {
                        let content = read_braced_group(&mut chars);
                        output.push_str(&latex_to_plain_text(&content));
                    } else {
                        output.push_str("\\text");
                    }
                }
                "sqrt" => {
                    if matches!(chars.next(), Some('{')) {
                        let radicand = read_braced_group(&mut chars);
                        output.push('√');
                        output.push('(');
                        output.push_str(&latex_to_plain_text(&radicand));
                        output.push(')');
                    } else {
                        output.push('√');
                    }
                }
                _ => {
                    if let Some(symbol) = latex_symbol(&command) {
                        output.push_str(symbol);
                    } else {
                        output.push('\\');
                        output.push_str(&command);
                    }
                }
            }
            continue;
        }
        if (ch == '^' || ch == '_') && matches!(chars.peek(), Some('{')) {
            chars.next();
            let group = read_braced_group(&mut chars);
            output.push(ch);
            output.push_str(&latex_to_plain_text(&group));
            continue;
        }
        if ch == '&' {
            continue;
        }
        output.push(ch);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_formula_renders_as_svg() {
        for model in ModelChoice::ALL {
            let degrees = if model.is_polynomial() {
                [1_usize, 9].as_slice()
            } else {
                [1_usize].as_slice()
            };
            for &degree in degrees {
                let formula = model_formula_info(UiLanguage::English, model, degree);
                let svg = String::from_utf8(
                    formula_svg_bytes(&formula.render_latex, false).unwrap_or_else(|error| {
                        panic!("{model:?} degree {degree} failed: {error}")
                    }),
                )
                .expect("SVG must be valid UTF-8");

                assert!(
                    svg.starts_with("<svg"),
                    "{model:?} degree {degree} must render SVG root"
                );
                assert!(
                    svg.contains("<path"),
                    "{model:?} degree {degree} must contain path glyphs"
                );
            }
        }
    }

    #[test]
    fn every_model_plain_text_stays_readable() {
        for model in ModelChoice::ALL {
            let degrees = if model.is_polynomial() {
                [1_usize, 9].as_slice()
            } else {
                [1_usize].as_slice()
            };
            for &degree in degrees {
                let formula = model_formula_info(UiLanguage::English, model, degree);
                let plain_text = formula.plain_text.trim();

                assert!(
                    !plain_text.is_empty(),
                    "{model:?} degree {degree} plain text must not be empty"
                );
                assert!(
                    !plain_text.contains('\\'),
                    "{model:?} degree {degree} plain text must not leak LaTeX commands: {plain_text}"
                );
            }
        }
    }

    #[test]
    fn basic_formula_svg_renders_valid_svg() {
        let svg = String::from_utf8(
            formula_svg_bytes(r"y = \frac{a}{x + 1}", false).expect("formula SVG must render"),
        )
        .expect("SVG must be valid UTF-8");

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn pseudo_voigt_render_formula_stays_multiline() {
        let formula = model_formula_info(UiLanguage::English, ModelChoice::PseudoVoigt, 1);
        let display_list = render_formula_display_list(&formula.render_latex, RatexColor::BLACK)
            .expect("pseudo-Voigt display list must build from aligned LaTeX");

        assert!(display_list.height + display_list.depth > 3.0);
    }

    #[test]
    fn pseudo_voigt_plain_text_is_human_readable() {
        let formula = model_formula_info(UiLanguage::English, ModelChoice::PseudoVoigt, 1);

        assert!(formula.plain_text.contains('\n'));
        assert!(formula.plain_text.contains("η"));
        assert!(!formula.plain_text.contains(r"\begin"));
    }
}
