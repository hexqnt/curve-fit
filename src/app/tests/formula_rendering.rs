use super::*;

#[test]
fn formula_svg_cache_distinguishes_formula_and_theme() {
    let mut app = CurveFitApp::default();

    let (light_uri, _) = app
        .cached_formula_svg(r"y = \frac{1}{x}", false)
        .expect("light formula SVG must render");
    let (dark_uri, _) = app
        .cached_formula_svg(r"y = \frac{1}{x}", true)
        .expect("dark formula SVG must render");
    let (other_uri, _) = app
        .cached_formula_svg(r"y = x^{2} + 1", false)
        .expect("different formula SVG must render");

    assert_ne!(light_uri, dark_uri);
    assert_ne!(light_uri, other_uri);
}
