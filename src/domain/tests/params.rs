use super::*;

#[test]
fn curve_params_reject_non_finite_values() {
    let values = vec![1.0, f64::NEG_INFINITY];
    let error = CurveParams::try_from_values(CurveFamily::Linear, values)
        .expect_err("non-finite parameters must be rejected");

    assert_eq!(
        error,
        InputError::NonFiniteParameter {
            family: CurveFamily::Linear,
            index: 1,
            value: f64::NEG_INFINITY,
        }
    );
}

#[test]
fn rational_curve_params_roundtrip_for_new_families() {
    let scenarios = [
        (
            CurveFamily::Rational33,
            vec![0.0, 0.3, 1.0, 0.2, 0.08, 0.01, 0.0],
        ),
        (
            CurveFamily::Rational44,
            vec![0.0, 0.0, 0.2, 0.8, 0.1, 0.05, 0.01, 0.0, 0.0],
        ),
        (
            CurveFamily::Rational55,
            vec![0.0, 0.0, 0.0, 0.2, 0.8, 0.1, 0.04, 0.01, 0.0, 0.0, 0.0],
        ),
    ];

    for (family, values) in scenarios {
        let params = CurveParams::try_from_values(family, values.clone())
            .expect("rational params must be accepted");
        assert_eq!(params.family(), family);
        assert_eq!(params.values(), values);
    }
}
