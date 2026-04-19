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
