use super::*;

#[test]
fn curve_family_all_matches_discriminant_order() {
    assert_eq!(CurveFamily::ALL.len(), CURVE_FAMILY_COUNT);
    for (index, family) in CurveFamily::ALL.iter().copied().enumerate() {
        assert_eq!(family as usize, index);
    }
}

#[test]
fn point_rejects_non_finite_values() {
    let error = Point::try_new(f64::NAN, 1.0).expect_err("NaN x must be rejected");
    assert!(matches!(
        error,
        InputError::NonFinitePoint {
            field: "x",
            value
        } if value.is_nan()
    ));

    let error = Point::try_new(1.0, f64::INFINITY).expect_err("Inf y must be rejected");
    assert_eq!(
        error,
        InputError::NonFinitePoint {
            field: "y",
            value: f64::INFINITY,
        }
    );
}

#[test]
fn points_require_at_least_two_values() {
    let points = vec![Point::try_new(0.0, 0.0).unwrap()];
    let error = Points::try_from(points).expect_err("must reject short vectors");

    assert_eq!(
        error,
        InputError::TooFewPoints {
            len: 1,
            min_required: 2,
        }
    );
}

#[test]
fn family_validation_checks_min_points_and_domain() {
    let points = Points::try_from(vec![
        Point::try_new(-1.0, 1.0).unwrap(),
        Point::try_new(1.0, 2.0).unwrap(),
    ])
    .unwrap();

    let error = CurveFamily::Power
        .validate_points(&points)
        .expect_err("power family requires x > 0");
    assert!(matches!(
        error,
        InputError::NonPositiveXForFamily {
            family: CurveFamily::Power,
            index: 0,
            value: -1.0
        }
    ));

    let error = CurveFamily::NaturalLog
        .validate_points(&points)
        .expect_err("natural log requires x > 0");
    assert!(matches!(
        error,
        InputError::NonPositiveXForFamily {
            family: CurveFamily::NaturalLog,
            index: 0,
            value: -1.0
        }
    ));

    let short_points = Points::try_from(vec![
        Point::try_new(1.0, 1.0).unwrap(),
        Point::try_new(2.0, 2.0).unwrap(),
    ])
    .unwrap();
    let error = CurveFamily::Quadratic
        .validate_points(&short_points)
        .expect_err("quadratic requires at least 3 points");
    assert!(matches!(
        error,
        InputError::TooFewPointsForFamily {
            family: CurveFamily::Quadratic,
            len: 2,
            min_required: 3
        }
    ));

    let error = CurveFamily::PseudoVoigt
        .validate_points(&short_points)
        .expect_err("pseudo-voigt requires at least 6 points");
    assert!(matches!(
        error,
        InputError::TooFewPointsForFamily {
            family: CurveFamily::PseudoVoigt,
            len: 2,
            min_required: 6
        }
    ));
}
