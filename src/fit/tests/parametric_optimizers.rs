use super::*;

#[test]
fn lbfgs_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &config,
    )
    .expect("linear fit must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn nelder_mead_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NelderMead(NelderMeadConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with Nelder-Mead must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn steepest_descent_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::SteepestDescent(SteepestDescentConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with steepest descent must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn sgd_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::Sgd(SgdConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with SGD must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn adam_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::Adam(
        AdamConfig::try_new(5_000, 2e-2).expect("adam test config must be valid"),
    );
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with Adam must succeed");

    assert!(result.mse < 1e-6, "adam mse={}", result.mse);
}

#[test]
fn newton_cg_fits_linear_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NewtonCg(NewtonCgConfig::default());
    let result = fit_curve_with_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
    )
    .expect("linear fit with Newton-CG must succeed");

    assert!(result.mse < 1e-10, "newton-cg mse={}", result.mse);
}

#[test]
fn newton_cg_supports_all_objective_metrics() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NewtonCg(NewtonCgConfig::default());
    for loss_metric in OptimizationLossMetric::ALL {
        let result = fit_curve_with_progress_and_optimizer_config_and_loss_metric(
            &points,
            CurveFamily::Linear,
            CurveParams::Linear { a: 0.2, b: 0.1 },
            &optimizer_config,
            loss_metric,
            |_iteration, _params| true,
        )
        .expect("fit with Newton-CG and selected objective metric must succeed");
        assert!(
            result.mse < 1e-6,
            "loss={loss_metric:?}, mse={}",
            result.mse
        );
    }
}

#[test]
fn lbfgs_fits_cubic_data() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| {
        0.4 * x * x * x - 0.8 * x * x + 1.2 * x + 0.5
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Cubic,
        CurveParams::Cubic {
            a: 0.1,
            b: 0.1,
            c: 0.1,
            d: 0.1,
        },
        &config,
    )
    .expect("cubic fit must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn lbfgs_fits_nonic_data() {
    let points = build_points(
        &[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        |x| {
            0.15 * x.powi(9) - 0.05 * x.powi(8) + 0.12 * x.powi(7) - 0.2 * x.powi(6)
                + 0.08 * x.powi(5)
                + 0.1 * x.powi(4)
                - 0.05 * x.powi(3)
                + 0.07 * x.powi(2)
                - 0.03 * x
                + 0.9
        },
    );
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Nonic,
        CurveParams::Nonic {
            a: 0.1,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
            g: 0.0,
            h: 0.0,
            i: 0.0,
            j: 0.0,
        },
        &config,
    )
    .expect("nonic fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_exponential_basic_data() {
    let points = build_points(&[0.0, 0.5, 1.0, 1.5, 2.0], |x| 0.7 + 2.4 * (-0.9 * x).exp());
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::ExponentialBasic,
        CurveParams::ExponentialBasic {
            a: 0.1,
            b: 1.0,
            c: 0.3,
        },
        &config,
    )
    .expect("exponential basic fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_exponential_linear_data() {
    let points = build_points(&[-1.0, -0.5, 0.0, 0.7, 1.4, 2.0], |x| {
        1.6 * (0.45 * x).exp() - 0.8 * x + 0.3
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::ExponentialLinear,
        CurveParams::ExponentialLinear {
            a: 1.0,
            b: 0.2,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("exponential + linear fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_arrhenius_data() {
    let points = build_points(&[0.5, 0.8, 1.0, 1.4, 2.0, 3.0], |x| 1.8 * (0.9 / x).exp());
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Arrhenius,
        CurveParams::Arrhenius { a: 1.0, b: 0.2 },
        &config,
    )
    .expect("arrhenius fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_inverse_data() {
    let points = build_points(&[0.5, 0.75, 1.0, 1.5, 2.0, 3.0], |x| 1.2 + 2.7 / x);
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Inverse,
        CurveParams::Inverse { a: 0.0, b: 1.0 },
        &config,
    )
    .expect("inverse fit must succeed");

    assert!(result.mse < 1e-10);
}

#[test]
fn lbfgs_fits_logistic_data() {
    let points = build_points(&[-2.0, -1.5, -1.0, -0.2, 0.4, 0.8, 1.2, 1.8], |x| {
        4.0 / (1.0 + (-2.2 * (x - 0.7)).exp())
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Logistic,
        CurveParams::Logistic {
            a: 3.0,
            b: 1.0,
            c: 0.0,
        },
        &config,
    )
    .expect("logistic fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_gompertz_data() {
    let points = build_points(&[-2.0, -1.2, -0.5, 0.0, 0.5, 1.0, 1.6, 2.2], |x| {
        4.8 * (-(-1.6 * (x - 0.4)).exp()).exp()
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Gompertz,
        CurveParams::Gompertz {
            a: 3.5,
            b: 0.8,
            c: 0.0,
        },
        &config,
    )
    .expect("gompertz fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_bi_exponential_data() {
    let points = build_points(&[0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.8, 3.6, 4.5], |x| {
        1.8 * (-2.4 * x).exp() + 0.7 * (-0.35 * x).exp() + 0.2
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::BiExponential,
        CurveParams::BiExponential {
            a1: 1.0,
            k1: 1.0,
            a2: 0.4,
            k2: 0.1,
            c: 0.0,
        },
        &config,
    )
    .expect("bi-exponential fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_damped_sinusoid_data() {
    let points = build_points(
        &[
            0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6,
        ],
        |x| 1.9 * (-0.25 * x).exp() * (2.4 * x + 0.35).sin() - 0.2,
    );
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::DampedSinusoid,
        CurveParams::DampedSinusoid {
            a: 1.4,
            k: 0.2,
            omega: 2.0,
            phi: 0.0,
            c: 0.0,
        },
        &config,
    )
    .expect("damped sinusoid fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_lorentzian_data() {
    let points = build_points(&[-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0], |x| {
        0.4 + 2.5 / (1.0 + ((x - 0.3) / 0.8).powi(2))
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Lorentzian,
        CurveParams::Lorentzian {
            a: 2.0,
            x0: 0.0,
            gamma: 1.0,
            c: 0.0,
        },
        &config,
    )
    .expect("lorentzian fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_natural_log_data() {
    let points = build_points(&[0.5, 0.8, 1.2, 1.8, 2.5, 3.2], |x| 1.5 * (x / 0.7).ln());
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::NaturalLog,
        CurveParams::NaturalLog { a: 1.0, b: 1.0 },
        &config,
    )
    .expect("natural log fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_michaelis_menten_data() {
    let points = build_points(&[0.5, 1.0, 2.0, 4.0, 8.0], |x| (3.5 * x) / (1.8 + x));
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::MichaelisMenten,
        CurveParams::MichaelisMenten { vmax: 2.0, km: 1.0 },
        &config,
    )
    .expect("michaelis-menten fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_hyperbolic_tangent_data() {
    let points = build_points(&[-2.0, -1.0, -0.4, 0.0, 0.6, 1.1, 1.8], |x| {
        2.2 * (1.3 * (x - 0.35)).tanh() - 0.4
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::HyperbolicTangent,
        CurveParams::HyperbolicTangent {
            a: 1.5,
            b: 0.8,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("hyperbolic tangent fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_arctangent_step_data() {
    let points = build_points(&[-2.0, -1.2, -0.6, 0.0, 0.5, 1.0, 1.8], |x| {
        2.0 * (1.5 * (x - 0.2)).atan() + 0.1
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::ArctangentStep,
        CurveParams::ArctangentStep {
            a: 1.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("arctangent step fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_softplus_data() {
    let points = build_points(&[-2.0, -1.0, -0.2, 0.3, 0.8, 1.4, 2.0], |x| {
        1.8 * super::softplus(2.0 * (x - 0.4)) - 0.35
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Softplus,
        CurveParams::Softplus {
            a: 1.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("softplus fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_power_data() {
    let points = build_points(&[0.5, 1.0, 1.5, 2.0, 3.0], |x| 1.7 * x.powf(1.35));
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Power,
        CurveParams::Power { a: 1.0, b: 1.0 },
        &config,
    )
    .expect("power fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_gaussian_data() {
    let points = build_points(&[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], |x| {
        2.1 * (-(x - 0.4).powi(2) / (2.0 * 0.7 * 0.7)).exp()
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Gaussian,
        CurveParams::Gaussian {
            a: 1.0,
            b: 0.0,
            c: 1.0,
        },
        &config,
    )
    .expect("gaussian fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_rational_11_data() {
    let true_params = CurveParams::Rational11 {
        a: 1.3,
        b: 0.4,
        c: 0.12,
        d: -0.2,
    };
    let points = build_points(&[-2.0, -1.2, -0.4, 0.0, 0.6, 1.4, 2.2], |x| {
        true_params.evaluate(x)
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Rational11,
        CurveParams::Rational11 {
            a: 0.5,
            b: 0.1,
            c: 0.0,
            d: 0.0,
        },
        &config,
    )
    .expect("rational 1/1 fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_rational_22_data() {
    let true_params = CurveParams::Rational22 {
        a: 0.2,
        b: 0.9,
        c: 0.6,
        d: 0.15,
        e: 0.03,
    };
    let points = build_points(&[-2.0, -1.5, -1.0, -0.2, 0.4, 1.0, 1.8, 2.6], |x| {
        true_params.evaluate(x)
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Rational22,
        CurveParams::Rational22 {
            a: 0.0,
            b: 0.4,
            c: 0.1,
            d: 0.0,
            e: 0.0,
        },
        &config,
    )
    .expect("rational 2/2 fit must succeed");

    assert!(result.mse < 1e-8);
}

#[test]
fn lbfgs_fits_emg_positive_tau_data() {
    let true_params = CurveParams::Emg {
        a: 2.4,
        mu: 0.15,
        sigma: 0.45,
        tau: 0.7,
        c: 0.1,
    };
    let points = build_points(&[-1.5, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.5, 2.0], |x| {
        true_params.evaluate(x)
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Emg,
        CurveParams::Emg {
            a: 1.0,
            mu: 0.0,
            sigma: 0.8,
            tau: 0.3,
            c: 0.0,
        },
        &config,
    )
    .expect("emg fit (tau>0) must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn lbfgs_fits_emg_negative_tau_data() {
    let true_params = CurveParams::Emg {
        a: 2.2,
        mu: 0.25,
        sigma: 0.5,
        tau: -0.8,
        c: -0.05,
    };
    let points = build_points(&[-1.8, -1.3, -0.8, -0.3, 0.1, 0.5, 0.9, 1.4, 1.9], |x| {
        true_params.evaluate(x)
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::Emg,
        CurveParams::Emg {
            a: 1.2,
            mu: 0.0,
            sigma: 0.9,
            tau: -0.2,
            c: 0.0,
        },
        &config,
    )
    .expect("emg fit (tau<0) must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn lbfgs_fits_pseudo_voigt_data() {
    let true_params = CurveParams::PseudoVoigt {
        a: 2.8,
        x0: 0.2,
        sigma: 0.5,
        gamma: 0.9,
        eta: 1.0,
        c: 0.2,
    };
    let points = build_points(&[-2.0, -1.3, -0.8, -0.2, 0.3, 0.8, 1.4, 2.0], |x| {
        true_params.evaluate(x)
    });
    let config = LbfgsConfig::default();
    let result = fit_curve(
        &points,
        CurveFamily::PseudoVoigt,
        CurveParams::PseudoVoigt {
            a: 1.5,
            x0: 0.0,
            sigma: 0.8,
            gamma: 0.6,
            eta: 0.0,
            c: 0.0,
        },
        &config,
    )
    .expect("pseudo-voigt fit must succeed");

    assert!(result.mse < 1e-6);
}

#[test]
fn new_families_support_all_optimizers_smoke() {
    let scenarios = [
        (
            CurveFamily::Rational11,
            CurveParams::Rational11 {
                a: 1.3,
                b: 0.4,
                c: 0.12,
                d: -0.2,
            },
            CurveParams::Rational11 {
                a: 0.4,
                b: 0.1,
                c: 0.0,
                d: 0.0,
            },
            vec![-2.0, -1.4, -0.8, -0.2, 0.5, 1.2, 2.0],
        ),
        (
            CurveFamily::Rational22,
            CurveParams::Rational22 {
                a: 0.2,
                b: 0.9,
                c: 0.6,
                d: 0.15,
                e: 0.03,
            },
            CurveParams::Rational22 {
                a: 0.0,
                b: 0.2,
                c: 0.2,
                d: 0.0,
                e: 0.0,
            },
            vec![-2.0, -1.5, -0.9, -0.2, 0.4, 1.1, 1.8, 2.6],
        ),
        (
            CurveFamily::Emg,
            CurveParams::Emg {
                a: 2.4,
                mu: 0.15,
                sigma: 0.45,
                tau: -0.7,
                c: 0.1,
            },
            CurveParams::Emg {
                a: 1.2,
                mu: 0.0,
                sigma: 0.8,
                tau: -0.2,
                c: 0.0,
            },
            vec![-1.8, -1.3, -0.8, -0.3, 0.1, 0.6, 1.1, 1.7],
        ),
        (
            CurveFamily::PseudoVoigt,
            CurveParams::PseudoVoigt {
                a: 2.6,
                x0: 0.2,
                sigma: 0.5,
                gamma: 0.8,
                eta: 1.2,
                c: 0.15,
            },
            CurveParams::PseudoVoigt {
                a: 1.4,
                x0: 0.0,
                sigma: 0.9,
                gamma: 0.6,
                eta: 0.0,
                c: 0.0,
            },
            vec![-2.0, -1.4, -0.9, -0.3, 0.2, 0.7, 1.3, 2.0],
        ),
    ];
    let optimizers = [
        OptimizerConfig::Lbfgs(LbfgsConfig::default()),
        OptimizerConfig::NelderMead(NelderMeadConfig::default()),
        OptimizerConfig::SteepestDescent(SteepestDescentConfig::default()),
        OptimizerConfig::NewtonCg(NewtonCgConfig::default()),
        OptimizerConfig::Sgd(SgdConfig::try_new(3_000, 3e-3).expect("SGD config must be valid")),
        OptimizerConfig::Adam(AdamConfig::try_new(3_000, 3e-3).expect("Adam config must be valid")),
    ];

    for (family, true_params, initial_params, xs) in scenarios {
        let points = build_points(&xs, |x| true_params.evaluate(x));
        let (start_mse, _start_rmse) = calculate_metrics(&points, &initial_params);

        for optimizer in &optimizers {
            let result =
                fit_curve_with_optimizer_config(&points, family, initial_params.clone(), optimizer)
                    .expect("fit for new family/optimizer combination must succeed");
            assert!(
                result.mse.is_finite(),
                "mse must be finite for {family:?}/{optimizer:?}"
            );
            assert!(
                result.mse < start_mse,
                "optimizer must improve initial fit for {family:?}/{optimizer:?}: start={start_mse}, final={}",
                result.mse
            );
        }
    }
}

#[test]
fn fit_curve_validates_positive_x_domain() {
    let points = build_points(&[-1.0, 1.0, 2.0], |x| x);
    let config = LbfgsConfig::default();
    let error = fit_curve(
        &points,
        CurveFamily::Power,
        CurveParams::Power { a: 1.0, b: 1.0 },
        &config,
    )
    .expect_err("power family must reject x <= 0");

    assert!(matches!(
        error,
        super::FitError::InvalidInput(InputError::NonPositiveXForFamily {
            family: CurveFamily::Power,
            ..
        })
    ));
}

#[test]
fn fit_curve_can_be_cancelled_via_progress_callback() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let config = LbfgsConfig::default();
    let result = fit_curve_with_progress(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &config,
        |_iteration, _params| false,
    );

    assert!(matches!(result, Err(FitError::Cancelled)));
}

#[test]
fn fit_curve_with_optimizer_config_can_be_cancelled_via_progress_callback() {
    let points = build_points(&[-2.0, -1.0, 0.0, 1.0, 2.0], |x| 2.5 * x - 0.75);
    let optimizer_config = OptimizerConfig::NelderMead(NelderMeadConfig::default());
    let result = fit_curve_with_progress_and_optimizer_config(
        &points,
        CurveFamily::Linear,
        CurveParams::Linear { a: 0.2, b: 0.1 },
        &optimizer_config,
        |_iteration, _params| false,
    );

    assert!(matches!(result, Err(FitError::Cancelled)));
}
