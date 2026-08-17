import A12_tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN as _base_mod
import defaults
from A12_tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN import tabpfn1d_0p3sin_6p5pi_x_plus_0p5x_GPvsPFN as _base_run
from load_experimental_data import (
    generate_tabpfn_1d_linear_homoscedastic_data,
    tabpfn_1d_linear_function,
)


def tabpfn1d_linear_homoscedastic_GPvsPFN(
    num_runs=defaults.NUM_RUNS,
    num_test=5000,
    train_size=40,
    dimensions=1,
    x_bounds=None,
    test_x_bounds=None,
    test_outside_margin=0.0,
    slope=1.0,
    intercept=0.0,
    num_inits=defaults.TRAINER_NUM_INITS,
    num_epochs=defaults.TRAINER_NUM_EPOCHS,
    lr=defaults.TRAINER_LR,
    convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_epochs=defaults.TRAINER_MIN_EPOCHS,
    min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device=defaults.TRAINER_GP_DEVICE,
    amp_device=defaults.TRAINER_AMP_DEVICE,
    save_path="./results/A18_linear_homoscedastic",
    title=None,
    standardize_X=defaults.STANDARDIZE_X,
    standardize_y=defaults.STANDARDIZE_Y,
    x_standardize_method=defaults.X_STANDARDIZE_METHOD,
    noise_train=0.0,
    noise_test=0.0,
    noise_type=defaults.NOISE_TYPE,
    seed=defaults.SEED,
    seed_trainer=defaults.SEED_TRAINER,
    gp_dtype=defaults.DTYPE_GP,
    pfn_dtype=defaults.DTYPE_PFN,
    trainer_info=True,
    run_models=None,
    log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
    plot_1d_comparison=True,
    warnings_ignore=defaults.WARNINGS_IGNORE,
):
    if warnings_ignore:
        import warnings
        warnings.filterwarnings("ignore")

    if x_bounds is None:
        x_bounds = [-1.0, 1.0]
    if test_x_bounds is None:
        test_x_bounds = [x_bounds[0] - test_outside_margin, x_bounds[1] + test_outside_margin]
    if title is None:
        title = "linear_homoscedastic"

    old_gen = _base_mod.generate_tabpfn_1d_sin_2pi_x_plus_x_data
    old_true = _base_mod.tabpfn_1d_sin_2pi_x_plus_x_function

    def _gen(*, n_train, n_test, dimensions, x_bounds, test_x_bounds, train_noise, test_noise, noise_type, seed):
        return generate_tabpfn_1d_linear_homoscedastic_data(
            n_train=n_train,
            n_test=n_test,
            dimensions=dimensions,
            x_bounds=x_bounds,
            test_x_bounds=test_x_bounds,
            slope=slope,
            intercept=intercept,
            train_noise=train_noise,
            test_noise=test_noise,
            noise_type=noise_type,
            seed=seed,
        )

    def _true(X):
        return tabpfn_1d_linear_function(X, slope=slope, intercept=intercept)

    _base_mod.generate_tabpfn_1d_sin_2pi_x_plus_x_data = _gen
    _base_mod.tabpfn_1d_sin_2pi_x_plus_x_function = _true
    try:
        return _base_run(
            num_runs=num_runs,
            num_test=num_test,
            train_size=train_size,
            dimensions=dimensions,
            x_bounds=x_bounds,
            test_x_bounds=test_x_bounds,
            test_outside_margin=test_outside_margin,
            num_inits=num_inits,
            num_epochs=num_epochs,
            lr=lr,
            convergence_patience=convergence_patience,
            min_epochs=min_epochs,
            min_loss_change=min_loss_change,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            initializer_class=initializer_class,
            gp_device=gp_device,
            amp_device=amp_device,
            save_path=save_path,
            title=title,
            standardize_X=standardize_X,
            standardize_y=standardize_y,
            x_standardize_method=x_standardize_method,
            noise_train=noise_train,
            noise_test=noise_test,
            noise_type=noise_type,
            seed=seed,
            seed_trainer=seed_trainer,
            gp_dtype=gp_dtype,
            pfn_dtype=pfn_dtype,
            trainer_info=trainer_info,
            run_models=run_models,
            log_lbfgs_inner=log_lbfgs_inner,
            plot_1d_comparison=plot_1d_comparison,
            warnings_ignore=warnings_ignore,
        )
    finally:
        _base_mod.generate_tabpfn_1d_sin_2pi_x_plus_x_data = old_gen
        _base_mod.tabpfn_1d_sin_2pi_x_plus_x_function = old_true


if __name__ == "__main__":
    tabpfn1d_linear_homoscedastic_GPvsPFN(
        num_runs=5,
        train_size=40,
        dimensions=1,
        save_path="./results/A18_linear_homoscedastic",
    )
