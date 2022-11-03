import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  # for building SVR model
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from pyearth import Earth
from .bmr import BMR


def get_rf_params(x, y):
    param_grid = {"n_estimators": [10, 100, 300, 500, 750, 1000, 1500], "min_samples_split": [2, 3, 5]}
    rf = RandomForestRegressor()
    sh = HalvingGridSearchCV(rf, param_grid, cv=3, factor=3, n_jobs=-1).fit(x, y)
    return sh.best_params_


def get_svr_params(x, y):
    param_grid = {"C": [0.1, 1, 10, 100, 300, 500, 750, 1000], "degree": [1, 2], "epsilon": [0.01, 0.1, 1, 10]}
    svr = SVR(kernel="rbf")
    sh = HalvingGridSearchCV(svr, param_grid, cv=5, factor=3, n_jobs=-1).fit(x, y)
    return sh.best_params_


def get_mars_params(x, y):
    param_grid = {"max_terms": [1, 2, 3, 5, 10, 100, 200, 500], "max_degree": [1, 2, 3, 4, 5]}
    mars = Earth()
    sh = HalvingGridSearchCV(mars, param_grid, cv=3, factor=3, n_jobs=-1).fit(x, y)
    return sh.best_params_


def get_bmr_params(x, y, M, degree):
    epsilon_trial = (np.mean(np.std(x, axis=0))) / np.sqrt(x.shape[0]) * 3
    n_trial = x.shape[0]
    param_grid = {
        "epsilon": [epsilon_trial * t for t in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]],
        "min_n_pts": [int(n_trial * t) for t in [0.01, 0.05, 0.1, 0.2, 0.3]],
    }
    bmr = BMR(min_n_pts=n_trial / 10, M=M, substitution_policy="nearest", degree=degree, epsilon=epsilon_trial)
    sh = HalvingGridSearchCV(bmr, param_grid, cv=3, factor=3, n_jobs=-1).fit(x, y)
    return sh.best_params_


def get_rf_model(x, y, x_test, y_test, params, scaler):
    model = RandomForestRegressor(**params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def get_bmr_model(x, y, x_test, y_test, M, degree, params, scaler):
    model = BMR(M=M, substitution_policy="nearest", degree=degree, **params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def get_catboost_model(x, y, x_test, y_test, scaler):
    train_dataset = cb.Pool(x, y)
    test_dataset = cb.Pool(x_test, y_test)
    model = cb.CatBoostRegressor(loss_function="RMSE", verbose=0)
    model.fit(train_dataset)
    pred_sc = model.predict(test_dataset)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def get_svr_model(x, y, x_test, y_test, params, scaler):
    model = SVR(kernel="rbf", **params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def get_mars_model(x, y, x_test, y_test, params, scaler):
    model = Earth(**params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def run_methods(X_train, y_train, X_test, y_test, std_scaler_y):
    # run bmr: deg1
    bmr_M = 4
    bmr1_params = get_bmr_params(X_train, y_train, M=bmr_M, degree=1)
    score_bmr1, mape_bmr1, pred_bmr1, model_bmr1 = get_bmr_model(
        X_train, y_train, X_test, y_test, M=bmr_M, degree=1, params=bmr1_params, scaler=std_scaler_y
    )
    # run bmr: deg2
    bmr2_params = get_bmr_params(X_train, y_train, M=bmr_M, degree=2)
    score_bmr2, mape_bmr2, pred_bmr2, model_bmr2 = get_bmr_model(
        X_train, y_train, X_test, y_test, M=bmr_M, degree=2, params=bmr2_params, scaler=std_scaler_y
    )

    # run bmr: deg3
    bmr3_params = get_bmr_params(X_train, y_train, M=bmr_M, degree=3)
    score_bmr3, mape_bmr3, pred_bmr3, model_bmr3 = get_bmr_model(
        X_train, y_train, X_test, y_test, M=bmr_M, degree=3, params=bmr3_params, scaler=std_scaler_y
    )

    # run bmr: deg4
    bmr4_params = get_bmr_params(X_train, y_train, M=bmr_M, degree=4)
    score_bmr4, mape_bmr4, pred_bmr4, model_bmr4 = get_bmr_model(
        X_train, y_train, X_test, y_test, M=bmr_M, degree=4, params=bmr2_params, scaler=std_scaler_y
    )

    # run randomforest
    rf_params = get_rf_params(X_train, y_train)
    score_rf, mape_rf, pred_rf, model_rf = get_rf_model(
        X_train, y_train, X_test, y_test, params=rf_params, scaler=std_scaler_y
    )

    # run SVR
    svr_params = get_svr_params(X_train, np.ravel(y_train))
    score_svr, mape_svr, pred_svr, model_svr = get_svr_model(
        X_train, y_train, X_test, y_test, params=svr_params, scaler=std_scaler_y
    )

    # run MARS
    mars_params = get_mars_params(X_train, y_train)
    score_mars, mape_mars, pred_mars, model_mars = get_mars_model(
        X_train, y_train, X_test, y_test, params=mars_params, scaler=std_scaler_y
    )

    # run CatBoost
    score_cb, mape_cb, pred_cb, model_cb = get_catboost_model(X_train, y_train, X_test, y_test, scaler=std_scaler_y)

    return (
        [score_bmr1, score_bmr2, score_bmr3, score_bmr4, score_rf, score_svr, score_mars, score_cb],
        [mape_bmr1, mape_bmr2, mape_bmr3, mape_bmr4, mape_rf, mape_svr, mape_mars, mape_cb],
        [pred_bmr1, pred_bmr2, pred_bmr3, pred_bmr4, pred_rf, pred_svr, pred_mars, pred_cb],
        [model_bmr1, model_bmr2, model_bmr3, model_bmr4, model_rf, model_svr, model_mars, model_cb],
    )
