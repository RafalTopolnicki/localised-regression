import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  # for building SVR model
import catboost as cb
from LocalRegression.bmlr import BMLR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from pyearth import Earth


def model_random_forest_params(x, y):
    param_grid = {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [2, 3, 5]}
    rf = RandomForestRegressor()
    sh = HalvingGridSearchCV(rf, param_grid, cv=3, factor=2).fit(x, y)
    return sh.best_params_


def model_random_forest(x, y, x_test, y_test, params, scaler):
    model = RandomForestRegressor(**params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_bmlr_params(x, y, cut, M, degree):
    epsilon_trial = (np.mean(np.std(x, axis=0))) / np.sqrt(x.shape[0]) * 3
    param_grid = {"epsilon": [epsilon_trial * x for x in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]]}
    bmlr = BMLR(cut=cut, M=M, substitution_policy="nearest", degree=degree)
    sh = HalvingGridSearchCV(bmlr, param_grid, cv=3, factor=3).fit(x, y)
    return sh.best_params_


def model_bmlr(x, y, x_test, y_test, cut, M, degree, params):
    model = BMLR(cut=cut, M=M, substitution_policy="nearest", degree=degree, **params)
    model.fit(x, y)
    pred = model.predict(x_test)
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_catboost(x, y, x_test, y_test, scaler):
    train_dataset = cb.Pool(x, y)
    test_dataset = cb.Pool(x_test, y_test)
    model = cb.CatBoostRegressor(loss_function="RMSE", verbose=0)
    model.fit(train_dataset)
    pred_sc = model.predict(test_dataset)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_svr_params(x, y):
    param_grid = {"C": [0.1, 1, 10, 100, 500, 1000], "degree": [2], "epsilon": [0.01, 0.1, 1, 10]}
    svr = SVR(kernel="rbf")
    sh = HalvingGridSearchCV(svr, param_grid, cv=3, factor=3).fit(x, y)
    return sh.best_params_


def model_svr(x, y, x_test, y_test, params, scaler):
    model = SVR(kernel="rbf", **params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_mars_params(x, y):
    param_grid = {"max_terms": [5, 10, 100, 200], "max_degree": [1, 2, 3, 4]}
    mars = Earth()
    sh = HalvingGridSearchCV(mars, param_grid, cv=3, factor=3).fit(x, y)
    return sh.best_params_


def model_mars(x, y, x_test, y_test, params, scaler):
    model = Earth(**params)
    model.fit(x, y)
    pred_sc = model.predict(x_test)
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1))
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model

def run_ml_methods(X, y, X_test, y_test, std_scaler_y):
    # run randomforest
    rf_params = model_random_forest_params(X, y)
    opt_score_rf, opt_mape_rf, opt_pred_rf, opt_model_rf = model_random_forest(X, y, X_test, y_test,
                                                                               params=rf_params,
                                                                               scaler=std_scaler_y)
    # run SVR
    svr_params = model_svr_params(X, y)
    opt_score_svr, opt_mape_svr, opt_pred_svr, opt_model_svr = model_svr(X, y, X_test, y_test,
                                                                         params=svr_params,
                                                                         scaler=std_scaler_y)
    # run MARS
    mars_params = model_mars_params(X, y)
    opt_score_mars, opt_mape_mars, opt_pred_mars, opt_model_mars = model_mars(X, y, X_test, y_test,
                                                                         params=mars_params,
                                                                         scaler=std_scaler_y)
    # run CatBoost
    opt_score_cb, opt_mape_cb, opt_pred_cb, opt_model_cb = model_catboost(X, y, X_test, y_test,
                                                                         scaler=std_scaler_y)
    return [opt_score_rf, opt_score_svr, opt_score_mars, opt_score_cb,
            opt_mape_rf, opt_mape_svr, opt_mape_mars, opt_mape_cb]
