import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR # for building SVR model
import catboost as cb
from LocalRegression.bmlr import BMLR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from pyearth import Earth


def model_random_forest_params(x, y):
    param_grid = {'n_estimators': [10, 100, 500, 1000],
                  'min_samples_split': [2, 3, 5]}
    rf = RandomForestRegressor()
    sh = HalvingGridSearchCV(rf, param_grid, cv=3, factor=2).fit(x, y)
    return sh.best_params_


def model_random_forest(x, y, x_test, y_test, params):
    model = RandomForestRegressor(**params)
    model.fit(x, y)
    pred = model.predict(x_test)
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_bmlr_params(x, y, cut, M, degree):
    epsilon_trial = (np.mean(np.std(x, axis=0))) / np.sqrt(x.shape[0]) * 3
    param_grid = {'epsilon': [epsilon_trial * x for x in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]]}
    bmlr = BMLR(cut=cut, M=M, substitution_policy='nearest', degree=degree)
    sh = HalvingGridSearchCV(bmlr, param_grid, cv=3, factor=3).fit(x, y)
    return sh.best_params_


def model_bmlr(x, y, x_test, y_test, cut, M, degree, params):
    model = BMLR(cut=cut, M=M, substitution_policy='nearest', degree=degree, **params)
    model.fit(x, y)
    pred = model.predict(x_test)
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_catboost(x, y, x_test, y_test):
    train_dataset = cb.Pool(x, y)
    test_dataset = cb.Pool(x_test, y_test)
    model = cb.CatBoostRegressor(loss_function='RMSE', verbose=0)
    model.fit(train_dataset)
    pred = model.predict(test_dataset)
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_svr_params(x, y):
    param_grid = {'C': [0.1, 1, 10, 100, 500, 1000],
                  'degree': [2],
                  'epsilon': [0.01, 0.1, 1, 10]
                  }
    svr = SVR(kernel='rbf')
    sh = HalvingGridSearchCV(svr, param_grid, cv=3, factor=3).fit(x, y)
    return sh.best_params_


def model_svr(x, y, x_test, y_test, params):
    model = SVR(kernel='rbf', **params)
    model.fit(x, y)
    pred = model.predict(x_test)
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model


def model_mars_params(x, y):
    param_grid = {'max_terms': [5, 10, 100, 200],
                 'max_degree': [1, 2, 3, 4]}
    mars = Earth()
    sh = HalvingGridSearchCV(mars, param_grid, cv=3, factor=3).fit(x, y)
    return sh.best_params_


def model_mars(x, y, x_test, y_test, params):
    model = Earth(**params)
    model.fit(x,y)
    pred = model.predict(x_test)
    score = mean_squared_error(y_test, pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, pred)
    return score, mape, pred, model
