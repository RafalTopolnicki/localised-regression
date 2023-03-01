import argparse
import pickle
import sys

import numpy as np
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.svm import SVR  # for building SVR model
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.model_selection import train_test_split

sys.path.insert(0,'..')

from BMR.bmr import *
from pyearth import Earth

N_JOBS = 4

def gen_data(n, type, noise):
    if type == 'friedman1':
        return make_friedman1(n_samples=n, noise=noise)
    if type == 'friedman2':
        return make_friedman2(n_samples=n, noise=noise)
    if type == 'friedman3':
        return make_friedman3(n_samples=n, noise=noise)

def get_mars_params(x, y):
    param_grid = {"max_terms": [0, 1, 2, 3, 5, 10], "max_degree": [0, 1, 2, 3, 4]}
    mars = Earth()
    sh = GridSearchCV(mars, param_grid, cv=3, n_jobs=N_JOBS).fit(x, y)
    return sh.best_params_, sh.cv_results_

def get_svr_params(x, y):
    param_grid = {"C": [0.1, 1, 10, 100, 300, 500], "degree": [1, 2, 3, 4],
                  "epsilon": [0, 0.0001, 0.001, 0.01, 0.1, 1], "kernel": ['linear', 'rbf']}
    svr = SVR(kernel="rbf")
    sh = HalvingGridSearchCV(svr, param_grid, cv=3, factor=3, n_jobs=N_JOBS).fit(x, y)
    return sh.best_params_, sh.cv_results_

def get_bmr_params(x, y, M, degree, substitution_policy="nearest", in_ball_model='linear'):
    min_pts = 10
    epsilons = [0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 10, 20, 50]
    min_n_pts = [min_pts]
    standard_scaler = True
    param_grid = {
        "epsilon": epsilons,
        "min_n_pts": min_n_pts,
    }
    bmr = BMR(min_n_pts=10, M=M, substitution_policy=substitution_policy, 
              degree=degree, epsilon=0.5, in_ball_model=in_ball_model, standard_scaler=standard_scaler)
    sh = GridSearchCV(bmr, param_grid, cv=3, n_jobs=N_JOBS, verbose=0).fit(x, y)
    params = sh.best_params_
    params['M'] = M
    params['substitution_policy'] = substitution_policy
    params['in_ball_model'] = in_ball_model
    params['degree'] = degree
    params['standard_scaler'] = standard_scaler
    print(params)
    return params, sh.cv_results_

def run_experiment(args):
    n = args.n;
    type = args.type
    mcloops = args.M
    noise = args.noise
    test_size = args.test_size
    substitution_policy = args.mode
    method_label = args.method
    filename_base = f'{OUTPUT_DIR}/{method_label}_n={n}_type={type}_noise={noise}_testsize={test_size}_{substitution_policy}'
    filename_pickle = f'{filename_base}.pickle'
    filename_txt = f'{filename_base}.txt'

    #generate one sample to set method parameters
    X, y = gen_data(n=n, type=type, noise=noise)
    if method_label == 'BMR1':
        params, cv_results = get_bmr_params(X, y, M=10, degree=1, substitution_policy=substitution_policy)
    if method_label == 'BMR2':
        params, cv_results = get_bmr_params(X, y, M=10, degree=2, substitution_policy=substitution_policy)
    if method_label == 'SVR':
        params, cv_results = get_svr_params(X, y[:, 0])
    if method_label == 'SVRdef':
        params, cv_results = None, None
    if method_label == 'MARS':
        params, cv_results = get_mars_params(X, y)
    if method_label == 'MARSdef':
        params, cv_results = None, None
    if method_label == 'LR':
        params, cv_results = None, None
    # save params to file
    pickle.dump([params, cv_results], open(filename_pickle, 'wb'))
    
    results = []
    for loop in range(mcloops):
        #if loop % 10 == 0:
        print(f'Running loop {loop}/{mcloops} for {filename_txt}')
        # run all methods on new data set
        X, y = gen_data(n=n, type=type, noise=noise)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        # fit models
        if method_label in ['BMR1', 'BMR2']:
            method = BMR(**params)
            method.fit(X_train, y_train)
        if method_label == 'SVR':
            method = SVR(**params)
            method.fit(X_train, y_train)
        if method_label == 'SVRdef':
            method = SVR()
            method.fit(X_train, y_train)
        if method_label == 'MARS':
            method = Earth(**params)
            method.fit(X_train, y_train)
        if method_label == 'MARSdef':
            method = Earth()
            method.fit(X_train, y_train)
        if method_label == 'LR':
            method = LinearRegression()
            method.fit(X_train, y_train)
        # make predictions
        pred = method.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        results.append(mse)

    # save results
    np.savetxt(filename_txt, results)

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=['BMR1', 'BMR2', 'LR', 'SVR', 'MARS', 'MARSdef', 'SVRdef'], required=True, help="which method to use")
parser.add_argument("--n", type=int, required=True, help="sample size")
parser.add_argument("--test_size", type=float, required=0.1, help="test sample size")
parser.add_argument("--type", choices=['friedman1', 'friedman2', 'friedman3'], required=True, help="Friedman model")
parser.add_argument("--noise", type=float, required=True, help="noise")
parser.add_argument("--M", type=int, required=True, help="number of MC loops")
parser.add_argument("--mode", type=str, required=False, default="nearest", help="substitution policy for BMR")
parser.add_argument("--seed", type=int, required=False, default=1234, help="random seed")
args = parser.parse_args()

OUTPUT_DIR = 'csv.friedman'

np.random.seed(seed=args.seed)

run_experiment(args)
