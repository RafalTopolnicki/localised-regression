import argparse
import pickle
import sys

import numpy as np
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.svm import SVR  # for building SVR model

sys.path.insert(0,'..')

from BMR.bmr import *
from pyearth import Earth
import pandas as pd

N_JOBS = 1

def gen_model(X, a, b, c):
    return a * X + b * X ** 2 + c * X ** 3

def gen_data(n, a, b, c, eps):
    if Xdist == 'U':
        rng = ss.uniform(loc=-4, scale=8)
    if Xdist == 'N':
        rng = ss.norm()
    X = rng.rvs(size=(n, 1))
    y = gen_model(X=X, a=a, b=b, c=c)
    if eps>0:
        y += ss.norm(loc=0, scale=eps).rvs(size=(n, 1))
    return X, y

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
    n_pts = x.shape[0]
    min_pts = 4
    if degree == 2:
        min_pts = 10
    epsilons = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    min_n_pts = [max(int(n_pts*2*e/8), min_pts) for e in epsilons]
    param_grid = {
        "epsilon": epsilons,
        "min_n_pts": min_n_pts,
    }
    bmr = BMR(min_n_pts=10, M=M, substitution_policy=substitution_policy, 
              degree=degree, epsilon=0.5, in_ball_model=in_ball_model)
    sh = GridSearchCV(bmr, param_grid, cv=3, n_jobs=N_JOBS, verbose=0).fit(x, y)
    params = sh.best_params_
    params['M'] = M
    params['substitution_policy'] = substitution_policy
    params['in_ball_model'] = in_ball_model
    params['degree'] = degree
    return params, sh.cv_results_

# generate points in which prediction is made
grid_points = np.arange(-4, 4.1, 0.1)
mesh_pts = grid_points.reshape(-1, 1)

def run_experiment(args):
    n = args.n; a = args.a
    b = args.b; c = args.c
    eps = args.eps; mcloops = args.M
    substitution_policy = args.mode
    method_label = args.method
    alpha = 0.05
    filename_base = f'{OUTPUT_DIR}/{method_label}_1d_n={n}_a={a:.3f}_b={b:.3f}_c={c:.3f}_eps={eps:.4f}_{substitution_policy}_X{Xdist}'
    filename_csv = f'{filename_base}.csv'
    filename_pickle = f'{filename_base}.pickle'
    
    X_pred = mesh_pts
    y_true = gen_model(X_pred, a, b, c)

    #generate one sample to set method parameters
    X, y = gen_data(n=n, a=a, b=b, c=c, eps=eps)
    if method_label == 'BMR1':
        params, cv_results = get_bmr_params(X, y, M=5, degree=1, substitution_policy=substitution_policy)
    if method_label == 'BMR2':
        params, cv_results = get_bmr_params(X, y, M=5, degree=2, substitution_policy=substitution_policy)
    if method_label == 'SVR':
        params, cv_results = get_svr_params(X, y[:, 0])
    if method_label == 'MARS':
        params, cv_results = get_mars_params(X, y[:, 0])
    if method_label == 'LR':
        params, cv_results = None, None
    # save params to file
    pickle.dump([params, cv_results], open(filename_pickle, 'wb'))
    
    results = []
    betas = []
    intercepts = []
    for loop in range(mcloops):
        #if loop % 10 == 0:
        print(f'Running loop {loop}/{mcloops} for {filename_csv}')
        # run all methods on new data set
        X, y = gen_data(n=n, a=a, b=b, c=c, eps=eps)
        # fit models
        if method_label in ['BMR1', 'BMR2']:
            method = BMR(**params)
            method.fit(X, y)
        if method_label == 'SVR':
            method = SVR(**params)
            method.fit(X, y[:, 0])
        if method_label == 'MARS':
            method = Earth(**params)
            method.fit(X, y)
        if method_label == 'LR':
            method = LinearRegression()
            method.fit(X, y)
        # make predictions
        pred = method.predict(X_pred)
        if len(pred.shape) > 1:
            pred = pred[:, 0]
        results.append(pred)

        # save coefficients
        if method_label in ['BMR1', 'BMR2']:
            beta, intercept = method.coefficients(X_pred)
            betas.append(beta)
            intercepts.append(intercept)
        if method_label == 'LR':
            betas.append(method.coef_)
            intercepts.append(method.intercept_)

    coeff = {}
    if method_label in ['BMR1', 'BMR2']:
        n_betas = np.array(betas).shape[2]
        for beta in range(n_betas):
            coeff[f'BMR_beta{beta+1}_mean'] = np.mean(np.array(betas), axis=0)[:, beta]
            coeff[f'BMR_beta{beta+1}_low'] = np.quantile(np.array(betas), q=alpha/2, axis=0)[:, beta]
            coeff[f'BMR_beta{beta+1}_up'] = np.quantile(np.array(betas), q=1-alpha/2, axis=0)[:, beta]
            coeff[f'BMR_beta{beta+1}_len'] =  np.array(coeff[f'BMR_beta{beta+1}_up'])-np.array(coeff[f'BMR_beta{beta+1}_low'])
        coeff['BMR_I_mean'] = np.mean(np.array(intercepts), axis=0)
        coeff['BMR_I_low'] = np.quantile(np.array(intercepts), q=alpha/2, axis=0)
        coeff['BMR_I_up'] = np.quantile(np.array(intercepts), 1-alpha/2, axis=0)
        coeff['BMR_I_len'] = np.array(coeff['BMR_I_up']) - np.array(coeff['BMR_I_low'])
    if method_label == 'LR':
        coeff['LR_beta1_mean'] = np.mean(np.array(betas), axis=0)[:, 0].tolist()*X_pred.shape[0]
        coeff['LR_beta1_low'] = np.quantile(np.array(betas), q=alpha/2, axis=0)[:, 0].tolist()*X_pred.shape[0]
        coeff['LR_beta1_up'] = np.quantile(np.array(betas), q=1-alpha/2, axis=0)[:, 0].tolist()*X_pred.shape[0]
        coeff['LR_beta1_len'] = np.array(coeff['LR_beta1_up']) - np.array(coeff['LR_beta1_low'])
        coeff['LR_I_mean'] = np.mean(np.array(intercepts), axis=0).tolist()*X_pred.shape[0]
        coeff['LR_I_low'] = np.quantile(np.array(intercepts), q=alpha/2, axis=0).tolist()*X_pred.shape[0]
        coeff['LR_I_up'] = np.quantile(np.array(intercepts), q=1-alpha/2, axis=0).tolist()*X_pred.shape[0]
        coeff['LR_I_len'] = np.array(coeff['LR_I_up']) - np.array(coeff['LR_I_low'])
       
    # collect the results and prepare the csv
    df0 = pd.DataFrame([mesh_pts[:, 0]]).transpose()
    df0.columns = ['x']
    dfs = [df0]
    dat = np.array(results).transpose()
    ci_low = np.quantile(dat, q=alpha/2, axis=1)
    ci_up = np.quantile(dat, q=1-alpha/2, axis=1)
    ci_mid = np.quantile(dat, q=0.5, axis=1)
    ci_mean = np.mean(dat, axis=1)
    mse = np.mean((dat - y_true.reshape(-1,1))**2, axis=1)
    df = pd.DataFrame([ci_mean, ci_mid, ci_low, ci_up, ci_up-ci_low, mse]).transpose()
    df.columns = [f'{method_label}_CI_mean', f'{method_label}_CI_mid', f'{method_label}_CI_low', f'{method_label}_CI_up', f'{method_label}_CI_len', f'{method_label}_MSE']
    dfs.append(df)
    if method_label in ['BMR1', 'BMR2', 'LR']:
        dfs.append(pd.DataFrame(coeff))
    df = pd.concat(dfs, axis=1)
    df.to_csv(filename_csv, index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=['BMR1', 'BMR2', 'LR', 'SVR', 'MARS'], required=True, help="which method to use")
parser.add_argument("--n", type=int, required=True, help="sample size")
parser.add_argument("--a", type=float, required=True, help="param a")
parser.add_argument("--b", type=float, required=True, help="param b")
parser.add_argument("--c", type=float, required=True, help="param c")
parser.add_argument("--eps", type=float, required=True, help="noise")
parser.add_argument("--M", type=int, required=True, help="number of MC loops")
parser.add_argument("--X", choices=['U', 'N'], required=True, help="X distribution")
parser.add_argument("--mode", type=str, required=False, default="nearest", help="substitution policy for BMR")
parser.add_argument("--seed", type=int, required=False, default=1234, help="random seed")
args = parser.parse_args()

OUTPUT_DIR = 'csv'

Xdist = args.X
np.random.seed(seed=args.seed)

run_experiment(args)
