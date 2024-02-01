import numpy as np
import copy
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from .ballmapper import BallMapper
from .polynomial_regression import PolynomialRegression


def optimize_min_n_pts(
    X,
    y,
    epsilon,
    val_prop=0.2,
    M=10,
    degree=1,
    inball_model="linear",
    verbose=True,
    random_state=42,
    mercy=2,
    n_jobs=-1,
):
    if verbose:
        print(f"Optimize min_n_pts for epsilon {epsilon}")
    min_n_pts = X.shape[1] + 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_prop, random_state=random_state)

    # start with minimal number of points and increase it by 50%
    best_score = np.Inf
    best_model = None
    best_mse_iter = 0
    best_mse_pts = None
    mse_iter = 0
    while min_n_pts < X.shape[0]:
        if mse_iter > 0:
            min_n_pts = max(int(1.5 * min_n_pts), min_n_pts + 1)
        model = BMR(epsilon=epsilon, min_n_pts=min_n_pts, M=M, degree=degree, in_ball_model=inball_model)
        model.fit(X_train, y_train, n_jobs=n_jobs)
        score = model.score_mse(X_test, y_test, n_jobs=n_jobs)
        if score <= best_score:
            best_score = score
            best_model = copy.copy(model)
            best_mse_iter = mse_iter
            best_mse_pts = min_n_pts
        if verbose:
            print(
                f"BMR optimization: iter={mse_iter} min_n_pts={min_n_pts} score={score:.3f} best_score={best_score:.3f} best_n_pts={best_mse_pts}"
            )
        if mse_iter - best_mse_iter > mercy:
            if verbose:
                print(f"BMR Optimization terminated: min_n_pts={best_mse_pts}")
            break
        mse_iter += 1
    return best_mse_pts, best_model


class BMR:
    def __init__(self, epsilon, min_n_pts, M, degree=1, max_pca_components=None, in_ball_model="linear",
                 n_jobs=1):
        self.epsilon = epsilon
        self.min_n_pts = min_n_pts
        self.degree = degree
        self.max_pca_components = max_pca_components
        self.in_ball_model = in_ball_model
        self.fitted = False
        self.global_model = PolynomialRegression(degree=self.degree, in_ball_model=in_ball_model)
        self.ball_mappers = []
        self.M = M
        self.n_jobs = n_jobs

    def create_and_fit_single(self, x, y):
        bm = SingleBMR(
            epsilon=self.epsilon,
            min_n_pts=self.min_n_pts,
            degree=self.degree,
            max_pca_components=self.max_pca_components,
            in_ball_model=self.in_ball_model,
        )
        bm.fit(x, y)
        return bm

    def fit(self, x, y, n_jobs=None):
        if n_jobs is None:
            n_jobs = self.n_jobs
        self.global_model.fit(x, y)
        if n_jobs == 1:
            self.ball_mappers = [self.create_and_fit_single(x, y) for i in range(self.M)]
        else:
            self.ball_mappers = Parallel(n_jobs=n_jobs)(
                delayed(self.create_and_fit_single)(x, y) for i in range(self.M)
            )

        # self.ball_mappers = [self.create_and_fit_single(x, y) for i in range(self.M)]
        # for bm in self.ball_mappers:
        #   bm.fit(x, y)
        self.fitted = True

    def predict(self, x, n_jobs=None, return_all_preds=False, return_local_mask=False):
        # preds = []
        # prediction_masks = []
        # #for bm in self.ball_mappers:
        # #   pred, mask = bm.predict(x)
        # #   preds.append(pred)
        # #   prediction_masks.append(mask)
        #
        # #res = Parallel(n_jobs=n_jobs)(delayed(self.predict_on_single)(i, x) for i in range(self.M))
        # #res = Parallel(n_jobs=n_jobs)(delayed(self.predict_on_single)(bm, x) for bm in self.ball_mappers)
        if n_jobs is None:
            n_jobs = self.n_jobs
        res = Parallel(n_jobs=n_jobs)(delayed(bm.predict)(x) for bm in self.ball_mappers)
        preds = [r[0] for r in res]
        prediction_masks = [r[1] for r in res]
        if return_all_preds:
            return preds
        # average over mappers
        avg_pred = np.zeros(len(x))
        counts = np.zeros(len(x))
        for pred, mask in zip(preds, prediction_masks):
            avg_pred[mask] += pred[mask]
            counts[mask] += 1
        # this will generate NaNs due to /0 but that is fine
        count_mask = counts > 0
        avg_pred[count_mask] /= counts[count_mask]
        # if count = 0 replace with global model
        count_mask = np.logical_not(count_mask)
        # see if global model must be used for any datapoint
        if np.any(count_mask):
            avg_pred[count_mask] = self.global_model.predict(x[count_mask])
        if return_local_mask:
            return avg_pred, count_mask
        return avg_pred

    def coefficients(self, points, alpha=0.05):
        alpha_half = alpha / 2
        point_coeffs = [[] for _ in range(len(points))]

        for bm in self.ball_mappers:
            # for each mapper find balls to which each point belongs
            points_balls = bm.model.find_balls(points, nearest_neighbour_extrapolation=True)

            for point_id, ball_ids in enumerate(points_balls):
                for ball_id in ball_ids:
                    if bm.model.balls[ball_id]["model"] is not None:
                        point_coeffs[point_id].append(bm.model.balls[ball_id]["model"].coefficiants())
        point_coeffs_low = np.array([np.quantile(pc, alpha_half, axis=0) for pc in point_coeffs])
        point_coeffs_high = np.array([np.quantile(pc, 1 - alpha_half, axis=0) for pc in point_coeffs])
        point_coeffs_mid = np.array([np.median(pc, axis=0) for pc in point_coeffs])
        return point_coeffs_low, point_coeffs_mid, point_coeffs_high

    def bands(self, points, alpha=0.05, n_jobs=1):
        alpha_half = alpha / 2.0
        preds = self.predict(points, n_jobs=n_jobs, return_all_preds=True)
        band_low = np.nanquantile(preds, alpha_half, axis=0)
        band_high = np.nanquantile(preds, 1 - alpha_half, axis=0)
        return band_low, band_high

    def get_params(self, deep=True):
        out = dict()
        out["epsilon"] = self.epsilon
        out["M"] = self.M
        out["min_n_pts"] = self.min_n_pts
        out["in_ball_model"] = self.in_ball_model
        out["n_jobs"] = self.n_jobs
        return out

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, x, y):
        yhat = self.predict(x)
        return r2_score(y, yhat)

    def score_mse(self, x, y, n_jobs=1):
        yhat = self.predict(x, n_jobs=n_jobs)
        return mean_squared_error(yhat, y, squared=False)


class SingleBMR:
    """
    Class to represent BallMapperRegression
    """

    def __init__(self, epsilon, min_n_pts, degree, max_pca_components, in_ball_model="linear"):
        """

        :param epsilon: radius of epsilon net
        :param min_n_pts: minimal number of points required inside each ball
        :param M: number of constructed BallMapper graphs
        :param degree: degree of polynomial used to construct the regression models inside a ball
        :param max_pca_components: max number of PCA components determined in each ball.
                Use 'None' if PCA should not be performed
        """
        if in_ball_model not in ["linear", "elasticnet"]:
            raise ValueError(f"in_ball_model must be linear of elasticnet. {in_ball_model} found instead")
        self.in_ball_model = in_ball_model
        self.epsilon = epsilon
        self.min_n_pts = min_n_pts

        self.npts = None  # number of points
        self.dpts = None  # dimensionality of the point data
        self.degree = degree
        self.max_pca_components = max_pca_components
        self.model = None

    def fit(self, x, y):
        self.npts = x.shape[0]  # number of points
        self.dpts = x.shape[1]  # dimension of points
        # fit one BMR
        self.model = BallMapper(points=x, epsilon=self.epsilon, shuffle=True)
        for ball_id, ball in self.model.balls.items():
            ball_pts_ind = ball["points_covered"]
            # if given ball covers more than min_n_pts points simply build the regression model inside
            n_ball_pts = len(ball_pts_ind)
            if n_ball_pts >= self.min_n_pts:
                x_ball = x[ball_pts_ind, :]
                y_ball = y[ball_pts_ind]
                localmodel = PolynomialRegression(
                    degree=self.degree, max_pca_components=self.max_pca_components, in_ball_model=self.in_ball_model
                )
                localmodel.fit(x_ball, y_ball)
                self.model.balls[ball_id]["model"] = localmodel
            else:
                # in global mode, a global model (i.e. one trained on whole data) set is used to
                # represent regression inside the ball
                self.model.balls[ball_id]["model"] = None

    def predict(self, x_test):
        npts_test = x_test.shape[0]  # number of test points
        yhat = np.zeros(npts_test)
        counts = np.zeros(npts_test)
        # get a list of nodes to which all test points belong
        # for each point a list of nodes ids is returned
        found_ball_idxs = self.model.find_balls(x_test, nearest_neighbour_extrapolation=True)
        # loop over balls to which all test points belong, this is in fact loop over test points
        for pt_id, ball_idxs in enumerate(found_ball_idxs):
            # get slice but keep information about dimension
            xp = x_test[[pt_id], :]
            if ball_idxs[0] is not None:
                # given test point can belong to many balls, loop over all of those
                # each of these balls covers several points from the trainig set
                # here we get a list of training points ids
                for ball_idx in ball_idxs:
                    # don't make predictions using global model
                    if self.model.balls[ball_idx]["model"] is not None:
                        yhat[pt_id] += self.model.balls[ball_idx]["model"].predict(xp)
                        counts[pt_id] += 1.0
            else:
                pass
        # # # search for point for which no predictions was made
        # in that case we have to use global model
        pred_mask = counts > 0
        # average predictions
        yhat = np.array(yhat)
        counts = np.array(counts)
        yhat[pred_mask] /= counts[pred_mask]
        nan_mask = counts == 0
        yhat[nan_mask] = None
        return yhat, pred_mask

    #
    #
    # # def summary(self):
    # #     print(f'Number of balls: {[len(mapper.balls) for mapper in self.ball_mappers]}')
    # #     for mapper_id, mapper in enumerate(self.ball_mappers):
    # #         print(f'Mapper {mapper_id}')
    # #         for ball_id in mapper.balls:
    # #             beta = mapper.balls[ball_id]['model']._model.coef_[0, :]
    # #             intercept = mapper.balls[ball_id]['model']._model.intercept_
    # #             merged = False
    # #             if 'merged' in mapper.balls[ball_id]:
    # #                 merged = mapper.balls[ball_id]['merged']
    # #                 size = len(mapper.balls[ball_id]['points_covered'])
    # #             print(f'BM={mapper_id} node={ball_id} #points {size}, '
    # #                   f'pos={mapper.balls[ball_id]["position"]}, '
    # #                   f'beta={beta}, '
    # #                   f'intercept={intercept[0]}, merged={merged}')
    #
    #
