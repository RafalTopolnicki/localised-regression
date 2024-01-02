import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from .ballmapper import BallMapper
from .polynomial_regression import PolynomialRegression

import sys

class FakeScaler:
    def __init__(self):
        pass
    def fit(self, x):
        pass
    def transform(self, x):
        return x
    def fit_transform(self, x):
        return x

class BMR:
    """
    Class to represent BallMapperRegression
    """

    def __init__(self, epsilon, min_n_pts, M, degree=1, standard_scaler = False,
                 max_pca_components=None,
                 in_ball_model='linear'):
        """

        :param epsilon: radius of epsilon net
        :param min_n_pts: minimal number of points required inside each ball
        :param M: number of constructed BallMapper graphs
        :param degree: degree of polynomial used to construct the regression models inside a ball
        :param max_pca_components: max number of PCA components determined in each ball.
                Use 'None' if PCA should not be performed
        """
        if in_ball_model not in ['linear', 'elasticnet']:
            raise ValueError(f'in_ball_model must be linear of elasticnet. {in_ball_model} found instead')
        self.in_ball_model = in_ball_model
        self.epsilon = epsilon
        self.min_n_pts = min_n_pts
        self.M = M

        self.npts = None  # number of points
        self.dpts = None  # dimensionality of the point data
        self.ball_mappers = []  # list of Ball Mappers
        self.fitted = False
        self.degree = degree
        self.standard_scaler = standard_scaler
        if self.standard_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = FakeScaler()
        self.max_pca_components = max_pca_components
        self.global_model = PolynomialRegression(degree=self.degree, in_ball_model=in_ball_model)

    def fit(self, x, y):
        self.npts = x.shape[0]  # number of points
        self.dpts = x.shape[1]  # dimension of points
        self.ball_mappers = []

        # scale data
        xscaled = self.scaler.fit_transform(x)
        self.global_model.fit(xscaled, y)
        # iterate over the averaging
        for loop_id in range(self.M):
            mapper = BallMapper(points=xscaled, epsilon=self.epsilon, shuffle=True)
            for ball_id, ball in mapper.balls.items():
                ball_pts_ind = ball['points_covered']
                # if given ball covers more than min_n_pts points simply build the regression model inside
                n_ball_pts = len(ball_pts_ind)
                if n_ball_pts >= self.min_n_pts:
                    x_ball = xscaled[ball_pts_ind, :]
                    y_ball = y[ball_pts_ind]
                    model = PolynomialRegression(degree=self.degree, max_pca_components=self.max_pca_components,
                                                 in_ball_model=self.in_ball_model)
                    model.fit(x_ball, y_ball)
                    mapper.balls[ball_id]['model'] = model
                else:
                    # in global mode, a global model (i.e. one trained on whole data) set is used to
                    # represent regression inside the ball
                    mapper.balls[ball_id]['model'] = None
            self.ball_mappers.append(mapper)
        self.fitted = True

    def predict(self, x_test):
        if not self.fitted:
            raise ValueError("Cannot run predict(). Run fit() first")

        # scale data first
        x_test_scaled = self.scaler.transform(x_test)

        npts_test = x_test_scaled.shape[0]  # number of test points
        yhat = np.zeros(npts_test)
        counts = np.zeros(npts_test)
        # iterate over all mappers
        for mapper in self.ball_mappers:
            # get a list of nodes to which all test points belong
            # for each point a list of nodes ids is returned
            found_ball_idxs = mapper.find_balls(x_test_scaled, nearest_neighbour_extrapolation=True)
            # loop over balls to which all test points belong, this is in fact loop over test points
            for pt_id, ball_idxs in enumerate(found_ball_idxs):
                # get slice but keep information about dimension
                xp = x_test_scaled[[pt_id], :]
                if ball_idxs[0] is not None:
                    # given test point can belong to many balls, loop over all of those
                    # each of these balls covers several points from the trainig set
                    # here we get a list of training points ids
                    for ball_idx in ball_idxs:
                        # don't make predictions using global model
                        if mapper.balls[ball_idx]["model"] is not None:
                            yhat[pt_id] += mapper.balls[ball_idx]["model"].predict(xp)
                            counts[pt_id] += 1.0
                else:
                    pass
        # search for point for which no predictions was made
        # in that case we have to use global model
        no_pred_mask = (counts == 0)
        if np.sum(no_pred_mask) > 0:
            counts[no_pred_mask] = 1
            yhat[no_pred_mask] = self.global_model.predict(x_test_scaled[no_pred_mask])
        # average predictions
        yhat = np.array(yhat)
        counts = np.array(counts)
        yhat /= counts
        return yhat

    def coefficients(self, points):
        coeffs = []
        intercepts = []

        for mapper in self.ball_mappers:
            # for each mapper find balls to which each point belongs
            points_balls = mapper.find_balls(points, nearest_neighbour_extrapolation=True)
            coeffs_mapper = []
            intercepts_mapper = []

            for ball_ids, point in zip(points_balls, points):
                point_coeffs = []
                point_intercepts = []
                for ball_id in ball_ids:
                    point_coeffs.append(mapper.balls[ball_id]['model']._model.coef_[0, :])
                    point_intercepts.append(mapper.balls[ball_id]['model']._model.intercept_)
                coeffs_mapper.append(np.mean(point_coeffs, axis=0))
                intercepts_mapper.append(np.mean(point_intercepts))

            coeffs.append(coeffs_mapper)
            intercepts.append(intercepts_mapper)

        coeffs = np.mean(coeffs, axis=0)
        intercepts = np.mean(intercepts, axis=0)
        return coeffs, intercepts

    # def summary(self):
    #     print(f'Number of balls: {[len(mapper.balls) for mapper in self.ball_mappers]}')
    #     for mapper_id, mapper in enumerate(self.ball_mappers):
    #         print(f'Mapper {mapper_id}')
    #         for ball_id in mapper.balls:
    #             beta = mapper.balls[ball_id]['model']._model.coef_[0, :]
    #             intercept = mapper.balls[ball_id]['model']._model.intercept_
    #             merged = False
    #             if 'merged' in mapper.balls[ball_id]:
    #                 merged = mapper.balls[ball_id]['merged']
    #                 size = len(mapper.balls[ball_id]['points_covered'])
    #             print(f'BM={mapper_id} node={ball_id} #points {size}, '
    #                   f'pos={mapper.balls[ball_id]["position"]}, '
    #                   f'beta={beta}, '
    #                   f'intercept={intercept[0]}, merged={merged}')


    def get_params(self, deep=True):
        out = dict()
        out["epsilon"] = self.epsilon
        out["M"] = self.M
        out["min_n_pts"] = self.min_n_pts
        out['in_ball_model'] = self.in_ball_model
        out['standard_scaler'] = self.standard_scaler
        return out

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, x, y):
        yhat = self.predict(x)
        return r2_score(y, yhat)
    def score_rmse(self, x, y):
        yhat = self.predict(x)
        return mean_squared_error(yhat, y, squared=False)