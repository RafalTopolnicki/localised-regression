import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from .ballmapper import BallMapper
from .polynomial_regression import PolynomialRegression


class BMR:
    """
    Class to represent BallMapperRegression
    """

    def __init__(self, epsilon, min_n_pts, M, substitution_policy="global", degree=1, max_pca_components=None):
        """

        :param epsilon: radius of epsilon net
        :param min_n_pts: minimal number of points required inside each ball
        :param M: number of constructed BallMapper graphs
        :param substitution_policy: what to do when number of points inside a ball is < min_n_pts
                options are:
                'global': use global model
                'nearest': merge small ball with other nearest ball
        :param degree: degree of polynomial used to construct the regression models inside a ball
        :param max_pca_components: max number of PCA components determined in each ball.
                Use 'None' if PCA should not be performed
        """
        self.epsilon = epsilon
        self.min_n_pts = min_n_pts
        self.M = M
        if substitution_policy not in ["global", "nearest"]:
            raise ValueError(f"Substitution policy {substitution_policy} not implemented")
        self.substitution_policy = substitution_policy

        self.npts = None  # number of points
        self.dpts = None  # dimensionality of the point data
        self.ball_mappers = []  # list of Ball Mappers
        self.in_sample_remse = None
        self.fitted = False
        self.return_nans = False
        self.degree = degree
        self.max_pca_components = max_pca_components
        self.global_model = PolynomialRegression(degree=self.degree)

    def fit(self, x, y):
        self.__fit__(x=x, y=y)

    def __fit__(self, x, y):
        self.npts = x.shape[0]  # number of points
        self.dpts = x.shape[1]  # dimension of points
        self.ball_mappers = []

        # fit global model
        if self.substitution_policy == "global":
            self.global_model.fit(x, y)

        for loop_id in range(self.M):
            bm = BallMapper(points=x, coloring_df=pd.DataFrame(y), epsilon=self.epsilon, shuffle=True)
            self.ball_mappers.append(bm)

            for node_id in bm.Graph.nodes:
                ball_pts_ind = bm.Graph.nodes[node_id]["points covered"]
                # if given ball covers more than min_n_pts points simply build the regression model inside
                if len(ball_pts_ind) >= self.min_n_pts:
                    x_ball = x[ball_pts_ind, :]
                    y_ball = y[ball_pts_ind]
                    model = PolynomialRegression(degree=self.degree, max_pca_components=self.max_pca_components)
                    model.fit(x_ball, y_ball)
                    bm.Graph.nodes[node_id]["model"] = model  # store model inside graph node
                else:
                    # umber of points inside a ball is smaller then required
                    if self.substitution_policy == "global":
                        # in global mode, a global model (i.e. one trained on whole data) set is used to
                        # represent regression inside the ball
                        bm.Graph.nodes[node_id]["model"] = self.global_model
                    else:
                        # find the nearest big ball containing at least min_n_pts points inside
                        min_dist = np.Inf
                        min_id = None
                        small_ball_location = np.mean(x[ball_pts_ind, :], axis=0)
                        for big_node_id in bm.Graph.nodes:
                            big_pts_ids = bm.Graph.nodes[big_node_id]["points covered"]
                            if len(big_pts_ids) >= self.min_n_pts:
                                big_ball_location = np.mean(x[big_pts_ids, :], axis=0)
                                dist = np.linalg.norm(small_ball_location - big_ball_location)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_id = big_node_id
                        if min_id is None:
                            # raise ValueError(f'All balls in BM contain less than {self.min_n_pts} points. '
                            #                  f'Reduce value of min_pts parameter')
                            # print(f'Warning. All balls in BM contain less than {self.min_n_pts} points. '
                            #      f'Reduce value of min_pts parameter by approx 20% to {int(self.min_n_pts*0.8)}')
                            self.min_n_pts = int(self.min_n_pts * 0.8)
                            self.fit(x, y)
                            break
                        big_pts_ids = bm.Graph.nodes[min_id]["points covered"]
                        # join the points from big ball and query ball and fit the linear model
                        all_pts_ids = big_pts_ids + ball_pts_ind
                        model_big = PolynomialRegression(degree=self.degree, max_pca_components=self.max_pca_components)
                        model_big.fit(x[all_pts_ids, :], y[all_pts_ids])
                        bm.Graph.nodes[node_id]["model"] = model_big
        # set flag that BMLR was fitted aready
        self.fitted = True

    def predict(self, x_test):
        if not self.fitted:
            raise ValueError("Cannot run predict(). Run fit() first")

        npts_test = x_test.shape[0]  # number of test points
        yhat = np.zeros(npts_test)
        counts = np.zeros(npts_test)
        # iterate over all mappers
        for bm in self.ball_mappers:
            # get a list of nodes to which all test points belong
            # for each point a list of nodes ids is returned
            ball_idxs = bm.find_balls(x_test, nearest_neighbour_extrapolation=True)
            # loop over balls to which all test points belong, this is in fact loop over test points
            for pt_id, ball_idx in enumerate(ball_idxs):
                # get slice but keep information about dimension
                xp = x_test[[pt_id], :]
                if ball_idx[0] is not None:
                    # given test point can belong to many balls, loop over all of those
                    # each of these balls covers several points from the trainig set
                    # here we get a list of training points ids
                    for node_idx in ball_idx:
                        yhat[pt_id] += bm.Graph.nodes[node_idx]["model"].predict(xp)
                        counts[pt_id] += 1.0
                else:
                    pass
        yhat = np.array(yhat)
        counts = np.array(counts)
        yhat /= counts
        return yhat

    def get_params(self, deep=True):
        out = dict()
        out["epsilon"] = self.epsilon
        out["M"] = self.M
        out["substitution_policy"] = self.substitution_policy
        out["min_n_pts"] = self.min_n_pts
        return out

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, x, y):
        yhat = self.predict(x)
        return mean_squared_error(yhat, y, squared=False)
