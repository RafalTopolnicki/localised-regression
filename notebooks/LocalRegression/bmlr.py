import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from .ballmapper import BallMapper

class BMLR:
    def __init__(self, cut, M, epsilon=1, substitution_policy='global'):
        self.cut = cut
        self.M = M
        self.substitution_policy = substitution_policy
        self.npts = None
        self.dpts = None
        self.lm_global = ElasticNet()
        self.ball_mappers = []
        self.in_sample_remse = None
        self.fitted = False
        self.epsilon = epsilon
        self.return_nans = False

    def fit(self, x, y):
        self.__fit__(x=x, y=y, epsilon=self.epsilon, M=self.M)

    def __fit__(self, x, y, epsilon, M):
        self.epsilon = epsilon
        self.M = M
        self.npts = x.shape[0]
        self.dpts = x.shape[1]
        self.ball_mappers = []
        self.fitted = True

        # fit global model
        if self.substitution_policy == 'global':
            self.lm_global.fit(x, y)

        for r_ in range(M):
            bm = BallMapper(points=x, coloring_df=pd.DataFrame(y), epsilon=epsilon, shuffle=True)
            self.ball_mappers.append(bm)
            for node_id in bm.Graph.nodes:
                ball_pts_ind = bm.Graph.nodes[node_id]['points covered']
                if len(ball_pts_ind) >= self.cut:
                    x_ball = x[ball_pts_ind, :]
                    y_ball = y[ball_pts_ind]
                    lm = ElasticNet().fit(x_ball, y_ball)
                    bm.Graph.nodes[node_id]['beta'] = lm.coef_
                    bm.Graph.nodes[node_id]['intercept'] = lm.intercept_
                else:
                    if self.substitution_policy == 'global':
                        bm.Graph.nodes[node_id]['beta'] = self.lm_global.coef_
                        bm.Graph.nodes[node_id]['intercept'] = self.lm_global.intercept_
                    else:
                        # find nearest big ball
                        min_dist = np.Inf
                        min_id = None
                        small_ball_mean = np.mean(x[ball_pts_ind, :], axis=0)
                        for big_node_id in bm.Graph.nodes:
                            big_pts_ids = bm.Graph.nodes[big_node_id]['points covered']
                            #if len(big_pts_ids) >= self.cut:
                            big_ball_mean = np.mean(x[big_pts_ids, :], axis=0)
                            dist = np.linalg.norm(small_ball_mean - big_ball_mean)
                            if dist < min_dist:
                                min_dist = dist
                                min_id = big_node_id
                        # make Linear Model in big ball
                        # TODO: This could be rearrange
                        # 1. first build models only in big-balls
                        # 2. iterate over small balls and copy the model from big balls
                        # thanks to that we don't have to fit linear model inside big-balls multiple times
                        # if min_id == None:
                        #     raise ValueError(f'All ball contain less than cut={self.cut} '
                        #                      f'points. Decrease cut or increase epsilon!')
                        #     #self.return_nans = True
                        #     #continue
                        big_pts_ids = bm.Graph.nodes[min_id]['points covered']
                        # join the points from big ball and query ball and fit the linear model
                        all_pts_ids = big_pts_ids + ball_pts_ind
                        #lm_big = LinearRegression().fit(x[all_pts_ids, :], y[all_pts_ids])
                        lm_big = ElasticNet().fit(x[all_pts_ids, :], y[all_pts_ids])
                        bm.Graph.nodes[node_id]['beta'] = lm_big.coef_
                        bm.Graph.nodes[node_id]['intercept'] = lm_big.intercept_    

    def predict(self, x_test):
        if not self.fitted:
            raise ValueError('Cannot run predict(). Run fit() first')

        npts_test = x_test.shape[0]
        yhat = np.zeros(npts_test)
        counts = np.zeros(npts_test)
        # iterate over all mappers
        for bm in self.ball_mappers:
            # get a list of nodes to which all test points belongs
            # for each point a list of nodes ids is returned
            ball_idxs = bm.find_balls(x_test, nearest_neighbour_extrapolation=True)
            # loop over balls to which all test points belongs, this is in fact loop over test points
            for pt_id, ball_idx in enumerate(ball_idxs):
                xp = x_test[pt_id, :]
                if ball_idx[0] is not None:
                    # given test point can belong to many balls, loop over all of those
                    # each of these balls covers several points from the trainig set
                    # here we get a list of training points ids
                    for node_idx in ball_idx:
                        yhat[pt_id] += np.matmul(bm.Graph.nodes[node_idx]['beta'], xp) + bm.Graph.nodes[node_idx]['intercept']
                        counts[pt_id] += 1.0
                else:
                    pass
        yhat = np.array(yhat)
        #counts = np.array(counts).reshape(-1, 1) # FIXME: why reshape?
        counts = np.array(counts)
        yhat /= counts
        return yhat
    
    def get_params(self, deep=True):
        out = dict()
        out['epsilon'] = self.epsilon
        out['M'] = self.M
        out['substitution_policy'] = self.substitution_policy
        out['cut'] = self.cut
        return out
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def score(self, x, y):
        yhat = self.predict(x)
        rmse = mean_squared_error(yhat, y, squared=False)
        return rmse
