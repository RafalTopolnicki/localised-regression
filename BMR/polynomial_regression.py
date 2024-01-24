from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import numpy as np


class PolynomialRegression:
    """
    Class for polynomial regression used by BMR
    """

    def __init__(self, degree=2, max_pca_components=None, in_ball_model="linear"):
        """

        :param degree: polynomial degree used in the fit
        :param max_pca_components: max number of PCA components determined in each ball.
                Use 'None' if PCA should not be performed
        """
        self.degree = degree
        self.max_pca_components = max_pca_components

        self._poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        if in_ball_model == "linear":
            self._model = LinearRegression()
        else:
            self._model = ElasticNet(alpha=0.001, max_iter=10000)
        self._pca = None

    def fit(self, x, y):
        if self.max_pca_components is not None:
            pca_components = np.min([self.max_pca_components, x.shape[0], x.shape[1]])
            self._pca = PCA(n_components=pca_components)
            x = self._pca.fit_transform(x)
        x_transf = self._poly_features.fit_transform(x)
        self._model.fit(x_transf, y)

    def predict(self, x):
        if self.max_pca_components is not None:
            x = self._pca.transform(x)
        x_transf = self._poly_features.transform(x)
        return self._model.predict(x_transf)

    def coefficiants(self):
        return self._model.coef_.tolist() + [self._model.intercept_.tolist()]

    def get_params(self, deep=False):
        out = {'degree': self.degree}
        return out

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        return self

    def score(self, x, y):
        yhat = self.predict(x)
        return r2_score(y, yhat)

