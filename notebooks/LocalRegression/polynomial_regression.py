from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import numpy as np


class PolynomialRegression:
    def __init__(self, degree, max_pca_components=None):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.model = ElasticNet(alpha=0.001)  # this is just to avid super large coeff estimates for ties
        self.pca = None
        self.max_pca_components = max_pca_components

    def fit(self, x, y):
        if self.pca is not None:
            pca_components = np.min([self.max_pca_components, x.shape[0], x.shape[1]])
            self.pca = PCA(n_components=pca_components)
            x = self.pca.fit_transform(x)
        x_transf = self.poly_features.fit_transform(x)
        self.model.fit(x_transf, y)

    def predict(self, x):
        if self.pca is not None:
            x = self.pca.transform(x)
        x_transf = self.poly_features.transform(x)
        return self.model.predict(x_transf)
