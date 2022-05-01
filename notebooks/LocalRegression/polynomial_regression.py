from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression():
    def __init__(self, degree):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.model = ElasticNet(alpha=0.001) # this is just to avid super large coeff estimates for ties
        #self.model = LinearRegression()

    def fit(self, X, y):
        X_transf = self.poly_features.fit_transform(X)
        self.model.fit(X_transf, y)

    def predict(self, X):
        X_transf = self.poly_features.transform(X)
        return self.model.predict(X_transf)