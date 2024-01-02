
from BMR.bmr import *
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.svm import SVR  # for building SVR model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(1)

Xdist = 'U'

def fun(X, a1, a2, a3):
    y = a1*X[:, 0] + a2*X[:, 1] + a3*X[:, 0]*X[:, 1]
    return y.reshape(-1, 1)

def get_step(n, a1, a2, a3, eps):
    rng = ss.uniform(loc=-4, scale=8)
    X = rng.rvs(size=(n, 2))
    # shuffle
    # random.shuffle(X)
    y = fun(X, a1, a2, a3)
    if eps>0:
        y += ss.norm(loc=0, scale=eps).rvs(size=(n, 1))
    return X, y[:, 0]


a1=-1
a2=1
a3=5
eps=0.5
X, y = get_step(n=1000, a1=a1, a2=a2, a3=a3, eps=eps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# epsilons = np.linspace(0.05, 1.0, 10)
# for min_n_pts in [5, 10, 20, 30]:
#     scores = []
#     for epsilon in epsilons:
#         bmr = BMR(min_n_pts=min_n_pts, M=10, degree=1, epsilon=epsilon)
#         bmr.fit(X_train, y_train)
#         score = bmr.score_rmse(X_test, y_test)
#         scores.append(score)
#
#     plt.plot(epsilons, scores, 'o-', label=min_n_pts)
# plt.legend()
# plt.show()

epsilon = 1
min_n_pts = 3
bmr = BMR(min_n_pts=min_n_pts, M=1, degree=1, epsilon=epsilon)
bmr.fit(X, y)

lr = LinearRegression()
lr.fit(X, y)
svr = SVR(kernel="rbf")
svr.fit(X, y)

xs = np.linspace(-4, 4, 20)
xv, yv = np.meshgrid(xs, xs)
Xpred = np.vstack([xv.ravel(), yv.ravel()]).transpose()

ypred = bmr.predict(Xpred)
ypredlr = lr.predict(Xpred)
ypredsvr = svr.predict(Xpred)
ytrue = fun(Xpred, a1, a2, a3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xv, yv, ytrue.reshape(xv.shape), alpha=0.5)
ax.plot_surface(xv, yv, ypredlr.reshape(xv.shape), alpha=0, linewidth=0.5, edgecolors='red')
ax.plot_surface(xv, yv, ypredsvr.reshape(xv.shape), alpha=0, linewidth=0.5, edgecolors='green')
ax.plot_surface(xv, yv, ypred.reshape(xv.shape), alpha=0, linewidth=0.5, edgecolors='black')
#plt.show()
