from BMR.bmr import *
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import matplotlib
from sklearn.svm import SVR  # for building SVR model
from sklearn.linear_model import LinearRegression

np.random.seed(1)

Xdist = 'U'

def fun(X, a1, b1, a2, b2, x):
    X[X > x] = X[X > x]
    y = a1*X + b1
    y[X>x] = a2*X[X>x] + b2
    return y

def get_step(n, a1, b1, a2, b2, x, eps):
    rng = ss.uniform(loc=-4, scale=8)
    X = rng.rvs(size=(n, 1))
    # shuffle
    # random.shuffle(X)
    y = fun(X, a1, b1, a2, b2, x)
    if eps>0:
        y += ss.norm(loc=0, scale=eps).rvs(size=(n, 1))
    return X, y


a1=-1
b1=0
a2=1
b2=10
x=0
eps=0.5
X, y = get_step(n=200, a1=a1, b1=b1, a2=a2, b2=b2, x=x, eps=eps)

Xpred = np.array(np.linspace(-4, 4, 500)).reshape(-1, 1)

bmr = BMR(min_n_pts=30, M=20, substitution_policy='global', degree=1, epsilon=1)
#bmr = BMR(min_n_pts=50, M=1, substitution_policy='nearest', degree=1, epsilon=10)
bmr.fit(X, y)
lr = LinearRegression()
lr.fit(X, y)
svr = SVR(kernel="rbf")
svr.fit(X, y)

ypred = bmr.predict(Xpred)
ypredlr = lr.predict(Xpred)
ypredsvr = svr.predict(Xpred)
ytrue = fun(Xpred, a1, b1, a2, b2, x)

#plt.plot(X, y, 'o')
# create rgb value mapping
keys = list(bmr.ball_mappers[0].balls.keys())
cmap=cm.rainbow(np.array(keys)/np.max(keys))
# create cmap
my_cmap = matplotlib.colors.ListedColormap(cmap, name='my_colormap')
# create norm

for key in keys:
    pids = []
    for pid in bmr.ball_mappers[0].balls[key]['points_covered']:
        pids.append(pid)
    plt.plot(X[pids], y[pids], 'o', color=cmap[key], label=f'BM {key}')


plt.plot(Xpred, ypredsvr, '-', label='SVR Approximation')
plt.plot(Xpred, ypred, 'o', color='black', label='BMR Approximation')
plt.plot(Xpred, ypredlr, '-', color='gray', label='Global Linear Model')
plt.plot(Xpred, ytrue, '-', color='black', label='True function')

plt.legend()
plt.show()

print(bmr.summary())