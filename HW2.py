import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler

### Pandas로 데이터 불러오기.

#Constants for Data Import
NUM_FEATURE = 4 # Number of Features in each sample

#Dataset Download
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
dataset = df.values

#Iris name to integer codes
iris_names = np.unique(dataset[:,NUM_FEATURE])
for i,name in enumerate(iris_names):
    dataset[np.where(dataset[:,NUM_FEATURE] == name),NUM_FEATURE] = i

X = dataset[:,:NUM_FEATURE].astype(float)
# Using Only Two Features (Sepal length & Petal Length)
X = X[:,[0,2]]
y = dataset[:,NUM_FEATURE].astype(int)

print('[System] Data import success')

def plot_decision_regions(X, y, classifier, ax, graph_margin = 1, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - graph_margin, X[:, 0].max() + graph_margin
    x2_min, x2_max = X[:, 1].min() - graph_margin, X[:, 1].max() + graph_margin
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Petal Length')
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        ax.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')


## Training Set / Testing Set 분리.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

## Normalization vs Standardization
# Normalization
clf_norm = Normalizer()
clf_norm.fit(X_train)
X_train_norm = clf_norm.transform(X_train)
X_test_norm = clf_norm.transform(X_test)

# Standardization
clf_std = StandardScaler()
clf_std.fit(X_train)
X_train_std = clf_std.transform(X_train)
X_test_std = clf_std.transform(X_test)

# X_train, X_test
# X_train_norm, X_test_norm
# X_train_std, X_test_std

### Implement Logistic regression
class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1, batch_size=-1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y):
        """ Fit training data.
        batch size를 기준으로 입력된 X 값을 batch size로 나눠서 학습을 시킴.
        X 사이즈가 batch size로 딱 나누어 떨어지지 않는 경우,
         맨 마지막 남는 데이터는 batch size 보다 크기가 작아도 그대로 학습을 진행.
         ex) X.shape[0] = 150, batch_size = 100
         epoch1-1 : X[0:100]
         epoch1-2 : X[100:150]
         epoch2-1 : X[0:100]
         epoch2-2 : X[100:150]
       대신 batch size로 -1을 넣는 경우 그냥 batch Gradient Descent 방식을 사용.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        if self.batch_size == -1: # Batch Gradient
            self.batch_size = X.shape[0]
        for i in range(self.n_iter):
            if X.shape[0] % self.batch_size == 0:  # training set 크기가 batch_size로 딱 나눠지는 경우
                batch_n_iter = int(X.shape[0] / self.batch_size)  # batch를 돌려아 하는 횟수.
                # 딱 나눠지는 경우에는 배치 사이즈 그대로 train을 시킴.
                for b in range(batch_n_iter):
                    smallX = X[self.batch_size * b : self.batch_size * (b + 1), :]
                    smally = y[self.batch_size * b : self.batch_size * (b + 1)]
                    net_input = self.net_input(smallX)
                    output = self.activation(net_input)
                    errors = (smally - output)
                    self.w_[1:] += self.eta * smallX.T.dot(errors)
                    self.w_[0] += self.eta * errors.sum()
                    cost = -smally.dot(np.log(output)) - ((1 - smally).dot(np.log(1 - output)))
                    self.cost_.append(cost)

            else:  # training set 크기가 batchsize로 딱 나눠지지 않는 경우
                batch_n_iter = int(np.floor(X.shape[0] / self.batch_size))
                # 배치 사이즈 그대로 train을 시키고,
                for b in range(batch_n_iter):
                    smallX = X[self.batch_size * b : self.batch_size * (b + 1), :]
                    smally = y[self.batch_size * b : self.batch_size * (b + 1)]
                    net_input = self.net_input(smallX)
                    output = self.activation(net_input)
                    errors = (smally - output)
                    self.w_[1:] += self.eta * smallX.T.dot(errors)
                    self.w_[0] += self.eta * errors.sum()
                    cost = -smally.dot(np.log(output)) - ((1 - smally).dot(np.log(1 - output)))
                    self.cost_.append(cost)
                # 나머지를 전부 넣어줌.
                smallX = X[self.batch_size * batch_n_iter :, :]
                smally = y[self.batch_size * batch_n_iter :]
                net_input = self.net_input(smallX)
                output = self.activation(net_input)
                errors = (smally - output)
                self.w_[1:] += self.eta * smallX.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                cost = -smally.dot(np.log(output)) - ((1 - smally).dot(np.log(1 - output)))
                self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

class LogisticRegressionGD_3C(object):
    """Class 3개인 데이터를 위한 classifier.
    3개의 LogisticRegressionGD Class를 만든뒤 One Versus All 방식으로 Class를 분류."""
    def __init__(self, eta=0.001, n_iter=1000, random_state=1, batch_size = -1):
        self.clf = [LogisticRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state, batch_size = batch_size),
                    LogisticRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state, batch_size = batch_size),
                    LogisticRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state, batch_size = batch_size)]

    def fit(self, X, y):
        # y값이 -1과 1이 아니라 0,1,2 의 세 숫자로 구성되기에 각 classifier에 맞는 형태로 변형시켜주어야 함.
        y_3C = 0 * np.ones([y.size, 3]) # sample 수 X 3 크기로 전부 0인 array를 만들고,
        for s, c in enumerate(y): # 모든 sample에 대해서
            y_3C[s, c] = 1  # index y에 해당하는 값만 1로 바꿔줌.
            # 이렇게 하면 Class0인 경우 [1,-1,-1], Class1인 경우 [-1,1,-1] 형태의 array가 만들어짐.

        for c in range(3):
            self.clf[c].fit(X,y_3C[:,c])
        return self

    def predict(self, X):
        result = np.zeros([X.shape[0],3])
        for c in range(3):
            result[:,c] = self.clf[c].net_input(X) # predict를 바로 하는 것이 아니라 net_input 값만 받아둠.
        return np.argmax(result, axis=1)

# eta_set = [0.1, 0.01, 0.001, 0.0001]
# batch_set = [1,2,4,8,16,32,120]
#
# output = pd.DataFrame(np.zeros([4,7]))
#
# fig1, ax = plt.subplots(4,7)
# for it in range(10):
#     print('iteration :' + str(it))
#     ## Training Set / Testing Set 분리.
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)
#     for ie, e in enumerate(eta_set):
#         for ib, b in enumerate(batch_set):
#             clf = LogisticRegressionGD_3C(random_state=None, eta=e, batch_size=b)
#             clf.fit(X_train,y_train)
#             #plot_decision_regions(X,y,clf,ax[ie,ib])
#             output[ib][ie] +=  accuracy_score(y_test,clf.predict(X_test))
#             print('eta = ' + str(e) + ' batch size = ' + str(b))
#
# plt.matshow(output/10)
# plt.colorbar()
# print(output/10)


#
#
#
# clf = LogisticRegressionGD_3C(random_state=3,batch_size=2)
# clf.fit(X,y)
# fig1, ax = plt.subplots()
# plot_decision_regions(X,y,clf,ax)
#
# clf1 = LogisticRegressionGD_3C(random_state=3,batch_size=-1)
# clf1.fit(X,y)
# fig2, ax = plt.subplots()
# plot_decision_regions(X,y,clf1,ax)
#
# fig3, ax = plt.subplots()
# from sklearn.linear_model import LogisticRegression
# clf2 = LogisticRegression(C=100)
# clf2.fit(X,y)
# plot_decision_regions(X,y,clf2,ax)
# plt.show()

