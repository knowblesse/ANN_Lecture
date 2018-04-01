import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%matplotlib inline

#Constants for Data Import
NUM_FEATURE = 4 # Number of Features in each sample
NUM_SAMPLE_PER_CLASS = 50
NUM_CLASS = 3
NUM_TEST_SAMPLE_PER_CLASS = 10

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


class AdalineClassifier(object):
    """ADAptive LInear NEuron classifier"""

    def __init__(self, eta=0.01, n_iter=10, random_state=None, eta_schedule = False):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.random_state = random_state
        self.eta_schedule = eta_schedule
        self.trial = 0

    def fit(self, X_train, y_train, batchsize):
        """Fit training data.""" # 이후 test 값들은 cost function 만들때만 사용할꺼임.
        self._initialize_weights(X_train.shape[1]) # feature number
        self.cost_train = []
        for i in range(self.n_iter):
            cost = []
            if X_train.shape[0] % batchsize == 0: #training set 크기가 batchsize로 딱 나눠지는 경우
                batch_n_iter = int(X_train.shape[0] / batchsize) # batch를 돌려아 하는 횟수.
                # 딱 나눠지는 경우에는 배치 사이즈 그대로 train을 시킴.
                for b in range(batch_n_iter):
                    cost.append(self._update_weights(X_train[batchsize * b : batchsize *(b + 1),:],
                                                     y_train[batchsize * b : batchsize *(b + 1)]))
            else: #training set 크기가 batchsize로 딱 나눠지지 않는 경우
                batch_n_iter = int(np.floor(X_train.shape[0] / batchsize))
                # 배치 사이즈 그대로 train을 시키고,
                for b in range(batch_n_iter):
                    cost.append(self._update_weights(X_train[batchsize * b : batchsize * (b + 1),:],
                                                     y_train[batchsize * b : batchsize * (b + 1)]))
                # 나머지를 전부 넣어줌.
                cost.append(self._update_weights(X_train[batchsize*batch_n_iter :, :],
                                                 y_train[batchsize*batch_n_iter :]))
            avg_cost = np.mean(cost)
            self.cost_train.append(avg_cost)
        return self

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        if self.eta_schedule: # for learning rate scheduling
            eta_ = 1.0 / (1.0 / self.eta + self.trial)
            self.trial += 1
        else:
            eta_ = self.eta
        self.w_[1:] += eta_ * xi.T.dot(error)
        self.w_[0] += eta_ * error.sum()
        cost = 0.5 * (error ** 2).sum()
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X


class Multi_AdalineClassifier(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=None, eta_schedule = False):  # make 3 adaline classifiers.
        self.clf = [AdalineClassifier(eta=eta, n_iter=n_iter, random_state=random_state, eta_schedule = eta_schedule),
                    AdalineClassifier(eta=eta, n_iter=n_iter, random_state=random_state, eta_schedule = eta_schedule),
                    AdalineClassifier(eta=eta, n_iter=n_iter, random_state=random_state, eta_schedule = eta_schedule)]

    def fit(self, X_train, targetY_train, batchsize):  # train each classifiers with appropriate samples
        self.numClass = targetY_train.shape[1]
        for c in range(self.numClass):
            self.clf[c].fit(X_train, targetY_train[:, c], batchsize)

    def predict(self, X):
        self.output = np.zeros([X.shape[0], self.numClass])
        self.result = np.zeros([X.shape[0]])
        for c in range(self.numClass):
            self.output[:, c] = self.clf[c].net_input(X)
        for s in range(X.shape[0]):
            self.result[s] = np.where(self.output[s, :] == np.amax(self.output[s, :]))[0]
        return self.result


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, ax = None):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
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

print('[System] Class Creation success')





eta_set = [0.1, 0.01, 0.001, 0.0001]
batch_set = [1,2,4,8,16,32,120]
n_iter = 20

# Divide train(0.8)/test(0.2) set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

targetY_train = -1 * np.ones([y_train.size, np.unique(y_train).__len__()])
for s, c in enumerate(y_train):
    targetY_train[s, c] = 1

targetY_test = -1 * np.ones([y_test.size, np.unique(y_test).__len__()])
for s, c in enumerate(y_test):
    targetY_test[s, c] = 1
#####
# 3.1
#####
fig1, ax = plt.subplots(nrows=4, ncols=3)
mclf = [None] * 4
for ie, e in enumerate(eta_set):
    mclf[ie] = Multi_AdalineClassifier(eta = e, n_iter = n_iter, eta_schedule = True)
    mclf[ie].fit(X_train,targetY_train,120)
    for c in range(NUM_CLASS):
        ax[ie,c].plot(range(n_iter),np.log(mclf[ie].clf[c].cost_train))
        ax[ie,c].set_xlabel('Epochs')
        ax[ie,c].set_ylabel('log(Sum-squared-error)')
        ax[ie,c].set_title('Adaline(Class' + str(c) + 'vs rest) - eta=' + str(e))
    # Performance
    y_pred = mclf[ie].predict(X_test)
    print('Eta = ' + str(e))
    print('     Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('     Misclassified samples: %d' % (y_test != y_pred).sum())

plt.show()

fig2, ax = plt.subplots(nrows=1, ncols=4)
for ie, _ in enumerate(eta_set):
    plot_decision_regions(X_train, y_train,mclf[ie],ax = ax[ie])
plt.show()
#####
# 3.1
#####
fig1, ax = plt.subplots(nrows=7, ncols=3)
mclf = [None] * 7
for ib, b in enumerate(batch_set):
    mclf[ib] = Multi_AdalineClassifier(eta = 0.0001, n_iter = n_iter, eta_schedule = True)
    mclf[ib].fit(X_train,targetY_train,b)
    for c in range(NUM_CLASS):
        ax[ib,c].plot(range(n_iter),np.log(mclf[ib].clf[c].cost_train))
        ax[ib,c].set_xlabel('Epochs')
        ax[ib,c].set_ylabel('log(Sum-squared-error)')
        ax[ib,c].set_title('Adaline(Class' + str(c) + 'vs rest) - batchsize=' + str(b))
    # Performance
    y_pred = mclf[ib].predict(X_test)
    print('Batchsize = ' + str(b))
    print('     Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('     Misclassified samples: %d' % (y_test != y_pred).sum())

plt.show()

fig2, ax = plt.subplots(nrows=1, ncols=7)
for ib, _ in enumerate(batch_set):
    plot_decision_regions(X_train, y_train,mclf[ib],ax = ax[ib])
plt.show()






#####
#3.2
#####
#mclf_no_scaling = Multi_AdalineClassifier(eta = e, n_iter = n_iter, eta_schedule = True)
#
# for e in eta_set:
#     for b in batch_set:
#         print('eta_set = ' + str(e))
#         print('batch_set = ' + str(b))
#         mclf = Multi_AdalineClassifier(eta = e, n_iter = n_iter)
#         mclf.fit(X_train,targetY_train,X_test,targetY_test,b)
#         plt.plot(mclf.clf[0].cost_train)
#         # Performance
#         # Cost Function Curve
#         # Misclassification Error Curve
#         # Decision Boundary Plot
#         def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

#
# result = mclf.predict(X)
# print(result)
#
#
# score = 0
# tot_result = []
# for s in range(150):
#     result = [clf[0].predict_value(X[s,:]), clf[1].predict_value(X[s,:]), clf[2].predict_value(X[s,:])]
#     result = np.where(result == np.amax(result))
#     tot_result.append(result)
#     if y[s] == result:
#         score += 1
# print(score)
#
