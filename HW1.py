import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

    def __init__(self, eta=0.01, n_iter=50, random_state = 1):  # Constructor
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):  # Training
        # Dataset details
        self.numFeature = X.shape[1]

        # Weight Set
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + self.numFeature)
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = net_input
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def predict_value(self, X):
        # Changed to quantitatively Score multiple Adaline
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def net_input(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]



targetY = -1 * np.ones([y.size, np.unique(y).size])
for s, c in enumerate(y):
    targetY[s, c] = 1


class Multi_AdalineClassifier(object):
    def __init__(self, eta=0.01, n_iter=50, random_state = 1):
        self.clf = [AdalineClassifier(eta = eta, n_iter = n_iter, random_state=random_state),
                    AdalineClassifier(eta = eta, n_iter = n_iter, random_state=random_state),
                    AdalineClassifier(eta = eta, n_iter = n_iter, random_state=random_state)]

    def fit(self, X, targetY):
        self.numClass = targetY.shape[1]
        for c in range(self.numClass):
            self.clf[c].fit(X, targetY[:, c])

    def predict(self, X):
        self.output = np.zeros([X.shape[0],self.numClass])
        self.result = np.zeros([X.shape[0]])
        for c in range(self.numClass):
            self.output[:,c] = self.clf[c].predict_value(X)
        for s in range(X.shape[0]):
            self.result[s] = np.where(self.output[s,:] == np.amax(self.output[s,:]))[0]
        return self.result

mclf = Multi_AdalineClassifier(eta = 0.0001, n_iter = 10000)
mclf.fit(X,targetY)
result = mclf.predict(X)
print(result)

# clf = AdalineClassifier(eta = 0.0001, n_iter = 100)
# clf.fit(X,targetY[:,1])
# output2 = clf.predict_value(X)


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=1, stratify=y)


"""

score = 0
tot_result = []
for s in range(150):
    result = [clf[0].predict_value(X[s,:]), clf[1].predict_value(X[s,:]), clf[2].predict_value(X[s,:])]
    result = np.where(result == np.amax(result))
    tot_result.append(result)
    if y[s] == result:
        score += 1
print(score)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), mar
ker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
# plt.savefig('images/02_11.png', dpi=300)
plt.show()
"""