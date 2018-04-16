import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



NUM_FEATURE = 4 # Number of Features in each sample



#Dataset Download
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
dataset = df.values

#Iris name to integer codes
iris_names = np.unique(dataset[:,NUM_FEATURE])
for i,name in enumerate(iris_names):
    dataset[np.where(dataset[:,NUM_FEATURE] == name),NUM_FEATURE] = i

X = dataset[:,:NUM_FEATURE].astype(float)
y = dataset[:,NUM_FEATURE].astype(int)

featureLabels = ['sepal length', 'sepal width', 'petal length', 'petal width']

print('[System] Data import success')


## dividing training/testing set
def train_test_split(X, y, test_size=0.2):
    rng = np.random.RandomState(None)
    unique = np.unique(y)
    numClass = len(unique)
    numSamplePerClass = [np.sum(y==c) for c in unique]

    X_train, X_test = np.empty((0,X.shape[1])), np.empty((0,X.shape[1]))
    y_train, y_test = np.array([]), np.array([])

    for c in range(numClass):
        indexset = np.array([i for i, x in enumerate(y == unique[c]) if x])
        numTestset = int(np.round(numSamplePerClass[c] * test_size))
        index_test = indexset[np.random.permutation(numSamplePerClass[c])[:numTestset]]
        index_train = indexset[np.random.permutation(numSamplePerClass[c])[numTestset::]]
        X_test = np.append(X_test,X[index_test,:],axis=0)
        X_train = np.append(X_train,X[index_train,:],axis=0)
        y_test = np.append(y_test,y[index_test],axis=0)
        y_train = np.append(y_train,y[index_train],axis=0)

    return X_train, X_test, np.int32(y_train), np.int32(y_test)
