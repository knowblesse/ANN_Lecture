import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
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



# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

## 4개의 feature가 관여되기에 전부 Standardization을 진행
# Standardization
clf_std = StandardScaler()
clf_std.fit(X_train)
X_train_std = clf_std.transform(X_train)
X_test_std = clf_std.transform(X_test)



## Covariance 구하기

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\n Eigenvalues \n%s' %eigen_vals)


## eigen vector의 설명력 구하기.
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 5), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 5), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xticks(range(1,5))
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)



w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis],
               eigen_pairs[2][1][:, np.newaxis],
               eigen_pairs[3][1][:, np.newaxis]))
print('Matrix W:\n', w)


## k feature에 맞게 다르게 나눈 eigen pair를 넣어둘 것.
# HW에서는 training set을 그대로 두고 사용할 feature index만 주면 되었는데 여기서는 X를 변환시켜주어야 함.


newX_train_PCA = []
newX_test_PCA = []

for i in range(4): # k=1부터 4까지 해서
    newX_train_PCA.append(X_train_std.dot(w[:,0:i+1]))
    newX_test_PCA.append(X_test_std.dot(w[:,0:i+1]))


#################
### LDA
# Caluate Mean Vectors
mean_vecs = []
for c in range(3):
    mean_vecs.append(np.mean(X_train_std[y_train == c], axis = 0))
    print('MV %s: %s \n' %(c, mean_vecs[c-1]))

S_W = np.zeros((NUM_FEATURE,NUM_FEATURE))
for c, mv in zip(range(3), mean_vecs):
    class_scatter = np.zeros((NUM_FEATURE, NUM_FEATURE))
    for row in X_train_std[y_train == c]:
        row, mv = row.reshape(NUM_FEATURE, 1), mv.re



"""
필요한 것
- 3개의 feature extraction method
x
k 개의 feature dimensions
2 개의 classifier

한 그래프 내에 k 개의 feature dimension

왼쪽은 logistic
오른쪽은 SVM

"""

# Feature Selection 1 : Sequential Backward Selection
fs1 = SBS(clf, k_features = 1, scoring=accuracy_score, test_size=0.25, random_state=1)
fs1.fit(X_train_std, y_train)

# Validation Set Result
k_feat = [len(k) for k in fs1.subsets_]

fig = plt.figure(figsize=(14,7))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('Sequential Backward Selection')

ax = fig.subplots(1,2)
ax[0].set_title('Accuracy of Validation Set')
ax[0].plot(k_feat, fs1.scores_, marker='o')
ax[0].set_ylim(0.8, 1.02)
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Selected Feature Indices')
ax[0].set_xticks(k_feat)
ax[0].set_xticklabels(fs1.subsets_)
ax[0].grid()

# Testing Set Result
result = []
for sub in fs1.subsets_:
    clf_test = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
    clf_test.fit(X_train_std[:,sub],y_train)
    result.append(accuracy_score(y_test,clf_test.predict(X_test_std[:,sub])))

ax[1].set_title('Accuracy of Test Set')
ax[1].plot(k_feat, result, marker='o')
ax[1].set_ylim(0.8, 1.02)
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Selected Feature Indices')
ax[1].set_xticks(k_feat)
ax[1].set_xticklabels(fs1.subsets_)
ax[1].grid()






## dividing training/testing set
def train_test_split(X, y, test_size=0.2):
    rng = np.random.RandomState(None)
    unique = np.unique(y)
    numClass = len(unique)
    numSamplePerClass = [np.sum(y==c) for c in unique]

    X_train, X_test, y_train, y_test = [], [], [], []

    for c in range(numClass):
        indexset = [i for i, x in enumerate(y == unique[c]) if x]
        numTestset = int(np.round(numSamplePerClass[c] * test_size))
        index_test = indexset[np.random.permutation(numSamplePerClass[c])[:numTestset]]
        index_train = indexset[np.random.permutation(numSamplePerClass[c])[numTestset::]]
        X_test.append(X[index_test,:])
        X_train.append(X[index_train,:])
        y_test.append(y[index_test])
        y_train.append(y[index_train])

    return X_train, X_test, y_train, y_test






    X_train, X_test, y_train, y_test =
    if

    C = 1

