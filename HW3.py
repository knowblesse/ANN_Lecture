import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from itertools import combinations

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

# Sequential backward Selection (강의자료 코드 그대로. 코드 이해를 위해 주석을 달아둠.)
class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,test_size=0.25, random_state=1):
        self.scoring = scoring # 어떤 방식으로 clf의 performance를 체크할지.
        self.estimator = clone(estimator) # 사용할 estimator. 중간에 fit 하여 predict 하는 부분이 있어서 따로 clone을 해둠.
        self.k_features = k_features # 남길 feature의 수. = 사용할 feature의 수. k_features 개 만큼 남을때까지 feature를 지움.
        self.test_size = test_size
        self.random_state = random_state
    def fit(self, X, y):
        # 들어온 데이터 셋을 다시 train과 test set으로 나누고
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        # 데이터 셋의 feature 수를 dim에 저장.
        dim = X_train.shape[1]
        # 하나의 feature만 있는 clf의 결과를 self.score_ 값에 저장.
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    def transform(self, X):
        return X[:, self.indices_]
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # indices에 넣은 feature 들로만 fit을 진행해서 score 값을 얻어낸다.
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

## Random Forest Classifier

# Classifier 자체는 HW2의 파라미터를 그대로 사용.
clf = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)

# Feature Selection 1 : Sequential Backward Selection
fs1 = SBS(clf, k_features = 1, scoring=accuracy_score, test_size=0.25, random_state=1)
fs1.fit(X_train_std, y_train)

# Validation Set Result
k_feat = [len(k) for k in fs1.subsets_]

fig = plt.figure(figsize=(10,20))
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

fig.show()

# 현재 전체 데이터 사이즈가 150개 밖에 되지 않는다.
# 따라서 feature selection 을 할때 40개/Class * 3Class 샘플만 사용이 된다.
# 이런 경우 score 값이 각 사용한 feature 조건 별로 유사하게 나올 가능성이 있다.
# 그런데 SBS.fit 내에서 np.argmax() 함수는 두 개 이상의 조건에서 max값이 나오면 무조건 앞의 조건을 선택한다.
# 때문에 feature index가 더 작은 숫자라면(더 앞에 있다면) 뒤 조건에 비해서 선택될 가능성이 높기에 수정이 필요할 수도 있다.

# Feature Selection 2 : feature_importances_ from RandomForestClassifier

# 먼저 각 feature들의 importance를 구한다.
fs2 = RandomForestClassifier(n_estimators=500,random_state=1)
fs2.fit(X_train_std, y_train)
importance = fs2.feature_importances_

fig = plt.figure(figsize=(10,10))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('feature_importances_')

ax = fig.add_subplot(111)
ax.bar(range(X_train_std.shape[1]), importance, align='center')
ax.set_xlim(-1, X_train.shape[1])
ax.set_xlabel('Features')
ax.set_xticklabels(featureLabels)
ax.set_xticks(range(X_train.shape[1]))

ax.set_ylabel('Importance')

fig.show()

# 위의 결과를 토대로 k=4 부터 importance가 가장 낮은 feature들을 빼면서 SBS의 subsets_와 같은 변수를 만든다.
# 이 변수를 통해서 k=4부터 k=1까지 subset을 만들고 이를 training 통해 performance를 확인.
subset = []
i_importance = np.argsort(importance)[::-1]
for i in range(4,0,-1):
    subset.append(np.sort(tuple(i_importance[:i])))

# Testing Set Result
result_train = []
result_test = []

for sub in subset:
    clf = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
    clf.fit(X_train_std[:,sub],y_train)
    result_train.append(accuracy_score(y_train,clf.predict(X_train_std[:,sub])))
    result_test.append(accuracy_score(y_test, clf.predict(X_test_std[:, sub])))


fig = plt.figure(figsize=(10,20))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('feature_importances_ Method')

ax = fig.subplots(1,2)
ax[0].set_title('Accuracy of Validation Set')
ax[0].plot(k_feat, result_train, marker='o')
ax[0].set_ylim(0.8, 1.02)
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Selected Feature Indices')
ax[0].set_xticks(k_feat)
ax[0].set_xticklabels(subset)
ax[0].grid()

ax[1].set_title('Accuracy of Test Set')
ax[1].plot(k_feat, result_test, marker='o')
ax[1].set_ylim(0.8, 1.02)
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Selected Feature Indices')
ax[1].set_xticks(k_feat)
ax[1].set_xticklabels(subset)
ax[1].grid()

fig.show()

