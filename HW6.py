import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


## Load Iris Data set
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

## Implement K-mean algorithm

# Class definition
class KMeans(object):
    # KMean Algorithm을 구현.
    def __init__(self, n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        # 주어진 조건으로 fit 만 진행하는 함수.

        # 코드 간결성을 위해 변수 지정.
        datasize = np.shape(X)[0]
        datadimension = np.shape(X)[1]

        # 총 n_init 번 돌려야 함.
        centroidset = np.zeros([self.n_init,self.n_clusters,datadimension])
        errorscore = np.zeros(self.n_init)
        for iterations in range(self.n_init):
            # initial centroid 지정
            centroid = self.initialize(X)
            meandistance = 0
            for run in range(self.max_iter):
                # 현재 centroid를 기반으로 모든 sample 들의 euclidean distance를 구함.
                distances = np.zeros([datasize,self.n_clusters])
                for i in range(self.n_clusters):
                    diff = np.tile(centroid[i,:],[datasize,1]) - X
                    distances[:,i] = np.sum(diff**2,1)**0.5

                # 이렇게 구한 euclidean distance를 기반으로 가장 가까운 centroid 라벨을 부여.
                labels = np.argmin(distances, 1)

                # meandistance가 tolerance보다 작으면 관두고
                meandistance = np.mean(np.min(distances,1))
                if  meandistance< self.tol:
                    break
                # 여전히 크면 새로운 라벨을 기반으로 새로운 centroid를 선택함.
                else:
                    for i in range(self.n_clusters):
                        centroid[i,:] = np.mean(X[labels==i,:],0)
            # 다 돌린뒤 centroid 와 score를 저장.
            centroidset[iterations,:,:] = centroid
            errorscore[iterations] = meandistance
        # 가장 성능이 좋았던 centroid를 선택.
        self.centroid = centroidset[np.argmin(errorscore),:]
        return self

    def fit_predict(self, X):
        # 주어진 조건으로 fit과 함께 해당하는 sample의 label을 출력하는 함수.

        # do fit
        self.fit(X)

        # variables
        n_clusters = self.n_clusters
        centroid = self.centroid
        datasize = np.shape(X)[0]

        # 모든 sample 들의 euclidean distance를 구함.
        distances = np.zeros([datasize, n_clusters])
        for i in range(n_clusters):
            diff = np.tile(centroid[i, :], [datasize, 1]) - X
            distances[:, i] = np.sum(diff ** 2, 1) ** 0.5

        # distances를 기반으로 가장 가까운 centroid 라벨을 부여.
        labels = np.argmin(distances, 1)
        return labels

    def initialize(self,X):
        # 초기 centroid를 지정하는 함수.

        #rgen 여기에서 초기화.
        rgen = np.random.RandomState(self.random_state)

        # 코드 간결성을 위해 변수 지정
        datasize = np.shape(X)[0]
        datadimension = np.shape(X)[1]

        # 걍 아무거나 랜덤으로 정함.
        if self.init == 'random':
            centroid = X[rgen.permutation(datasize)[0:self.n_clusters], :]
        # k-mean 플플 알고리즘 적용.
        elif self.init == 'k-means++':
            # centroid 값 넣을 변수 생성
            centroid = np.zeros([self.n_clusters,datadimension])
            # 첫 centroid는 아무거나 정함
            centroidindex = rgen.permutation(datasize)[0]
            centroid[0,:] = X[centroidindex, :]
            # 선택한 sample은 X에서 제외.
            newX = np.delete(X,centroidindex,0)

            # 나머지 centroid를 정하는 알고리즘 시작
            for i in range(self.n_clusters - 1):
                # 모든 sample 들의 euclidean distance를 구함.
                distances = np.zeros([datasize -1 -i, i + 1])
                for j in range(i+1):
                    diff = np.tile(centroid[j, :], [datasize -1 -i, 1]) - newX
                    distances[:, j] = np.sum(diff ** 2, 1) ** 0.5

                # 이 distance 중에서 최소 값을 기준으로 다음 centroid로 뽑힐 확률을 계산.
                probabilities = np.min(distances, 1) / np.sum(np.min(distances,1))
                # [0,1] 범위를 distance 값을 사용해서 나눔.
                randomrange = np.cumsum(probabilities)
                # rand(1) 값보다 처음으로 큰 randomrange 값을 index로 고름.
                newindex = np.argmax(randomrange > rgen.rand(1))

                # centroid 에 넣기.
                centroid[i+1,:] = newX[newindex,:]

                # 넣은 sample은 빼기
                newX = np.delete(newX,newindex,0)
        # 그런 옵션은 없습니다 고객님.
        else:
            raise ValueError(self.init + ' is not appropriate parameter!')
        return centroid

class FCM(object):
    # Fuzzy C KMean Algorithm을 구현.
    def __init__(self, n_clusters=3, fuzzy_coef_m = 2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0):
        self.n_clusters = n_clusters
        self.fuzzy_coef_m = fuzzy_coef_m
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        # 주어진 조건으로 fit 만 진행하는 함수.

        # 코드 간결성을 위해 변수 지정.
        datasize = np.shape(X)[0]
        datadimension = np.shape(X)[1]

        ######
        fuzzy_coef_m = self.fuzzy_coef_m


        # 총 n_init 번 돌려야 함.
        centroidset = np.zeros([self.n_init,self.n_clusters,datadimension])
        errorscore = np.zeros(self.n_init)
        for iterations in range(self.n_init):
            # initial centroid 지정
            centroid = self.initialize(X)
            meandistance = 0
            for run in range(self.max_iter):
                # 현재 centroid를 기반으로 모든 sample 들의 euclidean distance를 구함.
                distances = np.zeros([datasize,self.n_clusters])
                for i in range(self.n_clusters):
                    diff = np.tile(centroid[i,:],[datasize,1]) - X
                    distances[:,i] = np.sum(diff**2,1)**0.5

                # 이렇게 구한 euclidean distance를 기반으로 w 값을 구함.
                labels = np.zeros([datasize,self.n_clusters])
                distances = distances + 0.000001
                for i in range(self.n_clusters):
                    labels[:,i] = np.sum(
                        (np.tile(distances[:,i].reshape([-1,1]),[1,self.n_clusters]) / distances) ** (2/(self.fuzzy_coef_m-1))
                    ,1)

                labels = labels ** -1

                # meandistance가 tolerance보다 작으면 관두고
                meandistance = np.mean(np.min(distances,1))
                if  meandistance< self.tol:
                    break
                # 여전히 크면 새로운 라벨을 기반으로 새로운 centroid를 선택함.
                else:
                    for i in range(self.n_clusters):
                        centroid[i,:] = ( np.sum(X[:,i] * labels[:,i]) )  / np.sum(labels[:,i])
            # 다 돌린뒤 centroid 와 score를 저장.
            centroidset[iterations,:,:] = centroid
            errorscore[iterations] = meandistance
        # 가장 성능이 좋았던 centroid를 선택.
        self.centroid = centroidset[np.argmin(errorscore),:]
        return self

    def fit_predict(self, X):
        # 주어진 조건으로 fit과 함께 해당하는 sample의 label을 출력하는 함수.

        # do fit
        self.fit(X)

        # variables
        n_clusters = self.n_clusters
        centroid = self.centroid
        datasize = np.shape(X)[0]

        # 현재 centroid를 기반으로 모든 sample 들의 euclidean distance를 구함.
        distances = np.zeros([datasize, self.n_clusters])
        for i in range(self.n_clusters):
            diff = np.tile(centroid[i, :], [datasize, 1]) - X
            distances[:, i] = np.sum(diff ** 2, 1) ** 0.5

            # 이렇게 구한 euclidean distance를 기반으로 w 값을 구함.
            labels = np.zeros([datasize, self.n_clusters])
            distances = distances + 0.000001
            for i in range(self.n_clusters):
                labels[:, i] = np.sum(
                    (np.tile(distances[:, i].reshape([-1, 1]), [1, self.n_clusters]) / distances) ** (2 / (self.fuzzy_coef_m - 1))
                    , 1)

            labels = labels ** -1
        return labels

    def initialize(self,X):
        # 초기 centroid를 지정하는 함수.

        #rgen 여기에서 초기화.
        rgen = np.random.RandomState(self.random_state)

        # 코드 간결성을 위해 변수 지정
        datasize = np.shape(X)[0]
        datadimension = np.shape(X)[1]

        # 걍 아무거나 랜덤으로 정함.
        if self.init == 'random':
            centroid = X[rgen.permutation(datasize)[0:self.n_clusters], :]
        # k-mean 플플 알고리즘 적용.
        elif self.init == 'k-means++':
            # centroid 값 넣을 변수 생성
            centroid = np.zeros([self.n_clusters,datadimension])
            # 첫 centroid는 아무거나 정함
            centroidindex = rgen.permutation(datasize)[0]
            centroid[0,:] = X[centroidindex, :]
            # 선택한 sample은 X에서 제외.
            newX = np.delete(X,centroidindex,0)

            # 나머지 centroid를 정하는 알고리즘 시작
            for i in range(self.n_clusters - 1):
                # 모든 sample 들의 euclidean distance를 구함.
                distances = np.zeros([datasize -1 -i, i + 1])
                for j in range(i+1):
                    diff = np.tile(centroid[j, :], [datasize -1 -i, 1]) - newX
                    distances[:, j] = np.sum(diff ** 2, 1) ** 0.5

                # 이 distance 중에서 최소 값을 기준으로 다음 centroid로 뽑힐 확률을 계산.
                probabilities = np.min(distances, 1) / np.sum(np.min(distances,1))
                # [0,1] 범위를 distance 값을 사용해서 나눔.
                randomrange = np.cumsum(probabilities)
                # rand(1) 값보다 처음으로 큰 randomrange 값을 index로 고름.
                newindex = np.argmax(randomrange > rgen.rand(1))

                # centroid 에 넣기.
                centroid[i+1,:] = newX[newindex,:]

                # 넣은 sample은 빼기
                newX = np.delete(newX,newindex,0)
        # 그런 옵션은 없습니다 고객님.
        else:
            raise ValueError(self.init + ' is not appropriate parameter!')
        return centroid


#
# # K mean 잘 작동하는 지 확인.
# km = KMeans(init='k-means++',n_init=10, max_iter=100)
# ys = km.fit_predict(X)
#
# plt.scatter(X[ys==0,0],X[ys==0,1],c='r')
# plt.scatter(X[ys==1,0],X[ys==1,1],c='g')
# plt.scatter(X[ys==2,0],X[ys==2,1],c='b')
# plt.scatter(km.centroid[0,0],km.centroid[0,1],s=250,c='r',marker='*',edgecolors='k')
# plt.scatter(km.centroid[1,0],km.centroid[1,1],s=250,c='g',marker='*',edgecolors='k')
# plt.scatter(km.centroid[2,0],km.centroid[2,1],s=250,c='b',marker='*',edgecolors='k')
#


fcm = FCM(init='k-means++',random_state=10,fuzzy_coef_m=3,tol=1e-10)
print('done')
labels = fcm.fit_predict(X)
plt.scatter(X[labels[:,0]>0.4,0],X[labels[:,0]>0.4,1],c='r')
plt.scatter(X[labels[:,1]>0.4,0],X[labels[:,1]>0.4,1],c='g')
plt.scatter(X[labels[:,2]>0.4,0],X[labels[:,2]>0.4,1],c='b')
plt.show()
