import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the Housing Dataset
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',
                 header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = df.values

## Draw Scatter Plot & Histogram
fig = plt.figure(figsize=(10,20))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('Scatter & Histogram')

ax = fig.subplots(13,13)

for row in range(13):
    for col in range(13):
        if row == col:
            # 대각선 라인에 있으면 histogram 을 그리도록
            ax[row,col].hist(dataset[:,row])
        else:
            ax[row,col].scatter(dataset[:,col],dataset[:,row],marker='.')
        # subplot의 x축 y축 값들 표기를 위해서 ylabel(세로축)과 title(가로축)을 사용.
        if col == 0:
            ax[row,col].set_ylabel(df.columns[row])
        if row == 0:
            ax[row,col].set_title(df.columns[col])



## simple linear regression

# Class definition
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)



# Function Definition

def lin_regplot(X, y, model, ax):
    ax.scatter(X, y, c='steelblue', marker='.', edgecolor='white', s=70)
    ax.plot(X, model.predict(X), color='black', lw=2)
    return




# Standardize y value
y = df['MEDV'].values
sc_y = StandardScaler()
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# Value holder for weights
weights = np.zeros([13,1])

# Figure for scatter plot & decision line
fig1 = plt.figure(figsize=(10,20))
fig1.patch.set_facecolor('lightcyan')
fig1.suptitle('Simple Linear Regression')
ax1 = fig1.subplots(7,2)

# Figure for residual plot
fig2 = plt.figure(figsize=(10,20))
fig2.patch.set_facecolor('lightcyan')
fig2.suptitle('Simple Linear Regression-Residuals')
ax2 = fig2.subplots(7,2)

# Draw plots
for row in range(7):
    for col in range(2):
        idx = 2*row + col
        if idx != 13: # 전체 subplot의 수는 14개. 마지막 것은 비워두어야 함.
            # Standardize the X
            X = dataset[:, idx].reshape(-1,1)
            sc_x = StandardScaler()
            X_std = sc_x.fit_transform(X)
            # Do the regression
            lr = LinearRegressionGD()
            lr.fit(X_std, y_std)
            # Save weight(slope)
            weights[idx] = lr.w_[1]
            #weights[idx] = r2_score(y_std,lr.predict(X_std))
            # Plot
            lin_regplot(X_std,y_std,lr,ax1[row,col])
            ax1[row,col].set_title(df.columns[idx])
            # Plot residuals
            ax2[row,col].scatter(lr.predict(X_std),lr.predict(X_std) - y_std, c='steelblue', marker='.')
            ax2[row,col].set_xlabel('Predicted values')
            ax2[row,col].set_ylabel('Residuals')
            ax2[row,col].hlines(y=0, xmin=-5, xmax=5, color='black', lw=2)
            ax2[row, col].set_title(df.columns[idx])

fig1.tight_layout()
fig2.tight_layout()

# Print Correlation
for k in range(13):
    print(df.columns[k] + str(weights[k]))

# Find highly correlated features
sorted_features = np.argsort(np.abs(weights.T))
print("Top 3 most highly correlated Features")
t3feature = np.empty(3,dtype='int32')
for k in range(3):
    idx = sorted_features[0,12-k]
    t3feature[k] = idx
    print(df.columns[idx] + str(weights[idx]))

# create X matrix
multi_X = dataset[:,t3feature]
# Standardize the X
sc_multi_X = StandardScaler()
multi_X_std = sc_multi_X.fit_transform(multi_X)
# Do the regression
lr = LinearRegressionGD()
lr.fit(multi_X_std, y_std)
# Plot
fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(111)
ax.scatter(lr.predict(multi_X_std),lr.predict(multi_X_std) - y_std, c='steelblue', marker='.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.hlines(y=0, xmin=-5, xmax=5, color='black', lw=2)

print("R2 Score : %.3f",r2_score(y_std,lr.predict(multi_X_std)))


#################################
## cubic

cubic = PolynomialFeatures(degree=3)

# Value holder for r2 value
scores = np.zeros([13,1])

# Figure for scatter plot & decision line
fig = plt.figure(figsize=(15,20))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('Polinomial Regression')
ax = fig.subplots(7,2)

# Draw plots
for row in range(7):
    for col in range(2):
        idx = 2*row + col
        if idx != 13: # 전체 subplot의 수는 14개. 마지막 것은 비워두어야 함.
            # Standardize the X
            X = dataset[:, idx].reshape(-1,1)
            # Make Cubic Terms
            X_cubic = cubic.fit_transform(X)
            sc_x = StandardScaler()
            X_std = sc_x.fit_transform(X_cubic)
            # Do the regression
            lr = LinearRegression()
            lr.fit(X_std, y_std)
            # Plot
            # X 데이터가 순차적으로 되어 있지 않기에 plot 함수를 그대로 사용하면 거미줄이 연성된다.
            # 고로, argsort 함수로 X 값을 순차적으로 정렬한 뒤, 이 정렬기준을 그대로 적용해서 y_std 값을 feed 해주면 제대로 그려짐.
            singleX = sc_x.fit_transform(X)
            order = np.argsort(singleX, 0).T[0]
            ax[row,col].scatter(singleX, y_std, c='steelblue', marker='.', edgecolor='white', s=70)
            ax[row,col].plot(singleX[order], lr.predict(X_std)[order], color='black', lw=2)
            ax[row,col].set_title(df.columns[idx])




# Figure for scatter plot & decision line
fig = plt.figure(figsize=(10,20))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('Polynomial(Cubic) Linear Regression')
ax = fig.subplots(7,2)

# Draw plots
for row in range(7):
    for col in range(2):
        idx = 2*row + col
        if idx != 13: # 전체 subplot의 수는 14개. 마지막 것은 비워두어야 함.
            # Standardize the X
            X = dataset[:, idx].reshape(-1,1)
            # Make Cubic Terms
            X_cubic = cubic.fit_transform(X)
            sc_x = StandardScaler()
            X_std = sc_x.fit_transform(X_cubic)
            # Do the regression
            lr = LinearRegression()
            lr.fit(X_std, y_std)
            # Save fitting score
            scores[idx] = r2_score(y_std,lr.predict(X_std))
            # Plot residuals
            ax[row,col].scatter(lr.predict(X_std),lr.predict(X_std) - y_std, c='steelblue', marker='.')
            ax[row,col].set_xlabel('Predicted values')
            ax[row,col].set_ylabel('Residuals')
            ax[row,col].hlines(y=0, xmin=-5, xmax=5, color='black', lw=2)
            ax[row, col].set_title(df.columns[idx])

fig.tight_layout()

# Print Correlation(r2 score)
for k in range(13):
    print(df.columns[k] + str(scores[k]))

# Find highly correlated features
sorted_features = np.argsort(np.abs(scores.T))
print("Top 3 most highly correlated Features")
t3feature = np.empty(3,dtype='int32')
for k in range(3):
    idx = sorted_features[0,12-k]
    t3feature[k] = idx
    print(df.columns[idx] + str(scores[idx]))


# create cubic matrix of each feature
# PolynomialFeatures를 사용해서 총 3개의 선택된 feature에 대해 각각 0차항부터 3차항까지의 array를 만든다.
# 그 후 이 array를 하나로 합쳐서 총 4(0차~3차) x 3(feature 3개) = 12열의 어레이를 만든다.
multi_X_cubic = np.empty([np.size(multi_X,0),4*3])
for i in range(3):
    X = dataset[:,t3feature[i]].reshape(-1,1)
    X_cubic = cubic.fit_transform(X)
    multi_X_cubic[:,4*i:4*(i+1)] = X_cubic

# Standardize the X
sc_multi_X = StandardScaler()
multi_X_std = sc_multi_X.fit_transform(multi_X_cubic)
# Do the regression
lr = LinearRegression()
lr.fit(multi_X_std, y_std)
# Plot
fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(111)
ax.scatter(lr.predict(multi_X_std),lr.predict(multi_X_std) - y_std, c='steelblue', marker='.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.hlines(y=0, xmin=-5, xmax=5, color='black', lw=2)


############################
##### Random Forest
############################

# Value holder for r2 value
scores = np.zeros([13,1])

# Figure for residual plot
fig = plt.figure(figsize=(10,20))
fig.patch.set_facecolor('lightcyan')
fig.suptitle('Random Forest Regression')
ax = fig.subplots(7,2)

# Draw plots
for row in range(7):
    for col in range(2):
        idx = 2*row + col
        if idx != 13: # 전체 subplot의 수는 14개. 마지막 것은 비워두어야 함.
            # Standardize the X
            X = dataset[:, idx].reshape(-1,1)
            sc_x = StandardScaler()
            X_std = sc_x.fit_transform(X)
            # Do the regression
            forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
            forest.fit(X_std, y_std)
            # Save fitting score
            scores[idx] = r2_score(y_std,forest.predict(X_std))
            # Plot residuals
            ax[row,col].scatter(forest.predict(X_std),forest.predict(X_std) - y_std, c='steelblue', marker='.')
            ax[row,col].set_xlabel('Predicted values')
            ax[row,col].set_ylabel('Residuals')
            ax[row,col].hlines(y=0, xmin=-5, xmax=5, color='black', lw=2)
            ax[row, col].set_title(df.columns[idx])

fig.tight_layout()




# Print Correlation(r2 score)
for k in range(13):
    print(df.columns[k] + str(scores[k]))

# Find highly correlated features
sorted_features = np.argsort(np.abs(scores.T))
print("Top 3 most highly correlated Features")
t3feature = np.empty(3,dtype='int32')
for k in range(3):
    idx = sorted_features[0,12-k]
    t3feature[k] = idx
    print(df.columns[idx] + str(scores[idx]))




### Multi

# create X matrix
multi_X = dataset[:,t3feature]
# Standardize the X
sc_multi_X = StandardScaler()
multi_X_std = sc_multi_X.fit_transform(multi_X)
# Do the regression
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(multi_X_std, y_std)
# Plot
fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(111)
ax.scatter(forest.predict(multi_X_std),forest.predict(multi_X_std) - y_std, c='steelblue', marker='.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.hlines(y=0, xmin=-5, xmax=5, color='black', lw=2)

print("R2 Score : %.3f",r2_score(y_std,forest.predict(multi_X_std)))