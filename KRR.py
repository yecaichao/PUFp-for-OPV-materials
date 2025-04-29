import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linspace
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("1.csv")
df = pd.DataFrame(data)
data.columns
feature = data.drop([PCE_max(%)],axis=1)
print(feature.columns)
target = data['PCE_max(%)']
feature.shape[1]
print(len(data))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature.values, target.values,test_size=0.20, random_state=111)
print(len(X_train))
print(len(X_test))

pipe_kkr = make_pipeline(StandardScaler(),
                         KernelRidge())

param_range = [2, 6, 8, 10, 20]

tuned_parameters = [{'kernelridge__kernel': ['rbf', 'laplacian', 'linear'], 'kernelridge__gamma': (1/2**np.arange(5, 18, 1.0))**2,
                      'kernelridge__alpha': 2**np.arange(-40, 0, 1.0)}]

gs = GridSearchCV(estimator=pipe_kkr,
                  param_grid=tuned_parameters,
                  cv=10,
                  n_jobs=30)

gs = gs.fit(X_train, y_train)
print(gs.best_params_)
y_pred = gs.predict(X_train)
y_pred_test = gs.predict(X_test)
plt.plot(y_train, y_pred, "o", label="Train")
plt.plot(y_test, y_pred_test, "o", label="Test")
X = np.linspace(-2,22,1000)
y = np.linspace(-2, 22, 1000)
plt.plot(X, y, "_", linewidth=2)
plt.xlim(-2, 22)
plt.ylim(-2, 22)
plt.axhline(y=0, lw=1, ls="--", color="r")
plt.axvline(x=0, lw=1, ls="--", color="r")
plt.legend()
plt.xlabel("Experiment PCE%")
plt.ylabel("KRR predict PCE%")
MSE = mean_squared_error(y_train,y_pred)
RMSE = np.sqrt(mean_squared_error(y_train,y_pred))
MAE = mean_absolute_error(y_train,y_pred)
CC = np.corrcoef(y_train,y_pred)[0, 1]
R2 = r2_score(y_train,y_pred)
print("MSE: %s \t RMSE: %s\t MAE: %s\t R2: %s\t CC: %s " % (MSE,RMSE, MAE,R2, CC))
MSE1 = mean_squared_error(y_test,y_pred_test)
RMSE1 = np.sqrt(mean_squared_error(y_test,y_pred_test))
MAE1 = mean_absolute_error(y_test,y_pred_test)
CC = np.corrcoef(y_test,y_pred_test)[0, 1]
R2 = r2_score(y_test,y_pred_test)
print("MSE: %s \t RMSE: %s\t MAE: %s\t R2:%s\t CC: %s" % (MSE1,RMSE1, MAE1,R2, CC))
