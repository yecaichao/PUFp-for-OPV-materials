import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
data = pd.read_csv("T_15_fea_descriptor_HOMO_LUMO.csv")
data.columns
feature = data.drop(['PCE_max(%)', 'PCE_ave(%)'],axis=1)
print(feature.columns)
target = data['PCE_max(%)']
feature.shape[1]
#feature.values[0:20]
print(len(data))
X_train, X_test, y_train, y_test = train_test_split(feature.values, target.values,test_size=0.2, random_state=111)
print(len(X_train))
print(len(X_test))
# nan_pos = np.where(np.isnan(X_train))
# print(nan_pos)
# nan_pos2 = np.where(np.isnan(X_test))
# print(nan_pos2)

param_range = [2, 6, 8, 10, 20, 30, 40, 50]
tuned_parameters = [{'max_depth': param_range,'max_features':['auto', 'sqrt', 'log2'], 'n_estimators':range(5,200,5)}]
gs = GridSearchCV(estimator=RandomForestRegressor(random_state=111),
                 param_grid=tuned_parameters,
                 cv=10,
                 n_jobs=-1)
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
plt.ylabel("Randon Forest PEC%")
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
plt.show()



