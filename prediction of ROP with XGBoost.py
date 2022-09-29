import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

# Read Data into Data frame
well_df=pd.read_csv(r"C:\Users\User\Desktop\Machine Learning Project\ROP Prediction\welldata_NonOutlier.csv")

# Split data into features and Target
X=well_df.drop(["ROP(1 m)"], axis=1)
y=well_df.iloc[:, -1].values.reshape(-1,1)
X1=well_df.drop(["ROP(1 m)"], axis=1)
y1=well_df.iloc[:, -1].values.reshape(-1,1)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1,test_size=0.25, random_state=42)

# Standardise data
scaleX=StandardScaler().fit(X_train1)
scaleY=StandardScaler().fit(y_train1)
X_train_stand=scaleX.transform(X_train1)
X_test_stand=scaleX.transform(X_test1)
y_train_stand=scaleY.transform(y_train1)
y_test_stand=scaleY.transform(y_test1)

# Define a function that returns a data frame of actual values and model's predicted values
def get_preds(y_test, y_preds):
    y_test=pd.DataFrame(y_test)
    y_test.rename(columns={0:'Actual'}, inplace=True)
    y_preds=pd.DataFrame(y_preds)
    y_preds.rename(columns={0:'Predicted'}, inplace=True)
    predictions=pd.concat([y_test, y_preds], axis=1)
    return predictions

#XGBoost model library
from xgboost import XGBRegressor
#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train1, y_train1):
    param_tuning = {
        'learning_rate': [0.01,0.05,0.1,0.15,0.2],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 10, 100],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : range(100,1100,100),
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train1,y_train1)
    
    return gsearch.best_params_
hyperParameterTuning(X_train1, y_train1)
import warnings
warnings.filterwarnings(action="ignore")
xgb_model = XGBRegressor(
        objective = 'reg:squarederror',
        colsample_bytree = 0.5,
        learning_rate = 0.02,
        max_depth = 10,
        min_child_weight = 1,
        n_estimators = 1000,
        subsample = 0.7)

%time xgb_model.fit(X_train1, y_train1, early_stopping_rounds=5, eval_set=[(X_train1, y_train1)], verbose=False)

y_pred_xgb = xgb_model.predict(X_test1)

mse_xgb = mean_absolute_error(y_test1, y_pred_xgb)

print("MAE: ", mse_xgb)

y_pred_xgb_train = xgb_model.predict(X_train1)
y_pred_xgb = xgb_model.predict(X_test1)

xgb_prediction=get_preds(y_test1, y_pred_xgb)
xgb_prediction.head(10)

plt.figure(figsize=(20,6))
plt.plot(xgb_prediction['Actual'][:100])
plt.plot(xgb_prediction['Predicted'][:100])
plt.ylabel('ROP')
plt.legend(['Actual', 'Predicted'], loc='best')
plt.title('Random Forest')
plt.show()

#EVALUATION OF MODEL
print('root mean squared error is {}'.format(np.sqrt(mean_squared_error(y_test1,y_pred_xgb))))
print('Mean absolute error is {}'.format(mean_absolute_error(y_test1,y_pred_xgb)))
print('R2 score is {}'.format(r2_score(y_test1,y_pred_xgb)))
print('TRAIN root mean squared error is {}'.format(np.sqrt(mean_squared_error(y_train1,y_pred_xgb_train))))
print('TRAIN Mean absolute error is {}'.format(mean_absolute_error(y_train1,y_pred_xgb_train)))
print('TRAIN R2 score is {}'.format(r2_score(y_train1,y_pred_xgb_train)))

xgb_prediction.corr()
plt.figure(figsize=(20,8))
plt.title('Mean absolute error = {}   and   R2 score = {}'.format(mean_absolute_error(y_test1,y_pred_xgb),r2_score(y_test1,y_pred_xgb)))
plt.plot(np.arange(0,300,1),np.arange(0,300,1),color='green', linestyle='dashed',label='Predicted R.O.P = True R.O.P')
sns.scatterplot(x=xgb_prediction["Actual"],y=xgb_prediction["Predicted"])
plt.xlabel('Actual ROP')
plt.ylabel('Predicted ROP');