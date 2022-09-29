# Import libraries
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

#split data into target and Features
X1=well_df.iloc[:, 1:-1].values
y1=well_df.iloc[:, -1].values.reshape(-1,1)
#split data into train and test set
X_train, X_test, y_train, y_test= train_test_split(X1, y1, test_size=0.2, random_state=42)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:',  X_test.shape,  y_test.shape)

# Standardize data
scaleX=StandardScaler()
X_train1=scaleX.fit_transform(X_train)
X_test1=scaleX.transform(X_test)
scaleY=StandardScaler()
y_train1=scaleY.fit_transform(y_train)
y_test1=scaleY.transform(y_test)

# define function that returns a data frame of actual ROP values and the model's predicted values
def get_preds(y_test, y_preds):
    y_test=pd.DataFrame(y_test)
    y_test.rename(columns={0:'Actual'}, inplace=True)
    y_preds=pd.DataFrame(y_preds)
    y_preds.rename(columns={0:'Predicted'}, inplace=True)
    predictions=pd.concat([y_test, y_preds], axis=1)
    return predictions

# import deep learning libraies
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense , Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# Develop model
model3 = Sequential(
    [ 
        Dense(256,  activation  = 'relu'),
        Dense(128, activation  = 'relu'),
        Dense(32, activation  = 'relu'),
        Dense(16, activation  = 'relu'),
        Dense(1,  activation  = 'linear')
    ])
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
callback3 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history3=model3.fit(X_train1, y_train1, batch_size=32, epochs=300, validation_data=(X_train1, y_train1), callbacks=[callback3])

plot_model(model3, "Model 3.png", show_shapes = True)

# Make predictions using model
model3_preds=model3.predict(X_test1)
model3_preds_train=model3.predict(X_train1)

model3_mse=mse(y_test1, model3_preds).numpy()
model3_mse

plt.figure(figsize=(20,6))
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

# return back to unstandardize data
y_test_new= scaleY.inverse_transform(y_test1)
model3_inverse = scaleY.inverse_transform(model3_preds)
model3_predictions=get_preds(y_test_new, model3_inverse)
model3_predictions.head(9)

# Model's Evaluation
plt.figure(figsize=(20,6))
plt.plot(model3_predictions['Actual'][:100])
plt.plot(model3_predictions['Predicted'][:100])
plt.ylabel('ROP')
plt.legend(['Actual', 'Predicted'], loc='best')
plt.title('ANN Model 3')
plt.show()
print('root mean squared error is {}'.format(np.sqrt(mean_squared_error(y_test1,model3_preds))))
print('Mean absolute error is {}'.format(mean_absolute_error(y_test1,model3_preds)))
print('R2 score is {}'.format(r2_score(y_test1,model3_preds)))

plt.figure(figsize=(20,6))
plt.plot(model3_preds_train[:100])
plt.plot(y_train1[:100])
plt.ylabel('ROP')
plt.legend(['Actual', 'Predicted'], loc='best')
plt.title('ANN Model 2 Train')
plt.show()

print('Train root mean squared error is {}'.format(np.sqrt(mean_squared_error(y_train1,model2_preds_train))))
print('Train Mean absolute error is {}'.format(mean_absolute_error(y_train1,model2_preds_train)))
print('Train R2 score is {}'.format(r2_score(y_train1,model2_preds_train)))