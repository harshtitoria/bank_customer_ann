import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('bank_data.csv')
x = dataset.iloc[ : , 3:13].values
y = dataset.iloc[: ,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
x[:,2] = labelencoder_x_1.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features =[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,activation = "relu",kernel_initializer = "uniform",input_dim = 11))
    classifier.add(Dense(units = 6,activation = "relu",kernel_initializer = "uniform"))
    classifier.add(Dense(units = 1,activation = "sigmoid",kernel_initializer = "uniform"))
    classifier.compile(optimizer = optimizer,loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'optimizer':["adam","rmsprop"],"batch_size":[25,32],"epochs":[100,500]}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = "accuracy",n_jobs = -1,cv = 10)
grid_search = grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




