## Okay, so this is the Churn problem for banks
## i.e. to predict if a customer leaves the bank or not

# Keras is soooo easy!!!!
# model could be improved by increasing the number of hidden layers and units per layer

## code is split into 3 parts, preprocessing, processing, post processing

layer1 = 6
layer2 = 6
numepochs = 30
batchsize = 10
kfolds = 10
## Let's begin bitch

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
###  PREPROCESSING ###

data = pd.read_csv('Churn_Modelling.csv') # read the file
X = data.values[:,3:13] # remove useless features
Y = data.values[:,13] # labels

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1]) # encode the country
X[:,2] = le.fit_transform(X[:,2]) # encode the sex
Y[:] = le.fit_transform(Y[:]) # encode the labels

enc = preprocessing.OneHotEncoder(categorical_features = [1]) # one hot encoding for the country
X = enc.fit_transform(X).toarray()
X = X[:,1:] # remove 1st column to avoid dummy variable trap
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = 0.2, random_state = 79) # split data into test and train

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train) # scale the data
X_test = scaler.fit_transform(X_test)


### PROCESSING ###


def build_Classifier(optimizer):
    
    model = Sequential()    
    model.add(Dense(units = layer1, input_dim = 11, init = 'uniform', activation = 'relu'))
    model.add(Dropout(rate = 0.1, seed = 73))
    model.add(Dense(units = layer2, init = 'uniform', activation = 'relu'))
    model.add(Dropout(rate = 0.1, seed = 23))
    model.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))    
    #model = Sequential([Dense(units = layer1, input_dim = 11, init = 'uniform'),Activation('relu'),Dense(units = layer2, init = 'uniform'),Activation('relu'),Dense(units = 1, init = 'uniform'),Activation('sigmoid')]) # create the model
    model.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics=['accuracy']) # compile the model
    return model

    
# K FOLD 
'''    
model = KerasClassifier(build_fn = build_Classifier, batch_size = batchsize, nb_epoch = numepochs)
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = kfolds,n_jobs = -1) # kfold regularization
mean = accuracies.mean()
variance = accuracies.std()
'''

## TUNING THE ANN

model = KerasClassifier(build_fn = build_Classifier)
parameters = {'batch_size':[25,32], 'nb_epoch':[60,100], 'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = kfolds)
grid_search = grid_search.fit(X = X_train, y = Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#model.fit(X_train,Y_train,epochs = numepochs, batch_size = batchsize) # fit the model

#predictions = model.predict(X_test) # predict the test data

#predictions = (predictions>=0.5) # probability to boolean


### POSTPROCESSING ###

#Y_test = (Y_test == 1)
#accu = 0

#
#for i in range(np.size(predictions)):
#    if Y_train[i] == predictions[i]:
#        accu+=1
#accu = float(accu)/ np.size(predictions) # Accuracy
#
#print 'Accuracy is: ',accu 
#
