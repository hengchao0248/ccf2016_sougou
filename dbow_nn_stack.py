'''dbow-nn stack for education/age/gender'''

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import KFold
from gensim.models import Doc2Vec
from collections import OrderedDict

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import re
import cfg

#-----------------------myfunc-----------------------
def myAcc(y_true,y_pred):
    y_pred = np.argmax(y_pred,axis=1)
    return np.mean(y_true == y_pred)

#-----------------------load dataset----------------------
df_all = pd.read_csv(cfg.data_path + 'all_v2.csv',encoding='utf8',usecols=['Id','Education','age','gender'],nrows=200000)
ys = {}
for label in ['Education','age','gender']:
    ys[label] = np.array(df_all[label])
    
model = Doc2Vec.load(cfg.data_path + 'dbow_d2v.model')
X_sp = np.array([model.docvecs[i] for i in range(200000)])

#----------------------dbowd2v stack for Education/age/gender---------------------------
df_stack = pd.DataFrame(index=range(len(df_all)))
TR = 100000
n = 5

X = X_sp[:TR]
X_te = X_sp[TR:]

feat = 'dbowd2v'
for i,lb in enumerate(['Education','age','gender']):
    num_class = len(pd.value_counts(ys[lb]))
    y = ys[lb][:TR]
    y_te = ys[lb][TR:]
    
    stack = np.zeros((X.shape[0],num_class))
    stack_te = np.zeros((X_te.shape[0],num_class))
    
    for k,(tr,va) in enumerate(KFold(len(y),n_folds=n)):
        print('{} stack:{}/{}'.format(datetime.now(),k+1,n))
        nb_classes = num_class
        X_train = X[tr]
        y_train = y[tr]
        X_test = X_te
        y_test = y_te

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)


        model = Sequential()
        model.add(Dense(300,input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.1))
        model.add(Activation('tanh'))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        history = model.fit(X_train, Y_train,shuffle=True,
                            batch_size=128, nb_epoch=35,
                            verbose=2, validation_data=(X_test, Y_test))
        y_pred_va = model.predict_proba(X[va])
        y_pred_te = model.predict_proba(X_te)
        print('va acc:',myAcc(y[va],y_pred_va))
        print('te acc:',myAcc(y_te,y_pred_te))
        stack[va] += y_pred_va
        stack_te += y_pred_te
    stack_te /= n
    stack_all = np.vstack([stack,stack_te])
    for l in range(stack_all.shape[1]):
        df_stack['{}_{}_{}'.format(feat,lb,l)] = stack_all[:,l]
df_stack.to_csv(cfg.data_path + 'dbowd2v_stack_20W.csv',encoding='utf8',index=None)
print(datetime.now(),'save dbowd2v stack done!')