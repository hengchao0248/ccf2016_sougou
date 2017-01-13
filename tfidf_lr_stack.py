'''tfidf-lr stack for education/age/gender'''

import pandas as pd
import numpy as np
import jieba
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from datetime import datetime
import cfg

#-----------------------myfunc-----------------------
def myAcc(y_true,y_pred):
    y_pred = np.argmax(y_pred,axis=1)
    return np.mean(y_true == y_pred)
#-----------------------load data--------------------

df_all = pd.read_csv(cfg.data_path + 'all_v2.csv',encoding='utf8',nrows=200000)
ys = {}
for label in ['Education','age','gender']:
    ys[label] = np.array(df_all[label])

class Tokenizer():
    def __init__(self):
        self.n = 0
    def __call__(self,line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1,2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i+gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('='*20)
            print(tokens)
        self.n += 1
        if self.n%10000==0:
            print(self.n)
        return tokens

tfv = TfidfVectorizer(tokenizer=Tokenizer(),min_df=3,max_df=0.95,sublinear_tf=True)
X_sp = tfv.fit_transform(df_all['query'])
pickle.dump(X_sp,open(cfg.data_path + 'tfidf_10W.feat','wb'))

df_stack = pd.DataFrame(index=range(len(df_all)))

#-----------------------stack for education/age/gender------------------
for lb in ['Education','age','gender']:
    print(lb)
    TR = 100000
    num_class = len(pd.value_counts(ys[lb]))
    n = 5

    X = X_sp[:TR]
    y = ys[lb][:TR]
    X_te = X_sp[TR:]
    y_te = ys[lb][TR:]

    stack = np.zeros((X.shape[0],num_class))
    stack_te = np.zeros((X_te.shape[0],num_class))

    for i,(tr,va) in enumerate(KFold(len(y),n_folds=n)):
        print('%s stack:%d/%d'%(str(datetime.now()),i+1,n))
        clf = LogisticRegression(C=3)
        clf.fit(X[tr],y[tr])
        y_pred_va = clf.predict_proba(X[va])
        y_pred_te = clf.predict_proba(X_te)
        print('va acc:',myAcc(y[va],y_pred_va))
        print('te acc:',myAcc(y_te,y_pred_te))
        stack[va] += y_pred_va
        stack_te += y_pred_te
    stack_te /= n
    stack_all = np.vstack([stack,stack_te])
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_{}_{}'.format(lb,i)] = stack_all[:,i]

df_stack.to_csv(cfg.data_path + 'tfidf_stack_20W.csv',index=None,encoding='utf8')
print(datetime.now(),'save tfidf stack done!')