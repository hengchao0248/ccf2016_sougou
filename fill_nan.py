'''1.concat train data and test data 
   2.use lr to fill null label'''

import pandas as pd
import numpy as np
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import pickle
import cfg

#----------------------load data--------------------------------
df_tr = []
for i,line in enumerate(open(cfg.data_path + 'user_tag_query.10W.TRAIN',encoding='GB18030')):
    segs = line.split('\t')
    row = {}
    row['Id'] = segs[0]
    row['age'] = int(segs[1])
    row['gender'] = int(segs[2])
    row['Education'] = int(segs[3])
    row['query'] = '\t'.join(segs[4:])
    df_tr.append(row)
df_tr = pd.DataFrame(df_tr)

df_te=[]
for i,line in enumerate(open(cfg.data_path + 'user_tag_query.10W.TEST',encoding='GB18030')):
    segs = line.split('\t')
    row = {}
    row['Id'] = segs[0]
    row['query'] = '\t'.join(segs[1:])
    df_te.append(row)
df_te = pd.DataFrame(df_te)

print(df_tr.shape)
print(df_te.shape)

df_all = df_tr
# df_all = pd.concat([df_tr,df_te]).fillna(1)
# df_all.index = range(len(df_all))

for lb in ['Education','age','gender']:
    df_all[lb] = df_all[lb] - 1
    print(df_all.iloc[:100000][lb].value_counts())
    
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
            print(self.n,end=' ')
        return tokens    

tfv = TfidfVectorizer(tokenizer=Tokenizer(),min_df=3,max_df=0.95,sublinear_tf=True)
X_sp = tfv.fit_transform(df_all['query'])
# pickle.dump(X_sp,open(root + 'tfidf_10W.pkl','wb'))
print(len(tfv.vocabulary_))
X_all = X_sp

#-----------------------------fill nan-------------------------------------
'''填充空值'''
for lb,idx in [('Education',0),('age',2),('gender',3)]:
    tr = np.where(df_all[lb]!=-1)[0]
    va = np.where(df_all[lb]==-1)[0]
lb = 'Education'
idx = 0
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
df_all.iloc[va,idx] = LogisticRegression(C=1).fit(X_all[tr],df_all.iloc[tr,idx]).predict(X_all[va])

lb = 'age'
idx = 2
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
df_all.iloc[va,idx] = LogisticRegression(C=2).fit(X_all[tr],df_all.iloc[tr,idx]).predict(X_all[va])

lb = 'gender'
idx = 3
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
df_all.iloc[va,idx] = LogisticRegression(C=2).fit(X_all[tr],df_all.iloc[tr,idx]).predict(X_all[va])

df_all = pd.concat([df_all,df_te]).fillna(0)
df_all.to_csv(cfg.data_path + 'all_v2.csv',index=None)