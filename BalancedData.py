# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 02:39:42 2021

@author: Duarte Lopes
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#Modelos
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

#Oversampling
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 

#Metricas
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
#%% Processamento da data antes de elaboracao de modelos

##SEED
n_seed=42
##Carregamento e tratamento dos dados referentes Vinhos Brancos do nosso dataset
data=pd.read_excel (r'C:\Users\HP\Desktop\Bio_20_21_4Ano\DACO\Projeto_WINES\DACO_Project\DACO_Wine_DataSet\winequality-white.xlsx')
# Separacao entre Parametrso e Etiquetas 
features = data.drop(['quality'],axis=1)
labels= data['quality']
labels=labels.values.reshape(-1,1)


# Separaco da data entre 75/25 Treino/Teste
num_rows, num_cols = features.shape
p=num_rows*3/4
lastElem=int(np.floor(num_rows*3/4));
features_training=features.iloc[0:lastElem,:]
labels_training=labels[0:lastElem,:]


# Las 25% is only for testing porposes 
features_testing=features.iloc[lastElem:num_rows,:]
labels_testing=labels[lastElem:num_rows,:] 



## oversampling unicamente na data de treino ja que nao ]e suposto altear a data de teste 
ros = RandomOverSampler(random_state=n_seed)
balance_trainingFeatures, balanceLabels = ros.fit_resample(features_training, labels_training)
balance_trainingLabels=pd.Series(balanceLabels)
print(balance_trainingLabels.value_counts())
balance_trainingLabels=balance_trainingLabels.values.reshape(-1,1)

#Pre Processamento da Data
scaler=StandardScaler()
scaler.fit(balance_trainingFeatures)
scaler.fit(features_testing)
x_train_normalized=scaler.transform(balance_trainingFeatures)
x_test_normalized=scaler.transform(features_testing)

#Kfolds
kfold=KFold(n_splits=5, random_state=n_seed, shuffle=True)
for train_idx, vali_idx in kfold.split(x_train_normalized):
         print(train_idx, vali_idx)

#Testar os diferentes modelos sem otimizar e visualizar qual o melhor 
models=[LogisticRegression(random_state=n_seed),LinearSVC(random_state=n_seed),SVC(kernel='rbf',random_state=n_seed),RandomForestClassifier(random_state=n_seed)]
model_names=['LogisticRegression','LinearSVM','rbfSVM','RandomForestClassifier']
acc=[]
d={}
for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train_normalized,balance_trainingLabels)
    pred=clf.predict(x_test_normalized)
    acc.append(accuracy_score(pred,labels_testing))
d={"Model Option":model_names,'Accuracy':acc}
acc_frame=pd.DataFrame(d)
acc_frame
sns.barplot(y='Model Option',x='Accuracy',data=acc_frame)
sns.factorplot(x='Model Option',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)
 

## typipe = Pipeline([ ('tfidf', TfidfVectorizer()), ('ros', RandomOverSampler()), ('oversampler', SMOTE()), ('clf', LinearSVC()), ])

