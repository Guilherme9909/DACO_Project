# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 02:39:42 2021

@author: Duarte Lopes
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.anova import anova_lm

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
from sklearn.preprocessing import StandardScaler
#%% Processamento da data antes de elaboracao de modelos

##SEED
n_seed=42
##Carregamento e tratamento dos dados referentes Vinhos Brancos do nosso dataset
data=pd.read_excel (r'C:\Users\HP\Desktop\Bio_20_21_4Ano\DACO\Projeto_WINES\DACO_Wine_DataSet\winequality-white.xlsx')
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
ros = RandomOverSampler(random_state=42)
balance_trainingFeatures, balanceLabels = ros.fit_resample(features_training, labels_training)
balance_trainingLabels=pd.Series(balanceLabels)
print(balance_trainingLabels.value_counts())
balance_trainingLabels=balance_trainingLabels.values.reshape(-1,1)

#Pre Processamento da Data
scaler=StandardScaler()
scaler.fit(balance_trainingFeatures)
scaler.fit(features_testing)
x_train_normalized=scaler.transform(balance_trainingFeatures)
x_train_normalized=scaler.transform(features_testing)

#Kfolds
kfold=KFold(n_splits=5, random_state=n_seed, shuffle=True)
for train_idx, vali_idx in kfold.split(x_train_normalized):
         print(train_idx, vali_idx)
 

## typipe = Pipeline([ ('tfidf', TfidfVectorizer()), ('ros', RandomOverSampler()), ('oversampler', SMOTE()), ('clf', LinearSVC()), ])

