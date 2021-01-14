# -*- coding: utf-8 -*-
"""
DACO PROJECT
White Wine Quality Evaluation
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

#Pre Processamento da Data
scaler=StandardScaler()
scaler.fit(features_training)
scaler.fit(features_testing)
x_train_normalized=scaler.transform(features_training)
x_test_normalized=scaler.transform(features_testing)

#Kfolds
kfold=KFold(n_splits=5, random_state=n_seed, shuffle=True)
for train_idx, vali_idx in kfold.split(x_train_normalized):
        print(train_idx, vali_idx)
    
#Testar os diferentes modelos sem otimizar e visualizar qual o melhor 
models=[LinearRegression(random_state=n_seed), LogisticRegression(random_state=n_seed),LinearSVC(random_state=n_seed),SVC(kernel='rbf',random_state=n_seed),RandomForestClassifier(random_state=n_seed)]
model_names=['Linar Regression','LogisticRegression','LinearSVM','rbfSVM''RandomForestClassifier']

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train_normalized,labels_training)
    pred=clf.predict(x_test_normalized)
    acc.append(accuracy_score(pred,labels_testing))

