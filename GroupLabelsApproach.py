# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 01:51:23 2021

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
#%%#Groupping diffrentes labels using quality
#3,4=Fraco(0)
#5,6=BOM(1)
#7,8,9=MuitoBom(2)

##SEED
n_seed=42
##Carregamento e tratamento dos dados referentes Vinhos Brancos do nosso dataset
data=pd.read_excel (r'C:\Users\HP\Desktop\Bio_20_21_4Ano\DACO\Projeto_WINES\DACO_Project\DACO_Wine_DataSet\winequality-white.xlsx')
data_array=pd.DataFrame(data).to_numpy()
features=data_array[:,0:11]
labels=data_array[:,11].reshape(-1,1)
# VINHOS FRACOS
labels=np.where(labels<5,0,labels);
#VINHOS BONS
labels=np.where(labels>4,1,labels);
#VINHOS BONS
labels=np.where(labels>6,2,labels);

# Separaco da data entre 75/25 Treino/Teste
num_rows, num_cols = features.shape
p=num_rows*3/4
lastElem=int(np.floor(num_rows*3/4));
features_training=features[0:lastElem,:]
labels_training=labels[0:lastElem,:]

# Las 25% is only for testing porposes 
features_testing=features[lastElem:num_rows,:]
labels_testing=labels[lastElem:num_rows,:]  


#Pre Processamento da Data
scaler=StandardScaler()
scaler.fit(features_training)
scaler.fit(features_testing)
x_train_normalized=scaler.transform(features_training)
x_test_normalized=scaler.transform(features_testing)


#Kfolds
#75-25 dentros dos 75 de Treino
n_slips=5
kfold=KFold(n_splits=n_slips, random_state=n_seed, shuffle=True)        
acc=[]
d={}
for train_idx, vali_idx in kfold.split(features_training):
      
    #Testar os diferentes modelos sem otimizar e visualizar qual o melhor 
    models=[LogisticRegression(random_state=n_seed),LinearSVC(random_state=n_seed),SVC(kernel='rbf',random_state=n_seed),RandomForestClassifier(random_state=n_seed)]
    model_names=['LogisticRegression','LinearSVM','rbfSVM','RandomForestClassifier']
    
    for model in range(len(models)):
        clf=models[model]
        clf.fit(x_train_normalized[train_idx],labels_training[train_idx])
        pred=clf.predict(x_train_normalized[vali_idx])
        acc.append(accuracy_score(pred,labels_training[vali_idx]))
    
    
#Average of the accuracy
acc_average=[]
for i in range(4):
    average=(acc[i]+acc[i+4]+acc[i+8]+acc[i+12]+acc[i+16])/n_slips
    acc_average.append(average)
    
d={"Model Option":model_names,'Accuracy':acc_average}
acc_frame=pd.DataFrame(d)
print(acc_frame)
sns.barplot(y='Model Option',x='Accuracy',data=acc_frame)
sns.factorplot(x='Model Option',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)


