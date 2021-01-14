# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 01:51:23 2021

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
#%%#Groupping diffrentes labels using quality
#3,4=Fraco(0)
#5,6=BOM(1)
#7,8,9=MuitoBom(2)

##SEED
n_seed=42
##Carregamento e tratamento dos dados referentes Vinhos Brancos do nosso dataset
data=pd.read_excel (r'C:\Users\HP\Desktop\Bio_20_21_4Ano\DACO\Projeto_WINES\DACO_Wine_DataSet\winequality-white.xlsx')
data_array=pd.DataFrame(data).to_numpy()
features=data_array[:,0:11]
labels=data_array[:,11].reshape(-1,1)
# VINHOS FRACOS
labels=np.where(labels<5,0,labels);
#VINHOS BONS
labels=np.where(labels>4,1,labels);
#VINHOS MUITO BONS
labels=np.where(labels>6,2,labels);

# Separaco da data entre 75/25 Treino/Teste
num_rows, num_cols = features.shape
p=num_rows*3/4
lastElem=int(np.floor(num_rows*3/4));
features_training=features[0:lastElem,:]
labels_training=labels[0:lastElem,:]


#Kfolds
#75-25 dentros dos 75 de Treino
kfold=KFold(n_splits=5, random_state=n_seed, shuffle=True)
for train_idx, vali_idx in kfold.split(features_training):
        print(train_idx, vali_idx)


# Las 25% is only for testing porposes 
features_testing=features[lastElem:num_rows,:]
labels_testing=labels[lastElem:num_rows,:]  
# Name of the feautures 
features_names=[]
for col in data.columns: 
    features_names.append(str(col))