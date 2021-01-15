# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:09:42 2021

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
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
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
#VINHOS Averg
labels=np.where(labels==5,1,labels);
labels=np.where(labels==6,1,labels);
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

#Every Model Otimized
# LogReg
# Parameters
s='newton-cg'
c=0.1
p='l2'
LogReg = LogisticRegression(random_state=n_seed,multi_class='multinomial',dual=False, solver=s, C=c, penalty=p)
y_predLog = LogReg.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
LogRegReport=classification_report(labels_testing, y_predLog, labels=[0,1,2])
LogRegConfusion=confusion_matrix(labels_testing, y_predLog, labels=[0,1,2])
print("Best model for LogReg : Solver = " + str(s) + " and C = " + str(c) + "and Penalty="+ str(p))
print("Classification Report")
print(LogRegReport)
print("Confusion Matrix")
print(LogRegConfusion)

#SVM
#Parameters
ke='rbf'
c=5
g=10
modelSVC = SVC(random_state=n_seed, kernel=ke, C=c, gamma=g)
y_predSVC = modelSVC.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
SVMReport=classification_report(labels_testing, y_predSVC, labels=[0,1,2])
SVMConfusion=confusion_matrix(labels_testing, y_predSVC, labels=[0,1,2])
print("Best model for SVM : Kernel = " + str(ke) + " and C = " + str(c) + "and Gamma="+ str(g))
print("Classification Report")
print(SVMReport)
print("Confusion Matrix")
print(SVMConfusion)

#Linear SVM
#Parameters
l='squared_hinge'
c=10
SVC_linear = LinearSVC(random_state=n_seed, loss=l, C=c)
y_predSVC_linear = SVC_linear.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
SVM_linearReport=classification_report(labels_testing, y_predSVC_linear, labels=[0,1,2])
SVM_linearConfusion=confusion_matrix(labels_testing, y_predSVC_linear, labels=[0,1,2])
print("Best model for LinearSVM : Loss = " + str(l) + " and C = " + str(c))
print("Classification Report")
print(SVM_linearReport)
print("Confusion Matrix")
print(SVM_linearConfusion)

#RandomForest
#Parameters
m='auto'
n=100
RandomForest = RandomForestClassifier(random_state=n_seed, max_features=m, n_estimators=n)
y_predRandomForest = RandomForest.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
RandomForestReport=classification_report(labels_testing, y_predRandomForest, labels=[0,1,2])
RandomForestConfusion=confusion_matrix(labels_testing, y_predRandomForest, labels=[0,1,2])
print("Best model for Random Forest : n_estimators = " + str(n) + " and MaxFeatures = " + str(m))
print("Classification Report")
print(RandomForestReport)
print("Confusion Matrix")
print(RandomForestConfusion)