# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:19:40 2021

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#Oversampling
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 

#Metricas
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification
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

#Pre Processamento da Data
scaler=StandardScaler()
scaler.fit(features_training)
scaler.fit(features_testing)
x_train_normalized=scaler.transform(features_training)
x_test_normalized=scaler.transform(features_testing)

#Every Model Otimized
# LogReg
# Parameters
s='saga'
c=0.0011
p='none'
LogReg = LogisticRegression(random_state=n_seed,multi_class='multinomial',dual=False, solver=s, C=c, penalty=p)
y_predLog = LogReg.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
LogRegReport=classification_report(labels_testing, y_predLog, labels=[3, 4, 5, 6, 7, 8, 9])
LogRegConfusion=confusion_matrix(labels_testing, y_predLog, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for LogReg : Solver = " + str(s) + " and C = " + str(c) + "and Penalty="+ str(p))
print("Classification Report")
print(LogRegReport)
print("Confusion Matrix")
print(LogRegConfusion)

#SVM
#Parameters
ke='rbf'
c=100
g=0.92
modelSVC = SVC(random_state=n_seed, kernel=ke, C=c, gamma=g)
y_predSVC = modelSVC.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
SVMReport=classification_report(labels_testing, y_predSVC, labels=[3, 4, 5, 6, 7, 8, 9])
SVMConfusion=confusion_matrix(labels_testing, y_predSVC, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for SVM : Kernel = " + str(ke) + " and C = " + str(c) + "and Gamma="+ str(g))
print("Classification Report")
print(SVMReport)
print("Confusion Matrix")
print(SVMConfusion)

#Linear SVM
#Parameters
l='squared_hinge'
c=0.98
SVC_linear = LinearSVC(random_state=n_seed, loss=l, C=c)
y_predSVC_linear = SVC_linear.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
SVM_linearReport=classification_report(labels_testing, y_predSVC_linear, labels=[3, 4, 5, 6, 7, 8, 9])
SVM_linearConfusion=confusion_matrix(labels_testing, y_predSVC_linear, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for LinearSVM : Loss = " + str(l) + " and C = " + str(c))
print("Classification Report")
print(SVM_linearReport)
print("Confusion Matrix")
print(SVM_linearConfusion)

#RandomForest
#Parameters
m='auto'
n=300
RandomForest = RandomForestClassifier(random_state=n_seed, max_features=m, n_estimators=n)
y_predRandomForest = RandomForest.fit(x_train_normalized,labels_training.ravel()).predict(x_test_normalized)
RandomForestReport=classification_report(labels_testing, y_predRandomForest, labels=[3, 4, 5, 6, 7, 8, 9])
RandomForestConfusion=confusion_matrix(labels_testing, y_predRandomForest, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for Random Forest : n_estimators = " + str(n) + " and MaxFeatures = " + str(m))
print("Classification Report")
print(RandomForestReport)
print("Confusion Matrix")
print(RandomForestConfusion)