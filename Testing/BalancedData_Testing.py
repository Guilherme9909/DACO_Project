# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:34:12 2021

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
#%% Avaliar a performance final com a data de teste 
##SEED
n_seed=42
##Carregamento e tratamento dos dados referentes Vinhos Brancos do nosso dataset
data=pd.read_excel (r'D:\Universidade\4_Ano_1_Semestre\DACO\Projeto\DACO_Project\DACO_Wine_DataSet\winequality-white.xlsx')
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

#Every Model Otimized
classes_names = ['3', '4', '5','6','7','8','9']
# LogReg
# Parameters
s='lbfgs'
c=1000
p='l2'
LogReg = LogisticRegression(random_state=n_seed,multi_class='multinomial',dual=False, solver=s, C=c, penalty=p)
y_predLog = LogReg.fit(x_train_normalized,balance_trainingLabels.ravel()).predict(x_test_normalized)
LogRegReport=classification_report(labels_testing, y_predLog, target_names=classes_names )
LogRegConfusion=confusion_matrix(labels_testing, y_predLog, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for LogReg : Solver = " + str(s) + " and C = " + str(c) + "and Penalty="+ str(p))
print("Acuracy="+str(accuracy_score(labels_testing, y_predLog )))
print("Classification Report")
print(LogRegReport)
print("Confusion Matrix")
print(LogRegConfusion)

#SVM
#Parameters
ke='rbf'
c=10
g=1.5
modelSVC = SVC(random_state=n_seed, kernel=ke, C=c, gamma=g)
y_predSVC = modelSVC.fit(x_train_normalized,balance_trainingLabels.ravel()).predict(x_test_normalized)
SVMReport=classification_report(labels_testing, y_predSVC, labels=[3, 4, 5, 6, 7, 8, 9] )
SVMConfusion=confusion_matrix(labels_testing, y_predSVC, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for SVM : Kernel = " + str(ke) + " and C = " + str(c) + "and Gamma="+ str(g))
print("Acuracy="+str(accuracy_score(labels_testing, y_predSVC)))
print("Classification Report")
print(SVMReport)
print("Confusion Matrix")
print(SVMConfusion)

#Linear SVM
#Parameters
l='squared_hinge'
c=1.5
SVC_linear = LinearSVC(random_state=n_seed, loss=l, C=c)
y_predSVC_linear = SVC_linear.fit(x_train_normalized,balance_trainingLabels.ravel()).predict(x_test_normalized)
SVM_linearReport=classification_report(labels_testing, y_predSVC_linear, target_names=classes_names )
SVM_linearConfusion=confusion_matrix(labels_testing, y_predSVC_linear, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for LinearSVM : Loss = " + str(l) + " and C = " + str(c))
print("Acuracy="+str(accuracy_score(labels_testing, y_predSVC_linear)))
print("Classification Report")
print(SVM_linearReport)
print("Confusion Matrix")
print(SVM_linearConfusion)

#RandomForest
#Parameters
m='sqrt'
n=500
RandomForest = RandomForestClassifier(random_state=n_seed, max_features=m, n_estimators=n)
y_predRandomForest = RandomForest.fit(x_train_normalized,balance_trainingLabels.ravel()).predict(x_test_normalized)
RandomForestReport=classification_report(labels_testing, y_predRandomForest, target_names=classes_names)
RandomForestConfusion=confusion_matrix(labels_testing, y_predRandomForest, labels=[3, 4, 5, 6, 7, 8, 9])
print("Best model for Random Forest : n_estimators = " + str(n) + " and MaxFeatures = " + str(m))
print("Acuracy="+str(accuracy_score(labels_testing, y_predRandomForest)))
print("Classification Report")
print(RandomForestReport)
print("Confusion Matrix")
print(RandomForestConfusion)
#%%
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

labels=pd.DataFrame(labels)
# Las 25% is only for testing porposes 
features_testing=features.iloc[lastElem:num_rows,:]
labels_testing=labels[lastElem:num_rows,:]
labels_testing=pd.DataFrame(labels_testing)
print(labels.value_counts())
print(features_testing.value_counts())
