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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#Oversampling
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 

#Metricas
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import accuracy_score,confusion_matrix
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

#Kfolds
n_slips=5
kfold=KFold(n_splits=n_slips, random_state=n_seed, shuffle=True)        
acc=[]
d={}
model_names=['LogisticRegression','LinearSVM','rbfSVM','RandomForestClassifier']
for train_idx, vali_idx in kfold.split(x_train_normalized):
        
        #Testar os diferentes modelos sem otimizar e visualizar qual o melhor
        
        #Otimizaar modelo 1 a 1 
        
        #LOGISTIC REGRESSION 
        LogReg_params={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['none','l2'],'solver':[ 'lbfgs', 'newton-cg', 'sag', 'saga']}
        
        LogReg=GridSearchCV(estimator=LogisticRegression(random_state=n_seed,multi_class='multinomial',dual=False),param_grid=LogReg_params,scoring='accuracy',cv=5)
        LogReg.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel())
        bestLogReg_params=LogReg.best_params_
        bestAccLogReg=LogReg.best_score_
        resultsLogReg=LogReg.cv_results_


print(bestLogReg_params)
print(bestAccLogReg)
# =============================================================================
# d={"Model Option":model_names,'Accuracy':bestAccLogReg}
# acc_frame=pd.DataFrame(d)
# print(acc_frame)
# sns.barplot(y='Model Option',x='Accuracy',data=acc_frame)
# sns.factorplot(x='Model Option',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)
# =============================================================================
