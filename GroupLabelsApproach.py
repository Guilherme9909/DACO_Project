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


#Kfolds
#75-25 dentros dos 75 de Treino
#Kfolds
n_slips=5
kfold=KFold(n_splits=n_slips, random_state=n_seed, shuffle=True)        
acc=[]
d={}
model_names=['LogisticRegression','LinearSVM','SVM','RandomForestClassifier'] 

bestLogReg_params=[]
bestSVM_params=[]
bestSVC_linear_params=[]
bestRandomForest_params=[]
for train_idx, vali_idx in kfold.split(x_train_normalized):
        
        #Testar os diferentes modelos sem otimizar e visualizar qual o melhor
        
        #Otimizaar modelo 1 a 1 
        
        #LOGISTIC REGRESSION 
        solver=['newton-cg', 'lbfgs','sag', 'saga']
        C=[0.001, 0.01, 0.1, 1, 10, 1000]
        penalty=['none','l2']

        for s in solver:
            for c in C:
                    for p in penalty:
                        LogReg = LogisticRegression(random_state=n_seed,multi_class='multinomial',dual=False, solver=s, C=c, penalty=p)
                        y_predLog = LogReg.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                        accuracyLogReg=accuracy_score(labels_training[vali_idx],y_predLog)
                        bestLogReg_params.append([float(accuracyLogReg), s, c, p])
                        
        
        #SVM 
        kernel=['linear','rbf','sigmoid']
        C=[0.1,0.90,1.0,5,10,100]
        gamma=[0.1,0.90,0.92,0.96,0.98,1.0,1.5,10]
        k_=0
        i_=0
        for ke in kernel:
            for c in C:
                    for g in gamma:
                        modelSVC = SVC(random_state=n_seed, kernel=ke, C=c, gamma=g)
                        y_predSVC = modelSVC.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                        accuracySVC=accuracy_score(labels_training[vali_idx],y_predSVC)
                        bestSVM_params.append([float(accuracySVC), ke, c, g])
        
        
        #SVM Linear
        loss=['hinge', 'squared_hinge']
        C=[0.1,0.5,0.90,0.95,1.0,1.5,10,50,100]

        for l in loss:
            for c in C:
                    SVC_linear = LinearSVC(random_state=n_seed, loss=l, C=c)
                    y_predSVC_linear = SVC_linear.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                    accuracySVC_linear=accuracy_score(labels_training[vali_idx],y_predSVC_linear)
                    bestSVC_linear_params.append([float(accuracySVC_linear), l, c])
                    
        
        #RandomForestClassifier
        n_estimators =[100,200,300,500,700,1000]
        max_features=['auto','log2']

        for n in n_estimators:
            for m in max_features:
                    RandomForest = RandomForestClassifier(random_state=n_seed, max_features=m, n_estimators=n)
                    y_predRandomForest = RandomForest.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                    accuracyRandomForest=accuracy_score(labels_training[vali_idx],y_predRandomForest)
                    bestRandomForest_params.append([float(accuracyRandomForest), n, m])
                    

        print("Fold DONE")
        


#%%
bestLogReg_params = np.array( bestLogReg_params)
bestSVM_params = np.array(bestSVM_params)
bestSVC_linear_params = np.array(bestSVC_linear_params)
bestRandomForest_params = np.array(bestRandomForest_params)

LogReg_accuracy_avg=[]
for i in range(48):
    average=(float(bestLogReg_params[i,0])+float(bestLogReg_params[i+48,0])+float(bestLogReg_params[i+48*2,0])+float(bestLogReg_params[i+48*3,0])+float(bestLogReg_params[i+48*4,0]))/5
    LogReg_accuracy_avg.append(average)
AccLogReg=LogReg_accuracy_avg [np.argmax(LogReg_accuracy_avg)]

bestIndex=np.argmax(LogReg_accuracy_avg)

best_S =bestLogReg_params [bestIndex,1]
best_C =bestLogReg_params [bestIndex,2]
best_P =bestLogReg_params [bestIndex,3]
 
print("Best model for LogReg : Solver = " + str(best_S) + " and C = " + str(best_C) + "and Penalty="+ str(best_P) + " with accuracy (%) = " + str(LogReg_accuracy_avg [bestIndex]))

LinearSVM_accuracy_avg=[]
for i in range(18):
    average=(float(bestSVC_linear_params[i,0])+float(bestSVC_linear_params[i+18,0])+float(bestSVC_linear_params[i+18*2,0])+float(bestSVC_linear_params[i+18*3,0])+float(bestSVC_linear_params[i+18*4,0]))/5
    LinearSVM_accuracy_avg.append(average)

AccLinearSVM=LinearSVM_accuracy_avg [np.argmax(LinearSVM_accuracy_avg)]

bestIndex=np.argmax(LinearSVM_accuracy_avg) 

best_L =bestSVC_linear_params [bestIndex,1]
best_C =bestSVC_linear_params [bestIndex,2]
    
print("Best model for LinearSVM : Loss = " + str(best_L) + " and C = " + str(best_C) + " with accuracy (%) = " + str(LinearSVM_accuracy_avg [bestIndex]))

SVM_accuracy_avg=[]
for i in range(144):
    average=(float(bestSVM_params[i,0])+float(bestSVM_params[i+144,0])+float(bestSVM_params[i+144*2,0])+float(bestSVM_params[i+144*3,0])+float(bestSVM_params[i+144*4,0]))/5
    SVM_accuracy_avg.append(average)

AccSVM=SVM_accuracy_avg [np.argmax(SVM_accuracy_avg)]

bestIndex=np.argmax(SVM_accuracy_avg)

best_K =bestSVM_params [bestIndex,1]
best_C =bestSVM_params [bestIndex,2]
best_G =bestSVM_params [bestIndex,3]
   
print("Best model for SVM : Kernel = " + str(best_K) + " and C = " + str(best_C) + "and Gamma="+ str(best_G) + " with accuracy (%) = " + str(SVM_accuracy_avg [bestIndex]))

RandomForest_accuracy_avg=[]
for i in range(12):
    average=(float(bestRandomForest_params[i,0])+float(bestRandomForest_params[i+12,0])+float(bestRandomForest_params[i+12*2,0])+float(bestRandomForest_params[i+12*3,0])+float(bestRandomForest_params[i+12*4,0]))/5
    RandomForest_accuracy_avg.append(average)
AccRandomForest=RandomForest_accuracy_avg [np.argmax(RandomForest_accuracy_avg)]

bestIndex=np.argmax(RandomForest_accuracy_avg) 

best_N =bestRandomForest_params [bestIndex,1]
best_M =bestRandomForest_params [bestIndex,2]      
print("Best model for Random Forest : n_estimators = " + str(best_N) + " and MaxFeatures = " + str(best_M) +  " with accuracy (%) = " + str(RandomForest_accuracy_avg   [bestIndex]))
      
bestAcc=[]
bestAcc.append(AccLogReg)
bestAcc.append(AccLinearSVM)
bestAcc.append(AccSVM)
bestAcc.append(AccRandomForest)

d={"Model Option":model_names,'Accuracy':bestAcc}
acc_frame=pd.DataFrame(d)
print(acc_frame)
sns.barplot(y='Model Option',x='Accuracy',data=acc_frame)
sns.factorplot(x='Model Option',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)