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
from sklearn.neural_network import MLPClassifier


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
        C=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
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
        C=[0.001,0.01,0.1,0.90,0.92,0.96,0.98,1.0,1.2,1.5,10,100]
        gamma=[0.001,0.01,0.1,0.90,0.92,0.96,0.98,1.0,1.2,1.5,10,100]
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
        C=[0.001,0.01,0.1,0.90,0.92,0.96,0.98,1.0,1.2,1.5,10,100]

        for l in loss:
            for c in C:
                    SVC_linear = LinearSVC(random_state=n_seed, loss=l, C=c)
                    y_predSVC_linear = SVC_linear.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                    accuracySVC_linear=accuracy_score(labels_training[vali_idx],y_predSVC_linear)
                    bestSVC_linear_params.append([float(accuracySVC_linear), l, c])
                    
        
        #RandomForestClassifier
        n_estimators =[100,200,300,500,700,1000]
        max_features=['auto','sqrt','log2']

        for n in n_estimators:
            for m in max_features:
                    RandomForest = RandomForestClassifier(random_state=n_seed, max_features=m, n_estimators=n)
                    y_predRandomForest = RandomForest.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                    accuracyRandomForest=accuracy_score(labels_training[vali_idx],y_predRandomForest)
                    bestRandomForest_params.append([float(accuracyRandomForest), n, m])
                    

        print("Fold DONE")
        

#%% Neural Networks
bestNN_params=[]

I=[1, 5, 10, 20, 50, 80, 100, 130, 150]
C=['relu', 'logistic', 'tanh']
S=['lbfgs', 'adam']

for train_idx, vali_idx in kfold.split(x_train_normalized):
        for c in C:
            for s in S:
                for i in I:
                    NN=MLPClassifier(hidden_layer_sizes=i, activation=c, solver=s, random_state=42, max_iter=500).fit(features_training,labels_training.ravel())
                    y_predNN=NN.fit(x_train_normalized[train_idx],labels_training[train_idx].ravel()).predict(x_train_normalized[vali_idx])
                    accuracyNN=accuracy_score(labels_training[vali_idx],y_predNN)
                    bestNN_params.append([float(accuracyNN), c, s, i])

bestNN_params=np.array(bestNN_params)

#%%

LogReg_accuracy_avg=[]
for i in range(56):
    average=(float(bestLogReg_params[i,0])+float(bestLogReg_params[i+56,0])+float(bestLogReg_params[i+56*2,0])+float(bestLogReg_params[i+56*3,0])+float(bestLogReg_params[i+56*4,0]))/5
    LogReg_accuracy_avg.append(average)
AccLogReg=LogReg_accuracy_avg [np.argmax(LogReg_accuracy_avg)]

bestIndex=np.argmax(LogReg_accuracy_avg)

best_S =bestLogReg_params [bestIndex,1]
best_C =bestLogReg_params [bestIndex,2]
best_P =bestLogReg_params [bestIndex,3]
 
print("Best model for LogReg : Solver = " + str(best_S) + " and C = " + str(best_C) + "and Penalty="+ str(best_P) + " with accuracy (%) = " + str(LogReg_accuracy_avg [bestIndex]))

LinearSVM_accuracy_avg=[]
for i in range(24):
    average=(float(bestSVC_linear_params[i,0])+float(bestSVC_linear_params[i+24,0])+float(bestSVC_linear_params[i+24*2,0])+float(bestSVC_linear_params[i+24*3,0])+float(bestSVC_linear_params[i+24*4,0]))/5
    LinearSVM_accuracy_avg.append(average)

AccLinearSVM=LinearSVM_accuracy_avg [np.argmax(LinearSVM_accuracy_avg)]

bestIndex=np.argmax(LinearSVM_accuracy_avg) 

best_L =bestSVC_linear_params [bestIndex,1]
best_C =bestSVC_linear_params [bestIndex,2]
    
print("Best model for LinearSVM : Loss = " + str(best_L) + " and C = " + str(best_C) + " with accuracy (%) = " + str(LinearSVM_accuracy_avg [bestIndex]))

SVM_accuracy_avg=[]
for i in range(432):
    average=(float(bestSVM_params[i,0])+float(bestSVM_params[i+432,0])+float(bestSVM_params[i+432*2,0])+float(bestSVM_params[i+432*3,0])+float(bestSVM_params[i+432*4,0]))/5
    SVM_accuracy_avg.append(average)

AccSVM=SVM_accuracy_avg [np.argmax(SVM_accuracy_avg)]

bestIndex=np.argmax(SVM_accuracy_avg)

best_K =bestSVM_params [bestIndex,1]
best_C =bestSVM_params [bestIndex,2]
best_G =bestSVM_params [bestIndex,3]
   
print("Best model for SVM : Kernel = " + str(best_K) + " and C = " + str(best_C) + "and Gamma="+ str(best_G) + " with accuracy (%) = " + str(SVM_accuracy_avg [bestIndex]))

RandomForest_accuracy_avg=[]
for i in range(18):
    average=(float(bestRandomForest_params[i,0])+float(bestRandomForest_params[i+18,0])+float(bestRandomForest_params[i+18*2,0])+float(bestRandomForest_params[i+18*3,0])+float(bestRandomForest_params[i+18*4,0]))/5
    RandomForest_accuracy_avg.append(average)
AccRandomForest=RandomForest_accuracy_avg [np.argmax(RandomForest_accuracy_avg)]

bestIndex=np.argmax(RandomForest_accuracy_avg) 

best_N =bestRandomForest_params [bestIndex,1]
best_M =bestRandomForest_params [bestIndex,2]      
print("Best model for Random Forest : n_estimators = " + str(best_N) + " and MaxFeatures = " + str(best_M) +  " with accuracy (%) = " + str(RandomForest_accuracy_avg   [bestIndex]))
      
NN_accuracy_avg=[]
for i in range(48):
    average=(float(bestNN_params[i,0])+float(bestNN_params[i+48,0])+float(bestNN_params[i+48*2,0])+float(bestNN_params[i+48*3,0])+float(bestNN_params[i+48*4,0]))/5
    NN_accuracy_avg.append(average)
AccNN=NN_accuracy_avg [np.argmax(NN_accuracy_avg)]

bestIndex=np.argmax(NN_accuracy_avg)

best_C =bestNN_params [bestIndex,1]
best_S =bestNN_params [bestIndex,2]
best_I =bestNN_params [bestIndex,3]

print("Best model for NN : Activation = " + str(best_C) + " and Solver = " + str(best_S) + "and HiddenLayerSizes="+ str(best_I) + " with accuracy (%) = " + str(NN_accuracy_avg [bestIndex]))

bestAcc=[]
bestAcc.append(AccLogReg)
bestAcc.append(AccLinearSVM)
bestAcc.append(AccSVM)
bestAcc.append(AccRandomForest)
bestAcc.append(AccNN)

model_names=['LogisticRegression','LinearSVM','SVM','RandomForestClassifier','NN'] 

d={"Model Option":model_names,'Accuracy':bestAcc}
acc_frame=pd.DataFrame(d)
print(acc_frame)
sns.barplot(y='Model Option',x='Accuracy',data=acc_frame)
sns.factorplot(x='Model Option',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)