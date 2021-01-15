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

#Metricas
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, confusion_matrix  # for classification

#%% Processamento da data antes de elaboracao de modelos

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

#Pre Processamento da Data
scaler=StandardScaler()
scaler.fit(features_training)
scaler.fit(features_testing)
x_train_normalized=scaler.transform(features_training)
x_test_normalized=scaler.transform(features_testing)

#Remoção de features menos revelantes
clf=RandomForestClassifier()
clf = clf.fit(features_training, labels_training)
ft_importance=clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
x_train_normalized = model.transform(x_train_normalized)
x_train_normalized.shape

# Name of the feautures 
features_names=[]
for col in data.columns: 
    features_names.append(str(col))
features_names=features_names[0:11]
d={"Features Names":features_names,'Importance':ft_importance}
acc_frame=pd.DataFrame(d)
print(acc_frame)
sns.barplot(y='Features Names',x='Importance',data=acc_frame)

#%%
LogReg = LogisticRegression(random_state=n_seed,multi_class='multinomial',dual=False, solver='saga', C=0.001, penalty='none')
y_predLog = LogReg.fit(x_train_normalized).predict(x_train_normalized)
accuracyLogReg=accuracy_score(y_predLog,labels_training)
confusion_matrix_LogReg=confusion_matrix(labels_training,y_predLog)

