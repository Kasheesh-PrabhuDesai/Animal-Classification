# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:32:54 2020

@author: kashe
"""


from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,f_classif,RFE
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier




names=['animal-name','hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']

df = pd.read_csv('zoo-data.csv',names=names)
corr = df.corr()
sns.heatmap(corr)
profile = ProfileReport(df)

profile.to_file('animal-classification.html')




df.drop(['animal-name'],axis=1,inplace=True)


array = df.values
x = array[:,:-1]
y = array[:,-1]


model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 10)
fit = rfe.fit(x, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=7)

models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results=[]
names=[]

for name,model in models:
    
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy') 
    results.append(cv_result)
    names.append(name)
    
    msg = "%s: %f %f"%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    
print('\t')

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM',
SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)

print('\t')

normpipeline = []
normpipeline.append(('NormalizedLR',Pipeline([('Normalizer',MinMaxScaler()),('LR',LogisticRegression(solver='liblinear'))])))
normpipeline.append(('NormalizedLDA',Pipeline([('Normalizer',MinMaxScaler()),('LDA',LinearDiscriminantAnalysis())])))
normpipeline.append(('NormalizedKNN',Pipeline([('Normalizer',MinMaxScaler()),('KNN',KNeighborsClassifier())])))
normpipeline.append(('NormalizedCART',Pipeline([('Normalizer',MinMaxScaler()),('CART',DecisionTreeClassifier())])))
normpipeline.append(('NormalizedNB',Pipeline([('Normalizer',MinMaxScaler()),('NB',GaussianNB())])))
normpipeline.append(('NormalizedSVC',Pipeline([('Normalizer',MinMaxScaler()),('SVM',SVC(gamma='auto'))])))
results = []
names = []
for name, model in normpipeline:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)

solver=['liblinear','newton-cg','lbfgs']
param_grid = dict(solver=solver)
kfold=KFold(n_splits=10,random_state=7,shuffle=True)
model=LogisticRegression()
scaler = StandardScaler()
rescaledx = scaler.fit_transform(x_train)
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,scoring='accuracy')
grid_result = grid.fit(rescaledx,y_train)
print(grid_result.best_score_,grid_result.best_params_)

predictor = LogisticRegression(solver='liblinear')
scaler = StandardScaler()
rescaledx = scaler.fit_transform(x_train)
rescaledxt = scaler.transform(x_test)
predictor.fit(rescaledx,y_train)
predictions = predictor.predict(rescaledxt)
print(accuracy_score(y_test,predictions)*100)

c_values=[1,2,3,4,5,6]
kernel_values=['linear','poly','sigmoid','rbf']
model=SVC(gamma='auto')
param_grid=dict(C=c_values,kernel=kernel_values)
kfold=KFold(n_splits=10,random_state=7,shuffle=True)
scaler = StandardScaler()
rescaledx = scaler.fit_transform(x_train)
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,scoring='accuracy')
grid_result = grid.fit(rescaledx,y_train)
print(grid_result.best_score_,grid_result.best_params_)

predictor = SVC(C=2,kernel='rbf')
scaler = StandardScaler()
rescaledx = scaler.fit_transform(x_train)
rescaledxt = scaler.transform(x_test)
predictor.fit(rescaledx,y_train)
predictions = predictor.predict(rescaledxt)
print(accuracy_score(y_test,predictions)*100)

predictor = DecisionTreeClassifier()
scaler = StandardScaler()
rescaledx = scaler.fit_transform(x_train)
rescaledxt = scaler.transform(x_test)
predictor.fit(rescaledx,y_train)
predictions = predictor.predict(rescaledxt)
print(accuracy_score(y_test,predictions)*100)



models = []
models.append(('DT',DecisionTreeClassifier()))
models.append(('RT',RandomForestClassifier()))
models.append(('AC',AdaBoostClassifier()))
models.append(('ET',ExtraTreesClassifier()))
models.append(('GT',GradientBoostingClassifier()))

names=[]
result=[]

for name,model in models:
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    names.append(name)
    result.append(cv_result)
    msg = '%s: %f (%f)'%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    
print('\t')
pipeline = []
pipeline.append(('ScaledDT',Pipeline([('Scale',StandardScaler()),('DT',DecisionTreeClassifier())])))
pipeline.append(('ScaledRT',Pipeline([('Scale',StandardScaler()),('RT',RandomForestClassifier())])))
pipeline.append(('ScaledAC',Pipeline([('Scale',StandardScaler()),('AC',AdaBoostClassifier())])))
pipeline.append(('ScaledET',Pipeline([('Scale',StandardScaler()),('ET',ExtraTreesClassifier())])))
pipeline.append(('ScaledGT',Pipeline([('Scale',StandardScaler()),('GT',GradientBoostingClassifier())])))

for name,model in pipeline:
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    names.append(name)
    result.append(cv_result)
    msg = '%s: %f (%f)'%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    
predictor = DecisionTreeClassifier()
predictor.fit(x_train,y_train)
predictions = predictor.predict(x_test)
print(accuracy_score(y_test,predictions)*100)

predictor = ExtraTreesClassifier()
predictor.fit(x_train,y_train)
predictions = predictor.predict(x_test)
print(accuracy_score(y_test,predictions)*100)
