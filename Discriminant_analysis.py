import pandas as pd
import numpy as np

import sklearn.preprocessing as skl_pre
import sklearn.discriminant_analysis as skl_da
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_m

import seaborn as sb
from matplotlib import pyplot as plt

data = pd.read_csv('training_data_vt2025.csv').reset_index()

data['workday'] = data['weekday'] & ~data['holiday']
data['daynight'] = 2*((data['hour_of_day'] < 21) & (data['hour_of_day'] > 14))+((data['hour_of_day']<15)&(data['hour_of_day']>7))
data['precipitation'] = 2*(data['precip']<4)+((data['precip']<0.5) & (data['precip']>0.1))

inLabels = [
    #'hour_of_day',
    #'day_of_week', #bra för precision
    #'month',   #bra för precision
    #'holiday',
    #'weekday', 
    'workday',  #bra för accuracy
    #'summertime',
    #temp',
    'dew',  #bra för accuracy
    'humidity', #bra för accuracy
    #'precip',
    #'snow', bara 0
    #'snowdepth',
    #'windspeed',
    #'cloudcover',  #bra för precision
    #'visibility',
    'daynight'
]

outLabels = [
    'increase_stock'
]

train, test = skl_ms.train_test_split(data,test_size=0.5,random_state=45)

reg = skl_da.QuadraticDiscriminantAnalysis()
param = {'reg_param': np.arange(0,1,0.01)}

reg = skl_ms.GridSearchCV(estimator=reg,param_grid=param, scoring='accuracy')

reg.fit(train[inLabels],np.ravel(train[outLabels]))

true = test[outLabels]
pred = reg.predict(test[inLabels])

print(reg.best_params_)
print(f'accuracy score: {reg.score(test[inLabels],true)}')
print(f'f1 score: {skl_m.f1_score(true,pred,pos_label="high_bike_demand")}')
print(f'recall score: {skl_m.recall_score(true,pred,pos_label="high_bike_demand")}')
print(f'precision score: {skl_m.precision_score(true,pred,pos_label="high_bike_demand")}')

df_cm = pd.DataFrame(skl_m.confusion_matrix(true,pred), index=['high bike demand','low bike demand'], columns=['high bike demand','low bike demand'])
sb.heatmap(df_cm,annot=True,fmt='.4g',cmap='crest').set(xlabel='predicted',ylabel='true')
plt.show()