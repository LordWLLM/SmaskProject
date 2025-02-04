import pandas as pd
import numpy as np

import sklearn.preprocessing as skl_pre
#import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
#import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_m

import seaborn as snus
from matplotlib import pyplot as plt

data = pd.read_csv('training_data_vt2025.csv').reset_index()

data = pd.read_csv('training_data_vt2025.csv').reset_index()
data['windspeed'] = data['windspeed']>25
data['workday'] = data['weekday'] & ~data['holiday']

inLabels = [
    'hour_of_day',
    'day_of_week',
    'month',
    #'holiday',
    #'weekday', 
    'workday',
    'summertime',
    'temp',
    'dew',
    #'humidity',
    #'weather',w
    'precip',
    #'snow', bara 0
    'snowdepth',
    'windspeed',
    'cloudcover',
    'visibility',
]

outLabels = [
    'increase_stock'
]

train, test = skl_ms.train_test_split(data,test_size=0.2)

reg = skl_da.QuadraticDiscriminantAnalysis(reg_param=0.1)
reg.fit(train[inLabels],np.ravel(train[outLabels]))

true = test[outLabels]
pred = reg.predict(test[inLabels])

cm = skl_m.confusion_matrix(true,pred)
print(skl_m.accuracy_score(true,pred))

df_cm = pd.DataFrame(cm, index=['high bike demand','low bike demand'], columns=['high bike demand','low bike demand'])
#skl_m.ConfusionMatrixDisplay(cm,display_labels=['high_bike_demand','low_bike_demand']).plot()
snus.heatmap(df_cm,annot=True,fmt='.4g',cmap='crest').set(xlabel='predicted',ylabel='true')
plt.show()