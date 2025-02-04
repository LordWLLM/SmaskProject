import pandas as pd
import numpy as np

import concurrent.futures as cf

from itertools import combinations

import sklearn.preprocessing as skl_pre
#import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
#import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_m


data = pd.read_csv('training_data_vt2025.csv').reset_index()

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
    'precip',
    #'snow', bara 0
    #'snowdepth',
    'windspeed',
    'cloudcover',
    'visibility',
]

AllComb = [np.array(list(combinations(inLabels,r))) for r in range(1,len(inLabels))]

outLabels = [
    'increase_stock'
]

attemptStore = {}
numTries = 100
montCarl = {}
iteration = 0
large = 2**len(inLabels)

def fun(label):
    train, test = skl_ms.train_test_split(data,test_size=0.5)

    reg = skl_da.QuadraticDiscriminantAnalysis()
    reg.fit(train[label],np.ravel(train[outLabels]))

    true = test[outLabels]
    pred = reg.predict(test[label])

    return skl_m.accuracy_score(true,pred)

for labels in AllComb:
    for label in labels:
        iteration+=1
        print(f'{iteration}/{large}')
        print(label)
        with cf.ThreadPoolExecutor() as ex:
            with np.errstate(divide='ignore'):
                attemptStore[str(label)] = list(ex.map(fun,[label]*numTries))
        montCarl[str(label)] = sum(attemptStore[str(label)])/len(attemptStore[str(label)])

print(max(montCarl, key=montCarl.get))
print(max(montCarl.values()))

#df_cm = pd.DataFrame(cm, index=['low_bike_demand','high_bike_demand'], columns=['low_bike_demand','high_bike_demand'])
#skl_m.ConfusionMatrixDisplay(cm,display_labels=['low_bike_demand','high_bike_demand']).plot()
#snus.heatmap(df_cm,annot=True,fmt='.4g',cmap='flare').set(xlabel='predicted',ylabel='true')
#plt.show()