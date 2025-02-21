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
data['daynight'] = 2*((data['hour_of_day'] < 21) & (data['hour_of_day'] > 14))+((data['hour_of_day']<15)&(data['hour_of_day']>7))

inLabels = [
    'hour_of_day',
    #'day_of_week', # ta bort
    #'month',
    #'holiday',  # ta bort
    #'weekday',  # ta bort
    'workday',  # bättre än weekday och holiday
    #'summertime',
    #'temp',
    'dew',
    'humidity', #beror på temp och dew ta bort
    #'precip',
    #'snow', #bara 0 ta bort
    #'snowdepth', #1542 st kanske ta bort
    #'windspeed',
    #'cloudcover',
    #'visibility',
    'daynight'
]
'''
for label in inLabels:
    high = list(data.loc[data["increase_stock"]=="high_bike_demand",label])
    low = list(data.loc[data["increase_stock"]=="low_bike_demand",label])
    print(f'{label} high mean: {np.mean(high)}, std: {np.std(high)}')
    print(f'{label} low mean:  {np.mean(low)}, std:{np.std(low)}\n')
print(f'high amount: {len(data.loc[data["increase_stock"]=="high_bike_demand"])}, low amount: {len(data.loc[data["increase_stock"]=="low_bike_demand"])}')


'''
AllComb = [np.array(list(combinations(inLabels,r))) for r in range(1,len(inLabels))]

outLabels = [
    'increase_stock'
]

numTries = 100
montCarl = {}
iteration = 0
large = 2**len(inLabels)

def fun(label):
    train, test = skl_ms.train_test_split(data,test_size=0.2)

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
            store = list(ex.map(fun,[label]*numTries))
            print([min(store),max(store),float(np.std(store))])
        montCarl[str(label)] = [sum(store)/len(store),np.std(store)]

print(max(montCarl, key=montCarl.get))
print(max(montCarl.values()))