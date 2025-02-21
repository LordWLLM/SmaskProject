import pandas as pd
import numpy as np

import seaborn as sb
from matplotlib import pyplot as plt

import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_m


data = pd.read_csv('training_data_vt2025.csv').reset_index()

train, test = skl_ms.train_test_split(data,test_size=0.5,random_state=45)

true = test['increase_stock']=='high_bike_demand'
pred = (test['hour_of_day']>=8) & (test['hour_of_day']<=20)

print(f'accuracy score: {skl_m.accuracy_score(true,pred)}')
print(f'f1 score: {skl_m.f1_score(true,pred)}')
print(f'recall score: {skl_m.recall_score(true,pred)}')
print(f'precision score: {skl_m.precision_score(true,pred)}')

#cm = skl_m.confusion_matrix(true,pred)
#skl_m.ConfusionMatrixDisplay(cm).plot()


df_cm = pd.DataFrame(skl_m.confusion_matrix(true,pred), index=['high bike demand','low bike demand'], columns=['high bike demand','low bike demand'])
sb.heatmap(df_cm,annot=True,fmt='.4g',cmap='crest').set(xlabel='predicted',ylabel='true')
plt.show()