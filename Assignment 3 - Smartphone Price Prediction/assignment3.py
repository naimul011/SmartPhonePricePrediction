"""
@author: Tokey - 14.02.04.066

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
trainData = pd.read_csv('input/train.csv')
testData = pd.read_csv('input/test.csv')


trainData['Release Date'] = pd.to_datetime(trainData['Release Date'], format='%d-%m-%y')   
testData['Release Date'] = pd.to_datetime(testData['Release Date'], format='%d-%m-%y')

trainData['OperationalDays']=""
testData['OperationalDays']=""

dateLastTrain = pd.DataFrame({'Date':np.repeat(['01-05-18'],[len(trainData)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%d-%m-%y')  
dateLastTest = pd.DataFrame({'Date':np.repeat(['01-05-18'],[len(testData)]) })
dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%d-%m-%y')  



trainData['OperationalDays'] = dateLastTrain['Date'] - trainData['Release Date']
testData['OperationalDays'] = dateLastTest['Date'] - testData['Release Date']

trainData['OperationalDays'] = trainData['OperationalDays'].astype('timedelta64[D]').astype(int)
testData['OperationalDays'] = testData['OperationalDays'].astype('timedelta64[D]').astype(int)



trainData = trainData.drop('Release Date', axis=1)
testData = testData.drop('Release Date', axis=1)



#Regression on everything
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

import numpy
xTrain = pd.DataFrame({'OpenDays':testData['OperationalDays'].apply(numpy.log),
                      'Single Core Score':testData['Single Core Score'],
                      'Multi Core Score':testData['Multi Core Score'],
                      'DxO Mark Rating':testData['DxO Mark Rating']})
#xTrain = trainData.drop(['revenue'], axis=1)
#xTrain['OpenDays'] = xTrain['OpenDays'].apply(numpy.log)
yTrain = trainData['Price'].apply(numpy.log)
xTest = pd.DataFrame({'OpenDays':testData['OperationalDays'].apply(numpy.log),
                      'Single Core Score':testData['Single Core Score'],
                      'Multi Core Score':testData['Multi Core Score'],
                      'DxO Mark Rating':testData['DxO Mark Rating']})

from sklearn import linear_model

cls = RandomForestRegressor(n_estimators=150)
cls.fit(xTrain, yTrain)
pred = cls.predict(xTest)
pred = numpy.exp(pred)
r2=cls.score(xTrain, yTrain)

pred = cls.predict(xTest)
pred = numpy.exp(pred)
print("Accuracy : ")
print(r2*100)

pred

pred2 = []
for i in range(len(pred)):
    if pred[i] != float('Inf'):
        pred2.append(pred[i])

m = sum(pred2) / float(len(pred2))

for i in range(len(pred)):
    if pred[i] == float('Inf'):
        print("haha")
        pred[i] = m
        
        
testData = pd.read_csv("input/test.csv")

submission = pd.DataFrame({
        
        "Prediction": pred
    })
submission.to_csv('PredictedPrice.csv',header=True, index=False)        


