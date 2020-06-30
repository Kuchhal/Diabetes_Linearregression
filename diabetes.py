import numpy as np
from sklearn.linear_model import LinearRegression
data=np.genfromtxt('train.csv',delimiter=',')
x=data[:,0:10]
y=data[:,10].reshape(-1,1)
algl=LinearRegression()
algl.fit(x,y)
m=algl.coef_[0]
c=algl.intercept_[0]
data1=np.genfromtxt('test.csv',delimiter=',')
x_test=data1[:]
y_pred=algl.predict(x_test)
li=[]
for i in range(len(y_pred)):
    li.append(float(y_pred[i]))
np.savetxt('predictions.csv',li,fmt='%.5f',delimiter=',')
