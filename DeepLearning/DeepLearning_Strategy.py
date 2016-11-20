import numpy as np
import scipy
import matplotlib as plt
import pandas as pd
import datetime as dt
import math
import sklearn
from sklearn import cross_validation, metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale,MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import theano
import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dropout
import os



def dalgo(alldata, flename, alldate):
    global dataset
    global name
    global date
    global price,bsratio,bssize, depth, bavol, ofi, oir, posdev,negdev,fut_price

    np.random.seed(20)
    dataset=alldata
    name=filename
    date=alldate.astype(object)
    price1, price2,bsratio,bssize, depth, bavol, ofi, oir, posdev,negdev,fut_price=dataset[:,0],dataset[:,1],dataset[:,2],dataset[:,3]dataset[:,4],dataset[:,5],dataset[:,6]dataset[:,7],dataset[:,8],dataset[:,9],dataset[:,10]

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    #####define final ip/op sets here
    final_ip = np.vstack((price, bsratio, bssize, depth, bavol, ofi, oir, posdev, negdev)).T
    final_op = futprice

    ##########Encode Class values as integers############################
    encoder = LabelEncoder()
    encoder.fit(final_op)
    encoded_Y = encoder.transform(final_op)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    train_ip, train_op = final_ip[0:train_size, :], dummy_Y[0:train_size]
    test_ip, test_op = final_ip[train_size:len(final_ip), :], dummy_Y[train_size:len(final_ip)]
    train_ipr = np.reshape(train_ip, (train_ip.shape[0], 1, train_ip.shape[1]))
    test_ipr = np.reshape(test_ip, (test_ip.shape[0], 1, test_ip.shape[1]))



    #########Define RNN Model here#####################
    dimensions = train_ip.shape[1]
    model = Sequential()
    model.add(LSTM(20, input_dim=dimensions, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(30, input_dim=dimensions, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(20, input_dim=dimensions, activation='tanh', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(10, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_ipr, train_op, nb_epoch=100, batch_size=1)

    ###make predictions on test dataset############
    trainPredict=model.predict(train_ipr)
    testPredict=model.predict(test_ipr)

    op_bucket=np.asarray(np.zeros(testPredict.shape[0])).astype(float)
    for j in range(0 , len(testPredict)):
        op_bucket[j]=np.argmax(testPredict[k,:])

    test_op=final_op[train_size:len(final_ip)]
    date_op=date[train_size:len(final_ip)]
    ossories=(testPredict).T
    allop=np.vstack((date_op,test_op,ossories,op_bucket)).T
    allop1=pd.DataFrame(allop)

    dirpath='C:/Users/sujit/Desktop/python/DeepLearning/output.CSV'
    file=str(name)
    filepath=os.path.join(dirpath,file)
    allop1.to_csv(filepath)

#####Read inputs Features and Output Classes######
dataframe=pd.read_csv('C:\\Users\\sujit\\Desktop\\python\\DeepLearning\\NIFTYFEATURES.CSV')
dataframe1=np.asarray(dataframe).astype(object)
datetimes=dataframe1[:,0:3]
dataset1=dataframe1[:,3:13]
dataset1=dataset.astype(float)
date1=datetimes[:,0]
endpoints2=[]

for j in range(0, len(date1)+1):
    if j%72==0:
        endpoints2.append(j)

endpoints2.append(j)
endpoints3=np.asarray(endpoints2).astype(float)

testdata=dataset1[endpoints3[0]:endpoints3[40]]
datevals=datetimes[endpoints3[0]:endpoints3[40]]

for i in range(0,7):
    alldata=np.asarray(np.zeros(testdata.shape).astype(float))
    alldata=dataset1[endpoints3[i*8]:endpoints3[i*8+40]]
    alldatee=np.asarray(np.zeros(datevals.shape).astype(float))
    alldate=date1[endpoints3[i*8]:endpoints3[i*8+40]]
    filename=i
    dalgo(alldata, filename, alldate)








