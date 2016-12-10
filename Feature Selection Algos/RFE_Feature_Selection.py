from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib as plt



"Recursive Feature Elimination Model applied on a rolling basis on Order Book Features"

"Select the most explanatory features, no limit on number of features"
def allfeat(alldata, alldate):
    global Xtrain, Ytrain,model,date_des
    global rfecv,g_scores,chooseone,chosenvals,names
    global featurenames,start,end

    X=alldata[:,1:19].astype('float32')
    Y = alldata[:,19].astype('float32')
    date_des=alldate

    train_size=int(len(Y)*0.75)
    Xtrain,Ytrain=X[0:train_size,:], Y[0:train_size]
    model=LogisticRegression()
    rfecv=RFECV(estimator=model, step=1,cv=StratifiedKFold(Ytrain,2) ,scoring='accuracy',)
    rfecv.fit(Xtrain,Ytrain)
    print("Orignal number of features is %s" % X.shape[1])
    print("Feature CV ranking: %s") % rfecv.ranking_
    print("RFECV final number of features: %d" % rfecv.n_features_)
    print(rfecv.support_)
    print('')
    g_scores=rfecv.grid_scores_
    indices=np.argsort(g_scores)[::-1]
    print('Printing RFECV Rresults:')
    for f in range(X.shape[1]):
        print("%d. Number of Features: %d; Grid_Score: %f" % (f + 1, indices[f]+1, g_scores[indices[f]]))

    "Plot number of Features vs CrossValidation Score"

    plt.figure()
    plt.xlabel("Number of Features Selected")
    plt.ylabel("CrossValidation Score(nb of correct Classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
    plt.show()

    "Assign Corresponding Names to Output Feature along with Start and End dates of evaluation"
    chooseone=rfecv.ranking_
    chosenvals=[i for i,x in enumerate(chooseone) if x==1]

    names=['size','depth', 'depthL5', 'AvgBidAsk', 'OBI', 'VWAPPRICE', 'OFI', 'OIR', 'BIDASK Vol', 'Posdev', 'Negdev','Pos/neg', 'Stdev', 'Trades/Order', 'buycount', 'sellcount','Buy+Sell', 'Ret 5min','Ret 10-5min','Fur Ret',]
    start=date_des[train_size+1]
    end=date_des[len(date_des)-1]
    featurenames=[]
    featurenames.append(start)
    featurenames.append(end)
    for j in range(0, len(chosenvals)):
        featurenames.append(names[chosenvals[j]])

    return featurenames



"Select four best fetaures"
def selectfour(alldata, alldate):

    global Xtrain, Ytrain,model,rfe,choosefour,date_des
    global select4,names,fournames,fit,start,end

    X=alldata[:,1:19].astype('float32')
    Y = alldata[:,19].astype('float32')
    date_des=alldate

    train_size=int(len(Y)*0.75)
    Xtrain,Ytrain=X[0:train_size,:], Y[0:train_size]

    model=LogisticRegression()
    rfe=RFE(model,4)
    rfe.fit(Xtrain,Ytrain)
    print("Num Features: %d") %rfe.n_features_
    print("Selected Features: %s") % rfe.support_
    print("Feature ranking: %s") % rfe.ranking_
    print("Feature Scores: %s") % rfe.estimator_.score

    "Assign Corresponding Names to Output Feature along with Start and End dates of evaluation"

    choosefour=rfe.ranking_
    select4=[i for i, x in enumerate(choosefour) if x==1]
    names = ['size', 'depth', 'depthL5', 'AvgBidAsk', 'OBI', 'VWAPPRICE', 'OFI', 'OIR', 'BIDASK Vol', 'Posdev',
             'Negdev', 'Pos/neg', 'Stdev', 'Trades/Order', 'buycount', 'sellcount', 'Buy+Sell', 'Ret 5min',
             'Ret 10-5min', 'Fur Ret', ]

    start=date_des[train_size+1]
    end=date_des[len(date_des)-1]
    fournames=[]
    fournames.append(start)
    fournames.append(end)
    for j in range(o, len(select4)):
        fournames.append(names[select4[j]])


    return fournames



"Select seven best features"

def selectseven(alldata, alldate):

    global Xtrain, Ytrain,model,rfe,chooseseven,date_des
    global select7,names,sevennames,fit,start,end

    X=alldata[:,1:19].astype('float32')
    Y = alldata[:,19].astype('float32')
    date_des=alldate

    train_size=int(len(Y)*0.75)
    Xtrain,Ytrain=X[0:train_size,:], Y[0:train_size]

    model=LogisticRegression()
    rfe=RFE(model,7)
    rfe.fit(Xtrain,Ytrain)
    print("Num Features: %d") %rfe.n_features_
    print("Selected Features: %s") % rfe.support_
    print("Feature ranking: %s") % rfe.ranking_
    print("Feature Scores: %s") % rfe.estimator_.score

    "Assign Corresponding Names to Output Feature along with Start and End dates of evaluation"

    chooseseven=rfe.ranking_
    select7=[i for i, x in enumerate(chooseseven) if x==1]
    names = ['size', 'depth', 'depthL5', 'AvgBidAsk', 'OBI', 'VWAPPRICE', 'OFI', 'OIR', 'BIDASK Vol', 'Posdev',
             'Negdev', 'Pos/neg', 'Stdev', 'Trades/Order', 'buycount', 'sellcount', 'Buy+Sell', 'Ret 5min',
             'Ret 10-5min', 'Fur Ret', ]

    start=date_des[train_size+1]
    end=date_des[len(date_des)-1]
    sevennames=[]
    sevennames.append(start)
    sevennames.append(end)
    for j in range(o, len(select7)):
        sevennames.append(names[select7[j]])

    return sevennames



"Rolling feature selection"

dataframe=pd.read_csv('C:\\Users\\sujit\\Desktop\\python\\DeepLearning\\NIFTYFEATURES.CSV')
array=dataframe.values
date=array[:,0].astype('object')

###Find End Point of everyday###
endpoints2=[]
for j ni range(0,len(date)+1):
    if j%72==0
        endpoints2.append(j)

endpoints2.append(j)
endpoints3=np.asarray(endpoints2).astype(float)
testdata=array[endpoints3[0]:endpoints3[40]]
datevals=date[endpoints3[0]:endpoints3[40]]

startdate=[]
enddate=[]
allfeatures=[]
fantastic4=[]
fantastic7=[]


for i in range(0,7):
    alldata=np.asarray(np.zeros(testdata.shape).astype(float))
    alldata=array[endpoints3[i*10]:endpoints3[1*10+40]]
    alldate=np.asarray(np.zeros(datevals.shape).astype(object))
    alldate=date[endpoints3[i*10]:endpoints3[1*10+40]]
    startdate.append(alldate[1])
    enddate.append(alldate[len(alldate)-1])
    allfeatures.append(allfeat(alldata,alldate))
    fantastic4.append(selectfour(alldata, alldate))
    fantastic7.append(selectseven(alldata,alldate))

"O/P final set of features from RFE algorithm"

data1=pd.DataFrame(allfeatures)
data1.to_csv('C:/Users/sujit/Desktop/python/DeepLearning/bestexplained.CSV')

data2=pd.DataFrame(fantastic4)
data2.to_csv('C:/Users/sujit/Desktop/python/DeepLearning/best4.CSV')

data3=pd.DataFrame(fantastic7)
data3.to_csv('C:/Users/sujit/Desktop/python/DeepLearning/best7.CSV')
