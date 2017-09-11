import sys
import os
import numpy as np
import pandas as pd
import string
import collections as coll
#import matplotlib.pyplot as plt
from collections import Counter

def zoloss(Prediction,Test):
    number_wrong=0
    for i in range(len(Prediction)):
        if Prediction[i]!=Test[i,1]:
            number_wrong=number_wrong+1
    return(number_wrong)
def feat(train,feat_list):
    featX=np.zeros([np.size(train,axis=0),len(feat_list)])
    for i in range(0,np.size(train,axis=0)):
        for j in range (0,len(feat_list)):
            if feat_list[j,0] in train[i,2]:
                featX[i,j]=1
    onecol=np.ones([np.size(featX,axis=0),1])
    featX=np.concatenate((onecol,featX),axis=1)
    return(featX)
    
def predict(featX,lw,a):
    if a=="logistic":      
        prediction_Log=[]
        prediction_Log=np.matrix(featX*lw).astype(float)
        prediction_Log=1/(1+np.exp(-1.0*prediction_Log))
        prediction_Log=np.array(prediction_Log)
        for i in range(len(prediction_Log)):
            if prediction_Log[i]<0.5:
                prediction_Log[i]=0
            else:
                prediction_Log[i]=1
    if a=='svm':
        prediction_Log=[]
        prediction_Log=np.matrix(featX*lw).astype(float)
        prediction_Log=np.array(prediction_Log)
        for i in range(len(prediction_Log)):
            if prediction_Log[i]<0:
                prediction_Log[i]=-1
            else:
                prediction_Log[i]=1
    return(prediction_Log)

def logreg(featX,lamda,learn_rate,train):
    lw=np.matrix(np.zeros([np.size(featX,axis=1),1]))
    gd=[]
    max_iter=0
    
    while max_iter<100:
        ly_hat=np.matrix(featX*lw).astype(float)
        ly_hat=1/(1+np.exp(-1.0*ly_hat))
        diff=np.matrix(train[:,1])-np.matrix(np.transpose(ly_hat))
        gd=np.transpose(((diff)*featX)-np.transpose(lamda*lw))
        lw=lw+learn_rate*gd
        max_iter=max_iter+1
    return(lw) 

def svm(featX,lamda2,learn_rate2,train,test):
    sweights=np.matrix(np.zeros([np.size(featX,axis=1),1]))
    delta=np.transpose(np.array(featX))
    gradient2=np.zeros([len(sweights)])
    max_iter2=0
    #check=[]
    #old=np.array(sweights)
    y=np.array(train[:,1])
    y[y==0]=-1
    testsvm=np.array(test)
    testsvm[testsvm==0]=-1
    
    while max_iter2<100:
        delta=np.transpose(np.array(featX))
        sy_hat=predict(featX,sweights,'svm')
        sy_hat[sy_hat==0]=-1
        #check=np.multiply(np.transpose(np.matrix(train[:,1])),sy_hat)
        for i in range(np.size(featX,axis=0)):
                if y[i]*sy_hat[i]<1:
                    delta[:,i]=delta[:,i]*y[i]
                else:
                    delta[:,i]=0
        delta=np.sum(delta,axis=1)       
        gradient2=(lamda2*sweights-np.transpose(np.matrix(delta)))/len(train[:,1])
        #change=np.array(sweights)
        sweights=sweights-learn_rate2*gradient2
        max_iter2=max_iter2+1
    
    y=np.array(train)
    for i in range(np.size(y,axis=0)):
        if y[i,1]==0:
            y[i,1]=-1
    return(sweights)

def nbc(featX,testfeatX,train):
    noy=0
    for i in range(np.size(train,axis=0)):
        if train[i,1]==1:
            noy=noy+1        
    non=np.size(train,axis=0)-noy
    predX=np.zeros([np.size(featX,axis=1),4])
    for i in range(0,np.size(predX,axis=0)):
        for j in range(0,np.size(featX,axis=0)):
            if (train[j,1]==1 and featX[j,i]==1):   #positive, x exists
                predX[i,0]=predX[i,0]+featX[j,i]
            if (train[j,1]==0 and featX[j,i]==1):   #negitive, x exists
                predX[i,1]=predX[i,1]+featX[j,i]
            if (train[j,1]==1 and featX[j,i]==0):   #positive, x doesnt exist
                predX[i,2]=predX[i,2]+1
            if (train[j,1]==0 and featX[j,i]==0):   #negitive, x doesnt exist
                predX[i,3]=predX[i,3]+1
    ilkhood=np.zeros([np.size(predX,axis=0),np.size(predX,axis=1)])
    for i in range(np.size(predX,axis=0)):
            ilkhood[i,0]=float((predX[i,0]+1)/(noy+2))
            ilkhood[i,1]=float((predX[i,1]+1)/(non+2))
            ilkhood[i,2]=float(predX[i,2]/(noy))
            ilkhood[i,3]=float(predX[i,3]/(non))
    for i in range(np.size(ilkhood, axis = 0)):
        if ilkhood[i,0]==0:
            ilkhood[i,0]=1/(noy+2)
        if ilkhood[i,1]==0:
            ilkhood[i,1]=1/(non+2)
        if ilkhood[i,2]==0:
            ilkhood[i,2]=1/(noy+2)
        if ilkhood[i,3]==0:
            ilkhood[i,3]=1/(non+2)
    for i in range(np.size(predX,axis=0)):
            ilkhood[i,2]=float(1-ilkhood[i,0])
            ilkhood[i,3]=float(1-ilkhood[i,1])
    p_y1=(np.sum(predX[:,0])+np.sum(predX[:,2]))/np.sum(predX)
    p_y0=(np.sum(predX[:,1])+np.sum(predX[:,3]))/np.sum(predX)
    
    testmatrix=np.zeros([np.size(testfeatX,axis=0),np.size(testfeatX,axis=1)])
    inv=np.zeros([np.size(testfeatX,axis=0),np.size(testfeatX,axis=1)])
    for i in range(np.size(testfeatX,axis=0)):
        for j in range(np.size(testfeatX,axis=1)):
           if testfeatX[i,j]==1:
               testmatrix[i,j]=ilkhood[j,0]
               inv[i,j]=ilkhood[j,1]
           if testfeatX[i,j]==0:
               testmatrix[i,j]=ilkhood[j,2]
               inv[i,j]=ilkhood[j,3]
    num=np.prod(testmatrix,axis=1)
    denom=np.prod(inv,axis=1)
    prediction=np.divide((num*p_y1),(num*p_y1+denom*p_y0))
    prediction=prediction*p_y1
    for i in range(len(prediction)):
        if prediction[i]>0.5:
            prediction[i]=1
        else:
            prediction[i]=0
    return(prediction)
    
def main():
    #os.chdir("D:\Jennifr\HW\\3")
    #x=pd.read_csv("yelp_data.csv",sep='\t|\n',engine='python',header=None,names=('Number','isPositive','reviewText'))
    #train=x.iloc[0:1000,:] 
    train=pd.read_csv(sys.argv[1],sep='\t|\n',engine='python',header=None,names=('Number','Positive','reviewText'))
    test=pd.read_csv(sys.argv[2],sep='\t|\n',engine='python',header=None,names=('Number','Positive','reviewText'))
    model=sys.argv[3]
    train=np.array(train)
    test=np.array(test)
    #nbc2train=np.array(train[:,2])
    #nbc2test=np.array(test[:,2])
    
    for i in range(0,np.size(train,axis=0)):
        train[i,2]=str.lower(train[i,2])
        train[i,2]=" ".join(train[i,2].split())
        train[i,2]="".join(j for j in train[i,2] if j not in set(string.punctuation))
        train[i,2]=list(train[i,2].split(" "))
        train[i,2]=list(set(train[i,2]))
        
    for i in range(0,np.size(test,axis=0)):
        test[i,2]=str.lower(test[i,2])
        test[i,2]=" ".join(test[i,2].split())
        test[i,2]="".join(j for j in test[i,2] if j not in set(string.punctuation))
        test[i,2]=list(test[i,2].split(" "))
        test[i,2]=list(set(test[i,2]))
        
    '''for i in range(0,np.size(nbc2train,axis=0)):
        nbc2train[i]=str.lower(nbc2train[i])
        nbc2train[i]=" ".join(nbc2train[i].split())
        nbc2train[i]="".join(j for j in nbc2train[i] if j not in set(string.punctuation))
        nbc2train[i]=list(nbc2train[i].split(" "))
        nbc2train[i]=np.array(coll.Counter(nbc2train[i]).most_common())
    
    for i in range(0,np.size(nbc2test,axis=0)):
        nbc2test[i]=str.lower(nbc2test[i])
        nbc2test[i]=" ".join(nbc2test[i].split())
        nbc2test[i]="".join(j for j in nbc2test[i] if j not in set(string.punctuation))
        nbc2test[i]=list(nbc2test[i].split(" "))
        nbc2test[i]=np.array(coll.Counter(nbc2test[i]).most_common())'''
    
    bow=[]
    for i in range(0,np.size(train,axis=0)):
       bow=np.append(bow,train[i,2],axis=0)
    fr=coll.Counter(bow) 
    freq=fr.most_common()
    feat_list=np.array(freq[101:4101])
    
    featX=feat(train,feat_list)
    featY=feat(test,feat_list)
    testsvm=np.array(test)
    testsvm[testsvm==0]=-1
    y=np.array(train)
    for i in range(np.size(y,axis=0)):
        if y[i,1]==0:
            y[i,1]=-1
    
    if model=="1":
    #---------------------------------------------------------------------------------------Logistic Regression
        logweights=logreg(featX,0.01,0.01,train)
        prediction_logreg=predict(featY,logweights,"logistic")
        #alog=zoloss(prediction_logreg,test)
        zoloss_logreg=(zoloss(prediction_logreg,test)/len(test[:,1]))
        print("ZERO-ONE-LOSS-LR",zoloss_logreg)
    
    #---------------------------------------------------------------------------------------SVM
    if model=="2":
        sweightsl=svm(featX,0.01,0.5,train,test)
        prediction_svm_test1=predict(featY,sweightsl,'svm')
        #asvm=zoloss(prediction_svm_test1,testsvm)
        zoloss_svm=zoloss(prediction_svm_test1,testsvm)/len(testsvm[:,1])
        print("ZERO-ONE-LOSS-SVM",zoloss_svm)
    
    

main()

   
  
'''__________________________________ANALYSIS - BOTH 1st and 2nd Part
import os
import numpy as np
import pandas as pd
import string
import collections as coll
import matplotlib.pyplot as plt
from collections import Counter

def zoloss(Prediction,Test):
    number_wrong=0
    for i in range(len(Prediction)):
        if Prediction[i]!=Test[i,1]:
            number_wrong=number_wrong+1
    return(number_wrong)
def feat(train):
    featX=np.zeros([np.size(train,axis=0),len(feat_list)])
    for i in range(0,np.size(train,axis=0)):
        for j in range (0,len(feat_list)):
            if feat_list[j,0] in train[i,2]:
                featX[i,j]=1
    onecol=np.ones([np.size(featX,axis=0),1])
    featX=np.concatenate((onecol,featX),axis=1)
    return(featX)
def predict(featX,lw,a):
    if a=="logistic":      
        prediction_Log=[]
        prediction_Log=np.matrix(featX*lw).astype(float)
        prediction_Log=1/(1+np.exp(-1.0*prediction_Log))
        prediction_Log=np.array(prediction_Log)
        for i in range(len(prediction_Log)):
            if prediction_Log[i]<0.5:
                prediction_Log[i]=0
            else:
                prediction_Log[i]=1
    if a=='svm':
        prediction_Log=[]
        prediction_Log=np.matrix(featX*lw).astype(float)
        prediction_Log=np.array(prediction_Log)
        for i in range(len(prediction_Log)):
            if prediction_Log[i]<0:
                prediction_Log[i]=-1
            else:
                prediction_Log[i]=1
    return(prediction_Log)

def logreg(featX,lamda,learn_rate):
    lw=np.matrix(np.zeros([np.size(featX,axis=1),1]))
    gd=[]
    max_iter=0
    
    while max_iter<100:
        ly_hat=np.matrix(featX*lw).astype(float)
        ly_hat=1/(1+np.exp(-1.0*ly_hat))
        diff=np.matrix(train[:,1])-np.matrix(np.transpose(ly_hat))
        gd=np.transpose(((diff)*featX)-np.transpose(lamda*lw))
        lw=lw+learn_rate*gd
        max_iter=max_iter+1
    return(lw) 

def svm(featX,lamda2,learn_rate2):
    sweights=np.matrix(np.zeros([np.size(featX,axis=1),1]))
    delta=np.transpose(np.array(featX))
    gradient2=np.zeros([len(sweights)])
    max_iter2=0
    #check=[]
    #old=np.array(sweights)
    y=np.array(train[:,1])
    y[y==0]=-1
    testsvm=np.array(test)
    testsvm[testsvm==0]=-1
    
    while max_iter2<100:
        delta=np.transpose(np.array(featX))
        sy_hat=predict(featX,sweights,'svm')
        sy_hat[sy_hat==0]=-1
        #check=np.multiply(np.transpose(np.matrix(train[:,1])),sy_hat)
        for i in range(np.size(featX,axis=0)):
                if y[i]*sy_hat[i]<1:
                    delta[:,i]=delta[:,i]*y[i]
                else:
                    delta[:,i]=0
        delta=np.sum(delta,axis=1)       
        gradient2=(lamda2*sweights-np.transpose(np.matrix(delta)))/len(train[:,1])
        #change=np.array(sweights)
        sweights=sweights-learn_rate2*gradient2
        max_iter2=max_iter2+1
    
    y=np.array(train)
    for i in range(np.size(y,axis=0)):
        if y[i,1]==0:
            y[i,1]=-1
    return(sweights)

def nbc(featX,testfeatX):
    noy=0
    for i in range(np.size(train,axis=0)):
        if train[i,1]==1:
            noy=noy+1        
    non=np.size(train,axis=0)-noy
    predX=np.zeros([np.size(featX,axis=1),4])
    for i in range(0,np.size(predX,axis=0)):
        for j in range(0,np.size(featX,axis=0)):
            if (train[j,1]==1 and featX[j,i]==1):   #positive, x exists
                predX[i,0]=predX[i,0]+featX[j,i]
            if (train[j,1]==0 and featX[j,i]==1):   #negitive, x exists
                predX[i,1]=predX[i,1]+featX[j,i]
            if (train[j,1]==1 and featX[j,i]==0):   #positive, x doesnt exist
                predX[i,2]=predX[i,2]+1
            if (train[j,1]==0 and featX[j,i]==0):   #negitive, x doesnt exist
                predX[i,3]=predX[i,3]+1
    ilkhood=np.zeros([np.size(predX,axis=0),np.size(predX,axis=1)])
    for i in range(np.size(predX,axis=0)):
            ilkhood[i,0]=float((predX[i,0]+1)/(noy+2))
            ilkhood[i,1]=float((predX[i,1]+1)/(non+2))
            ilkhood[i,2]=float(predX[i,2]/(noy))
            ilkhood[i,3]=float(predX[i,3]/(non))
    for i in range(np.size(ilkhood, axis = 0)):
        if ilkhood[i,0]==0:
            ilkhood[i,0]=1/(noy+2)
        if ilkhood[i,1]==0:
            ilkhood[i,1]=1/(non+2)
        if ilkhood[i,2]==0:
            ilkhood[i,2]=1/(noy+2)
        if ilkhood[i,3]==0:
            ilkhood[i,3]=1/(non+2)
    for i in range(np.size(predX,axis=0)):
            ilkhood[i,2]=float(1-ilkhood[i,0])
            ilkhood[i,3]=float(1-ilkhood[i,1])
    p_y1=(np.sum(predX[:,0])+np.sum(predX[:,2]))/np.sum(predX)
    p_y0=(np.sum(predX[:,1])+np.sum(predX[:,3]))/np.sum(predX)
    
    testmatrix=np.zeros([np.size(testfeatX,axis=0),np.size(testfeatX,axis=1)])
    inv=np.zeros([np.size(testfeatX,axis=0),np.size(testfeatX,axis=1)])
    for i in range(np.size(testfeatX,axis=0)):
        for j in range(np.size(testfeatX,axis=1)):
           if testfeatX[i,j]==1:
               testmatrix[i,j]=ilkhood[j,0]
               inv[i,j]=ilkhood[j,1]
           if testfeatX[i,j]==0:
               testmatrix[i,j]=ilkhood[j,2]
               inv[i,j]=ilkhood[j,3]
    num=np.prod(testmatrix,axis=1)
    denom=np.prod(inv,axis=1)
    prediction=np.divide((num*p_y1),(num*p_y1+denom*p_y0))
    prediction=prediction*p_y1
    for i in range(len(prediction)):
        if prediction[i]>0.5:
            prediction[i]=1
        else:
            prediction[i]=0
    return(prediction)

def nbc2(featX,testfeatX):
    noy=0
    for i in range(np.size(train,axis=0)):
        if train[i,1]==1:
            noy=noy+1        
    non=np.size(train,axis=0)-noy
    predX=np.zeros([np.size(featX2,axis=1),6])
    for i in range(0,np.size(predX,axis=0)):
        for j in range(0,np.size(featX2,axis=0)):
            if (train[j,1]==1 and featX2[j,i]==0):   #positive, x 0
                predX[i,0]=predX[i,0]+1
            if (train[j,1]==1 and featX2[j,i]==1):   #positive, x 1
                predX[i,1]=predX[i,1]+1
            if (train[j,1]==1 and featX2[j,i]==2):   #positive, x 2
                predX[i,2]=predX[i,2]+1
            if (train[j,1]==0 and featX2[j,i]==0):   #negitive, x 0
                predX[i,3]=predX[i,3]+1
            if (train[j,1]==0 and featX2[j,i]==1):   #negitive, x 1 
                predX[i,4]=predX[i,4]+1
            if (train[j,1]==0 and featX2[j,i]==2):   #negitive, x 2 
                predX[i,5]=predX[i,5]+1
    ilkhood=np.zeros([np.size(predX,axis=0),np.size(predX,axis=1)])
    for i in range(np.size(predX,axis=0)):
            ilkhood[i,0]=float((predX[i,0]+1)/(noy+3))
            ilkhood[i,1]=float((predX[i,1]+1)/(noy+3))
            ilkhood[i,2]=float((predX[i,2]+1)/(noy+3))
            
    for i in range(np.size(ilkhood, axis = 0)):
        if ilkhood[i,0]==0:
            ilkhood[i,0]=1/(noy+2)
        if ilkhood[i,1]==0:
            ilkhood[i,1]=1/(non+2)
        if ilkhood[i,2]==0:
            ilkhood[i,2]=1/(noy+2)
        if ilkhood[i,3]==0:
            ilkhood[i,3]=1/(non+2)
    for i in range(np.size(predX,axis=0)):
            ilkhood[i,3]=float(1-ilkhood[i,0])
            ilkhood[i,4]=float(1-ilkhood[i,1])
            ilkhood[i,5]=float(1-ilkhood[i,2])
    
    p_y1=(np.sum(predX[:,0])+np.sum(predX[:,1])+np.sum(predX[:,2]))/np.sum(predX)
    p_y0=(np.sum(predX[:,3])+np.sum(predX[:,4])+np.sum(predX[:,5]))/np.sum(predX)
    
    testmatrix=np.zeros([np.size(featY2,axis=0),np.size(featY2,axis=1)])
    inv=np.zeros([np.size(featY2,axis=0),np.size(featY2,axis=1)])
    
    for i in range(np.size(featY2,axis=0)):
        for j in range(np.size(featY2,axis=1)):
            if featY2[i,j]==0:
               testmatrix[i,j]=ilkhood[j,0]
               inv[i,j]=ilkhood[j,3]
            if featY2[i,j]==1:
               testmatrix[i,j]=ilkhood[j,1]
               inv[i,j]=ilkhood[j,4]
            if featY2[i,j]==2:
               testmatrix[i,j]=ilkhood[j,2]
               inv[i,j]=ilkhood[j,5]
    
    num=np.prod(testmatrix,axis=1)
    denom=np.prod(inv,axis=1)
    prediction=np.divide((num*p_y1),(num*p_y1+denom*p_y0))
    prediction=prediction*p_y1
    for i in range(len(prediction)):
        if prediction[i]>0.5:
            prediction[i]=1
        else:
            prediction[i]=0
    return(prediction)

def feat2(nbc2train):
    featX2=np.zeros([np.size(nbc2train,axis=0),len(feat_list)])
    for i in range(0,len(nbc2train)):
        for j in range(0,len(feat_list)):
            if  feat_list[j,0] in list(nbc2train[i]):
                if coll.Counter(list(nbc2train[i]))[feat_list[j,0]]==1:    
                    featX2[i][j]=1
                else:
                    featX2[i,j]=2
    return(featX2)
#---------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------ACTUAL CODE STARTS HERE---------------------------------------
#---------------------------------------------------------------------------------------------------------------------
'''
os.chdir("D:\CS573 Data Mining\HW\\3")
x=pd.read_csv("yelp_data.csv",sep='\t|\n',engine='python',header=None,names=('Number','isPositive','reviewText'))
#train=x.iloc[0:1000,:]
#log_record=np.ones([6,12])
#svm_record=np.ones([6,12])
#nbc_record=np.ones([6,12])
import random
pool=list(range(np.size(x,axis=0)))
splitlen=int(len(pool)/10)
set1 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set1)
set2 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set2)
set3 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set3)
set4 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set4)
set5 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set5)
set6 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set6)
set7 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set7)
set8 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set8)
set9 = list(random.sample(pool, splitlen))
pool=set(pool)-set(set9)
set10 = list(random.sample(pool, splitlen))
actualset=list([set1,set2,set3,set4,set5,set6,set7,set8,set9,set10])
#correct till here---------------------------------------------------------
log_record=np.ones([6,12])
svm_record=np.ones([6,12])
nbc_record=np.ones([6,12])
j=0
for h in [0.01,0.03,0.05,0.08,0.1,0.15]:
    for k in range(10):
        test=(x[x['Number'].isin(actualset[k])])
        train=np.array(x.drop(test.index))
        train=pd.DataFrame(train)
        train=train.sample(n=int(h*np.size(x,axis=0)))
        train=np.array(train)
        test=np.array(test)
        nbc2train=np.array(train[:,2])
        nbc2test=np.array(test[:,2])
        for i in range(0,np.size(train,axis=0)):
            train[i,2]=str.lower(train[i,2])
            train[i,2]=" ".join(train[i,2].split())
            train[i,2]="".join(j for j in train[i,2] if j not in set(string.punctuation))
            train[i,2]=list(train[i,2].split(" "))
            train[i,2]=list(set(train[i,2]))
            
        for i in range(0,np.size(test,axis=0)):
            test[i,2]=str.lower(test[i,2])
            test[i,2]=" ".join(test[i,2].split())
            test[i,2]="".join(j for j in test[i,2] if j not in set(string.punctuation))
            test[i,2]=list(test[i,2].split(" "))
            test[i,2]=list(set(test[i,2]))
            
        for i in range(0,np.size(nbc2train,axis=0)):
            nbc2train[i]=str.lower(nbc2train[i])
            nbc2train[i]=" ".join(nbc2train[i].split())
            nbc2train[i]="".join(j for j in nbc2train[i] if j not in set(string.punctuation))
            nbc2train[i]=list(nbc2train[i].split(" "))
            #nbc2train[i]=np.array(coll.Counter(nbc2train[i]).most_common())
        
        for i in range(0,np.size(nbc2test,axis=0)):
            nbc2test[i]=str.lower(nbc2test[i])
            nbc2test[i]=" ".join(nbc2test[i].split())
            nbc2test[i]="".join(j for j in nbc2test[i] if j not in set(string.punctuation))
            nbc2test[i]=list(nbc2test[i].split(" "))
        
        bow=[]
        for i in range(0,np.size(train,axis=0)):
           bow=np.append(bow,train[i,2],axis=0)
        fr=coll.Counter(bow) 
        freq=fr.most_common()
        feat_list=np.array(freq[101:4101])
        
        featX=feat(train)
        featY=feat(test)
        testsvm=np.array(test)
        testsvm[testsvm==0]=-1
        y=np.array(train)
        for i in range(np.size(y,axis=0)):
            if y[i,1]==0:
                y[i,1]=-1
        #---------------------------------------------------------------------------------------Logistic Regression
        logweights=logreg(featX,0.01,0.01)
        prediction_logreg=predict(featY,logweights,"logistic")
        zoloss_logreg=(zoloss(prediction_logreg,test)/len(test[:,1]))
        log_record[j,k]=zoloss_logreg
        #---------------------------------------------------------------------------------------SVM
        sweightsl=svm(featX,0.01,0.5)
        prediction_svm_test1=predict(featY,sweightsl,'svm')
        zoloss_svm=zoloss(prediction_svm_test1,testsvm)/len(testsvm[:,1])
        svm_record[j,k]=zoloss_svm
        #---------------------------------------------------------------------------------------NBC
        nbc_predict=nbc(featX,featY)
        zoloss_nbc=0
        for i in range(len(nbc_predict)):
            if nbc_predict[i]!=test[i,1]:
                zoloss_nbc=zoloss_nbc+1
        zoloss_nbc=zoloss_nbc/len(test[:,1])
        nbc_record[j,k]=zoloss_nbc
        #-------------------------------------------------------------------------------------------------------END
        featX2=feat2(nbc2train)
        featY2=feat2(nbc2test)
        logweights=logreg(featX2,0.01,0.01)
        prediction_logreg=predict(featY2,logweights,"logistic")
        zoloss_logreg=(zoloss(prediction_logreg,test)/len(test[:,1]))
        log_record[j,k]=zoloss_logreg
        #---------------------------------------------------------------------------------------SVM
        sweightsl=svm(featX2,0.01,0.5)
        prediction_svm_test1=predict(featY2,sweightsl,'svm')
        zoloss_svm=zoloss(prediction_svm_test1,testsvm)/len(testsvm[:,1])
        svm_record[j,k]=zoloss_svm
        #---------------------------------------------------------------------------------------NBC
        nbc_predict=nbc2(featX2,featY)
        zoloss_nbc=0
        for i in range(len(nbc_predict)):
            if nbc_predict[i]!=test[i,1]:
                zoloss_nbc=zoloss_nbc+1
        zoloss_nbc=zoloss_nbc/len(test[:,1])
        nbc_record[j,k]=zoloss_nbc
        #-------------------------------------------------------------------------------------------------------END
    j=j+1
log_record[:,10]=np.mean(log_record,axis=1)
log_record[:,11]=np.std(log_record,axis=1)
log_record[:,11]=log_record[:,11]/10**0.5
svm_record[:,10]=np.mean(svm_record,axis=1)
svm_record[:,11]=np.std(svm_record,axis=1)
svm_record[:,11]=svm_record[:,11]/10**0.5
nbc_record[:,10]=np.mean(nbc_record,axis=1)
nbc_record[:,11]=np.std(nbc_record,axis=1)
nbc_record[:,11]=nbc_record[:,11]/10**0.5

seq=np.array([0.01,0.03,0.05,0.08,0.1,0.15])
log_means=log_record[:,10]
svm_means=svm_record[:,10]
nbc_means=nbc_record[:,10]

plt.plot(seq,np.mean(log_record,axis=1),'bo')
plt.errorbar(seq,np.mean(log_record,axis=1),yerr=log_record[:,11])
plt.plot(seq,np.mean(svm_record,axis=1),'go') 
plt.errorbar(seq,np.mean(svm_record,axis=1),yerr=svm_record[:,11])
plt.plot(seq,np.mean(nbc_record,axis=1),'ro')
plt.errorbar(seq,np.mean(nbc_record,axis=1),yerr=nbc_record[:,11])
plt.xlabel("Train set size (Percentage)")
plt.ylabel("Zero-One Error (Percentage)")
#plt.plot(seq,)
plt.axis([0.0123,0.25,0,1])
plt.show()







