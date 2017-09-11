import os
import numpy as np
import pandas as pd
import string
import collections as coll
import matplotlib.pyplot as plt
os.chdir("D:\CS573 Data Mining\HW\\2")
x=np.array(pd.read_csv("yelp_data.csv",sep='\t|\n',engine='python',header=None,names=('Number','Positive','reviewText')))
for i in range(0,np.size(x,axis=0)):
    x[i,2]=str.lower(x[i,2])
    x[i,2]=" ".join(x[i,2].split())
    x[i,2]="".join(j for j in x[i,2] if j not in set(string.punctuation))
    x[i,2]=list(x[i,2].split(" "))
    x[i,2]=list(set(x[i,2]))
x=pd.DataFrame(x)
one=list()
baseline=list()
index=0
means=np.zeros([6,10])

baseline_means=np.zeros([6,10])
qq=0
#train=x.iloc[0:1000,:]
for q in [1,5,10,20,50,90]:
    
    count=0
    one=list()
    baseline=list()
    while count<10:
        train=x.sample(frac=q/100)
        test=np.array(x.drop(train.index))
        train=np.array(train)
        basepred=np.zeros([np.size(test,axis=0)])
        bow=list()
        for i in range(0,np.size(train,axis=0)):
           bow=np.append(bow,train[i,2],axis=0)
        bagofwords=np.unique(bow)
        fr=coll.Counter(bow)
        freq=fr.most_common()
        feat_list=np.array(freq[100:600])
        featX=np.zeros([np.size(train,axis=0),len(feat_list)])
        for i in range(0,np.size(train,axis=0)):
            for j in range (0,len(feat_list)):
                if feat_list[j,0] in train[i,2]:
                    featX[i,j]=1
        noy=0
        for i in range(np.size(train,axis=0)):
            if train[i,1]==1:
                noy=noy+1        
        non=np.size(train,axis=0)-noy
        for i in range(np.size(basepred,axis=0)):
            if noy>non:
                basepred[i]=1
            else:
                basepred[i]=0
        predX=np.zeros([np.size(featX,axis=1),4])
        for i in range(0,np.size(predX,axis=0)):
            for j in range(0,np.size(featX,axis=0)):
                if (train[j,1]==1 and featX[j,i]==1):   #positive, x exists
                    predX[i,0]=predX[i,0]+featX[j,i]
                if (train[j,1]==0 and featX[j,i]==1):   #negitive, x exists
                    predX[i,1]=predX[i,1]+featX[j,i]
                if (train[j,1]==1 and featX[j,i]==0):   #positive, x doesnt exi
                    predX[i,2]=predX[i,2]+1
                if (train[j,1]==0 and featX[j,i]==0):   #negitive, x doesnt exi
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
        testfeatX=np.zeros([np.size(test,axis=0),len(feat_list)])
        for i in range(0,np.size(test,axis=0)):
            for j in range (0,len(feat_list)):
                if feat_list[j,0] in test[i,2]:
                    testfeatX[i,j]=1
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
        berror=0
        for i in range(np.size(test,axis=0)):
            if basepred[i]!=test[i,1]:
                berror=berror+1
        b=berror/np.size(test,axis=0)
        error=0
        for i in range(len(prediction)):
            if prediction[i]!=test[i,1]:
                error=error+1
        a=error/np.size(test,axis=0)
        one.append(a)
        baseline.append(b)
        means[qq,count]=a
        baseline_means[qq,count]=b
        count=count+1  
    qq=qq+1
    print("Mean",np.mean(one))
    print("Std Dev",np.std(one))
    print("Baseline Mean",np.mean(baseline))
    print("Baseline Std Dev",np.std(baseline))
answer=pd.DataFrame(test)
m_means=np.mean(means,axis=1)
b_means=np.mean(baseline_means,axis=1)
seq=np.array([1,5,10,20,50,90])
plt.plot(seq,m_means,'bo')
plt.plot(seq,m_means,'b-')
plt.plot(seq,b_means,'ro')
plt.plot(seq,b_means,'r-')
plt.xlabel("Train set size (Percentage)")
plt.ylabel("Zero-One Error (Percentage)")
plt.plot(seq,)
plt.axis([0,100,0,1])
plt.show()  
my_std=np.std(means,axis=1)
baseline_std=np.std(baseline_means,axis=1)
