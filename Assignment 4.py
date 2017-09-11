import sys
import string

from collections import Counter
import numpy as np
from random import randrange,sample


def zo_error(prediction,test):
    zero_error=0
    for i in range(len(test)):
        if prediction[i]!=test[i][0]:
            zero_error=zero_error+1
    return(zero_error/len(test))

def process_str(s):
    return s.translate(str.maketrans('','',string.punctuation)).lower().split()
    
    
def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), set(words)) )
    return dataset
    
    
def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

    
def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i
    
    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append( (item[0], vector) )

    return vectors
#______________________________________________________________________________________________________________________________vv_HERE_vv

def select_variable(traindata,p_value=None): #Output is a dictionary of "Index" and something else
    gini_list=list()                        #This is a list that saves gini_indexes for all 1000 variables
    if p_value==None:
        for i in range(len(traindata[0][1])):   #This iterates over each column (Variable)
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            gini_list.append(gini)          
        children=split(traindata,np.argmin(gini_list))  #Physical Split, not phantom
        return {"index":np.argmin(gini_list),"children":children}
    else:
        best_gini=10
        best_index=1000
        features=generate_feature_subset(traindata,int(p_value))
        for i in list(features):   #This iterates over each column (Variable)    <- This changed
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            if gini<best_gini:
                best_gini=gini
                best_index=i
        children=split(traindata,best_index)  #Physical Split, not phantom
        return {"index":best_index,"children":children}
     
def split(traindata,index):
    left=list()
    right=list()
    for i in range(len(traindata)):
        if traindata[i][1][index]==0:
            left.append(traindata[i])       #This has entire data rows
        else:
            right.append(traindata[i])      #This has entire data rows
    return left , right
        
def terminal(traindata):
    prediction=[traindata[y][0] for y in range(len(traindata))]
    positives=sum(prediction)
    negitives=len(prediction)-positives
    if positives<negitives:
        return (0)
    else:
        return (1)

def make_tree(dictionary,max_depth,min_split,current_depth,p_value=None):
    left,right=dictionary['children']
    del(dictionary['children'])
    if not left or not right:
        dictionary['left']=dictionary['right']=terminal(left+right)
        return
    if current_depth>=max_depth:
        dictionary['left'],dictionary['right']=terminal(left),terminal(right)
        return
    if len(left)<min_split:
        dictionary['left']=terminal(left)
    else:
        dictionary['left']=select_variable(left,p_value)
        make_tree(dictionary['left'],max_depth,min_split,current_depth+1)
    if len(right)<min_split:
        dictionary['right']=terminal(right)
    else:
        dictionary['right']=select_variable(right,p_value)
        make_tree(dictionary['right'],max_depth,min_split,current_depth+1)
        
#These are for DT mainly, and are also to be called for each of bagging and randomforest
def build_decisiontree(traindata,max_depth,min_split,p_value=None):#We have to make a dictionary, not an actual prediction.
    structure=select_variable(traindata,p_value)
    make_tree(structure,max_depth,min_split,1,p_value)
    return(structure)

def predict(tree,testfeaturematrix):  
    if testfeaturematrix[1][tree["index"]]==0:
        if isinstance(tree["left"],dict):
            return predict(tree["left"],testfeaturematrix)
        else:
            return tree["left"]
    if testfeaturematrix[1][tree["index"]]==1:
        if isinstance(tree["right"],dict):
            return predict(tree["right"],testfeaturematrix)
        else:
            return tree["right"] 

#The following are mainly for BAGGING:
def bootstrap(data):
    bootstrap=list()
    while len(bootstrap)<int(len(data)):
        i=randrange(len(data))
        bootstrap.append(data[i])
    return(bootstrap)
    
def bagging_prediction(list_of_trees,row):
    prediction_set=[predict(tree,row) for tree in list_of_trees]
    if sum(prediction_set)<float(len(prediction_set)/2):
        prediction=0
    else:
        prediction=1
    return(prediction)

def build_bagging(trainmatrix,max_depth,min_size,ntrees,testmatrix):
    trees=list()
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size)
        trees.append(single_tree)
    
    prediction=[bagging_prediction(trees,row) for row in testmatrix]   
    return(prediction)    

#The following are mainly for the RandomForest:
def generate_feature_subset(dataset,p_value=None):
    if p_value==None:    
        subset=sample(range(len(dataset[0][1])),(len(dataset[0][1]))**0.5)
    else:
        subset=sample(range(len(dataset[0][1])),p_value)
    return(subset)

def build_randomForest(trainmatrix,max_depth,min_size,ntrees,testmatrix,p_value=None):
    trees=list()
    if p_value==None:
        p_value=int(len(trainmatrix[0][1])**0.5)
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size,p_value)
        trees.append(single_tree)
    prediction=[bagging_prediction(trees,row) for row in testmatrix]                #We call bagging prediction, since just aggregating
    return prediction
    
#______________________________________________________________________________________________________________________________
def main():
    train_file=sys.argv[1]
    test_file=sys.argv[2]
    model=sys.argv[3]
    
    train=read_dataset(train_file)
    test=read_dataset(test_file)
    #top_ten = get_most_commons(train, skip=100, total=10)              #Dont need to print top ten words in this assignment
    common_words=get_most_commons(train,skip=100,total=1000)
    train_featurematrix=generate_vectors(train,common_words)
    test_featurematrix=generate_vectors(test,common_words)
    test_y=list()
    for i in range(len(test_featurematrix)):
        test_y.append(test_featurematrix[i][0])
    
    if model=="1":
        #___________________Decision Tree -PART(A)-___________________
        decisiontree=build_decisiontree(train_featurematrix,10,10)
        DT_pred=list()
        for i in range(len(test_featurematrix)):
            DT_pred.append(predict(decisiontree,test_featurematrix[i]))
        DT_error=zo_error(DT_pred,test_featurematrix)
        print("ZERO-ONE-LOSS-DT",DT_error)
    if model=="2":
        #___________________Bagged Tree -PART(B)-_____________________
        baggedtree=build_bagging(train_featurematrix,10,10,50,test_featurematrix)
        BT_error=zo_error(baggedtree,test_featurematrix)
        print("ZERO-ONE-LOSS-BT",BT_error)
    if model=="3":
        #___________________Random Forest -PART(C)-___________________
        randomForest=build_randomForest(train_featurematrix,10,10,50,test_featurematrix)
        RF_error=zo_error(randomForest,test_featurematrix)
        print("ZERO-ONE-LOSS-RF",RF_error)
main()





''' ANALYSIS _NORMAL 
import string
from collections import Counter
import numpy as np
import os
from random import randrange,shuffle,sample
import matplotlib.pyplot as plt

os.chdir("D:\Jennifer\HW\\4")


def zo_error(prediction,actual):
    zero_error=0
    for i in range(len(test)):
        if prediction[i]!=test[i][0]:
            zero_error=zero_error+1
    return(zero_error/len(test))

def process_str(s):
    return s.translate(str.maketrans('','',string.punctuation)).lower().split()
    
    
def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), set(words)) )
    return dataset
    
    
def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

    
def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i
    
    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append( (item[0], vector) )

    return vectors
#______________________________________________________________________________________________________________________________vv_HERE_vv

def select_variable(traindata,p_value=None): #Output is a dictionary of "Index" and something else
    gini_list=list()                        #This is a list that saves gini_indexes for all 1000 variables
    if p_value==None:
        for i in range(len(traindata[0][1])):   #This iterates over each column (Variable)
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            gini_list.append(gini)          
        children=split(traindata,np.argmin(gini_list))  #Physical Split, not phantom
        return {"index":np.argmin(gini_list),"children":children}
    else:
        best_gini=10
        best_index=1000
        features=generate_feature_subset(traindata,int(p_value))
        for i in list(features):   #This iterates over each column (Variable)    <- This changed
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            if gini<best_gini:
                best_gini=gini
                best_index=i
        children=split(traindata,best_index)  #Physical Split, not phantom
        return {"index":best_index,"children":children}
     
def split(traindata,index):
    left=list()
    right=list()
    for i in range(len(traindata)):
        if traindata[i][1][index]==0:
            left.append(traindata[i])       #This has entire data rows
        else:
            right.append(traindata[i])      #This has entire data rows
    return left , right
        
def terminal(traindata):
    prediction=[traindata[y][0] for y in range(len(traindata))]
    positives=sum(prediction)
    negitives=len(prediction)-positives
    if positives<negitives:
        return (0)
    else:
        return (1)

def make_tree(dictionary,max_depth,min_split,current_depth,p_value=None):
    left,right=dictionary['children']
    del(dictionary['children'])
    if not left or not right:
        dictionary['left']=dictionary['right']=terminal(left+right)
        return
    if current_depth>=max_depth:
        dictionary['left'],dictionary['right']=terminal(left),terminal(right)
        return
    if len(left)<min_split:
        dictionary['left']=terminal(left)
    else:
        dictionary['left']=select_variable(left,p_value)
        make_tree(dictionary['left'],max_depth,min_split,current_depth+1)
    if len(right)<min_split:
        dictionary['right']=terminal(right)
    else:
        dictionary['right']=select_variable(right,p_value)
        make_tree(dictionary['right'],max_depth,min_split,current_depth+1)
        
#These are for DT mainly, and are also to be called for each of bagging and randomforest
def build_decisiontree(traindata,max_depth,min_split,p_value=None):#We have to make a dictionary, not an actual prediction.
    structure=select_variable(traindata,p_value)
    make_tree(structure,max_depth,min_split,1,p_value)
    return(structure)

def predict(tree,testfeaturematrix):  
    if testfeaturematrix[1][tree["index"]]==0:
        if isinstance(tree["left"],dict):
            return predict(tree["left"],testfeaturematrix)
        else:
            return tree["left"]
    if testfeaturematrix[1][tree["index"]]==1:
        if isinstance(tree["right"],dict):
            return predict(tree["right"],testfeaturematrix)
        else:
            return tree["right"] 

#The following are mainly for BAGGING:
def bootstrap(data):
    bootstrap=list()
    while len(bootstrap)<int(len(data)):
        i=randrange(len(data))
        bootstrap.append(data[i])
    return(bootstrap)
    
def bagging_prediction(list_of_trees,row):
    prediction_set=[predict(tree,row) for tree in list_of_trees]
    if sum(prediction_set)<float(len(prediction_set)/2):
        prediction=0
    else:
        prediction=1
    return(prediction)

def build_bagging(trainmatrix,max_depth,min_size,ntrees,testmatrix):
    trees=list()
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size)
        trees.append(single_tree)
    
    prediction=[bagging_prediction(trees,row) for row in testmatrix]   
    return(prediction)    

#The following are mainly for the RandomForest:
def generate_feature_subset(dataset,p_value=None):
    if p_value==None:    
        subset=sample(range(len(dataset[0][1])),(len(dataset[0][1]))**0.5)
    else:
        subset=sample(range(len(dataset[0][1])),p_value)
    return(subset)

def build_randomForest(trainmatrix,max_depth,min_size,ntrees,testmatrix,p_value=None):
    trees=list()
    if p_value==None:
        p_value=int(len(trainmatrix[0][1])**0.5)
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size,p_value)
        trees.append(single_tree)
    prediction=[bagging_prediction(trees,row) for row in testmatrix]                #We call bagging prediction, since just aggregating
    return prediction
    
def svm(features, labels):
    # test sub-gradient SVM
    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T    
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1

    return w

def svm_pred(w, features):
    return np.where((features @ w) >= 0, 1, 0)    

def generate_vectors2(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)    
def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)
    
#______________________________________________________________________________________________________________________________

Decision_Tree=np.ones([4,12])
Bagged_Tree=np.ones([4,12])
RandomForest_Tree=np.ones([4,12])
SVM=np.ones([4,12])

x=read_dataset("yelp_data.csv")
pool=list(range(np.size(x,axis=0)))
splitlen=int(len(pool)/10)
set1 = list(sample(pool, splitlen))
pool=set(pool)-set(set1)
set2 = list(sample(pool, splitlen))
pool=set(pool)-set(set2)
set3 = list(sample(pool, splitlen))
pool=set(pool)-set(set3)
set4 = list(sample(pool, splitlen))
pool=set(pool)-set(set4)
set5 = list(sample(pool, splitlen))
pool=set(pool)-set(set5)
set6 = list(sample(pool, splitlen))
pool=set(pool)-set(set6)
set7 = list(sample(pool, splitlen))
pool=set(pool)-set(set7)
set8 = list(sample(pool, splitlen))
pool=set(pool)-set(set8)
set9 = list(sample(pool, splitlen))
pool=set(pool)-set(set9)
set10 = list(sample(pool, splitlen))
actualset=list([set1,set2,set3,set4,set5,set6,set7,set8,set9,set10])
j=0
for h in [0.025,0.05,0.125,0.25]:   #This is for Sample Sizes
    for k in range(10):
        print(k)
        test=[x[y] for y in actualset[k]]
        newactual=actualset[:k]+actualset[k+1:]
        train=list()
        for i in range(9):
            train=train+[x[y] for y in newactual[i]]
        train=sample(train,int(h*len(x)))
        
        common_words=get_most_commons(train,skip=100,total=1000)
        train_featurematrix=generate_vectors(train,common_words)
        test_featurematrix=generate_vectors(test,common_words)
        train_f,train_l=generate_vectors2(train, common_words)
        test_f, test_l = generate_vectors2(test, common_words)
        test_y=list()
        for i in range(len(test_featurematrix)):
            test_y.append(test_featurematrix[i][0])
        
        #___________________Decision Tree -PART(A)-___________________
        decisiontree=build_decisiontree(train_featurematrix,10,10)
        DT_pred=list()
        for i in range(len(test_featurematrix)):
            DT_pred.append(predict(decisiontree,test_featurematrix[i]))
        DT_error=zo_error(DT_pred,test_featurematrix)
        Decision_Tree[j,k]=DT_error
        
        #___________________Bagged Tree -PART(B)-_____________________
        baggedtree=build_bagging(train_featurematrix,10,10,50,test_featurematrix)
        BT_error=zo_error(baggedtree,test_featurematrix)
        Bagged_Tree[j,k]=BT_error
        
        #___________________Random Forest -PART(C)-___________________
        randomForest=build_randomForest(train_featurematrix,10,10,50,test_featurematrix)
        RF_error=zo_error(randomForest,test_featurematrix)
        RandomForest_Tree[j,k]=RF_error
        
        #___________________SVM________________________________________
        w = svm(train_f, train_l)
        test_pred = svm_pred(w, test_f)
        SVM_error=calc_error(test_pred, test_l)
        SVM[j,k]=SVM_error
        
    j=j+1
    print("-",j,"-")
    

Decision_Tree[:,10]=np.mean(Decision_Tree,axis=1)
Bagged_Tree[:,10]=np.mean(Bagged_Tree,axis=1)
RandomForest_Tree[:,10]=np.mean(RandomForest_Tree,axis=1)
SVM[:,10]=np.mean(SVM,axis=1)

Decision_Tree[:,11]=np.std(Decision_Tree[:,:-2],axis=1)
Bagged_Tree[:,11]=np.std(Bagged_Tree[:,:-2],axis=1)
RandomForest_Tree[:,11]=np.std(RandomForest_Tree[:,:-2],axis=1)
SVM[:,11]=np.std(SVM[:,:-2],axis=1)

seq=np.array([0.025,0.05,0.125,0.25])
Decision_Tree_means=Decision_Tree[:,10]
Bagged_Tree_means=Bagged_Tree[:,10]
RandomForest_Tree_means=RandomForest_Tree[:,10]
SVM_means=SVM[:,10]

plt.plot(seq,np.mean(Decision_Tree,axis=1),'bo')
plt.errorbar(seq,np.mean(Decision_Tree,axis=1),yerr=Decision_Tree[:,11])
plt.plot(seq,np.mean(Bagged_Tree,axis=1),'go') 
plt.errorbar(seq,np.mean(Bagged_Tree,axis=1),yerr=Bagged_Tree[:,11])
plt.plot(seq,np.mean(RandomForest_Tree,axis=1),'ro')
plt.errorbar(seq,np.mean(RandomForest_Tree,axis=1),yerr=RandomForest_Tree[:,11])
plt.plot(seq,np.mean(SVM,axis=1),'yo')
plt.errorbar(seq,np.mean(SVM,axis=1),yerr=SVM[:,11])
plt.xlabel("Features (Number)")
plt.ylabel("Zero-One Error (Percentage)")
plt.axis([0.023,0.27,0.07,0.40])
plt.show()


'''
'''ANALYSIS BONUS
import string
from collections import Counter
import numpy as np
import os
from random import randrange,shuffle,sample
import matplotlib.pyplot as plt

os.chdir("D:\Jennifer\HW\\4")


def zo_error(prediction,actual):
    zero_error=0
    for i in range(len(test)):
        if prediction[i]!=test[i][0]:
            zero_error=zero_error+1
    return(zero_error/len(test))

def process_str(s):
    return s.translate(str.maketrans('','',string.punctuation)).lower().split()
    
    
def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), set(words)) )
    return dataset
    
    
def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

    
def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i
    
    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append( (item[0], vector) )

    return vectors
#______________________________________________________________________________________________________________________________vv_HERE_vv

def select_variable(traindata,p_value=None): #Output is a dictionary of "Index" and something else
    gini_list=list()                        #This is a list that saves gini_indexes for all 1000 variables
    if p_value==None:
        for i in range(len(traindata[0][1])):   #This iterates over each column (Variable)
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            gini_list.append(gini)          
        children=split(traindata,np.argmin(gini_list))  #Physical Split, not phantom
        return {"index":np.argmin(gini_list),"children":children}
    else:
        best_gini=10
        best_index=1000
        features=generate_feature_subset(traindata,int(p_value))
        for i in list(features):   #This iterates over each column (Variable)    <- This changed
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            if gini<best_gini:
                best_gini=gini
                best_index=i
        children=split(traindata,best_index)  #Physical Split, not phantom
        return {"index":best_index,"children":children}
     
def split(traindata,index):
    left=list()
    right=list()
    for i in range(len(traindata)):
        if traindata[i][1][index]==0:
            left.append(traindata[i])       #This has entire data rows
        else:
            right.append(traindata[i])      #This has entire data rows
    return left , right
        
def terminal(traindata):
    prediction=[traindata[y][0] for y in range(len(traindata))]
    positives=sum(prediction)
    negitives=len(prediction)-positives
    if positives<negitives:
        return (0)
    else:
        return (1)

def make_tree(dictionary,max_depth,min_split,current_depth,p_value=None):
    left,right=dictionary['children']
    del(dictionary['children'])
    if not left or not right:
        dictionary['left']=dictionary['right']=terminal(left+right)
        return
    if current_depth>=max_depth:
        dictionary['left'],dictionary['right']=terminal(left),terminal(right)
        return
    if len(left)<min_split:
        dictionary['left']=terminal(left)
    else:
        dictionary['left']=select_variable(left,p_value)
        make_tree(dictionary['left'],max_depth,min_split,current_depth+1)
    if len(right)<min_split:
        dictionary['right']=terminal(right)
    else:
        dictionary['right']=select_variable(right,p_value)
        make_tree(dictionary['right'],max_depth,min_split,current_depth+1)
        
#These are for DT mainly, and are also to be called for each of bagging and randomforest
def build_decisiontree(traindata,max_depth,min_split,p_value=None):#We have to make a dictionary, not an actual prediction.
    structure=select_variable(traindata,p_value)
    make_tree(structure,max_depth,min_split,1,p_value)
    return(structure)

def predict(tree,testfeaturematrix):  
    if testfeaturematrix[1][tree["index"]]==0:
        if isinstance(tree["left"],dict):
            return predict(tree["left"],testfeaturematrix)
        else:
            return tree["left"]
    if testfeaturematrix[1][tree["index"]]==1:
        if isinstance(tree["right"],dict):
            return predict(tree["right"],testfeaturematrix)
        else:
            return tree["right"] 

#The following are mainly for BAGGING:
def bootstrap(data):
    bootstrap=list()
    while len(bootstrap)<int(len(data)):
        i=randrange(len(data))
        bootstrap.append(data[i])
    return(bootstrap)
    
def bagging_prediction(list_of_trees,row):
    prediction_set=[predict(tree,row) for tree in list_of_trees]
    if sum(prediction_set)<float(len(prediction_set)/2):
        prediction=0
    else:
        prediction=1
    return(prediction)

def build_bagging(trainmatrix,max_depth,min_size,ntrees,testmatrix):
    trees=list()
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size)
        trees.append(single_tree)
    
    prediction=[bagging_prediction(trees,row) for row in testmatrix]   
    return(prediction)    

#The following are mainly for the RandomForest:
def generate_feature_subset(dataset,p_value=None):
    if p_value==None:    
        subset=sample(range(len(dataset[0][1])),(len(dataset[0][1]))**0.5)
    else:
        subset=sample(range(len(dataset[0][1])),p_value)
    return(subset)

def build_randomForest(trainmatrix,max_depth,min_size,ntrees,testmatrix,p_value=None):
    trees=list()
    if p_value==None:
        p_value=int(len(trainmatrix[0][1])**0.5)
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size,p_value)
        trees.append(single_tree)
    prediction=[bagging_prediction(trees,row) for row in testmatrix]                #We call bagging prediction, since just aggregating
    return prediction
    
def svm(features, labels):
    # test sub-gradient SVM
    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T    
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1

    return w

def svm_pred(w, features):
    return np.where((features @ w) >= 0, 1, 0)    

def generate_vectors2(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)    
def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)
    
#______________________________________________________________________________________________________________________________

Decision_Tree=np.ones([4,12])
Bagged_Tree=np.ones([4,12])
RandomForest_Tree=np.ones([4,12])
SVM=np.ones([4,12])
Boosted_Tree=np.ones([4,12])

x=read_dataset("yelp_data.csv")
pool=list(range(np.size(x,axis=0)))
splitlen=int(len(pool)/10)
set1 = list(sample(pool, splitlen))
pool=set(pool)-set(set1)
set2 = list(sample(pool, splitlen))
pool=set(pool)-set(set2)
set3 = list(sample(pool, splitlen))
pool=set(pool)-set(set3)
set4 = list(sample(pool, splitlen))
pool=set(pool)-set(set4)
set5 = list(sample(pool, splitlen))
pool=set(pool)-set(set5)
set6 = list(sample(pool, splitlen))
pool=set(pool)-set(set6)
set7 = list(sample(pool, splitlen))
pool=set(pool)-set(set7)
set8 = list(sample(pool, splitlen))
pool=set(pool)-set(set8)
set9 = list(sample(pool, splitlen))
pool=set(pool)-set(set9)
set10 = list(sample(pool, splitlen))
actualset=list([set1,set2,set3,set4,set5,set6,set7,set8,set9,set10])
j=0
for h in [5,10,15,20]:   #This is for Sample Sizes
    for k in range(10):
        print(k)
        test=[x[y] for y in actualset[k]]
        newactual=actualset[:k]+actualset[k+1:]
        train=list()
        for i in range(9):
            train=train+[x[y] for y in newactual[i]]
        train=sample(train,int(0.25*len(x)))
        
        common_words=get_most_commons(train,skip=100,total=1000)
        train_featurematrix=generate_vectors(train,common_words)
        test_featurematrix=generate_vectors(test,common_words)
        train_f,train_l=generate_vectors2(train, common_words)
        test_f, test_l = generate_vectors2(test, common_words)
        test_y=list()
        for i in range(len(test_featurematrix)):
            test_y.append(test_featurematrix[i][0])
        
        #___________________Decision Tree -PART(A)-___________________
        decisiontree=build_decisiontree(train_featurematrix,10,10)
        DT_pred=list()
        for i in range(len(test_featurematrix)):
            DT_pred.append(predict(decisiontree,test_featurematrix[i]))
        DT_error=zo_error(DT_pred,test_featurematrix)
        Decision_Tree[j,k]=DT_error
        
        #___________________Bagged Tree -PART(B)-_____________________
        baggedtree=build_bagging(train_featurematrix,h,10,50,test_featurematrix)
        BT_error=zo_error(baggedtree,test_featurematrix)
        Bagged_Tree[j,k]=BT_error
        
        #___________________Random Forest -PART(C)-___________________
        randomForest=build_randomForest(train_featurematrix,h,10,50,test_featurematrix)
        RF_error=zo_error(randomForest,test_featurematrix)
        RandomForest_Tree[j,k]=RF_error
        
        #__________________Boosting___________________________________
        trainmatrix_y=np.zeros([len(train_featurematrix),1])
        for i in range(len(train_featurematrix)):
            for j in range(len(train_featurematrix)):
                trainmatrix_y[i,0]=train_featurematrix[i][0]
        
        testmatrix_y=np.zeros([len(test_featurematrix),1])
        for i in range(len(test_featurematrix)):
            for j in range(len(test_featurematrix)):
                testmatrix_y[i,0]=test_featurematrix[i][0]
        
        trainmatrix=np.zeros([len(train_featurematrix),len(train_featurematrix[1][1])])
        for i in range(len(train_featurematrix)):
            for j in range(len(train_featurematrix[1][1])):
                trainmatrix[i,j]=train_featurematrix[i][1][j]
        
        testmatrix=np.zeros([len(test_featurematrix),len(test_featurematrix[1][1])])
        for i in range(len(test_featurematrix)):
            for j in range(len(test_featurematrix[1][1])):
                testmatrix[i,j]=test_featurematrix[i][1][j]
        
        train_final=np.concatenate((trainmatrix,trainmatrix_y),axis=1)
        test_final=np.concatenate((testmatrix,testmatrix_y),axis=1)
        
        
        newtrain=np.array(trainmatrix)
        train_y=np.array(trainmatrix_y)
        
        n=len(train_featurematrix)
        weights=np.ones([n,1])
        weights=np.array(np.divide(weights,n))
        rules=list()
        alpha=list()
        import math
        
        for i in range(50):
            print(i)
            newtrain=np.multiply(weights,newtrain)
            withlabels=np.append(newtrain,train_y,axis=1)
            stump=build_decisiontree(withlabels,1,100000000)
            DT_pred=list()
            for i in range(len(newtrain)):
                DT_pred.append(predict(stump,newtrain[i]))
            errors=([DT_pred[i] != train_y[i] for i in range(len(newtrain))])
            for i in range(len(errors)):
                if errors[i]==True:
                    errors[i]=1
                else:
                    errors[i]=0
            e=(errors*np.asarray(weights).T).sum()
            #SOMETHING HERE
            alpha_t = 0.5 * math.log((1-e)/e)
            w=np.zeros([n,1])
            for i in range(n):
                if errors[i] == 1:
                    w[i,0] = weights[i,0] * math.exp(alpha_t)
                else: 
                    w[i,0] = weights[i,0] * math.exp(-alpha_t)
            weights=np.array(w/w.sum())
            rules.append(stump)
            alpha.append(alpha_t)
        
        #___________________SVM________________________________________
        w = svm(train_f, train_l)
        test_pred = svm_pred(w, test_f)
        SVM_error=calc_error(test_pred, test_l)
        SVM[j,k]=SVM_error
        
    j=j+1
    print("-",j,"-")
    

Decision_Tree[:,10]=np.mean(Decision_Tree,axis=1)
Bagged_Tree[:,10]=np.mean(Bagged_Tree,axis=1)
RandomForest_Tree[:,10]=np.mean(RandomForest_Tree,axis=1)
SVM[:,10]=np.mean(SVM,axis=1)
Boosted_Tree[:,10]=np.mean(Boosted_Tree,axis=1)

Decision_Tree[:,11]=np.std(Decision_Tree[:,:-2],axis=1)
Bagged_Tree[:,11]=np.std(Bagged_Tree[:,:-2],axis=1)
RandomForest_Tree[:,11]=np.std(RandomForest_Tree[:,:-2],axis=1)
SVM[:,11]=np.std(SVM[:,:-2],axis=1)

seq=np.array([5,10,15,20])
Decision_Tree_means=Decision_Tree[:,10]
Bagged_Tree_means=Bagged_Tree[:,10]
RandomForest_Tree_means=RandomForest_Tree[:,10]
SVM_means=SVM[:,10]
Boosted_means=Boosted_Tree[:,10]

plt.plot(seq,np.mean(Decision_Tree,axis=1),'bo')
plt.errorbar(seq,np.mean(Decision_Tree,axis=1),yerr=Decision_Tree[:,11])
plt.plot(seq,np.mean(Bagged_Tree,axis=1),'go') 
plt.errorbar(seq,np.mean(Bagged_Tree,axis=1),yerr=Bagged_Tree[:,11])
plt.plot(seq,np.mean(RandomForest_Tree,axis=1),'ro')
plt.errorbar(seq,np.mean(RandomForest_Tree,axis=1),yerr=RandomForest_Tree[:,11])
plt.plot(seq,np.mean(Boosted_Tree,axis=1),'ko')
plt.errorbar(seq,np.mean(Boosted_Tree,axis=1),yerr=Boosted_Tree[:,11])
plt.plot(seq,np.mean(SVM,axis=1),'yo')
plt.errorbar(seq,np.mean(SVM,axis=1),yerr=SVM[:,11])
plt.xlabel("Tree Depth (Number)")
plt.ylabel("Zero-One Error (Percentage)")
plt.axis([5,20,0,0.40])
plt.show()












'''