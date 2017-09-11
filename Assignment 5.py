from PIL import Image
from itertools import repeat
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import math
import pandas as pd
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import sys

def read_dataset_raw(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            dataset.append(line.strip().split(','))
    return dataset

def extract_column(matrix, i): # matrix = list
    return [row[i] for row in matrix]

def extract_multi_values_from_index(matrix, i): # matrix = list
    return [row[i :] for row in matrix]

def print_image_from_pixel_feat_matrix(dataset, i):
    arr = np.array(dataset[i])
    arr2 = [int(k) for k in arr]
    arr2 = np.array(arr2)
    arr2.resize((28,28))
    im = Image.fromarray(arr2)
    im.show()
    #im.save("C://Users//tende//Desktop//DM Assg 5//img.png","PNG")

def print_image_from_pixel_PCA(arr):
    arr.resize((28,28))
    im = Image.fromarray(arr)
    im.show()
    
def plot_points_based_on_class_label(x,y,label):
    if label == '0':
        plt.scatter(x, y,c='#ff6347') # Tomato
    if label == '1':
        plt.scatter(x, y,c='#8a2be2') # Blue Violet
    if label == '2':
        plt.scatter(x, y,c='#ffd700') # Gold
    if label == '3':
        plt.scatter(x, y,c='#ff69b4') # Deep Pink
    if label == '4':
        plt.scatter(x, y,c='#00ced1') # Dark Turquoise
    if label == '5':
        plt.scatter(x, y,c='#228b22') # Forest Green
    if label == '6':
        plt.scatter(x, y,c='#4682b4') # Steel Blue
    if label == '7':
        plt.scatter(x, y,c='#00bfff') # Deep Sky Blue
    if label == '8':
        plt.scatter(x, y,c='#000080') # Navy
    if label == '9':
        plt.scatter(x, y,c='#8b7765') # Peach Puff 

def plot_n_random_points(num):
    for i in range(num):
        randValue = rnd.sample(range(0,19999),1) #--------------------------------- generates a random number between 0 and 19999
        a=randValue[0] #----------------------------------------------------------- converts that value to integer
        plot_points_based_on_class_label(xy_coord[a][0],xy_coord[a][1],class_label[a])
    plt.show()
    
def euclidean_distance(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def ssd(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)

def plot_cluster(cluster):
    for i in range(len(cluster)):
        for lab_val in cluster[i]:
            plot_points_based_on_class_label(lab_val[1][0],lab_val[1][1],lab_val[0])

def calculate_centroid_for_cluster(cluster):
    p = cluster
    arr = np.ones((len(p),2))
    for i in range(len(p)):
        arr[i,0] = float(p[i][1][0])
        arr[i,1] = float(p[i][1][1]) 
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return [sum_x/length, sum_y/length]

def k_means(k,xy_coord):
    centroids = [[]*2]*k
    randValue = rnd.sample(range(0,19999),k)     
    for i in range(k):
        centroids[i] = xy_coord[randValue[i]]
    for count in repeat(None, 50):
        cl = []
        cluster = [[] for _ in range(k)]
        for i in range(len(xy_coord)): # every point from the dataset
            diff = [0]*k
            for j in range(len(centroids)): # is compared that which centroid its closest to! :)
                diff[j] = euclidean_distance(float(xy_coord[i][0]),float(centroids[j][0]),float(xy_coord[i][1]),float(centroids[j][1]))
            min_index = diff.index(min(diff))
            cluster[min_index].append([class_label[i]]+[xy_coord[i]])   
            cl.append(min_index)
        for p in range(len(cluster)):
            centroids[p] = calculate_centroid_for_cluster(cluster[p])
    return cluster,centroids, cl
    
def wcssd(cluster,centroids,k):
    wc_ssd = 0.0
    for i in range(0,k): # for all clusters 15
        x1 = float(centroids[i][0])
        y1 = float(centroids[i][1]) 
        sub_clus = cluster[i]
        for p in range(len(sub_clus)): 
            x2 = float(sub_clus[p][1][0])
            y2 = float(sub_clus[p][1][1])
            wc_ssd += ssd(x1,x2,y1,y2)
    return wc_ssd

def sample_dataset_with_k_values(class_label,k):
    list0 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "0"],k)
    list1 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "1"],k)
    list2 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "2"],k)
    list3 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "3"],k)
    list4 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "4"],k)
    list5 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "5"],k)
    list6 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "6"],k)
    list7 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "7"],k)
    list8 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "8"],k)
    list9 = rnd.sample( [i for i, x in enumerate(class_label) if x[0] == "9"],k)
    
    indices = list0 + list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8 + list9 
    subset = []    
    for i in range(len(indices)):
        subset.append([class_label[indices[i]]]+[xy_coord[i]]) 
    return subset
    
def compute_silholutte(cluster,k):
    silhoutte = 0.0
    for p in range(0,k): # for every cluster
        distance = 0.0 # consider the initial distance to be 0
        newList = [each_list[i] for i in [1] for each_list in cluster[p]] # extracted the x y co-ords into another list
        self_arr = np.asarray(newList) # converted that into a 2d array...umm string :|
        self_arr = self_arr.astype(np.float) # converted into array of float :)
        for i in range(0,np.size(self_arr, axis =0)): # you select a point from the self array
            distance_matrix = sc.cdist(self_arr,[self_arr[i]]) # and find its distance with every other point in that array i.e. that cluster
            distance = sum(distance_matrix) # so basically its an array of distances
            a = distance/len(cluster[p]) # taking sum of all thosedistance values
            min_dist = 9999999.99 # set some max dummy distance between that point to every other point for comparison
            d = 0.0
            b = 0.0
            for m in range(0,k): # now that you have that one point, you traverse over different clusters
                if m!= p: # except for itself, since you already have its distance
                    avg_dist = 0.0 # set the average distance to 0 i.e. b
                    newList2 = [each_list[x] for x in [1] for each_list in cluster[m]] # you pickup one cluster
                    foreign_arr = np.asarray(newList2) # convert it into an array...umm..of strings, again :|
                    foreign_arr = foreign_arr.astype(np.float) # and then into a float array :)
                    d = sc.cdist(foreign_arr,[self_arr[i]]) # and then find the distance from that self_Arr point to every point in this array i.e. cluster :)
                    sum_d = sum(d)
                    if sum_d<min_dist:
                        min_dist = sum_d
                        avg_dist = sum_d/np.size(foreign_arr,axis=0)
                        b = avg_dist
            silhoutte += (b-a)/max(b,a) 
    silhoutte /= 20000    
    return silhoutte
    
def compute_NMI(original_labels,k,dataset,cl_kmeans):
    distribution=np.zeros([len(set(original_labels)),k])
    #print(np.shape(distribution))
    for i in range(len(set(original_labels))):
        for j in range(k):
            number=[final for final in [initial for initial in range(len(dataset)) if cl_kmeans[initial]==j] if original_labels[final]==i]
            distribution[i,j]=len(number)
    distribution=distribution/len(original_labels)
    for i in range(distribution.shape[0]):
        for j in range(distribution.shape[1]):
            if distribution[i,j]==0:
                distribution[i,j]=0.000000001
    
    numerator=np.zeros([len(set(original_labels)),k])
    for i in range(numerator.shape[0]):
        for j in range(numerator.shape[1]):
            numerator[i,j]=distribution[i,j]*((math.log((distribution[i,j])/(np.sum(distribution[i,:])*np.sum(distribution[:,j])))))
    num=np.sum(numerator)
    #return numerator
    c_dist=np.sum(distribution,axis=1).reshape(distribution.shape[0],1)
    for i in range(len(c_dist)):
        c_dist[i,0]=c_dist[i,0]*math.log(c_dist[i,0])
    denom_1=-(np.sum(c_dist))
    g_dist=np.sum(distribution,axis=0).reshape(distribution.shape[1],1)
    for i in range(len(g_dist)):
        g_dist[i,0]=g_dist[i,0]*math.log(g_dist[i,0])
    denom_2=-(np.sum(g_dist))
    
    nmi=num/(denom_1+denom_2)
    return 2*nmi

def label_count(cluster,label):
    p = []
    for i in range(len(cluster)):
        p.append(cluster[i][0])
    return p.count(str(label))

def sumColumn(m, column):
    total = 0
    for row in range(len(m)):
        total += m[row][column]
    return total

def Silhoutte2(k,class_labels,dataset):
    sil=0.0
    for cluster_number in range(k):
        corresponding_vector=dataset[[x for x in range(len(dataset)) if class_labels[x]==cluster_number],:]
        a_i=sc.cdist(corresponding_vector,corresponding_vector)
        a_i=np.mean(a_i,axis=1).reshape(len(a_i),1)
        #print(cluster_number)
        
        rest_list=np.empty([len(corresponding_vector),1])
        corr_rest=[x for x in range(k) if x!=cluster_number]
        for rest in corr_rest:
           # print(corr_rest)
            corresponding_rest=dataset[[x for x in range(len(dataset)) if class_labels[x]==rest],:]
            b_i=sc.cdist(corresponding_vector,corresponding_rest)
         #   print(np.shape(b_i))
            b_i=np.array(np.mean(b_i,axis=1)).reshape(len(b_i),1)
            rest_list=np.append(rest_list,b_i,axis=1)
        #np.delete(rest_list,[0],axis=1)
        #print(np.shape(rest_list))
        b_i=np.min(rest_list[:,1:],axis=1)
        for i in range(a_i.shape[0]):
            sil+=((b_i[i]-a_i[i,0])/max(set([b_i[i],a_i[i,0]])))
    return sil/(dataset.shape[0])

def W_SSD2(k,class_labels,data,centroids):
    WC_SSD_List_Final=list()                    #Winthin Sum of Square Distance Starts here::::::
    for cluster in range(k):
        WC_SSD_list=list()
        corresponding_cluster_points=[y for y in range(len(class_labels)) if class_labels[y]==cluster]
        for i in corresponding_cluster_points:    
            WC_SSD_list.append(np.linalg.norm(data[i]-centroids[cluster]))
        WC_SSD_List_Final.append(np.sum(WC_SSD_list))
    wssd=(np.sum(WC_SSD_List_Final))
    return wssd 


#digits_raw = read_dataset_raw('C://Users//tende//Desktop//DM Assg 5//digits-raw.csv')
#class_label= extract_column(digits_raw,1)
#pixel_features = extract_multi_values_from_index(digits_raw,2)

digits_embedding=read_dataset_raw(sys.argv[1])
class_label= extract_column(digits_embedding,1)
xy_coord = extract_multi_values_from_index(digits_embedding,2)

k=int(sys.argv[2])
'''
#------------------------------------------------------------------------------ A.1 EXPLORATION : REPRESENT A GREY SCALE IMAGE FROM PIX MATRIX
print_image_from_pixel_feat_matrix(pixel_features,1) #------------------------- PRINTS 0
print_image_from_pixel_feat_matrix(pixel_features,3) #------------------------- PRINTS 1
print_image_from_pixel_feat_matrix(pixel_features,5) #------------------------- PRINTS 2
print_image_from_pixel_feat_matrix(pixel_features,7) #------------------------- PRINTS 3
print_image_from_pixel_feat_matrix(pixel_features,2) #------------------------- PRINTS 4
print_image_from_pixel_feat_matrix(pixel_features,0) #------------------------- PRINTS 5
print_image_from_pixel_feat_matrix(pixel_features,13) #------------------------ PRINTS 6
print_image_from_pixel_feat_matrix(pixel_features,15) #------------------------ PRINTS 7
print_image_from_pixel_feat_matrix(pixel_features,17) #------------------------ PRINTS 8
print_image_from_pixel_feat_matrix(pixel_features,4) #------------------------- PRINTS 9

#------------------------------------------------------------------------------ A.2 EXPLORATION : COLORING 1K RANDOM POINTS BASED ON THEIR CLASS LABELS
plot_n_random_points(1000)
'''
# ----------------------------------------------------------------------------- K MEANS

cluster,centroids,cl_kmeans = k_means(k,xy_coord)

#------------------------------------------------------------------------------ WC SSD
wc_ssd = wcssd(cluster,centroids,k)
print("SSD:",wc_ssd)
#------------------------------------------------------------------------------ SILHOLUTTE
silhoutte = compute_silholutte(cluster,k)

#NOTE: SILHOLUTTE TAKES LONGER THAN USUAL, THUS, FOR THE SAKE OF QUICK RESPONSE,I HAVE MAINTAINED
# A SUBSET OF MY CLUSTER OF ONLY 10000 ENTRIES, WHICH IS HALF THE DATASET, AND IS AS SHOWN BELOW

sub_coord = []
for i in range (0,10000):
    sub_coord.append(xy_coord[i])

sub_cluster,sub_centroids = k_means(k,sub_coord)
silhoutte = compute_silholutte(sub_cluster,k)

print("silhoutte for dataset of 10000:",silhoutte)
# OUTPUT AT CONSOLE 17

#------------------------------------------------------------------------------ NMI
nmi = compute_NMI(class_label,k,np.array(xy_coord),cl_kmeans)
print("NMI:",nmi)
'''
# ----------------------------------------------------------------------------- HIERARCHICAL

digits_raw=pd.read_csv("C://Users//tende//Desktop//DM Assg 5//digits-raw.csv",header=None)
digits_embedding=pd.read_csv(sys.argv[1],header=None,names=("image_id","class_label",'embedding_features_1','embedding_features_2'))


subsample_0=digits_embedding[digits_embedding.class_label==0].sample(10)
subsample_1=digits_embedding[digits_embedding.class_label==1].sample(10)
subsample_2=digits_embedding[digits_embedding.class_label==2].sample(10)
subsample_3=digits_embedding[digits_embedding.class_label==3].sample(10)
subsample_4=digits_embedding[digits_embedding.class_label==4].sample(10)
subsample_5=digits_embedding[digits_embedding.class_label==5].sample(10)
subsample_6=digits_embedding[digits_embedding.class_label==6].sample(10)
subsample_7=digits_embedding[digits_embedding.class_label==7].sample(10)
subsample_8=digits_embedding[digits_embedding.class_label==8].sample(10)
subsample_9=digits_embedding[digits_embedding.class_label==9].sample(10)

subsample=subsample_0.append([subsample_1,subsample_2,subsample_3,subsample_4,subsample_5,subsample_6,subsample_7,subsample_8,subsample_9])
subsample.reindex(np.random.permutation(subsample.index))
subsample_points=np.array(subsample[[2,3]])

subsample_centroids=np.array(subsample[[1,2,3]])
single_linkage=linkage(subsample_points,method="single")
dendrogram(single_linkage)

#PART C2:
complete_linkage=linkage(subsample_points,method="complete")
average_linkage=linkage(subsample_points,method="average")
dendrogram(complete_linkage)
dendrogram(average_linkage)

#PART C3:
single_k,complete_k,average_k=list(),list(),list()
single_ssd,complete_ssd,average_ssd=list(),list(),list()
single_sil,complete_sil,average_sil=list(),list(),list()
single_nml,complete_nml,average_nml=list(),list(),list()

single_link_fcluster=fcluster(single_linkage,10,'maxclust')
#plt.scatter(subsample_points[:,0],subsample_points[:,1],c=single_link_fcluster)

for k in [2,4,8,16,32]:# Single First
    
    single_k=fcluster(single_linkage,k,'maxclust')
    if single_k.shape != [len(single_k),1]:
        single_k=fcluster(single_linkage,k,'maxclust').reshape(len(single_k),1)
    centroid=list()
    ackerman=np.concatenate(single_k,axis=0)
    for i in set(ackerman):
        corresponding=[subsample_points[y,:] for y in range(len(single_k)) if subsample_centroids[y,0]==i]
        centroid.append(np.mean(corresponding,axis=0))
    single_ssd.append(W_SSD2(k,ackerman,subsample_points,centroid))
    single_sil.append(Silhoutte2(k,ackerman,subsample_points))
    single_nml.append(compute_NMI([subsample_centroids[y,0] for y in range(len(subsample_centroids))],k,subsample_points,single_k))
    


#------------------------------------------------------------------------------ NMI analysis
for k in [0,1,2,3,4]:
    k=3
    ana_cluster,ana_centroids,ana_cl = k_means(np.power(2,(k+1)),xy_coord)
    nmi = compute_NMI(class_label,np.power(2,(k+1)),np.array(xy_coord),ana_cl)
    print(nmi)

NMI ANA 2
list2467 = [i for i, x in enumerate(class_label) if x[0] == "2"] + [i for i, x in enumerate(class_label) if x[0] == "4"] + [i for i, x in enumerate(class_label) if x[0] == "6"] + [i for i, x in enumerate(class_label) if x[0] == "7"]
xy_coord_2467 = []
for i in range(len(list2467)):
    xy_coord_2467.append(xy_coord[list2467[i]])   
 
k=1
ana_cluster,ana_centroids,ana_cl = k_means(np.power(2,(k+1)),xy_coord)
nmi = compute_NMI(class_label,np.power(2,(k+1)),np.array(xy_coord_2467),ana_cl)
print(nmi)

#------------------------------------------------------------------------------ PCA
pixel_arr = np.asarray(pixel_features).astype(float)
# normalization, for some reason

mean_vector=np.mean(pixel_arr,axis=0).reshape(784,1) #This is the mean vector

#Normalization
digits_normalized=pixel_arr-np.transpose(mean_vector)

covar=np.cov(digits_normalized,rowvar=False)

eigen_values, eigen_vectors = np.linalg.eig(covar)

eigen_values =np.real(eigen_values)
eigen_vectors = np.real(eigen_vectors)

sorted_evalues = eigen_values.argsort()

eig_val_sorted=eigen_values[-sorted_evalues]
eig_vectors_sorted=eigen_vectors[:,-sorted_evalues]

req_sorted_evalues = eig_val_sorted[:10]
req_sorted_evectors = eig_vectors_sorted[:,0:10]
req_sorted_evectors = np.real(req_sorted_evectors)

PCA_processed = np.real(np.dot(np.transpose(req_sorted_evectors),np.transpose(digits_normalized)))

im0 = np.array(np.real(req_sorted_evectors)[:,0])
#print_image_from_pixel_PCA(im0)

im1 = np.array(np.real(req_sorted_evectors)[:,1])
#print_image_from_pixel_PCA(im1)

im2 = np.array(np.real(req_sorted_evectors)[:,2])
#print_image_from_pixel_PCA(im2)

im3 = np.array(np.real(req_sorted_evectors)[:,3])
#print_image_from_pixel_PCA(im3)

im4 = np.array(np.real(req_sorted_evectors)[:,4])
#print_image_from_pixel_PCA(im4)

im5 = np.array(np.real(req_sorted_evectors)[:,5])
#print_image_from_pixel_PCA(im5)

im6 = np.array(np.real(req_sorted_evectors)[:,6])
#print_image_from_pixel_PCA(im6)

im7 = np.array(np.real(req_sorted_evectors)[:,7])
#print_image_from_pixel_PCA(im7)

im8 = np.array(np.real(req_sorted_evectors)[:,8])
#print_image_from_pixel_PCA(im8)

im9 = np.array(np.real(req_sorted_evectors)[:,9])
#print_image_from_pixel_PCA(im9)

# Q3 and onwareds
eigen_vectors_3=req_sorted_evectors[:,0:2]
data_3=np.matrix(digits_normalized)*(eigen_vectors_3)

dig_embed=pd.read_csv("digits-embedding.csv",header=None,names=("image_id","class_label",'embedding_features_1','embedding_features_2'))
dig_embed=np.array(dig_embed)

x_1=[dig_embed[y,2] for y in range(len(dig_embed))]

y_1=[dig_embed[y,3] for y in range(len(dig_embed))]

x_2=[data_3[y,0] for y in range(len(data_3))]
y_2=[data_3[y,1] for y in range(len(data_3))]

cl=[dig_embed[y,1] for y in range(len(dig_embed))]


#plt.scatter(x_1,y_1,c=cl)
plt.scatter(x_2,y_2,c=cl)

for i in range(len(dig_embed)):
    dig_embed[i,2]=data_3[i,0]
    dig_embed[i,3]=data_3[i,1]


dig_embed_list=dig_embed.tolist()
#dig_embed[:,2]=data_3[:,0]


dig_embed_points=data_3.tolist()

wd_ssd_list=list()
sil_list=list()
nml_list=list()
for k in [2,4,8,16,32]:
    cl_kmeans=list()
    dig_embed_points=[dig_embed_list[i][2:] for i in range(len(dig_embed_list))]
    
    indexs_for_initial_centroids=sample(range(len(dig_embed_list)),k)
    
    centroids=[dig_embed_points[i] for i in indexs_for_initial_centroids] #Centroids have been initialized here
    
    
    for z in range(50):
        print(z)
        cl_kmeans=list()
        datapoints=np.array(dig_embed_points)
        centroid_array=np.array(centroids)
                                 
        for i in range(len(dig_embed_list)):  #This iterates over the 20,000 samples
            #print(i)
            distances=list()
            for j in range(len(centroids)):         #This iterates over the k number of centroids
                #print(j,"j")
                distances.append(np.linalg.norm(datapoints[i]-centroid_array[j])) #For each of the 20000 points, this generates a set of 10 distances
            cl_kmeans.append(np.argmin(distances))    #These are the class_labels
        
        for f in range(len(centroids)):
            corresponding_list=[y for y in range(len(cl_kmeans)) if cl_kmeans[y]==f]
            centroids[f][0]=np.mean([dig_embed_points[a][0] for a in corresponding_list])
            centroids[f][1]=np.mean([dig_embed_points[a][1] for a in corresponding_list])
        
    x=[dig_embed_list[i][2] for i in range(len(dig_embed_list))]
    y=[dig_embed_list[i][3] for i in range(len(dig_embed_list))]
    cl=[dig_embed_list[i][1] for i in range(len(dig_embed_list))]
    
    print("From Actual:")
    plt.scatter(x,y,c=cl)
    plt.show()
    
    print("From K-means:")
    plt.scatter(x,y,c=cl_kmeans)
    plt.show()
    
    w_ssd=W_SSD(k,cl_kmeans,datapoints,centroids)                   #This is W_SSD for the PART B1
    wd_ssd_list.append(w_ssd)   
    sil_coeff=Silhoutte_Final(k,cl_kmeans,datapoints)
    sil_list.append(sil_coeff)
    nml=NML([dig_embed_list[x][1] for x in range(len(dig_embed_list))],k,datapoints,cl_kmeans)
    nml_list.append(nml)

    
wd_ssd_list=np.array(wd_ssd_list)


seq=np.array([2,4,8,16,32])


plt.plot(seq,(wd_ssd_list),'b',label="WD SSD")
plt.plot(seq,(wd_ssd_list),'bx')
plt.xlabel("Clusters (Number)")
plt.ylabel("Validations")
plt.axis([2,32,0,max(wd_ssd_list)])
plt.legend()
plt.show()
#plt.errorbar(seq,np.mean(Decision_Tree,axis=1),yerr=Decision_Tree[:,11])
plt.plot(seq,(sil_list),'g',label="Silhoutte")
plt.plot(seq,(sil_list),'gx') 
#plt.errorbar(seq,np.mean(Bagged_Tree,axis=1),yerr=Bagged_Tree[:,11])
#plt.plot(seq,(nml_list),'r',label="NML")
#plt.plot(seq,(nml_list),'rx')
#plt.errorbar(seq,np.mean(RandomForest_Tree,axis=1),yerr=RandomForest_Tree[:,11])
#plt.plot(seq,np.mean(Boosted_Tree,axis=1),'ko')
#plt.errorbar(seq,np.mean(Boosted_Tree,axis=1),yerr=Boosted_Tree[:,11])
#plt.plot(seq,np.mean(SVM,axis=1),'yo')
#plt.errorbar(seq,np.mean(SVM,axis=1),yerr=SVM[:,11])
plt.xlabel("Clusters (Number)")
plt.ylabel("Validations")
plt.axis([2,32,0,max(sil_list)+0.2])
plt.legend()
plt.show()
        
#poggy's
digits_raw=pd.read_csv("C://Users//tende//Desktop//DM Assg 5//digits-raw.csv",header=None)


digits_raw=np.array(digits_raw)
digits=digits_raw[:,2:]
# done...pixel_array
mean_vector=np.mean(digits,axis=0).reshape(784,1) #This is the mean vector

#Normalization
digits_normalized=digits-np.transpose(mean_vector)

covariance_matrix=np.cov(digits_normalized,rowvar=False)

eig_values,eig_vectors=np.linalg.eig(covariance_matrix) #We have got the Eignen Values and Eigen Vectors

eig_vectors=np.real(eig_vectors)
eig_values_sorted=eig_values.argsort()
eig_val_sorted=eig_values[-eig_values_sorted]
eig_vectors_sorted=eig_vectors[:,-eig_values_sorted]

selected_eigen_vectors=eig_vectors_sorted[:,0:10]
selected_eigen_values=eig_val_sorted[:10]

selected_eig_vec=np.real(selected_eigen_vectors)
data_after_PCA=np.real(np.dot(selected_eigen_vectors.T,digits_normalized.T).T)

Image_0=np.array(np.real(selected_eigen_vectors)[:,0])
Image_0 = np.matrix(np.reshape(Image_0,[28,28])) 
cv2.imshow('0',Image_0)
'''

'''  
ssd_mean = np.mean(ssd_t)
#------------------------------------------------------------------------------ ANALYSIS
# debugging
w,h = 5,10
table = [[0 for x in range(w)] for y in range(h)] 
table = np.asarray(table).astype(float)
for k in [0,1,2,3,4]:
    for i in range(0,10):
        table[i][k] = i+k
             
#ssd ana...entire data-set
w,h = 5,10
ssd_table = [[0 for x in range(w)] for y in range(h)] 
ssd_table = np.asarray(ssd_table).astype(float)
for k in [0,1,2,3,4]:
    for i in range(0,10):
        ana_cluster,ana_centroids = k_means(np.power(2,(k+1)),xy_coord)
        ssd_table[i][k]= wcssd(ana_cluster, ana_centroids, np.power(2,(k+1)))

ssd_mean = np.mean(ssd_table, axis = 0)
ssd_std = np.std(ssd_table, axis = 0)

#ssd ana...2 4 6 7
list2467 = [i for i, x in enumerate(class_label) if x[0] == "2"] + [i for i, x in enumerate(class_label) if x[0] == "4"] + [i for i, x in enumerate(class_label) if x[0] == "6"] + [i for i, x in enumerate(class_label) if x[0] == "7"]
xy_coord_2467 = []
for i in range(len(list2467)):
    xy_coord_2467.append(xy_coord[list2467[i]])   

w,h = 5,10
ssd_table = [[0 for x in range(w)] for y in range(h)] 
ssd_table = np.asarray(ssd_table).astype(float)
for k in [0,1,2,3,4]:
    for i in range(0,10):
        ana_cluster,ana_centroids = k_means(np.power(2,(k+1)),xy_coord_2467)
        ssd_table[i][k]= wcssd(ana_cluster, ana_centroids, np.power(2,(k+1)))

ssd_mean = np.mean(ssd_table, axis = 0)
ssd_std = np.std(ssd_table, axis = 0)
#ssd ana...6 7
list67 = [i for i, x in enumerate(class_label) if x[0] == "6"] + [i for i, x in enumerate(class_label) if x[0] == "7"]
xy_coord_67 = []
for i in range(len(list67)):
    xy_coord_67.append(xy_coord[list67[i]])   

w,h = 5,10
ssd_table = [[0 for x in range(w)] for y in range(h)] 
ssd_table = np.asarray(ssd_table).astype(float)
for k in [0,1,2,3,4]:
    #print(k)
    for i in range(0,10):
        #print(i)
        ana_cluster,ana_centroids = k_means(np.power(2,(k+1)),xy_coord_67)
        ssd_table[i][k]= wcssd(ana_cluster, ana_centroids, np.power(2,(k+1)))
    #print("###################")
ssd_mean = np.mean(ssd_table, axis = 0)
ssd_std = np.std(ssd_table, axis = 0)
    
#silholutte ana...entire dataset
w,h = 5,10
slholutte_table = [[0 for x in range(w)] for y in range(h)] 
slholutte_table = np.asarray(slholutte_table).astype(float)
for k in [0,1,2,3,4]:
    for i in range(0,10):
        ana_cluster,ana_centroids = k_means(np.power(2,(k+1)),xy_coord)
        slholutte_table[i][k]= compute_silholutte(ana_cluster, np.power(2,(k+1)))

sil_mean = np.mean(slholutte_table, axis = 0)
sil_std = np.std(slholutte_table, axis = 0)

#silholutte ana...2 4 6 7
list2467 = [i for i, x in enumerate(class_label) if x[0] == "2"] + [i for i, x in enumerate(class_label) if x[0] == "4"] + [i for i, x in enumerate(class_label) if x[0] == "6"] + [i for i, x in enumerate(class_label) if x[0] == "7"]
xy_coord_2467 = []
for i in range(len(list2467)):
    xy_coord_2467.append(xy_coord[list2467[i]])    
    
w,h = 5,10
slholutte_table = [[0 for x in range(w)] for y in range(h)] 
slholutte_table = np.asarray(slholutte_table).astype(float)
for k in [0,1,2,3,4]:
    #print(k)
    for i in range(0,10):
        ana_cluster,ana_centroids = k_means(np.power(2,(k+1)),xy_coord_2467)
        slholutte_table[i][k]= compute_silholutte(ana_cluster, np.power(2,(k+1)))
        
sil_mean = np.mean(slholutte_table, axis = 0)
sil_std = np.std(slholutte_table, axis = 0)

#silholutte ana...6 7
list67 = [i for i, x in enumerate(class_label) if x[0] == "6"] + [i for i, x in enumerate(class_label) if x[0] == "7"]
xy_coord_67 = []
for i in range(len(list67)):
    xy_coord_67.append(xy_coord[list67[i]])  
    
w,h = 5,10
slholutte_table = [[0 for x in range(w)] for y in range(h)] 
slholutte_table = np.asarray(slholutte_table).astype(float)
for k in [0,1,2,3,4]:
    for i in range(0,10):
        ana_cluster,ana_centroids = k_means(np.power(2,(k+1)),xy_coord_67)
        slholutte_table[i][k] = compute_silholutte(ana_cluster, np.power(2,(k+1)))

sil_mean = np.mean(slholutte_table, axis = 0)
sil_std = np.std(slholutte_table, axis = 0)
'''
