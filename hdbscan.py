# hdbscan implementation

import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.manifold import TSNE
import seaborn as sns

#------------------------------

cols = ['Flow Duration', 'TotLen Fwd Pkts', 'Fwd Pkt Len Mean', 
        'Bwd Pkt Len Min', 'Bwd Pkt Len Std', 'Flow IAT Mean', 
        'Flow IAT Std', 'Flow IAT Min', 'Fwd IAT Mean', 
        'Fwd IAT Min', 'Bwd IAT Mean', 'Fwd PSH Flags', 
        'Fwd Pkts/s', 'Bwd Pkts/s', 'SYN Flag Cnt', 
        'ACK Flag Cnt', 'PSH Flag Cnt', 'Pkt Size Avg',
        'Subflow Fwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 
        'Active Mean','Active Min', 'Label']

#%%

df1 = pd.read_csv("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = {'Flow Duration' : float, 'TotLen Fwd Pkts' : float, 'Fwd Pkt Len Mean' : float, 
        'Bwd Pkt Len Min' : float, 'Bwd Pkt Len Std' : float, 'Flow IAT Mean' : float, 
        'Flow IAT Std' : float, 'Flow IAT Min' : float, 'Fwd IAT Mean' : float, 
        'Fwd IAT Min' : float, 'Bwd IAT Mean' : float, 'Fwd PSH Flags' : float, 
        'Fwd Pkts/s' : float, 'Bwd Pkts/s' : float, 'SYN Flag Cnt' : float, 
        'ACK Flag Cnt' : float, 'PSH Flag Cnt' : float, 'Pkt Size Avg' : float,
        'Subflow Fwd Byts' : float, 'Init Fwd Win Byts' : float, 'Init Bwd Win Byts' : float, 
        'Active Mean' : float,'Active Min' : float, 'Label' : object})

df2 = pd.read_csv("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = {'Flow Duration' : float, 'TotLen Fwd Pkts' : float, 'Fwd Pkt Len Mean' : float, 
        'Bwd Pkt Len Min' : float, 'Bwd Pkt Len Std' : float, 'Flow IAT Mean' : float, 
        'Flow IAT Std' : float, 'Flow IAT Min' : float, 'Fwd IAT Mean' : float, 
        'Fwd IAT Min' : float, 'Bwd IAT Mean' : float, 'Fwd PSH Flags' : float, 
        'Fwd Pkts/s' : float, 'Bwd Pkts/s' : float, 'SYN Flag Cnt' : float, 
        'ACK Flag Cnt' : float, 'PSH Flag Cnt' : float, 'Pkt Size Avg' : float,
        'Subflow Fwd Byts' : float, 'Init Fwd Win Byts' : float, 'Init Bwd Win Byts' : float, 
        'Active Mean' : float,'Active Min' : float, 'Label' : object})

data = pd.concat([df1,df2])
del df1,df2
df5 = pd.read_csv("Friday-23-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = {'Flow Duration' : float, 'TotLen Fwd Pkts' : float, 'Fwd Pkt Len Mean' : float, 
        'Bwd Pkt Len Min' : float, 'Bwd Pkt Len Std' : float, 'Flow IAT Mean' : float, 
        'Flow IAT Std' : float, 'Flow IAT Min' : float, 'Fwd IAT Mean' : float, 
        'Fwd IAT Min' : float, 'Bwd IAT Mean' : float, 'Fwd PSH Flags' : float, 
        'Fwd Pkts/s' : float, 'Bwd Pkts/s' : float, 'SYN Flag Cnt' : float, 
        'ACK Flag Cnt' : float, 'PSH Flag Cnt' : float, 'Pkt Size Avg' : float,
        'Subflow Fwd Byts' : float, 'Init Fwd Win Byts' : float, 'Init Bwd Win Byts' : float, 
        'Active Mean' : float,'Active Min' : float, 'Label' : object})
data=pd.concat([data,df5])
del df5
df6 = pd.read_csv("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = {'Flow Duration' : float, 'TotLen Fwd Pkts' : float, 'Fwd Pkt Len Mean' : float, 
        'Bwd Pkt Len Min' : float, 'Bwd Pkt Len Std' : float, 'Flow IAT Mean' : float, 
        'Flow IAT Std' : float, 'Flow IAT Min' : float, 'Fwd IAT Mean' : float, 
        'Fwd IAT Min' : float, 'Bwd IAT Mean' : float, 'Fwd PSH Flags' : float, 
        'Fwd Pkts/s' : float, 'Bwd Pkts/s' : float, 'SYN Flag Cnt' : float, 
        'ACK Flag Cnt' : float, 'PSH Flag Cnt' : float, 'Pkt Size Avg' : float,
        'Subflow Fwd Byts' : float, 'Init Fwd Win Byts' : float, 'Init Bwd Win Byts' : float, 
        'Active Mean' : float,'Active Min' : float, 'Label' : object})
data=pd.concat([data,df6])
del df6
df7 = pd.read_csv("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = {'Flow Duration' : float, 'TotLen Fwd Pkts' : float, 'Fwd Pkt Len Mean' : float, 
        'Bwd Pkt Len Min' : float, 'Bwd Pkt Len Std' : float, 'Flow IAT Mean' : float, 
        'Flow IAT Std' : float, 'Flow IAT Min' : float, 'Fwd IAT Mean' : float, 
        'Fwd IAT Min' : float, 'Bwd IAT Mean' : float, 'Fwd PSH Flags' : float, 
        'Fwd Pkts/s' : float, 'Bwd Pkts/s' : float, 'SYN Flag Cnt' : float, 
        'ACK Flag Cnt' : float, 'PSH Flag Cnt' : float, 'Pkt Size Avg' : float,
        'Subflow Fwd Byts' : float, 'Init Fwd Win Byts' : float, 'Init Bwd Win Byts' : float, 
        'Active Mean' : float,'Active Min' : float, 'Label' : object})
data=pd.concat([data,df7])
del df7
df9 = pd.read_csv("Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = {'Flow Duration' : float, 'TotLen Fwd Pkts' : float, 'Fwd Pkt Len Mean' : float, 
        'Bwd Pkt Len Min' : float, 'Bwd Pkt Len Std' : float, 'Flow IAT Mean' : float, 
        'Flow IAT Std' : float, 'Flow IAT Min' : float, 'Fwd IAT Mean' : float, 
        'Fwd IAT Min' : float, 'Bwd IAT Mean' : float, 'Fwd PSH Flags' : float, 
        'Fwd Pkts/s' : float, 'Bwd Pkts/s' : float, 'SYN Flag Cnt' : float, 
        'ACK Flag Cnt' : float, 'PSH Flag Cnt' : float, 'Pkt Size Avg' : float,
        'Subflow Fwd Byts' : float, 'Init Fwd Win Byts' : float, 'Init Bwd Win Byts' : float, 
        'Active Mean' : float,'Active Min' : float, 'Label' : object})
data=pd.concat([data,df9])
del df9

df3 = pd.read_csv("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = object)
df3.iloc[:, 0:23] = df3.iloc[:, 0:23].apply(pd.to_numeric, errors = 'coerce')
df3 = df3.dropna()

data = pd.concat([data,df3])
del df3

df4 = pd.read_csv("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = object)
df4.iloc[:, 0:23] = df4.iloc[:, 0:23].apply(pd.to_numeric, errors = 'coerce')
df4 = df4.dropna()

data=pd.concat([data,df4])
del df4

df8 = pd.read_csv("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", header = 0, usecols = cols, dtype = object)
df8.iloc[:, 0:23] = df8.iloc[:, 0:23].apply(pd.to_numeric, errors = 'coerce')
df8 = df8.dropna()

data=pd.concat([data,df8])
del df8

data = data.dropna() # drop not assigned values


data['Label'] = data['Label'].map(lambda x: 0 if 'Benign' in x else 1) # 0 is Benign, 1 is Malicious

rand = data.sample(frac=0.1, random_state=1) # sample 10% of the data randomly

del data



rows = rand.shape[0]
normal_rows = int(rows*0.6)
abnormal_rows = int(rows*0.4)


rand1 = rand[rand['Label'] == 1].sample(n = abnormal_rows) # take 40% of the data as malicious 
rand2 = rand[rand['Label'] == 0].sample(n = normal_rows) # take 60% of the data as benign

rand = pd.concat([rand1, rand2])

del (rand1, rand2)

#%%

X = rand.iloc[:, 0:23].values # X for testing purposes, should be output of classification threshold
y = rand.Label.to_frame().values # y for testing purposes, should be output of classification threshold

data_labels, label_counts = np.unique(rand.Label, return_counts = True)

#%%


#------------------------------




def metric(tp, tn, fp, fn):

    recall = tp/(tp + fn)
    
    precision = tp/(tp + fp)
    
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    
    f1 = 2*(recall*precision)/(recall + precision)
    
    return recall, precision, accuracy, f1



def getMetrics(labels, clusters, percent, y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for j in range(0, len(clusters)):
            cnt2 = 0
            cnt1 = 0
            for i in range(0, len(labels)):
                if y[i] != 'Benign' and labels[i] == clusters[j]:
                    cnt1 = cnt1 + 1 #abnormal
                elif y[i] == 'Benign' and labels[i] == clusters[j]:
                    cnt2 = cnt2 + 1 #normal
                    
            if cnt1 >= cnt2*percent: #more abnormal, cluster is abnormal
                #abnorm is positive
                tp = tp + cnt1 #cnt1 is number of abnormal
                fp = fp + cnt2 #cnt2 is number of normal
                print("Cluster", j, "\t-- Normal:\t", cnt2, "\tAbnormal:\t", cnt1, "\t-> ABNORMAL")
            else: #more normal, cluster is normal
                #norm is negative
                tn = tn + cnt2 #cnt2 is normal
                fn = fn + cnt1 #cnt1 is abnormal
                print("Cluster", j, "\t-- Normal:\t", cnt2, "\tAbnormal:\t", cnt1, "\t-> NORMAL")
            
    
    print("tn", tn, "tp", tp, "fn", fn, "fp", fp)
    
    recall, precision, accuracy, f1 = metric(tp, tn, fp, fn)
    
    print("recall", recall)
    print("precision", precision)
    print("accuracy", accuracy)
    print("F1 score", f1)



#------------------------------



clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10).fit(X) 
# X is the unlabelled data and Y is the list of labels

labels = clusterer.labels_

clusters = np.unique(labels)

getMetrics(labels, clusters, 0.8, Y) # obtain the accuracy, precision, recall, and F1-scores alongside the details of the clusters
# a cluster is malicious if the number of malicious labels within it is larger than 80% of the number of benign labels



#------------------------------



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

plt.scatter(*tsne_results.T) # data visualisation



#------------------------------



color_palette = sns.color_palette('Paired', len(X))
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*tsne_results.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25) # data visualisation with colored cluster memberships