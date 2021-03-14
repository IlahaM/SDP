#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df1 = pd.read_csv("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df2 = pd.read_csv("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df3 = pd.read_csv("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df4 = pd.read_csv("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df5 = pd.read_csv("Friday-23-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df6 = pd.read_csv("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df8 = pd.read_csv("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df9 = pd.read_csv("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
df10 = pd.read_csv("Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})

frames = [df1, df2, df3, df4, df5, df6, df8, df9, df10]
df = pd.concat(frames, sort=False)

print(df.shape)

df = df.dropna()

print(df.shape)

print(df.groupby('Label').size())


p1 = df[1:1500000]
p2 = df[2200000:3966760]
p3 = df[4500000:4800000]
p4 = df[5500000:6000000]
p5 = df[6900000:8261241]

parts = [p1, p2, p3, p4, p5]
df = pd.concat(parts, sort=False)

print(df.shape)
print(df.groupby('Label').size())

del df1, df2, df3, df4, df5, df6, df8, df9, df10


y = df['Label']

features = list(zip(df['Bwd Pkt Len Min'].astype(np.float64), df['Subflow Fwd Byts'].astype(np.float64),
                    df['TotLen Fwd Pkts'].astype(np.float64), df['Fwd Pkt Len Mean'].astype(np.float64),
                    df['Bwd Pkt Len Std'].astype(np.float64), df['Flow IAT Min'].astype(np.float64),
                    df['Fwd IAT Min'].astype(np.float64),     df['Flow IAT Mean'].astype(np.float64),
                    df['Flow Duration'].astype(np.float64),   df['Flow IAT Std'].astype(np.float64),
                    df['Active Min'].astype(np.float64),      df['Active Mean'].astype(np.float64),
                    df['Fwd IAT Min'].astype(np.float64),     df['Bwd IAT Mean'].astype(np.float64),
                    df['Fwd IAT Mean'].astype(np.float64),    df['Init Fwd Win Byts'].astype(np.float64),
                    df['ACK Flag Cnt'].astype(np.float64),    df['Fwd PSH Flags'].astype(np.float64),  
                    df['SYN Flag Cnt'].astype(np.float64),    df['Fwd Pkts/s'].astype(np.float64),
                    df['Bwd Pkts/s'].astype(np.float64),      df['Init Bwd Win Byts'].astype(np.float64), 
                    df['PSH Flag Cnt'].astype(np.float64),    df['Pkt Size Avg'].astype(np.float64)))


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.1, shuffle = True) # 70% training and 30% test

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors = 9, n_jobs = -1)

#Train the model using the training sets
knn.fit(X_train, y_train)

df = pd.read_csv("Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})

print(df.shape)

df = df.dropna()

print(df.shape)

print(df.groupby('Label').size())

df = df[1:1600000]

print(df.groupby('Label').size())


y = df['Label']

features = list(zip(df['Bwd Pkt Len Min'].astype(np.float64), df['Subflow Fwd Byts'].astype(np.float64),
                    df['TotLen Fwd Pkts'].astype(np.float64), df['Fwd Pkt Len Mean'].astype(np.float64),
                    df['Bwd Pkt Len Std'].astype(np.float64), df['Flow IAT Min'].astype(np.float64),
                    df['Fwd IAT Min'].astype(np.float64),     df['Flow IAT Mean'].astype(np.float64),
                    df['Flow Duration'].astype(np.float64),   df['Flow IAT Std'].astype(np.float64),
                    df['Active Min'].astype(np.float64),      df['Active Mean'].astype(np.float64),
                    df['Fwd IAT Min'].astype(np.float64),     df['Bwd IAT Mean'].astype(np.float64),
                    df['Fwd IAT Mean'].astype(np.float64),    df['Init Fwd Win Byts'].astype(np.float64),
                    df['ACK Flag Cnt'].astype(np.float64),    df['Fwd PSH Flags'].astype(np.float64),  
                    df['SYN Flag Cnt'].astype(np.float64),    df['Fwd Pkts/s'].astype(np.float64),
                    df['Bwd Pkts/s'].astype(np.float64),      df['Init Bwd Win Byts'].astype(np.float64), 
                    df['PSH Flag Cnt'].astype(np.float64),    df['Pkt Size Avg'].astype(np.float64)))

y_pred = knn.predict(features)
y_pred_prob = knn.predict_proba(features)
    

x_not_predicted = []
x_not_predicted_index_test = []
y_my_pred_index_test = []
not_pred_indexes = []
y_test_new = []
y_not_pred = []


x_pred = []
y_pred_needed = []
y_test_needed = []


for x in y:
    y_test_new.append(x)
    
    
for i in range(len(features)):
    if max(y_pred_prob[i]) < 0.8:
         x_not_predicted.append(features[i])              
         y_not_pred.append(y_test_new[i])
    else:
         x_pred.append(features[i])
         y_pred_needed.append(y_pred[i])
         y_test_needed.append(y_test_new[i])
		 
         

# Results for data passing threshold
print("Accuracy:", metrics.accuracy_score(y_test_needed, y_pred_needed))
print(metrics.confusion_matrix(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test_needed, y_pred_needed, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test_needed, y_pred_needed, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test_needed, y_pred_needed, labels=None, pos_label=1, average='weighted', sample_weight=None))

   
# Results for all data   
print("Accuracy:", metrics.accuracy_score(y_test_new, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test_new, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test_new, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test_new, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))

