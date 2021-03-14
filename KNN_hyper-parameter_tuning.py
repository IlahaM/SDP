#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt



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
df = pd.concat(frames)



del df1, df2, df3, df4, df5, df6, df8, df9, df10

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
df = pd.concat(parts)

print(df.shape)

print(df.groupby('Label').size())



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


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.3, shuffle = True)



#hyper-parameter tuning using grid search
from sklearn.model_selection import GridSearchCV

#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.array([3, 5, 7, 9])}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=3)

#fit model to data
knn_gscv.fit(features, y)



#check top performing n_neighbors value
knn_gscv.best_params_


#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_


print(df.groupby('Label').size())
print(df.shape)



#Create KNN Classifier with the parameter obtained from tuning
knn = KNeighborsClassifier(n_neighbors = 9)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))




