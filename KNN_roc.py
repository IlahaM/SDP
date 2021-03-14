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
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle



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


# In[ ]:


p1 = df[1:1500000]
p2 = df[2200000:3966760]
p3 = df[4500000:4800000]
p4 = df[5500000:6000000]
p5 = df[6900000:8261241]

parts = [p1, p2, p3, p4, p5]
df = pd.concat(parts)

print(df.shape)

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



X = np.array(features)

le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
y = le.fit_transform(y)

y = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])

n_classes = 14


# In[ ]:


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 9))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], threshold = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'green',
                'yellow', 'black', 'aquamarine', 'lime', 'pink', 'chocolate', 'aquamarine', 'orange'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

