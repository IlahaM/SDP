# -*- coding: utf-8 -*-


from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

df0 = pd.read_csv('C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv', dtype='object')
df1 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",  dtype='object')
df = pd.concat([df0,df1])
del df0,df1
df3 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",  dtype='object')
df=pd.concat([df,df3])
del df3
df4 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')
df=pd.concat([df,df4])
del df4
df5 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",  dtype={"Bwd Pkt Len Std": object, "Flow Byts/s": object})
df=pd.concat([df,df5])
del df5
df6 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')
df=pd.concat([df,df6])
del df6
# # df7 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')
df8 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",  dtype='object',engine='python')
df=pd.concat([df,df8])
del df8
df9 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')
df=pd.concat([df,df9])
del df9
df10 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",  dtype='object')
df=pd.concat([df,df10])
del df10

print(df.shape)

df = df.dropna()

print(df.shape)

print(df.groupby('Label').size())

########################## Dropping some Benign Values from dataset #################################


p1 = df[1:1500000]
p2 = df[2200000:3966760]
p3 = df[4500000:4800000]
p4 = df[5500000:6000000]
p5 = df[6900000:8261241]


parts = [p1, p2, p3, p4, p5]
df = pd.concat(parts, sort=False)
del p1,p2,p3,p4,p5
del parts

print(df.shape)

print(df.groupby('Label').size())

X=df.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y= df['Label'] 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=42, shuffle = True) # 80% training and 20% test


######scale the data##########
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)


#######################Param Tuning#############

#Define a hyper-parameter space to search

tuned_parameters = { 'solver': ['sgd','adam'], 
                     'learning_rate': ["adaptive"],
                      'max_iter': [400,600,800,1000], 
                      #'alpha': 10.0 ** -np.arange(2, 7),
                      'alpha': [1e-3, 1e-5, 1e-7],
                      'hidden_layer_sizes': [(8,8,8),(10,10,10),(12,12,12),(14,14,14)]}

scores = ['precision', 'recall']

print()

#clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
#Run the search
clf_grid = RandomSearchCV(MLPClassifier(), tuned_parameters, n_jobs=-1,cv=5)
clf_grid.fit(X_train, y_train)

#see the result
print("Best parameters set found on development set:")
print()
print(clf_grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf_grid.cv_results_['mean_test_score']
stds = clf_grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf_grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()


############### Train the Model #################

print(len(X_train))
print(len(X_test))
print(len(X))

clf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14, 14), max_iter=600,learning_rate='adaptive', random_state=1)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

print(max(y_pred_prob[0]))  #max of first row prob
print(y_pred[0])            #prediction of first row
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))




######treshold setting

counter = 0
for i in range(len(y_pred_prob)):
    if(max(y_pred_prob[i])) < 0.8:
        counter = counter + 1
        
print("not acc pred:", counter)


x_not_predicted = []
x_not_predicted_index_test = []
y_my_pred_index_test = []
not_pred_indexes = []
y_test_new = []
y_not_pred = []


x_pred = []
y_pred_needed = []
y_test_needed = []

for x in y_test:
    y_test_new.append(x)
    

    
for i in range(len(X_test)):
    if max(y_pred_prob[i]) < 0.8:
         #x_not_predicted_index_test.insert(i, X_test[i])   #has the same indexes as X_test having None-s
         x_not_predicted.append(X_test[i])              #not accurately pred
         y_not_pred.append(y_test_new[i])
         #y_my_pred_index_test.insert(i, None)    
         #not_pred_indexes.append(i)                        #indexes of X_test that have not been predicted
    elif(max(y_pred_prob[i])) >= 0.8:
         #y_my_pred_index_test.insert(i, y_pred[i])         #accurately predicted + None-s if not sure
         #x_not_predicted_index_test.insert(i, None)
         x_pred.append(X_test[i])
         y_pred_needed.append(y_pred[i])
         y_test_needed.append(y_test_new[i])


print(len(y_test))
print(len(X_test))
print(len(y_pred))
print(len(y_pred_prob))

print(len(x_not_predicted_index_test))
print(len(x_not_predicted))
print(len(y_my_pred_index_test))
print(len(not_pred_indexes))

print(len(x_not_predicted))
print(len(y_not_pred))

print(len(x_pred))
print(len(y_pred_needed))
print(len(y_test_needed))



print("Accuracy:", metrics.accuracy_score(y_test_needed, y_pred_needed))
print(metrics.confusion_matrix(y_test_needed, y_pred_needed))
print("Precision:", metrics.precision_score(y_test_needed, y_pred_needed, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test_needed, y_pred_needed, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test_needed, y_pred_needed, labels=None, pos_label=1, average='weighted', sample_weight=None))


############### save variables###################
import os
path = os.getcwd() + "\\"
print(path)

import pickle
with open(path+'tresh_x.pickle','wb') as f:
    pickle.dump(x_not_predicted, f)
    

with open(path+'tresh_y.pickle','wb') as f2:
    pickle.dump(y_not_pred, f2)
    
    
with open('tresh_y.pickle','rb') as f3:
    loaded=pickle.load(f3)
    
    

import pickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    pickle.dump(clf, fid)    

# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    ann_loaded = pickle.load(fid)
 
######seperate cross validation, fits model, trains######
scores = cross_val_score(clf, X, Y, cv=5)
print("Precision:", metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print(metrics.confusion_matrix(y_test, y_pred))


clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
#print(clf.score(X_test, y_test))
print(accuracy_score(y_test,y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None))
y_pred_prob = clf.predict_proba(X_test)
print(y_pred_prob[:, 1])


############################################ROC CURVES Start here#########################################################


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



ddf = pd.read_csv("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})

X=ddf.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y= ddf['Label'] 

le = preprocessing.LabelEncoder() #how do you know which label it converts to which 

# Converting string labels into numbers.
y = le.fit_transform(Y)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5) # 50% training and 50% test

cclf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14, 14), max_iter=600,learning_rate='adaptive', random_state=1)
cclf.fit(X_train, y_train)


y_pred_prob = cclf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of ANN')
plt.show()



#ROC2

dddf = pd.read_csv("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})
X=dddf.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y= dddf['Label'] 

le = preprocessing.LabelEncoder() #how do you know which label it converts to which 

# Converting string labels into numbers.
y = le.fit_transform(Y)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5) # 50% training and 50% test

cclf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14,14), max_iter=600,learning_rate='adaptive', random_state=1)
cclf.fit(X_train, y_train)


y_pred_prob = cclf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of ANN')
plt.show()



#ROC3

d3f = pd.read_csv("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", dtype={"Flow Byts/s": object, "Flow Pkts/s": object})

X=d3f.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y= d3f['Label'] 

le = preprocessing.LabelEncoder() #how do you know which label it converts to which 

# Converting string labels into numbers.
y = le.fit_transform(Y)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5) # 50% training and 50% test

cclf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14, 14), max_iter=600,learning_rate='adaptive', random_state=1)
cclf.fit(X_train, y_train)


y_pred_prob = cclf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of ANN')
plt.show()




#New Ones ROC4

d4f = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')

X=d4f.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y= d4f['Label'] 

le = preprocessing.LabelEncoder() #how do you know which label it converts to which 

# Converting string labels into numbers.
y = le.fit_transform(Y)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5) # 50% training and 50% test

cclf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14, 14), max_iter=600,learning_rate='adaptive', random_state=1)
cclf.fit(X_train, y_train)


y_pred_prob = cclf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of ANN')
plt.show()



#New Ones ROC4

d5f = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')

X=d5f.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y= d5f['Label'] 

le = preprocessing.LabelEncoder() #how do you know which label it converts to which 

# Converting string labels into numbers.
y = le.fit_transform(Y)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5) # 50% training and 50% test

cclf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14, 14), max_iter=600,learning_rate='adaptive', random_state=1)
cclf.fit(X_train, y_train)


y_pred_prob = cclf.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of ANN')
plt.show()

del d3f



#################################all ROC curves in one########################

from sklearn.preprocessing import label_binarize

X = np.array(X)

le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
y = le.fit_transform(Y)

y = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])

n_classes = 14



from sklearn.multiclass import OneVsRestClassifier


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

# Learn to predict each class against the other
clf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(14, 14,14), max_iter=600,learning_rate='adaptive', random_state=1)
classifier = OneVsRestClassifier(clf)
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


from scipy import interp
from itertools import cycle
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
            )

    
    
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')


plt.plot([0, 1], [0, 1], 'k--', lw=2)
#plt.xlim([0.0, 15.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", prop=fontP)
plt.show()


###########small test with 10th file###########
del X_test,y_test
del y_pred,y_pred_prob
del X_train,y_train

df7 = pd.read_csv("C:/Users/Sevinj/OneDrive - ADA University/University Files/Semester 8,9/SDP/DATASet/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", dtype='object')
X_test2=df7.loc[:, ['Bwd Pkt Len Min','Subflow Fwd Byts','TotLen Fwd Pkts','Bwd Pkt Len Std','Fwd Pkt Len Mean','Flow IAT Min',
            'Fwd IAT Min','Flow IAT Mean','Flow Duration','Flow IAT Std','Active Min','Active Mean','Fwd IAT Min', 
             'Bwd IAT Mean','Fwd IAT Mean','Init Fwd Win Byts','ACK Flag Cnt','Fwd PSH Flags','SYN Flag Cnt','Fwd Pkts/s',
             'Bwd Pkts/s','Init Bwd Win Byts','PSH Flag Cnt','Pkt Size Avg']]
Y_test2= df7['Label'] 

 
X_test2 = scaler.transform(X_test2)  
# apply same transformation to test data

y_pred2=clf.predict(X_test2)
print("Accuracy:", metrics.accuracy_score(Y_test2, y_pred2))
print(metrics.confusion_matrix(Y_test2, y_pred2))
print("Precision:", metrics.precision_score(Y_test2, y_pred2, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("Recall:", metrics.recall_score(Y_test2, y_pred2, labels=None, pos_label=1, average='weighted', sample_weight=None))
print("F-measure:", metrics.f1_score(Y_test2, y_pred2, labels=None, pos_label=1, average='weighted', sample_weight=None))


y_pred_prob = clf.predict_proba(X_test2)

counter = 0
for i in range(len(y_pred_prob)):
    if(max(y_pred_prob[i])) < 0.8:
        counter = counter + 1
        
print("not acc pred:", counter)


x_not_predicted = []
x_not_predicted_index_test = []
y_my_pred_index_test = []
not_pred_indexes = []
y_test_new = []
y_not_pred = []


x_pred = []
y_pred_needed = []
y_test_needed = []

for x in y_test:
    y_test_new.append(x)
    

    
for i in range(len(X_test)):
    if max(y_pred_prob[i]) < 0.8:
         #x_not_predicted_index_test.insert(i, X_test[i])   #has the same indexes as X_test having None-s
         x_not_predicted.append(X_test[i])              #not accurately pred
         y_not_pred.append(y_test_new[i])
         #y_my_pred_index_test.insert(i, None)    
         #not_pred_indexes.append(i)                        #indexes of X_test that have not been predicted
    elif(max(y_pred_prob[i])) >= 0.8:
         #y_my_pred_index_test.insert(i, y_pred[i])         #accurately predicted + None-s if not sure
         #x_not_predicted_index_test.insert(i, None)
         x_pred.append(X_test[i])
         y_pred_needed.append(y_pred[i])
         y_test_needed.append(y_test_new[i])



import os
path = os.getcwd() + "\\"
print(path)

import pickle
with open(path+'file10_tresh_x.pickle','wb') as f:
    pickle.dump(x_not_predicted, f)
    

with open(path+'file10_tresh_y.pickle','wb') as f2:
    pickle.dump(y_not_pred, f2)
    