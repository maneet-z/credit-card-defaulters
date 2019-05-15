# Google Colabotory has been used as the python development environment

#Mounting the drive for data
from google.colab import drive
drive.mount('/content/drive')
cd drive
cd My Drive

# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset using pandas lib

data = pd.read_csv('default_of_credit_card_clients.csv')
# Displaying the first five entries of the dataset
data.head()
# Renaming the col names in lower case for convenience

#Pre-processing
data.rename(columns=lambda X: X.lower(), inplace=True)
data.head()
# To visualize the mean, std, count of the dataset cols
data.describe()
# Removing unnecessary columns which do not contribute significantly in information
data.drop('id',axis=1,inplace=True)
# Renaming the class label to DefaultPayment for convenience
data.rename(columns={'default payment next month': 'DefaultPayment'}, inplace=True)
columns = data.columns.tolist()

# Printing total default cases and non default cases for better understanding of the dataset

#Preparing the dataset for training and testing
columns = [c for c in columns if c not in ['DefaultPayment']]

label = 'DefaultPayment'

X = data[columns]
Y = data[label]

print(X.shape)
print(Y.shape)

default = data[data['DefaultPayment'] == 1]
print("Total Default cases : {}".format(len(default)))

not_default = data[data['DefaultPayment']==0]
print("Total Non-Default cases : {}".format(len(not_default)))

# PLotting the histogram of each feature to understand the data representation

data.hist(figsize=(20,20))
plt.show()

# PLotting the histogram
cormat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=0.8,square = True)
plt.show()

# Splitting the dataset into train and test (70% for training and 30% for test)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=42)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape # To show the shape 

# Implementing the standard scaling as a pre-processing technique to scale the dataset

from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Implementing various classifiers to train the model

# 1. Implementing Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy for Logistic Regression is : ",metrics.accuracy_score(Y_test,y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)

#Class labels = 0 and 1
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred)) 

# 2. Implementing Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

randomf=RandomForestClassifier(n_estimators=100)


randomf.fit(X_train,Y_train)

y_pred=randomf.predict(X_test)

print("Accuracy for Random Forest Classifier is :",metrics.accuracy_score(Y_test, y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)

#Class labels = 0 and 1
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred)) 

# 3. Implementing the KNearestNeighbor Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score


# For n = 1
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print("Accuracy of KNN for N=1 : ",metrics.accuracy_score(Y_test,y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)



# For n = 3
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print("Accuracy of KNN for N=3 : ",metrics.accuracy_score(Y_test,y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

# For n = 5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print("Accuracy of KNN for N=5 : ",metrics.accuracy_score(Y_test,y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

# For n = 10
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
print("Accuracy of KNN for N=10 : ",metrics.accuracy_score(Y_test,y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
# Plotting the confusion matrix for n = 10

#Class labels = 0 and 1
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred)) 

# 4. Implementing Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()


clf = clf.fit(X_train,Y_train)


y_pred = clf.predict(X_test)

print("Accuracy for Decision Tree Classifier is :",metrics.accuracy_score(Y_test, y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)

#Class labels = 0 and 1
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Implementing the decision tree classifier with gini index
clf1 = DecisionTreeClassifier(criterion="gini", max_depth=3)


clf1 = clf1.fit(X_train,Y_train)


y_pred = clf1.predict(X_test)

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)

#Class labels = 0 and 1
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("Accuracy of DT with gini index is :",metrics.accuracy_score(Y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred)) 

# 5. Implementing Adaboost Classifier with Decision tree classifier as the defaukt base estimator

from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(n_estimators=100, learning_rate=0.000001)

model = abc.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)

# For class labels 0 and 1
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred)) 

# 6. Implementing SVM(Support Vector Machine)

from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, Y_train)  

y_pred = svclassifier.predict(X_test) 

print("Accuracy of SVM classifier is : ",metrics.accuracy_score(Y_test,y_pred))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred))  

precision = precision_score(Y_test,y_pred)
recall = recall_score(Y_test,y_pred)
f1 = f1_score(Y_test,y_pred)

print("Precision score : ",precision)
print("Recall score : ",recall)
print("F1 score : ",f1)

# Ploting the confusion matrix

#Class labels = 0 and 1
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
