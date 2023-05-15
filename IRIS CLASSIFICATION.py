# oasis infobyte #task1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the dataset in the CSV format
column_names= ['sepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
data = pd.read_csv('/content/Iris.csv',index_col=0)
data

#display the description of dataset
data.describe()

#visualize between each feature
sns.pairplot(data, hue='Species')

df = data.values
X = df[:,0:4]
y = df[:,4]
print(X)

#splitting the dataset for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)

#training the data
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)

#using logistic regression
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)

prediction2 = model_LR.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, prediction2)*100)
print(confusion_matrix(y_test, prediction2))

#  use DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)

prediction3 = model_DTC.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, prediction3)*100)
print(confusion_matrix(y_test, prediction3))

#testing the data to predict the output
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction2))

X_new = np.array([[3,2,1,0.2], [4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])
prediction = model_svc.predict(X_new)
print("Prediction of Species: {}".format(prediction))
