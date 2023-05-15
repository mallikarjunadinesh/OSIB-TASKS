import pandas as pd

data = pd.read_csv("/content/spam.csv",index_col=0)

data.info()

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

#loding the data

data = pd.read_csv("/content/spam.csv",encoding = "ISO-8859-1")

data

#displaying the header part of the dataset

data.head()

#give the description of dataset

data.describe()

#displaying visualizing between the ham and spam mails

sns.countplot(x='v1', data=data)

plt.title('Distribution of Spam and Non-Spam Messages')

plt.show()

mail_data = data.where(pd.notnull(data),'')

mail_data.drop('Unnamed: 3',axis=1, inplace=True)

mail_data.drop('Unnamed: 4',axis=1, inplace=True)

mail_data

#defining the spam and ham with 0 and 1

mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0

mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1

mail_data

X = mail_data['v2']

Y = mail_data['v1']

print(X)

print(Y)

#splitting the dataset to test and train

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 3)

print(X.shape)

print(X_train.shape)

print(X_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase= True)

X_train_features = feature_extraction.fit_transform(X_train)

X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')

Y_test = Y_test.astype('int')

print(X_test_features)

model = LogisticRegression()

# train the model

model.fit(X_train_features,Y_train)

# prediction on training data

prediction_on_training_data = model.predict(X_train_features)

accuracy_training_data = accuracy_score(Y_train,prediction_on_training_data)

accuracy_training_data

#testing the data

prediction_on_test_data = model.predict(X_test_features)

accuracy_test_data = accuracy_score(Y_test,prediction_on_test_data)

accuracy_test_data

#giving new data input for predicting

input_mail = ['WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.']

# convert text to feature vector

input_data_feature = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_feature)

print(prediction)

#predicting whether the given mail is pam or not

if (prediction == 1):

    print('ham mail')

else:

    print('spam mail')
