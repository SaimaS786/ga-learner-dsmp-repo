# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info())

# replace the $ symbol
columns = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for col in columns:
    df[col].replace({'\$': '', ',': ''}, regex=True,inplace=True)

# store independent variable
X = df.drop(['CLAIM_FLAG'],axis=1)

# store dependent variable
y = df['CLAIM_FLAG']

# Check the value counts
count = y.value_counts()
print(count)

# spliting the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.3,random_state=6)


# --------------
# Code starts here




X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)

X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)

# check missing values in X_train
print(pd.DataFrame({'total_missing': X_train.isnull().sum(), 'perc_missing': (X_train.isnull().sum()/7211)*100}))

# check missing values in X_test
print(pd.DataFrame({'total_missing': X_train.isnull().sum(), 'perc_missing': (X_train.isnull().sum()/3091)*100}))

# Code ends here


# --------------
# Code starts here

#X_train = X_train.drop([['YOJ','OCCUPATION']], axis = 1)
#X_test = X_test.drop([['YOJ','OCCUPATION']], axis = 1)

X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

# fill missing values with mean
X_train['AGE'].fillna((X_train['AGE'].mean()), inplace=True)
X_test['AGE'].fillna((X_train['AGE'].mean()), inplace=True)

X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)
X_test['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)

X_train['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)
X_test['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)

X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)
X_test['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)

print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    # Instantiate label encoder
    le = LabelEncoder()
    # fit and transform label encoder on X_train
    X_train[col]=le.fit_transform(X_train[col].astype(str))
    # transform label encoder on X_test
    X_test[col]=le.transform(X_test[col].astype(str))
print(X_train.shape)
print(X_test.shape)
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Instantiate logistic regression
model = LogisticRegression(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the score
score = accuracy_score(y_test,y_pred)
print('Accuracy_score:',score)
precision = precision_score(y_test,y_pred)
print("Precision Score:",precision)


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

smote = SMOTE(random_state = 9)
#X_train = smote.fit_sample(X_train)
#y_train = smote.fit_sample(y_train)

X_train, y_train = smote.fit_sample(X_train, y_train)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)

# Code ends here


