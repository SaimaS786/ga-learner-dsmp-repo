# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
path
df = pd.read_csv(path)
print(df.head(5))
print(df.shape)
#print(df.info())
#print(df.describe())

X = df[['ages', 'num_reviews', 'piece_count', 'play_star_rating', 'review_difficulty', 'star_rating', 'theme_name', 'val_star_rating', 'country']]
y = df['list_price']
#X = df.values
#y = df['list_price'].values

#df = pd.DataFrame(diabetes.data, columns=columns) # load the dataset as a pandas data frame
#y = diabetes.target # define the target variable (dependent variable) as y
##X_train, y_train = df.iloc[:,:1], df.iloc[:,1]
#X_test, y_test = test.iloc[:,:1], test.iloc[:,1]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 6)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here
#for x_col in X_train.columns: 
cols = X_train.columns
rows=3
columns=3
fig, axes = plt.subplots(nrows = rows, ncols = columns,figsize=(20,10))
for i in range(0,rows,1):
    for j in range(0,columns,1):
        col = cols[i*3 + j]
        axes[i,j].scatter(X[col],y)
        plt.ylabel(str.upper('Price'))
        plt.xlabel(str.upper(col))
        plt.xticks(rotation=45)
plt.show()

# code ends here


# --------------
# Code starts here
#print(X_train)

corr = X_train.corr()
print(corr.shape)
m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.5).any()
print(m)
X_train.drop(['play_star_rating','val_star_rating'], 1 ,inplace=True)
X_test.drop(['play_star_rating','val_star_rating'], 1 ,inplace=True)


# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
from sklearn import metrics
#from sklearn import r2_score

# Code starts here
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)
print(y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)
print(mse)

r2 = r2_score(y_test, y_pred)
print(r2)

# Code ends here


# --------------
# Code starts here




#residual = (y_test.values)-(y_pred) #.values)

residual = (y_test)-(y_pred) #.values)
print(residual.shape) 

plt.hist(residual, bins=20)
plt.show()
# Code ends here


