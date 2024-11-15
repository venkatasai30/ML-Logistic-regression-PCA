import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'C:\Users\venka\Downloads\data sets excel\adult.csv')

data.head()
data.shape
data.info()

data[data=='?']=np.nan


for col in ['workclass', 'occupation', 'native.country']:
    data[col].fillna(data[col].mode()[0], inplace=True)
    
data.isnull().sum()

x=data.drop(['income'], axis=1)
y=data['income']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

from sklearn import preprocessing
categorical=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature]=le.fit_transform(x_train[feature])
    x_test[feature]=le.transform(x_test[feature])
    
 from sklearn.preprocessing import StandardScaler
 scaler=StandardScaler()
 x_train=pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
 x_test=pd.DataFrame(scaler.fit_transform(x_test), columns=x.columns) 
 
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import accuracy_score
 logreg=LogisticRegression()
 logreg.fit(x_train, y_train)
 y_pred=logreg.predict(x_test)
 print('Logistic Regression accuracy score with all the features: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
 
 from sklearn.decomposition import PCA
 pca=PCA()
 x_train=pca.fit_transform(x_train)
 pca.explained_variance_ratio_
 
 x=data.drop(['income', 'hours.per.week'], axis=1)
 y=data['income']
 x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
 categorical=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
 for feature in categorical:
     le=preprocessing.LabelEncoder()
     x_train[feature]=le.fit_transform(x_train[feature])
     x_test[feature]=le.transform(x_test[feature])
x_train=pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
x_test=pd.DataFrame(scaler.fit_transform(x_test), columns=x.columns) 
logreg=LogisticRegression()
logreg.fit(x_train, y_train)
y_pred=logreg.predict(x_test)
print('Logistic Regression accuracy score with all the features: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    
x=data.drop(['income', 'hours.per.week', 'capital.loss'], axis=1)
 y=data['income']
 x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
 categorical=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
 for feature in categorical:
     le=preprocessing.LabelEncoder()
     x_train[feature]=le.fit_transform(x_train[feature])
     x_test[feature]=le.transform(x_test[feature])
x_train=pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
x_test=pd.DataFrame(scaler.fit_transform(x_test), columns=x.columns) 
logreg=LogisticRegression()
logreg.fit(x_train, y_train)
y_pred=logreg.predict(x_test)
print('Logistic Regression accuracy score with all the features: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
 



    

 