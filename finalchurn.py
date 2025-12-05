# %%
#Importing Library Files 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %%
#combining the datasets since it is splitted

rawdf1=pd.read_csv('customer_churn_dataset-testing-master.csv')
print(rawdf1.shape)
rawdf2=pd.read_csv('customer_churn_dataset-training-master.csv')
print(rawdf2.shape)
rawdf=pd.concat([rawdf1,rawdf2],axis=0)
rawdf.head()


# %%
rawdf.info()

# %%
rawdf.describe()

# %%
#Churn is target 
#dropping unnecessary columns 
df=rawdf.drop(columns=['CustomerID'],axis=1)
df.head()

# %%
#checking missing values 

df.isnull().sum()

# %%
df[df['Age'].isna()]

# %%
df=df.dropna()

# %%
#check duplicates 
df.duplicated().sum()

# %%
num_col= df.select_dtypes(include='number').columns

# %%
#check outliers
columns = num_col
plt.figure(figsize=(20,20))

for i, col in enumerate(columns, 1):
    plt.subplot(3,3,i)
    sns.boxplot(df[col])
    plt.title(col)
    
plt.show()

# %%
#no outliers
#check distributions 
plt.figure(figsize=(20,20))

for i, col in enumerate(columns, 1):
    plt.subplot(3,3,i)
    sns.histplot(df[col])
    plt.title(col)
    
plt.show()

# %%
#churn is imbalanced

# %%
df.shape

# %%
#splitting X and y 
y = df['Churn']
X = df.drop(columns='Churn')
X.head()

# %%
df.select_dtypes(include='object').columns

# %%
#Encoding categorical values 
cat_cols = ['Gender', 'Subscription Type', 'Contract Length']

OHE = OneHotEncoder(sparse_output=False)
OHE.fit(X[cat_cols])

feature_names = OHE.get_feature_names_out(cat_cols)

OHE_df = pd.DataFrame(OHE.transform(X[cat_cols]), columns=feature_names,index=X.index)

# Drop original categorical cols
X = X.drop(columns=cat_cols)

# Combine encoded 
X= pd.concat([X,OHE_df], axis=1)

X.head()

# %%
num_col

# %%
#Scaling numerical values 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scale_cols = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
X[scale_cols] = scaler.fit_transform(X[scale_cols])
X.head()

# %%
X.shape,y.shape

# %%
#split train & test 

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=11)
print(X_train.shape,X_test.shape)

# %%
#Logistic Regression model 

lr= LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)
predicted_y_test = lr.predict(X_test)

print("Accuracy:",accuracy_score(y_test, predicted_y_test))
print("Precision:",precision_score(y_test, predicted_y_test))
print("Recall:",recall_score(y_test, predicted_y_test))
print("F1:",f1_score(y_test, predicted_y_test))

# %%
cm=confusion_matrix(y_test, predicted_y_test)
from sklearn.metrics import ConfusionMatrixDisplay
display=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])
display.plot()
plt.show()

# %%
#KNN model 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
kpredict_y_test = knn.predict(X_test)

print("Accuracy:",accuracy_score(y_test, kpredict_y_test))
print("Precision:",precision_score(y_test, kpredict_y_test))
print("Recall:",recall_score(y_test, kpredict_y_test))
print("F1:",f1_score(y_test, kpredict_y_test))

# %%
cm2=confusion_matrix(y_test, kpredict_y_test)
from sklearn.metrics import ConfusionMatrixDisplay
display=ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=[1,0])
display.plot()
plt.show()

#KNN is better
# save the KNN model 

import pickle
pipeline = {
    'scaler': scaler,
    'encoder': OHE,
    'model': knn,
    'scale_cols': scale_cols,
    'cat_cols': cat_cols
}

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved DONE!")
# %%