
# coding: utf-8

# ## Practicing PCA

# #### Dataset downloaded from: https://archive.ics.uci.edu/ml/datasets/iris

# In[1]:


import pandas as pd
pd.__version__
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score


# #### Load Dataset

# In[2]:


col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']


# In[4]:


# http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.read_csv.html#pandas.read_csv

iris_ds = pd.read_csv ("iris.data", names = col_names)


# In[22]:


iris_ds.head(1)


# In[6]:


# http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.drop.html#pandas.DataFrame.drop

X = iris_ds.drop ('class', 1)
Y = iris_ds['class']


# In[7]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.2)


# In[19]:


len (X_train), len (Y_train), len (X_test), len (Y_test)


# In[21]:


X_train.size # number of rows x number of col 120 * 4


# #### Normalize

# In[9]:


# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# Normalize feature set

scaler = StandardScaler ()
X_train = scaler.fit_transform (X_train)
X_test = scaler.transform(X_test)


# In[10]:


# pca = PCA(n_components= this value can be changed to 2, 3 or 4 to see the corelation)
pca = PCA()

# Fit the model with X and apply the dimensionality reduction on X.
X_train = pca.fit_transform(X_train)

# Apply dimensionality reduction to X. X is projected on the first principal components previously extracted from a training set.
X_test = pca.transform (X_test)

exp_var = pca.explained_variance_ratio_
exp_var


# In[11]:


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

classifier = RandomForestClassifier(max_depth=2, random_state=0)  
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)  


# In[12]:


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

cm = confusion_matrix(Y_test, y_pred)  
print(cm)  
print('Accuracy',  accuracy_score(Y_test, y_pred))  

