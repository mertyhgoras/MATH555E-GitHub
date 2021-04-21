#!/usr/bin/env python
# coding: utf-8

# ## Question #1
# 
# There are four text files in the data folder
# 
# * Atatürk's "Nutuk" in Turkish
# * Dicken's novel "Great Expectations" in English
# * Flauberts' novel "Madam Bovary" in French
# * A text file `unknown.txt` in an unknown language
# 
# Your tasks are
# 
# * Calculate how many times each character (letter) appear in each text.
# * Calculate the character distributions, i.e. using the character counts, calculate the probability of each character appearing in the text.
# * Find the set of characters common to all three texts.
# * Using the common set and the KL-divergence, show that each language have different character distributions.
# * Determine the language of the text file `unknown.txt` KL-divergence measure.

# I will load the text files and clean up unnecessary extra alphabets.

# In[109]:


import pandas as pd
import os
import glob
from re import sub




corpus = [open(file).read() for file in file_list]

nutuk = open('ataturk_nutuk.txt',encoding='utf-8').read().lower()
g_exp=open('dickens_great_expectations.txt',encoding='utf-8').read().lower()
flaubert=open('flaubert_madame_bovary.txt',encoding='utf-8').read().lower()
un=open('unknown.txt',encoding='utf-8').read().lower()

def nospecial(text):
    import re
    text = re.sub(r"[^a-zA-Z0-9]+", "",text)
    return text

un1=nospecial(un)
g_exp1=nospecial(g_exp)
flaubert1=nospecial(flaubert)
nutuk1=nospecial(nutuk)


# First, we will calculate the number of letters in each text, and then print the common letters.
# 

# In[144]:


from collections import Counter

def counter(text1):
    import collections
    text1= collections.Counter(text1)
    return text1

un2 = counter(un1)
g_exp2=counter(g_exp1)
flaubert2=counter(flaubert1)
nutuk2=counter(nutuk1)

common = g_exp2 & flaubert2 & nutuk2
common


# now, ı will calculate probability distributions for common letters

# In[152]:



def prob(text2):
    N = len(common)
    P = counter(common)
    result = [P[x]/N for x in text2]
    return result
    

un3 = prob(un2)
g_exp3 = prob(g_exp2)
nutuk3=prob(nutuk2)
flaubert3=prob(flaubert2)
un3


# ## Question #2
# 
# For this question consider the [Car Evaluation Data Set](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) from UCI. Here is the [direct link](https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data) to the dataset.
# 
# Make [contingency tables](https://en.wikipedia.org/wiki/Contingency_table#:~:text=In%20statistics%2C%20a%20contingency%20table,%2C%20engineering%2C%20and%20scientific%20research.) of the columns (using [`crosstab`](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html) function from [pandas](https://pandas.pydata.org)) and figure out which pairs of columns are dependent and independent. Explain your result using statistical tests.

# In[8]:


import numpy as np
import pandas as pd
import pickle
import re
from sklearn import preprocessing

columns = ['class','buying','maint','doors','persons','daily_boot','safety']

data = pd.read_csv("car_data.csv",header=None)

data.columns = columns

data.head()


# In[9]:


pair1 = pd.crosstab(data['maint'],data['persons'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair1.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# In order not to calculate all the data by writing the same code over and over, I write the names and results of both pairs directly here.
# 
# class & maint: Chi-square test Statistics 0.000 p_value 1.000
# 
# class & doors: Chi-square test Statistics 0.000 p_value 1.000
# 
# class & persons: Chi-square test Statistics 0.000 p_value 1.000
# 
# class & daily_boot: Chi-square test Statistics 0.000 p_value 1.000
# 
# buying & maint: Chi-square test Statistics 0.000 p_value 1.000
# 
# buying & doors: Chi-square test Statistics 0.000 p_value 1.000
# 
# buying & persons: Chi-square test Statistics 0.000 p_value 1.000
# 
# buying & daily_boot: Chi-square test Statistics 0.000 p_value 1.000
# 
# maint & doors: Chi-square test Statistics 0.000 p_value 1.000
# 
# maint & persons: Chi-square test Statistics 0.000 p_value 1.000
# 
# maint & daily_boot: Chi-square test Statistics 0.000 p_value 1.000
# 
# doors & persons: Chi-square test Statistics 0.000 p_value 1.000
# 
# doors & daily_boot: Chi-square test Statistics 0.000 p_value 1.000
# 
# persons & daily_boot: Chi-square test Statistics 0.000 p_value 1.000

# In[71]:


pair2 = pd.crosstab(data['class'],data['safety'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair2.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# In[72]:


pair3 = pd.crosstab(data['buying'],data['safety'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair3.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# In[73]:


pair4 = pd.crosstab(data['maint'],data['safety'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair4.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# In[74]:


pair5 = pd.crosstab(data['doors'],data['safety'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair5.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# In[76]:


pair6 = pd.crosstab(data['persons'],data['safety'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair6.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# In[77]:


pair7 = pd.crosstab(data['daily_boot'],data['safety'])

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(pair7.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))


# Between the colmuns, all but safety are not linked to each other. Safety and daily_boot, doors are the most relevant rows.

# ## Question #3
# 
# For this question, use [Default of Credit Card Clients Data Set]() from UCI. Here is the [direct link](https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls) to the dataset.
# 
# Your tasks are
# 
# * Inspect the dataset.
# * Would it be appropriate to form a linear regression model to predict the `default payment next month` variable? Explain.
# * Form a [contingency table](https://en.wikipedia.org/wiki/Contingency_table#:~:text=In%20statistics%2C%20a%20contingency%20table,%2C%20engineering%2C%20and%20scientific%20research.) of the columns `SEX` vs `default payment next month` and `EDUCATION` vs `default payment next month`.
# * Are there statistically verifiable relationships between credit card defaults, the gender of and the education level the borrower? Which is stronger? Quantify your analysis using [Chi Square Test](https://en.wikipedia.org/wiki/Chi-squared_test).

# In[80]:


data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',header=1)
data.head()


# In[102]:


import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }

plt.matshow(data.corr(method='pearson'))
plt.xticks(range(25), data.columns.values, rotation='vertical')
plt.yticks(range(25), data.columns.values)
plt.xlabel('Pearson Correlation', fontdict=font)
plt.show()
data.corr(method='pearson')


# In the table above, we examined which rows are more useful. Pay0,..,Pay5 and BILL_AMT1,...,BILL_AMIT5 are highly correlated among themselves. This means we will use only one data from its datas. So which regression model is best for us? Linear or logistic?
# İt is depend on behaviour of last column.

# In[101]:


np.unique(data['default payment next month'])


# we have two values, not numerical so our regression model is logistic regression.

# In[103]:


pd.crosstab(data['SEX'],data['default payment next month'])


# In[104]:


pd.crosstab(data['EDUCATION'],data['default payment next month'])


# In[119]:


sex_default = pd.crosstab(data['SEX'],data['default payment next month'],normalize=0)

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(sex_default.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))
sex_default


# In[121]:


education_default = pd.crosstab(data['EDUCATION'],data['default payment next month'],normalize=0)

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(education_default.values)
print ('Chi-square test Statistics %0.3f p_value %0.3f'%(chi2,p))
education_default


# We could not get any consistent information from the above results, so here is no statistically verifiable correlation between education of the barrower and defaulting a loan, or sex of the barrower and defaulting a loan.

# ## Question #4
# 
# 
# For this question, use the [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) from UCI.  Here is the [direct link](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) to the dataset.
# 
# Your tasks are
# 
# * Form a [K-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) model for this dataset.
# * Test your model on random samples of your data and calculate its accuracy.
# * Repeat your calculation 100 times and give an interval of accuracy values leaving the best 2.5% and worst 2.5% accuracy values.
# * Is there a better way of doing this without repeating the calculation 100 times? Explain.
# * Find the best parameter $k$ for your dataset for the K-NN model.

# In[166]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
data.head()


# In[167]:


X = data.iloc[:,0:4]
y = data.iloc[:,4]


# In[236]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy_score(y_test, y_pred)


# Let's repeat the same process 100 times.
# 

# In[239]:


accuracies = [ y_pred for X in range(100)]
accuracies


# In[ ]:





# ## Question #5
# 
# For this question, we are going to use [Concrete Slump Test Dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test) from UCI. Here is the [direct link](https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data) to the dataset.
# 
# Your tasks are
# 
# * Form three separate linear regression model for the following dependent variables:
# 
#   - SLUMP (cm)
#   - FLOW (cm)
#   - 28-day Compressive Strength (Mpa)
#   
# * Compare how well these models fit.

# In[73]:


data1 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data')
data.head()


# We do not need 'no' column, we willremove it.

# In[74]:


X = data.iloc[:,1:8]
slump= data1['SLUMP(cm)']
flow= data1['FLOW(cm)']
cs = data1['Compressive Strength (28-day)(Mpa)']


# In[75]:


import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 15,
        }

plt.matshow(data1.corr(method='pearson'))
plt.xticks(range(11), data1.columns.values, rotation='vertical')
plt.yticks(range(11), data1.columns.values)
plt.xlabel('Pearson Correlation', fontdict=font)
plt.show()
data.corr(method='pearson')


# we have good correlation between FLOW and SLUMP
# 
# Let us start with FLOW.

# In[82]:


from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()

model.fit(X,flow)


# In[83]:


model = LinearRegression().fit(X, flow)

r_sq = model.score(X, flow)
print('coefficient of determination:', r_sq)


# The $R^2$-score is %50
# 
# Let us check SLUMP

# In[84]:


model = LinearRegression().fit(X, slump)

r_sq = model.score(X, slump)
print('coefficient of determination:', r_sq)


# The $R^2$-score is %32
# 
# Last part is COMPRESSİON

# In[85]:


model = LinearRegression().fit(X, cs)

r_sq = model.score(X, cs)
print('coefficient of determination:', r_sq)


# The $R^2$-score increased to 90%.

# In[ ]:




