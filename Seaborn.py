#!/usr/bin/env python
# coding: utf-8

# In[44]:


# Importing a CSV file into a Pandas DataFrame: 
import pandas as pd
df = pd.read_csv('StudentsPerformance.csv')


# In[54]:


# Handling missing values in a DataFrame:
df.dropna(inplace=True)

df.fillna(df.mean(), inplace=True)


# In[46]:


# Removing duplicate rows from a DataFrame:

df.drop_duplicates(inplace=True)


# In[48]:


# quick overview of the statistical summary of a DataFrame
summary = df.describe()
print(summary)


# In[56]:


# Create the histogram using Seaborn
import seaborn as sns
import pandas as pd
sns.distplot(df['math score'], kde=False)  # Set kde=False for a histogram, kde=True for a density plot
plt.xlabel('number of student')
plt.ylabel('mark')
plt.title('Histogram')
plt.show()


# In[50]:


#  how to calculate and visualize correlation
# between different columns in a DataFrame
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[51]:


# converting categorical variables into numerical representations using encoding techniques
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['reading score'] = label_encoder.fit_transform(df['reading score'])
df['math score'] = label_encoder.fit_transform(df['math score'])
df['writing score'] = label_encoder.fit_transform(df['writing score'])


# In[52]:


# Detect outliers using z-score
from scipy import stats
z_scores = stats.zscore(df['reading score'])
outliers = (z_scores > 3) | (z_scores < -3)

df_no1_outliers = df[~outliers]

from scipy import stats
z_scores = stats.zscore(df['math score'])
outliers = (z_scores > 3) | (z_scores < -3)

df_no2_outliers = df[~outliers]
from scipy import stats
z_scores = stats.zscore(df['writing score'])
outliers = (z_scores > 3) | (z_scores < -3)

df_no3_outliers = df[~outliers]


# In[53]:


#normalize numerical data in a DataFrame

from sklearn.preprocessing import MinMaxScaler

numerical_columns = ['reading score', 'math score', 'writing score'] 
scaler = MinMaxScaler()

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[ ]:




