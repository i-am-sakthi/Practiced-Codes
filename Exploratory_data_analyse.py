#!/usr/bin/env python
# coding: utf-8

# In[4]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\StudentsPerformance.csv"  
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(file_path, names=names)

print("First five rows:")
print(df.head())
print("\nLast five rows:")
print(df.tail())

print("\nSize of the dataset:", df.shape)

print("\nDescription of the dataset:")
print(df.describe())

print("\nNumber of unique values in each column:")
print(df.nunique())

scaler = MinMaxScaler()
normalized_df = df.copy()
columns_to_normalize = ['math score', 'reading score']
normalized_df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


sns.relplot(x='math score', y='reading score', hue='gender', data=df)
plt.title("Relation between Math Score and Reading Score")
plt.show()

sns.pairplot(df)
plt.title("Pairplot")
plt.show()


sns.distplot(df['math score'], kde=False)
plt.title("Distribution of Math Scores")
plt.show()


# In[ ]:





# In[7]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\StudentsPerformance.csv"  # Replace this with the actual file path on your system
df = pd.read_csv(file_path)

print("First five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nSize of the dataset:", df.shape)

print("\nDescription of the dataset:")
print(df.describe())

print("\nNumber of unique values in each column:")
print(df.nunique())

scaler = MinMaxScaler()
normalized_df = df.copy()
columns_to_normalize = ['math score', 'reading score']
normalized_df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

print("\nNormalized DataFrame:")
print(normalized_df.head())

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.relplot(x='math score', y='reading score', hue='gender', data=df)
plt.title("Relation between Math Score and Reading Score")
plt.show()

sns.pairplot(df)
plt.title("Pairplot")
plt.show()

sns.distplot(df['math score'], kde=False)
plt.title("Distribution of Math Scores")
plt.show()


# In[9]:


import pandas as pd
from sklearn.preprocessing import Binarizer

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  # Replace this with the actual file path on your system
df = pd.read_csv(file_path)

columns_to_binarize =['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']  # Replace column names with the ones you want to binarize

binarizer = Binarizer(threshold=0.0)  # You can adjust the threshold as needed
binarized_values = binarizer.fit_transform(df[columns_to_binarize])

binarized_df = pd.DataFrame(binarized_values, columns=columns_to_binarize)

print(binarized_df.head())


# In[10]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  # Replace this with the actual file path on your system
df = pd.read_csv(file_path)

columns_to_standardize = ['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']  # Replace column names with the ones you want to standardize

scaler = StandardScaler()
standardized_values = scaler.fit_transform(df[columns_to_standardize])

standardized_df = pd.DataFrame(standardized_values, columns=columns_to_standardize)

print(standardized_df.head())


# In[14]:


import pandas as pd
import random

file_path =   
df = pd.read_csv(file_path)

num_rows = len(df)
random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_rows)]

df['Glucose'] = random_colors
df['BloodPressure'] = random_colors
df['SkinThickness'] = random_colors
df['Insulin'] = random_colors
df['DiabetesPedigreeFunction'] = random_colors
df['Age'] = random_colors
df['Outcome'] = random_colors

print(df.head())


# In[15]:


import pandas as pd
import random
import matplotlib.pyplot as plt

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  # Replace this with the actual file path on your system
df = pd.read_csv(file_path)

num_rows = len(df)
random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_rows)]

df['Glucose'] = random_colors
df['BloodPressure'] = random_colors
df['SkinThickness'] = random_colors
df['Insulin'] = random_colors
df['DiabetesPedigreeFunction'] = random_colors
df['Age'] = random_colors
df['Outcome'] = random_colors

plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], c=df['label_color'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Colored Data Points')
plt.show()


# In[18]:



import pandas as pd
import random
import matplotlib.pyplot as plt

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\Dataset2.csv"  #  # Replace this with the actual file path on your system
df = pd.read_csv(file_path)

num_rows = len(df)
random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_rows)]

df['Glucose'] = random_colors
df['BloodPressure'] = random_colors
df['SkinThickness'] = random_colors
df['Insulin'] = random_colors
df['DiabetesPedigreeFunction'] = random_colors
df['Age'] = random_colors
df['Outcome'] = random_colors

plt.figure(figsize=(8, 6))
plt.scatter(df.index, df.index, c=df['Insulin'])  # Assuming you want to plot data points against their index
plt.xlabel('Data Point Index')
plt.ylabel('Data Point Index')
plt.title('Colored Data Points')
plt.show()


# In[19]:


# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Given sample dataset
X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_train = np.array([3, 5, 7, 7, 11, 13])

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Values to predict
X_test = np.array([10, 15, 20, 25]).reshape(-1, 1)

# Predict the values
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model using R-squared (R2)
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (R2) on the training set:", r2_train)

# Print the predicted values
print("Predicted values for X=10, 15, 20, 25:", y_pred)


# In[20]:


# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Given sample dataset
X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_train = np.array([3, 5, 7, 7, 11, 13])

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Values to predict
X_test = np.array([10, 15, 20, 25]).reshape(-1, 1)

# Predict the values
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model using R-squared (R2)
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (R2) on the training set:", r2_train)

# Print the predicted values
print("Predicted values for X=10, 15, 20, 25:", y_pred)


# In[ ]:





# In[2]:


import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\Dataset2.csv"
data = pd.read_csv(file_path)


# Extract features (X) and target variable (y)
X = data[['area', 'bedrooms', 'age']].values
y = data['price'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict prices for the given homes
homes_to_predict = X  # Use the same features for prediction
predicted_prices = model.predict(homes_to_predict)

# Output predicted prices
for i, price in enumerate(predicted_prices):
    print(f"Predicted price for home {i+1}: ${price:.2f}")

# Assess the accuracy of the model (e.g., using R-squared)
r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.2f}")


# In[3]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load data from CSV file

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\Dataset2.csv"
data = pd.read_csv(file_path)


# Check for missing values
if data.isnull().values.any():
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


# Extract features (X) and target variable (y)
X = data[['area', 'bedrooms', 'age']].values
y = data['price'].values


# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict prices for the given homes
homes_to_predict = X  # Use the same features for prediction
predicted_prices = model.predict(homes_to_predict)

# Output predicted prices
for i, price in enumerate(predicted_prices):
    print(f"Predicted price for home {i+1}: ${price:.2f}")

# Assess the accuracy of the model (e.g., using R-squared)
r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.2f}")


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\Book1_lab_task.csv"
data = pd.read_csv(file_path)

# Select a subset of features and the target variable
# Extract features (X) and target variable (y)
X = data[['Mileage', 'Avg_yrs']].values
y = data['Sell Price ($)'].values


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model using R-squared
accuracy = r2_score(y_test, y_pred)

print("Accuracy (R-squared):", accuracy)


# In[6]:


# Importing necessary libraries
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load the dataset from local system
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  #  # Replace this with the actual file path on your system
df = pd.read_csv(file_path)

# Generate a list of random colors
num_rows = len(df)
random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_rows)]

# Assign the random colors to the 'label' column (or any other column)
df['Glucose'] = random_colors
df['BloodPressure'] = random_colors
df['SkinThickness'] = random_colors
df['Insulin'] = random_colors
df['DiabetesPedigreeFunction'] = random_colors
df['Age'] = random_colors
df['Outcome'] = random_colors

# Plot the colored data points
plt.figure(figsize=(8, 6))
plt.scatter(df.index, df.index, c=df['Insulin'])  # Assuming you want to plot data points against their index
plt.xlabel('Data Point Index')
plt.ylabel('Data Point Index')
plt.title('Colored Data Points')
plt.show()


# In[7]:


# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Given sample dataset
X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_train = np.array([3, 5, 7, 7, 11, 13])

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Values to predict
X_test = np.array([10, 15, 20, 25]).reshape(-1, 1)

# Predict the values
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model using R-squared (R2)
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (R2) on the training set:", r2_train)

# Print the predicted values
print("Predicted values for X=10, 15, 20, 25:", y_pred)


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\insurance_data.csv"  #  # Replace this with the actual file path on your system
data = pd.read_csv(file_path)

# Prepare the data
X = data[['age']]
y = data['bought_insurance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[10]:


import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
data = {'X': [1, 2, 3, 4, 5],
        'Y': [10, 20, 15, 25, 30]}

# Create a DataFrame
df = pd.DataFrame(data)

# List of random colors
colors = ['#FF5733', '#33FF57', '#5733FF', '#FF33C4', '#33C4FF']

# Assign random color to each data point
df['Color'] = [random.choice(colors) for _ in range(len(df))]

# Display the DataFrame
print("Data with Random Colors:")
print(df)

# Plot data points with assigned colors
plt.scatter(df['X'], df['Y'], color=df['Color'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points with Random Colors')
plt.show()


# In[14]:


import pandas as pd

# Load the dataset
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv"  #  # Replace this with the actual file path on your system
data = pd.read_csv(file_path)

# Replace 'low', 'medium', 'high' with numerical values
salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
data['salary'] = data['salary'].replace(salary_mapping)



# Check for missing values
print(data.isnull().sum())

# Perform EDA to identify variables with clear impact on retention
# You can use methods like value_counts(), groupby(), or visualization techniques
# For example, data['left'].value_counts() to see the distribution of 'left' (1 for left, 0 for stayed)

import matplotlib.pyplot as plt

# Plot bar chart
plt.figure(figsize=(8, 6))
data.groupby('left')['salary'].mean().plot(kind='bar')
plt.title('Average Salary vs Retention')
plt.xlabel('Left (1 for left, 0 for stayed)')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)
plt.show()


# In[15]:


import matplotlib.pyplot as plt

# Plot bar chart
import pandas as pd

# Load the dataset
data = pd.read_csv('employee_retention_data.csv')



plt.figure(figsize=(8, 6))
data.groupby('left')['salary'].mean().plot(kind='bar')
plt.title('Average Salary vs Retention')
plt.xlabel('Left (1 for left, 0 for stayed)')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv"  #  # Replace this with the actual file path on your system
data = pd.read_csv(file_path)

# Calculate retention rate by department
retention_by_department = data.groupby('Department')['left'].mean()

# Plot bar chart
plt.figure(figsize=(10, 6))
retention_by_department.plot(kind='bar')
plt.title('Employee Retention Rate by Department')
plt.xlabel('Department')
plt.ylabel('Retention Rate')
plt.xticks(rotation=45)
plt.show()


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv"  #  # Replace this with the actual file path on your system
data = pd.read_csv(file_path)

# Narrow down the variables
selected_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                     'average_montly_hours', 'time_spend_company', 'Work_accident', 
                     'promotion_last_5years', 'salary']

# Prepare the data
X = data[selected_features]
y = data['left']

# Convert categorical variable 'salary' to dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:








# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\StudentsPerformance.csv"  
df = pd.read_csv(file_path)
print("First five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())
print("\nSize of the dataset:", df.shape)
print("\nDescription of the dataset:")
print(df.describe())
print("\nNumber of unique values in each column:")
print(df.nunique())
scaler = MinMaxScaler()
normalized_df = df.copy()
columns_to_normalize = ['math score', 'reading score']
normalized_df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

print("\nNormalized DataFrame:")
print(normalized_df.head())
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


sns.relplot(x='math score', y='reading score', hue='gender', data=df)
plt.title("Relation between Math Score and Reading Score")
plt.show()

sns.pairplot(df)
plt.title("Pairplot")
plt.show()

sns.distplot(df['math score'], kde=False)
plt.title("Distribution of Math Scores")
plt.show()


# In[2]:


import pandas as pd
from sklearn.preprocessing import Binarizer

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  
df = pd.read_csv(file_path)

columns_to_binarize =['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'] 
binarizer = Binarizer(threshold=0.0) 
binarized_values = binarizer.fit_transform(df[columns_to_binarize])

binarized_df = pd.DataFrame(binarized_values, columns=columns_to_binarize)

print(binarized_df.head())


# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  
df = pd.read_csv(file_path)

columns_to_standardize=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

scaler = StandardScaler()
standardized_values = scaler.fit_transform(df[columns_to_standardize])

standardized_df = pd.DataFrame(standardized_values, columns=columns_to_standardize)

print(standardized_df.head())


# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\pima-indians-diabetes.csv"  
df = pd.read_csv(file_path)

columns_to_standardize=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

scaler = StandardScaler()
standardized_values = scaler.fit_transform(df[columns_to_standardize])

standardized_df = pd.DataFrame(standardized_values, columns=columns_to_standardize)

print(standardized_df.head())


# In[2]:


import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
data = {'X': [1, 2, 3, 4, 5],
        'Y': [10, 20, 15, 25, 30]}
df = pd.DataFrame(data)
colors = ['#FF5733', '#33FF57', '#5733FF', '#FF33C4', '#33C4FF']
df['Color'] = [random.choice(colors) for _ in range(len(df))]
print("Data with Random Colors:")
print(df)
plt.scatter(df['X'], df['Y'], color=df['Color'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points with Random Colors')
plt.show()


# In[3]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_train = np.array([3, 5, 7, 7, 11, 13])
model = LinearRegression()
model.fit(X_train, y_train)
X_test = np.array([10, 15, 20, 25]).reshape(-1, 1)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (R2) on the training set:", r2_train)
print("Predicted values for X=10, 15, 20, 25:", y_pred)


# In[4]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\Dataset2.csv"
data = pd.read_csv(file_path)
if data.isnull().values.any():
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
X = data[['area', 'bedrooms', 'age']].values
y = data['price'].values
model = LinearRegression()
model.fit(X, y)
homes_to_predict = X 
predicted_prices = model.predict(homes_to_predict)
for i, price in enumerate(predicted_prices):
    print(f"Predicted price for home {i+1}: ${price:.2f}")
r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.2f}")


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\Book1_lab_task.csv"
data = pd.read_csv(file_path)
X = data[['Mileage', 'Avg_yrs']].values
y = data['Sell Price ($)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("Accuracy (R-squared):", accuracy)


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\insurance_data.csv"  
data = pd.read_csv(file_path)
X = data[['age']]
y = data['bought_insurance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[7]:


import pandas as pd
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv"  
data = pd.read_csv(file_path)
print(data.head())
print(data.isnull().sum())


# In[8]:


import pandas as pd
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv" 
data = pd.read_csv(file_path)
salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
data['salary'] = data['salary'].replace(salary_mapping)
print(data.isnull().sum())
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
data.groupby('left')['salary'].mean().plot(kind='bar')
plt.title('Average Salary vs Retention')
plt.xlabel('Left (1 for left, 0 for stayed)')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv" 
data = pd.read_csv(file_path)
retention_by_department = data.groupby('Department')['left'].mean()
plt.figure(figsize=(10, 6))
retention_by_department.plot(kind='bar')
plt.title('Employee Retention Rate by Department')
plt.xlabel('Department')
plt.ylabel('Retention Rate')
plt.xticks(rotation=45)
plt.show()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
file_path = "C:\\Users\\sakth\\OneDrive\\Desktop\\iot\\HR_comma_sep.csv"  
data = pd.read_csv(file_path)
selected_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                     'average_montly_hours', 'time_spend_company', 'Work_accident', 
                     'promotion_last_5years', 'salary']
X = data[selected_features]
y = data['left']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[13]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




