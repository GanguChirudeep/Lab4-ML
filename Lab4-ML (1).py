#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Define the data
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the dataset
print(df)


# In[2]:


import pandas as pd
from math import log2

# Define the dataset
data = pd.DataFrame({
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
})

# Calculate the entropy for the 'buys_computer' feature
def calculate_entropy(data):
    class_counts = data.value_counts(normalize=True)
    entropy = -sum(p * log2(p) for p in class_counts)
    return entropy

entropy_buys_computer = calculate_entropy(data['buys_computer'])
print(f'Entropy of "buys_computer": {entropy_buys_computer:.4f}')

# Calculate the entropy for each feature
feature_entropies = {}
for feature in data.columns[:-1]:  # Exclude the target variable 'buys_computer'
    unique_values = data[feature].unique()
    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        entropy = calculate_entropy(subset['buys_computer'])
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy
    feature_entropies[feature] = weighted_entropy

# Print the entropy for each feature
print('\nEntropy for each feature at the root node:')
for feature, entropy in feature_entropies.items():
    print(f'{feature}: {entropy:.4f}')


# In[4]:


import pandas as pd
import math

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Calculate the entropy of the target variable 'buys_computer' at the root node
total_samples = len(df)
yes_count = len(df[df['buys_computer'] == 'yes'])
no_count = len(df[df['buys_computer'] == 'no'])

entropy_root = 0

if yes_count > 0:
    p_yes = yes_count / total_samples
    entropy_root -= p_yes * math.log2(p_yes)
    
if no_count > 0:
    p_no = no_count / total_samples
    entropy_root -= p_no * math.log2(p_no)

# Calculate Information Gain for each feature
def calculate_entropy(attribute):
    entropy = 0
    attribute_counts = df.groupby(attribute)['buys_computer'].value_counts()
    total_samples = len(df)
    
    for value, count in attribute_counts.items():
        value_samples = len(df[df[attribute] == value[0]])
        p_value = value_samples / total_samples
        p_class = count / value_samples
        entropy -= p_value * p_class * math.log2(p_class)
    
    return entropy

information_gain = {}
for feature in df.columns[:-1]:  # Exclude the target variable
    entropy_feature = calculate_entropy(feature)
    information_gain[feature] = entropy_root - entropy_feature

# Identify the feature with the highest Information Gain as the root node
root_node = max(information_gain, key=information_gain.get)

print("Entropy at root node:", entropy_root)

print("\nInformation Gain for each feature:")
for feature, ig in information_gain.items():
    print(f"{feature}: {ig}")

print("The first feature to select for the decision tree:", root_node)


# In[5]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Separate the features (X) and the target variable (y)
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Perform one-hot encoding on the categorical features
X_encoded = pd.get_dummies(X, columns=['age', 'income', 'student', 'credit_rating'])

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# Get the training set accuracy
training_accuracy = model.score(X_encoded, y)
print("Training Set Accuracy:", training_accuracy)

# Get the depth of the constructed tree
tree_depth = model.get_depth()
print("Tree Depth:", tree_depth)


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Define the dataset as you provided
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Encode categorical features
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Define features and target variable
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Create and fit the decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=list(df.columns[:-1]), class_names=['No', 'Yes'])
plt.show()


#In[7]:

import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatalabel.xlsx")
df

#In[8]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Separate features (X) and target (y)
X = df[['embed_1', 'embed_2']]
y = df['Label']

# Split the data into training and test sets (70% training, 30% test)
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree classifier
model = DecisionTreeClassifier()

# Fit the model on the training data
model.fit(Tr_X, Tr_y)

# Training Set accuracy
train_accuracy = model.score(Tr_X, Tr_y)

# Test Set Accuracy
test_accuracy = model.score(Te_X, Te_y)

print(f"Training Set Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")
class_names = df['Label'].unique().astype(str).tolist()
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=['embed_1', 'embed_2'], class_names=class_names)
plt.title("Decision Tree")
plt.show()

#In[8]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
X = df[['embed_1', 'embed_2']]
y = df['Label']
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree classifier with max_depth constraint
model = DecisionTreeClassifier(max_depth=5)

# Fit the model on the training data
model.fit(Tr_X, Tr_y)

# Training Set accuracy
train_accuracy = model.score(Tr_X, Tr_y)

# Test Set Accuracy
test_accuracy = model.score(Te_X, Te_y)

print(f"Training Set Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

# Convert class labels to strings
class_names = df['Label'].unique().astype(str).tolist()

# Plot the Decision Tree with max_depth constraint
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=['embed_1', 'embed_2'], class_names=class_names)
plt.title("Decision Tree with max_depth=5")
plt.show()

#In[8]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



# Separate features (X) and target (y)
X = df[['embed_1', 'embed_2']]
y = df['Label']

# Split the data into training and test sets (70% training, 30% test)
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree classifier with entropy criterion
model_entropy = DecisionTreeClassifier(criterion="entropy")

# Fit the model on the training data
model_entropy.fit(Tr_X, Tr_y)

# Training Set accuracy with entropy criterion
train_accuracy_entropy = model_entropy.score(Tr_X, Tr_y)

# Test Set Accuracy with entropy criterion
test_accuracy_entropy = model_entropy.score(Te_X, Te_y)

print(f"Training Set Accuracy (Entropy Criterion): {train_accuracy_entropy}")
print(f"Test Set Accuracy (Entropy Criterion): {test_accuracy_entropy}")

# Convert class labels to strings
class_names = df['Label'].unique().astype(str).tolist()


# Plot the Decision Tree with entropy criterion
plt.figure(figsize=(10, 6))
plot_tree(model_entropy, filled=True, feature_names=['embed_1', 'embed_2'], class_names=class_names)
plt.title("Decision Tree with Entropy Criterion")
plt.show()
