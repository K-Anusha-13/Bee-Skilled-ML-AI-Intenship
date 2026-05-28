# Titanic Survival Prediction – Data Cleaning Project

# -----------------------------------
# IMPORT LIBRARIES
# -----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# -----------------------------------
# LOAD DATASET
# -----------------------------------

# Load the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------------
# DISPLAY FIRST 5 ROWS
# -----------------------------------

print("FIRST 5 ROWS OF DATASET:\n")
print(df.head())

# -----------------------------------
# DATASET INFORMATION
# -----------------------------------

print("\n\nDATASET INFO:\n")
print(df.info())

# -----------------------------------
# STATISTICAL SUMMARY
# -----------------------------------

print("\n\nSTATISTICAL SUMMARY:\n")
print(df.describe())

# -----------------------------------
# CHECK MISSING VALUES
# -----------------------------------

print("\n\nMISSING VALUES BEFORE CLEANING:\n")
print(df.isnull().sum())

# -----------------------------------
# HANDLE MISSING DATA
# -----------------------------------

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fill missing Cabin values with "Unknown"
df['Cabin'] = df['Cabin'].fillna("Unknown")

# -----------------------------------
# CHECK MISSING VALUES AGAIN
# -----------------------------------

print("\n\nMISSING VALUES AFTER CLEANING:\n")
print(df.isnull().sum())

# -----------------------------------
# LABEL ENCODING
# -----------------------------------

# Encode Sex column
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

print("\n\nENCODED SEX COLUMN:\n")
print(df[['Sex']].head())

# -----------------------------------
# ONE HOT ENCODING
# -----------------------------------

# One Hot Encode Embarked column
df = pd.get_dummies(df, columns=['Embarked'])

print("\n\nDATA AFTER ONE HOT ENCODING:\n")
print(df.head())

# -----------------------------------
# VISUALIZE AGE DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df['Age'], bins=30, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()

# -----------------------------------
# SAVE CLEANED DATASET
# -----------------------------------

df.to_csv("Cleaned_Titanic_Dataset.csv", index=False)

print("\n\nCleaned dataset saved successfully as 'Cleaned_Titanic_Dataset.csv'")