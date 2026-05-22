import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("diabetes.csv")

# ==============================
# BASIC DATASET INFORMATION
# ==============================

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFORMATION")
print(df.info())

# ==============================
# STATISTICAL SUMMARY
# ==============================

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ==============================
# CHECK MISSING VALUES
# ==============================

print("\nMISSING VALUES")
print(df.isnull().sum())

# ==============================
# CHECK DUPLICATES
# ==============================

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nSHAPE AFTER REMOVING DUPLICATES")
print(df.shape)

# ==============================
# DATA VISUALIZATION
# ==============================

# 1. Diabetes Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='diabetes', data=df)
plt.title("Diabetes Distribution")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.show()

# ==============================

# 2. Gender Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='gender', data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ==============================

# 3. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ==============================

# 4. BMI Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['bmi'], bins=20, kde=True)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

# ==============================

# 5. Blood Glucose Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['blood_glucose_level'], bins=20, kde=True)
plt.title("Blood Glucose Level Distribution")
plt.xlabel("Blood Glucose Level")
plt.ylabel("Frequency")
plt.show()

# ==============================

# 6. Gender vs Diabetes
plt.figure(figsize=(7,5))
sns.countplot(x='gender', hue='diabetes', data=df)
plt.title("Gender vs Diabetes")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ==============================

# 7. Smoking History Analysis
plt.figure(figsize=(10,5))
sns.countplot(x='smoking_history', data=df)
plt.title("Smoking History")
plt.xlabel("Smoking History")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# ==============================

# 8. Correlation Heatmap
plt.figure(figsize=(10,8))

# Convert categorical columns
df_encoded = df.copy()

df_encoded['gender'] = df_encoded['gender'].astype('category').cat.codes
df_encoded['smoking_history'] = df_encoded['smoking_history'].astype('category').cat.codes

sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# ==============================

# 9. Boxplot for BMI Outliers
plt.figure(figsize=(7,5))
sns.boxplot(x=df['bmi'])
plt.title("BMI Outliers")
plt.show()

# ==============================

# 10. Boxplot for Glucose Level
plt.figure(figsize=(7,5))
sns.boxplot(x=df['blood_glucose_level'])
plt.title("Blood Glucose Level Outliers")
plt.show()

# ==============================
# INSIGHTS
# ==============================

print("\nKEY INSIGHTS")

print("""
1. Higher blood glucose levels are strongly associated with diabetes.

2. BMI values are generally higher among diabetic patients.

3. Older age groups show increased diabetes occurrence.

4. HbA1c level has a strong positive correlation with diabetes.

5. Smoking history and hypertension may influence diabetes risk.
""")
