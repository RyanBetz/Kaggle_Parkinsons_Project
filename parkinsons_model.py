import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# Read in the CSV file
train_peptides = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_peptides.csv')
train_clinical_data = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
train_proteins = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_proteins.csv')
train_supplemental_clinical_data = pd.read_csv('amp-parkinsons-disease-progression-prediction/supplemental_clinical_data.csv')
# Merge the dataframes on visit_id, visit_month, and patient_id
merged_df = pd.merge(train_proteins, train_clinical_data, on=['visit_id', 'visit_month', 'patient_id'])
# Create scatter plot
sns.scatterplot(x='NPX', y='updrs_2', data=merged_df)
plt.title('Scatterplot of Protein Level (NPX) vs. UPDRS Part 2 Scores')
plt.xlabel('NPX')
plt.ylabel('UPDRS Part 3')
plt.show()

# Compute the correlation coefficients between NPX and each stage of UPDRS
corr_coeffs = merged_df[['NPX', 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].corr()['NPX'][1:]

# Sort the correlations in descending order
corr_coeffs_sorted = corr_coeffs.sort_values(ascending=False)

# Plot a bar chart to visualize the correlations
sns.barplot(x=corr_coeffs_sorted.index, y=corr_coeffs_sorted.values)
plt.title('Correlation between NPX and UPDRS')
plt.xlabel('UPDRS Stage')
plt.ylabel('Correlation Coefficient')

plt.show()
# Get unique protein codes
unique_proteins = merged_df['UniProt'].unique()

# Initialize empty lists for storing correlation values and protein names
corr_values = []
protein_names = []

# Get unique protein codes
unique_proteins = merged_df['UniProt'].unique()

# Initialize a dictionary to store the correlation values
corr_dict = {}

# Loop through unique protein codes
for protein in unique_proteins:
    # Filter the dataframe to only include rows for the current protein
    protein_df = merged_df[merged_df['UniProt'] == protein]

    # Compute the correlation matrix between NPX and UPDRS Part 3
    corr_matrix = protein_df[['NPX', 'updrs_3']].corr()

    # Extract the correlation value between the current protein and UPDRS Part 3
    corr_value = corr_matrix.iloc[0, 1]

    # Add the correlation value to the dictionary
    corr_dict[protein] = corr_value

# Sort the dictionary by correlation value in descending order
sorted_corr_dict = dict(sorted(corr_dict.items(), key=lambda x: x[1], reverse=True))

# Get the top 10 highest correlation unique protein codes
top_10 = dict(list(sorted_corr_dict.items())[:10])

# Get the correlation values for the top 10 unique protein codes
corr_values = list(top_10.values())

# Get the unique protein codes for the top 10 highest correlations
unique_proteins = list(top_10.keys())

# Create a bar chart of the correlation values
plt.bar(x=unique_proteins, height=corr_values)
plt.xticks(rotation=90)
plt.xlabel('Unique Protein Codes')
plt.ylabel('Correlation with UPDRS Part 3')
plt.title('Top 10 Unique Proteins Correlated with UPDRS Part 3')
plt.show()

# Merge the dataframes on visit_id, visit_month, and patient_id
merged_df_2 = pd.merge(train_clinical_data, train_proteins, on=['visit_id', 'visit_month', 'patient_id'], suffixes=('_clinical', '_protein'))

# Group the data by patient_id and UniProt, and select the max value of NPX for each group
max_npx_df = merged_df_2.groupby(['patient_id', 'UniProt'])['NPX'].min().reset_index()

# Pivot the table to have patient_id as the index and UniProt as the columns
pivoted_df = max_npx_df.pivot(index='patient_id', columns='UniProt', values='NPX')

# Replace null values with 0
pivoted_df.fillna(0, inplace=True)

# Select the features and target variable
features = pivoted_df
target = train_clinical_data.groupby('patient_id')['updrs_1'].min()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fit a Lasso regression model
model = Lasso(alpha=0.01)
model.fit(X_train, y_train)

# Print the model's R-squared score on the training and testing sets
print("Training set R-squared score:", model.score(X_train, y_train))
print("Testing set R-squared score:", model.score(X_test, y_test))







