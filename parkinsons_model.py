import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

# Merge the datasets using an outer join on visit_id
merged_df = pd.merge(train_proteins, train_clinical_data, on=['visit_id', 'visit_month', 'patient_id'])
merged_df = pd.merge(merged_df, train_peptides, on=['visit_id', 'visit_month', 'patient_id'])
merged_df = pd.merge(merged_df, train_supplemental_clinical_data, on='visit_id', how='outer', suffixes=('_scd', '_scd'))

# Print the resulting dataframe
print(merged_df.head())

