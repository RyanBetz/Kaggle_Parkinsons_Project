import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Read in the CSV file
train_peptides = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_peptides.csv')
train_clinical_data = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
train_proteins = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_proteins.csv')

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
# Compute the correlation between NPX and UPDRS Part 1 scores
corr = merged_df['NPX'].corr(merged_df['updrs_1'])

# Print the correlation value
print(f"The correlation between NPX and UPDRS Part 1 is: {corr:.3f}")

