import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the CSV files
train_proteins = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_proteins.csv')
train_clinical_data = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')

# Merge the dataframes based on visit_id, visit_month, and patient_id
merged_df = pd.merge(train_clinical_data, train_proteins, on=['visit_id', 'visit_month', 'patient_id'])

# Filter the merged dataframe to only include visit_month=6
merged_df = merged_df.query('visit_month == 0')

# Get the rows with the highest NPX value for each patient
max_npx_df = merged_df.loc[merged_df.groupby('patient_id')['NPX'].idxmax()]

# Plot the data stratified by medication status
sns.scatterplot(x='NPX', y='updrs_1', hue='upd23b_clinical_state_on_medication', data=max_npx_df)
plt.title('UPDRS Part 1 vs. NPX (n={})'.format(max_npx_df.shape[0]))
plt.xlabel('NPX')
plt.ylabel('UPDRS Part 1')
plt.show()
