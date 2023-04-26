import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Read in the CSV file
train_peptides = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_peptides.csv')

# Print the first five rows of the dataframe
print(train_peptides.head())
# Load the data from the csv file
train_peptides = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_peptides.csv')

# Group the data by patient_id and visit_month and sum the peptide abundance
train_peptides_grouped = train_peptides.groupby(['patient_id', 'visit_month']).agg({'PeptideAbundance': 'sum'})

# Print the resulting dataframe
print(train_peptides_grouped.head())

# Filter data for a single patient
patient_id = '35'
patient_df = train_peptides[train_peptides['patient_id'] == patient_id]

# Create scatter plot
sns.scatterplot(x='visit_month', y='PeptideAbundance', data=patient_df)
plt.title(f'Peptide Abundance vs. Visit Month for Patient {patient_id}')
plt.xlabel('Visit Month')
plt.ylabel('Peptide Abundance')

plt.show()
