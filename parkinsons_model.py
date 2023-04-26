import pandas as pd

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
