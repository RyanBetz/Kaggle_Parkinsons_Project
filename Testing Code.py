import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#%%
#open peptide file
train_peptides = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_peptides.csv')
train_peptides['PeptideAbundance'] = train_peptides['PeptideAbundance'].fillna(0)
print(train_peptides.shape)
train_peptides.head()
#%%
train_proteins = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_proteins.csv')
print(train_proteins.shape)
train_proteins.head()
#%%
train_clinical_data = pd.read_csv('amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
print(train_clinical_data.shape)
train_clinical_data.head()
#%%
merged_df = pd.merge(train_proteins, train_peptides, on = ['visit_id','patient_id','visit_month','UniProt'])
merged_df.head()
#%%
merged_df_2 = pd.merge(merged_df, train_clinical_data, on = ['visit_id','patient_id','visit_month'])
merged_df_2['PeptideAbundance_log'] = np.log(merged_df_2['PeptideAbundance'])
merged_df_2['NPX_log'] = np.log(merged_df_2['NPX'])
print(merged_df_2.shape)
merged_df_2.head()
#%%
fig, axs = plt.subplots(2,3,figsize = (10,10))
axs = axs.flatten()
distro = ['updrs_1','updrs_2','updrs_3','updrs_4', 'NPX_log', 'PeptideAbundance_log']
for idx, var in enumerate(distro):
    axs[idx].hist(merged_df_2[var].dropna(),bins = 30)
    axs[idx].set_title(var)
#%%
pivot_df = merged_df_2.pivot(index = 'visit_id', columns = 'Peptide', values = 'PeptideAbundance')
pivot_df.fillna(0, inplace = True)
print(pivot_df.shape)
pivot_df.head()
#%%
train_df = pd.merge(train_clinical_data, pivot_df, on='visit_id')
train_df.dropna(subset = ['updrs_1','updrs_2','updrs_3'],inplace= True)
print(train_df.isna().sum())
print(train_df.shape)
train_df.head()
#%%
X = train_df.loc[:,'AADDTWEPFASGK':]
Y = train_df.loc[:,'updrs_1']
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
