"""#Load data
import pandas as pd

data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

#Create different subsets

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats



# Group by sex
sex_groups = df.groupby('Sex')

#Plot FVC Progression for Different Subgroups
plt.figure(figsize=(10, 6))
for sex, group in sex_groups:
    # Calculate the mean FVC for each week in the group
    mean_fvc = group.groupby('Weeks')['FVC'].mean()
    plt.plot(mean_fvc.index, mean_fvc.values, label=f'Sex: {sex}')

plt.title('FVC Progression by Sex')
plt.xlabel('Weeks')
plt.ylabel('Mean FVC (ml)')
plt.legend()
plt.grid(True)
plt.show()


# Group by Smoking Status

smoking_groups = df.groupby('SmokingStatus')

# Plotting FVC progression by Smoking Status
plt.figure(figsize=(10, 6))
for status, group in smoking_groups:
    mean_fvc = group.groupby('Weeks')['FVC'].mean()
    plt.plot(mean_fvc.index, mean_fvc.values, label=f'Smoking Status: {status}')

plt.title('FVC Progression by Smoking Status')
plt.xlabel('Weeks')
plt.ylabel('Mean FVC (ml)')
plt.legend()
plt.grid(True)
plt.show()


f_value, p_value = stats.f_oneway(
    df[df['SmokingStatus'] == 'Currently smokes']['FVC'],
    df[df['SmokingStatus'] == 'Ex-smoker']['FVC'],
    df[df['SmokingStatus'] == 'Never smoked']['FVC']
)

print(f"P-value based on smoking status: {p_value}")

# Group by Age Category

age_bins = [40, 60, 70, 80]
age_labels = ['40-60', '60-70', '70-80']

df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

age_groups = df.groupby('Age_Group')

# Plotting FVC progression by Age Group
plt.figure(figsize=(10, 6))
for age_group, group in age_groups:
    mean_fvc = group.groupby('Weeks')['FVC'].mean()
    plt.plot(mean_fvc.index, mean_fvc.values, label=f'Age Group: {age_group}')

plt.title('FVC Progression by Age Group')
plt.xlabel('Weeks')
plt.ylabel('Mean FVC (ml)')
plt.legend()
plt.grid(True)
plt.show()

f_value, p_value = stats.f_oneway(
    df[df['Age_Group'] == '40-60']['FVC'],
    df[df['Age_Group'] == '60-70']['FVC'],
    df[df['Age_Group'] == '70-80']['FVC']
)

print(f"P-value based on age group: {p_value}")


# STEP 3 — Statistical Testing
from scipy import stats

male_data = df[df['Sex'] == 'Male']['FVC']
female_data = df[df['Sex'] == 'Female']['FVC']

# Perform t-test
t_stat, p_val = stats.ttest_ind(male_data, female_data)
print(f"T-test for FVC between Male and Female: T-stat = {t_stat}, P-value = {p_val}")
"""


"""#Clustering K-means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

df['SmokingStatus'] = df['SmokingStatus'].map({'Never smoked': 0, 'Ex-smoker': 1,'Currently smokes':2})
df['Sex'] = df['Sex'].map({'Male':0, 'Female':1})

df = df['Patient'].unique()

#Normalize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age','FVC','Sex','SmokingStatus']])
"""
#K-means
"""
# Metodo del gomito per trovare il numero ottimale di cluster
inertia = []
k_range = range(1, 50)  # Proviamo da 1 a 10 cluster

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot del metodo del gomito
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Metodo del Gomito')
plt.xlabel('Numero di Cluster')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()"""


"""kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Calcola il silhouette score
sil_score = silhouette_score(df_scaled, df['cluster'])



print(f"Silhouette Score per 2 cluster: {sil_score}")

# Step 3: 3D plot visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of Age, Sex, and Smoking Status
scatter = ax.scatter(df['Age'], df['Sex'], df['SmokingStatus'], c=df['cluster'], cmap='viridis', s=100)

# Labels and Title
ax.set_xlabel('Age')
ax.set_ylabel('Sex')
ax.set_zlabel('Smoking Status')
ax.set_title('3D Clustering of Patients (Age, Sex, Smoking Status)')

# Color bar for clusters
plt.colorbar(scatter, label='Cluster')

# Show the plot
plt.show()


# Centroids of the clusters for k=2
centroids = kmeans.cluster_centers_
print("Centroids for k=2:", centroids)

# Reverse the standardization to get the actual centroids in the original scale
centroids_original_scale = scaler.inverse_transform(centroids)
print("Original scale centroids for k=2:", centroids_original_scale)


# Remove Age from the data
df_reduced = df[['Sex', 'SmokingStatus']]  # Only keep Sex and Smoking Status

# Standardize the data again (since we changed the features)
scaler = StandardScaler()
df_scaled_reduced = scaler.fit_transform(df_reduced)

# Inertia for different values of k
inertia = []
k_range = range(2, 11)  # Test from k=2 to k=10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled_reduced)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Apply K-means with k=4 and k=5
kmeans_4 = KMeans(n_clusters=4, random_state=42)
df['cluster_4'] = kmeans_4.fit_predict(df_scaled_reduced)

kmeans_5 = KMeans(n_clusters=5, random_state=42)
df['cluster_5'] = kmeans_5.fit_predict(df_scaled_reduced)

kmeans_6 =  KMeans(n_clusters=6, random_state=42)
df['cluster_6'] = kmeans_6.fit_predict(df_scaled_reduced)

# Silhouette Scores for k=4 and k=5
from sklearn.metrics import silhouette_score

sil_score_4 = silhouette_score(df_scaled_reduced, df['cluster_4'])
sil_score_5 = silhouette_score(df_scaled_reduced, df['cluster_5'])
sil_score_6 = silhouette_score(df_scaled_reduced, df['cluster_6'])

print(f"Silhouette Score for k=4: {sil_score_4}")
print(f"Silhouette Score for k=5: {sil_score_5}")
print(f"Silhouette Score for k=6: {sil_score_6}")

# Visualize the clusters for k=4
plt.figure(figsize=(8, 6))
plt.scatter(df['Sex'], df['SmokingStatus'], c=df['cluster_4'], cmap='viridis', s=100)
plt.title('Clustering of Patients (Sex, Smoking Status) for k=4')
plt.xlabel('Sex')
plt.ylabel('Smoking Status')
plt.colorbar(label='Cluster')
plt.show()

# Visualize the clusters for k=5
plt.figure(figsize=(8, 6))
plt.scatter(df['Sex'], df['SmokingStatus'], c=df['cluster_5'], cmap='viridis', s=100)
plt.title('Clustering of Patients (Sex, Smoking Status) for k=5')
plt.xlabel('Sex')
plt.ylabel('Smoking Status')
plt.colorbar(label='Cluster')
plt.show()

# Visualize the clusters for k=6
plt.figure(figsize=(8, 6))
plt.scatter(df['Sex'], df['SmokingStatus'], c=df['cluster_6'], cmap='viridis', s=100)
plt.title('Clustering of Patients (Sex, Smoking Status) for k=6')
plt.xlabel('Sex')
plt.ylabel('Smoking Status')
plt.colorbar(label='Cluster')
plt.show()

# Check the centroids for k=5
centroids_5 = kmeans_5.cluster_centers_
print("Centroids for k=5:", centroids_5)

# Check the centroids for k=6
centroids_6 = kmeans_6.cluster_centers_
print("Centroids for k=6:", centroids_6)"""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Carica i dati
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# Mappare i valori categoriali in numerici
df['SmokingStatus'] = df['SmokingStatus'].map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2})
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})

# Raggruppa i dati per paziente (PatientId) e calcola la variazione di FVC nel tempo
df['FVC_change'] = df.groupby('Patient')['FVC'].diff()  # Differenza di FVC tra le settimane consecutive
df['FVC_change'] = df['FVC_change'].fillna(0)

# Ora possiamo aggiungere il clustering basato sulla variazione di FVC nel tempo
df_reduced = df[['FVC_change', 'Sex', 'SmokingStatus']]  # Usa FVC_change come caratteristica di clustering
# Sostituisci NaN con 0 per la settimana baseline (prima settimana per ogni paziente)
# Standardizza i dati
scaler = StandardScaler()
df_scaled_reduced = scaler.fit_transform(df_reduced)

# Inertia for different values of k
inertia = []
k_range = range(2, 20)  # Test from k=2 to k=10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled_reduced)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

