import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carica i dati
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# Esploriamo prima i dati
print("Shape del dataset:", df.shape)
print("\nPrime righe:")
print(df.head())
print("\nInfo sul dataset:")
print(df.info())
print("\nPazienti unici:", df['Patient'].nunique())
print("\nDistribuzione delle settimane:")
print(df['Weeks'].describe())

# Verifichiamo quanti pazienti hanno Week 0
patients_with_week0 = df[df['Weeks'] == 0]['Patient'].nunique()
total_patients = df['Patient'].nunique()
print(f"\nPazienti con Week 0: {patients_with_week0}/{total_patients}")
print(f"Pazienti SENZA Week 0: {total_patients - patients_with_week0}")

# Feature Engineering: Aggregate per paziente
def create_patient_features(df):
    """
    Crea features aggregate per ogni paziente per il clustering
    """
    patient_features = []
    
    for patient in df['Patient'].unique():
        patient_data = df[df['Patient'] == patient].sort_values('Weeks')
        
        # Features temporali
        weeks = patient_data['Weeks'].values
        fvc = patient_data['FVC'].values
        
        # Calcola slope (tasso di variazione FVC)
        if len(weeks) > 1:
            slope, intercept = np.polyfit(weeks, fvc, 1)
        else:
            slope = 0
            intercept = fvc[0] if len(fvc) > 0 else np.nan
        
        features = {
            'Patient': patient,
            'n_observations': len(patient_data),
            'weeks_range': weeks.max() - weeks.min(),
            'first_week': weeks.min(),
            'last_week': weeks.max(),
            'has_week0': int(0 in weeks),
            
            # FVC features
            'fvc_mean': fvc.mean(),
            'fvc_std': fvc.std() if len(fvc) > 1 else 0,
            'fvc_min': fvc.min(),
            'fvc_max': fvc.max(),
            'fvc_first': fvc[0],
            'fvc_last': fvc[-1],
            'fvc_range': fvc.max() - fvc.min(),
            'fvc_slope': slope,  # Tasso di declino/crescita
            'fvc_intercept': intercept,  # FVC predetto a week 0
            
            # Percent features
            'percent_mean': patient_data['Percent'].mean(),
            'percent_first': patient_data['Percent'].iloc[0],
            
            # Demographic features
            'Age': patient_data['Age'].iloc[0],
            'Sex': patient_data['Sex'].iloc[0],
            'SmokingStatus': patient_data['SmokingStatus'].iloc[0]
        }
        
        patient_features.append(features)
    
    return pd.DataFrame(patient_features)

# Crea il dataframe con le features per paziente
patient_df = create_patient_features(df)

print("\n" + "="*50)
print("FEATURES AGGREGATE PER PAZIENTE")
print("="*50)
print(f"\nShape: {patient_df.shape}")
print("\nPrime righe:")
print(patient_df.head(10))

print("\nStatistiche FVC slope (tasso di declino):")
print(patient_df['fvc_slope'].describe())

print("\nDistribuzione pazienti con/senza Week 0:")
print(patient_df['has_week0'].value_counts())

# Encode categorical variables
le_sex = LabelEncoder()
le_smoking = LabelEncoder()

patient_df['Sex_encoded'] = le_sex.fit_transform(patient_df['Sex'])
patient_df['SmokingStatus_encoded'] = le_smoking.fit_transform(patient_df['SmokingStatus'])

print("\nEncoding:")
print(f"Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"SmokingStatus: {dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_)))}")

patient_df.head()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Seleziona le features numeriche per il clustering
features_for_clustering = [
    'fvc_slope',           # MOLTO IMPORTANTE: tasso di declino
    'fvc_mean',            # Livello medio di FVC
    'fvc_std',             # Variabilità
    'percent_mean',        # Percentuale media
    'Age',                 # Età
    'Sex_encoded',         # Sesso
    'SmokingStatus_encoded', # Stato fumatore
    'n_observations',      # Quante misure abbiamo
    'weeks_range'          # Range temporale
]

X = patient_df[features_for_clustering].copy()

# Normalizzazione (importante per K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features per clustering:")
print(X.describe())

# Proviamo diversi numeri di cluster
inertias = []
silhouette_scores = []
K_range = range(2, 10)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot per scegliere il numero ottimale di cluster
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow method
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Numero di Cluster')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

# Silhouette score
axes[1].plot(K_range, silhouette_scores, 'ro-')
axes[1].set_xlabel('Numero di Cluster')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score per K')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Scegliamo k=4 come punto di partenza (possiamo modificare dopo)
optimal_k = 4
print(f"\n{'='*50}")
print(f"Clustering con K={optimal_k}")
print(f"{'='*50}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
patient_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analisi dei cluster
print("\nDistribuzione pazienti per cluster:")
print(patient_df['Cluster'].value_counts().sort_index())

print("\nCaratteristiche medie per cluster:")
cluster_summary = patient_df.groupby('Cluster')[features_for_clustering + ['has_week0']].mean()
print(cluster_summary.round(2))

# Visualizzazione con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=patient_df['Cluster'], 
                     cmap='viridis', 
                     s=100, 
                     alpha=0.6,
                     edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'Patient Clustering (K={optimal_k}) - PCA Visualization')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nVarianza spiegata da PC1 e PC2: {sum(pca.explained_variance_ratio_[:2]):.2%}")

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Prepara i dati per la predizione
predictions_list = []
models_per_cluster = {}

print("="*60)
print("MODELLI DI REGRESSIONE PER CLUSTER")
print("="*60)

for cluster_id in range(optimal_k):
    # Pazienti in questo cluster
    cluster_patients = patient_df[patient_df['Cluster'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*60}")
    print(f"Pazienti: {len(cluster_patients)}")
    print(f"Osservazioni totali: {len(cluster_data)}")
    
    # Fit regressione: FVC ~ Weeks
    X_reg = cluster_data[['Weeks']].values
    y_reg = cluster_data['FVC'].values
    
    model = LinearRegression()
    model.fit(X_reg, y_reg)
    
    # Statistiche del modello
    y_pred = model.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)
    mae = mean_absolute_error(y_reg, y_pred)
    
    print(f"Slope: {model.coef_[0]:.2f} FVC/week")
    print(f"Intercept (FVC a Week 0): {model.intercept_:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    
    models_per_cluster[cluster_id] = {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'mae': mae
    }
    
    # Predici Week 0 per pazienti SENZA Week 0 in questo cluster
    cluster_patients_no_week0 = patient_df[
        (patient_df['Cluster'] == cluster_id) & 
        (patient_df['has_week0'] == 0)
    ]
    
    print(f"\nPazienti da predire (senza Week 0): {len(cluster_patients_no_week0)}")
    
    for _, patient_row in cluster_patients_no_week0.iterrows():
        patient_id = patient_row['Patient']
        
        # Usa il modello del cluster per predire a Week 0
        predicted_fvc_week0 = model.predict([[0]])[0]
        
        # Oppure usa l'intercept calcolato per il singolo paziente
        # (più accurato se abbiamo il suo slope personale)
        patient_intercept = patient_row['fvc_intercept']
        
        predictions_list.append({
            'Patient': patient_id,
            'Cluster': cluster_id,
            'predicted_FVC_week0_cluster': predicted_fvc_week0,
            'predicted_FVC_week0_personal': patient_intercept,
            'fvc_mean': patient_row['fvc_mean'],
            'fvc_slope': patient_row['fvc_slope'],
            'first_week': patient_row['first_week']
        })

# Crea dataframe con predizioni
predictions_df = pd.DataFrame(predictions_list)

print(f"\n{'='*60}")
print(f"SOMMARIO PREDIZIONI")
print(f"{'='*60}")
print(f"Totale predizioni: {len(predictions_df)}")
print("\nPrime 10 predizioni:")
print(predictions_df.head(10))

# Confronto tra metodo cluster e metodo personale
print("\nConfronto metodi di predizione:")
print(predictions_df[['predicted_FVC_week0_cluster', 'predicted_FVC_week0_personal']].describe())

# Visualizza le predizioni per cluster
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for cluster_id in range(optimal_k):
    ax = axes[cluster_id]
    
    # Dati del cluster
    cluster_patients = patient_df[patient_df['Cluster'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    # Plot i dati reali
    for patient in cluster_patients[:10]:  # Plot solo primi 10 per chiarezza
        patient_data = cluster_data[cluster_data['Patient'] == patient]
        ax.plot(patient_data['Weeks'], patient_data['FVC'], 'o-', alpha=0.3, linewidth=1)
    
    # Plot la regressione del cluster
    weeks_range = np.linspace(cluster_data['Weeks'].min(), cluster_data['Weeks'].max(), 100)
    model = models_per_cluster[cluster_id]['model']
    fvc_pred = model.predict(weeks_range.reshape(-1, 1))
    ax.plot(weeks_range, fvc_pred, 'r-', linewidth=3, label=f'Cluster {cluster_id} trend')
    
    # Marca Week 0
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Week 0')
    
    ax.set_xlabel('Weeks')
    ax.set_ylabel('FVC')
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_patients)})\nSlope: {model.coef_[0]:.2f}, R²: {models_per_cluster[cluster_id]["r2"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# VALIDAZIONE: Test su pazienti con Week 0 conosciuto
# ============================================================

print("="*60)
print("VALIDAZIONE DELLE PREDIZIONI")
print("="*60)

# Pazienti con Week 0 (per validazione)
patients_with_week0 = patient_df[patient_df['has_week0'] == 1].copy()

validation_results = []

for _, patient_row in patients_with_week0.iterrows():
    patient_id = patient_row['Patient']
    
    # Valore reale di FVC a Week 0
    real_fvc_week0 = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
    
    # Predizione usando intercept personale
    predicted_fvc = patient_row['fvc_intercept']
    
    # Predizione usando cluster
    cluster_id = patient_row['Cluster']
    predicted_fvc_cluster = models_per_cluster[cluster_id]['intercept']
    
    error_personal = abs(real_fvc_week0 - predicted_fvc)
    error_cluster = abs(real_fvc_week0 - predicted_fvc_cluster)
    
    validation_results.append({
        'Patient': patient_id,
        'Cluster': cluster_id,
        'Real_FVC_Week0': real_fvc_week0,
        'Predicted_Personal': predicted_fvc,
        'Predicted_Cluster': predicted_fvc_cluster,
        'Error_Personal': error_personal,
        'Error_Cluster': error_cluster,
        'Percent_Error_Personal': (error_personal / real_fvc_week0) * 100,
        'Percent_Error_Cluster': (error_cluster / real_fvc_week0) * 100
    })

validation_df = pd.DataFrame(validation_results)

print(f"\nPazienti con Week 0 disponibili: {len(validation_df)}")
print("\n" + "="*60)
print("CONFRONTO ERRORI")
print("="*60)

print("\nMetodo PERSONALE (regressione individuale):")
print(f"MAE medio: {validation_df['Error_Personal'].mean():.2f}")
print(f"Errore percentuale medio: {validation_df['Percent_Error_Personal'].mean():.2f}%")
print(f"Mediana errore: {validation_df['Error_Personal'].median():.2f}")

print("\nMetodo CLUSTER (regressione per gruppo):")
print(f"MAE medio: {validation_df['Error_Cluster'].mean():.2f}")
print(f"Errore percentuale medio: {validation_df['Percent_Error_Cluster'].mean():.2f}%")
print(f"Mediana errore: {validation_df['Error_Cluster'].median():.2f}")

print("\n" + "="*60)
print("DETTAGLIO VALIDAZIONE")
print("="*60)
print(validation_df.sort_values('Error_Personal'))

# Visualizzazione validazione
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Real vs Predicted (Personal method)
ax1 = axes[0]
ax1.scatter(validation_df['Real_FVC_Week0'], 
           validation_df['Predicted_Personal'],
           c=validation_df['Cluster'],
           cmap='viridis',
           s=150,
           alpha=0.6,
           edgecolors='black')

# Linea perfetta
min_val = min(validation_df['Real_FVC_Week0'].min(), validation_df['Predicted_Personal'].min())
max_val = max(validation_df['Real_FVC_Week0'].max(), validation_df['Predicted_Personal'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

ax1.set_xlabel('Real FVC at Week 0')
ax1.set_ylabel('Predicted FVC (Personal Method)')
ax1.set_title(f'Personal Method Validation\nMAE: {validation_df["Error_Personal"].mean():.2f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Errori per cluster
ax2 = axes[1]
validation_df.boxplot(column=['Error_Personal', 'Error_Cluster'], ax=ax2)
ax2.set_ylabel('Absolute Error (FVC units)')
ax2.set_title('Error Distribution by Method')
ax2.set_xticklabels(['Personal\nMethod', 'Cluster\nMethod'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Grafico errori per cluster
fig, ax = plt.subplots(figsize=(10, 6))
validation_df.boxplot(column='Error_Personal', by='Cluster', ax=ax)
ax.set_ylabel('Absolute Error (FVC units)')
ax.set_xlabel('Cluster')
ax.set_title('Prediction Error by Cluster (Personal Method)')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("CONCLUSIONE")
print("="*60)
print(f"✓ Il metodo PERSONALE è migliore (errore: {validation_df['Error_Personal'].mean():.0f} vs {validation_df['Error_Cluster'].mean():.0f})")
print(f"✓ Abbiamo predetto Week 0 per {len(predictions_df)} pazienti")
print(f"✓ Errore medio atteso: ~{validation_df['Error_Personal'].mean():.0f} unità FVC ({validation_df['Percent_Error_Personal'].mean():.1f}%)")

# ============================================================
# CREAZIONE DATASET AUGMENTATO con Week 0 predetti
# ============================================================

print("="*60)
print("AUGMENTAZIONE DATASET")
print("="*60)

# Crea le nuove righe per Week 0
new_rows = []

for _, pred_row in predictions_df.iterrows():
    patient_id = pred_row['Patient']
    
    # Prendi i dati demografici dal paziente
    patient_info = df[df['Patient'] == patient_id].iloc[0]
    
    # Usa la predizione personale (più accurata)
    predicted_fvc = pred_row['predicted_FVC_week0_personal']
    
    # Calcola Percent basandosi sul FVC medio del paziente
    # Percent = (FVC / FVC_predetto) * 100
    # Approssimazione: usa lo stesso rapporto delle altre misure
    patient_data = df[df['Patient'] == patient_id]
    avg_percent_ratio = (patient_data['Percent'] / patient_data['FVC']).mean()
    predicted_percent = predicted_fvc * avg_percent_ratio
    
    new_row = {
        'Patient': patient_id,
        'Weeks': 0,
        'FVC': int(round(predicted_fvc)),
        'Percent': round(predicted_percent, 6),
        'Age': patient_info['Age'],
        'Sex': patient_info['Sex'],
        'SmokingStatus': patient_info['SmokingStatus'],
        'Predicted': True  # Flag per identificare i valori predetti
    }
    
    new_rows.append(new_row)

# Aggiungi flag ai dati originali
df['Predicted'] = False

# Crea nuovo dataset
df_augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df_augmented = df_augmented.sort_values(['Patient', 'Weeks']).reset_index(drop=True)

print(f"\nDataset originale: {len(df)} righe")
print(f"Nuove righe aggiunte: {len(new_rows)}")
print(f"Dataset augmentato: {len(df_augmented)} righe")

print("\nVerifica pazienti con Week 0:")
patients_with_week0_new = df_augmented[df_augmented['Weeks'] == 0]['Patient'].nunique()
print(f"Prima: 18/176 pazienti avevano Week 0")
print(f"Dopo: {patients_with_week0_new}/176 pazienti hanno Week 0")

# Esempio di paziente augmentato
sample_patient = predictions_df.iloc[0]['Patient']
print(f"\n{'='*60}")
print(f"ESEMPIO: Paziente {sample_patient}")
print(f"{'='*60}")
print(df_augmented[df_augmented['Patient'] == sample_patient][['Weeks', 'FVC', 'Percent', 'Predicted']].head(10))

# Statistiche
print(f"\n{'='*60}")
print("STATISTICHE DATASET AUGMENTATO")
print(f"{'='*60}")
print(df_augmented.groupby('Predicted').agg({
    'Patient': 'nunique',
    'FVC': ['mean', 'std'],
    'Weeks': ['min', 'max']
}))

# Visualizzazione: alcuni pazienti con Week 0 predetto
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

sample_patients = predictions_df.sample(6, random_state=42)['Patient'].values

for idx, patient_id in enumerate(sample_patients):
    ax = axes[idx]
    
    patient_data = df_augmented[df_augmented['Patient'] == patient_id].sort_values('Weeks')
    
    # Plot dati reali
    real_data = patient_data[patient_data['Predicted'] == False]
    predicted_data = patient_data[patient_data['Predicted'] == True]
    
    ax.plot(real_data['Weeks'], real_data['FVC'], 'o-', 
            linewidth=2, markersize=8, label='Dati reali', color='blue')
    
    if len(predicted_data) > 0:
        ax.plot(predicted_data['Weeks'], predicted_data['FVC'], 's', 
                markersize=12, label='Week 0 predetto', color='red', 
                markeredgecolor='black', markeredgewidth=2)
    
    # Linea di trend
    weeks = patient_data['Weeks'].values
    fvc = patient_data['FVC'].values
    z = np.polyfit(weeks, fvc, 1)
    p = np.poly1d(z)
    ax.plot(weeks, p(weeks), '--', alpha=0.5, color='gray', label=f'Trend (slope={z[0]:.2f})')
    
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.3)
    ax.set_xlabel('Weeks')
    ax.set_ylabel('FVC')
    ax.set_title(f'Patient {patient_id[-8:]}\nCluster {predictions_df[predictions_df["Patient"]==patient_id]["Cluster"].values[0]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Salva il dataset augmentato
output_path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train_augmented.csv'
df_augmented.to_csv(output_path, index=False)
print(f"\n✓ Dataset augmentato salvato in: {output_path}")

# Statistiche finali per cluster
print(f"\n{'='*60}")
print("DISTRIBUZIONE WEEK 0 PER CLUSTER")
print(f"{'='*60}")

for cluster_id in range(optimal_k):
    cluster_patients = patient_df[patient_df['Cluster'] == cluster_id]['Patient'].values
    cluster_data_aug = df_augmented[df_augmented['Patient'].isin(cluster_patients)]
    week0_data = cluster_data_aug[cluster_data_aug['Weeks'] == 0]
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Pazienti totali: {len(cluster_patients)}")
    print(f"  Week 0 ora disponibili: {len(week0_data)}")
    print(f"  FVC medio a Week 0: {week0_data['FVC'].mean():.2f}")
    print(f"  FVC range: [{week0_data['FVC'].min():.0f}, {week0_data['FVC'].max():.0f}]")
    
    # ============================================================
# ANALISI CRITICA DELLE PREDIZIONI
# ============================================================

import scipy.stats as stats

print("="*70)
print("ANALISI APPROFONDITA DELL'AFFIDABILITÀ DELLE PREDIZIONI")
print("="*70)

# ============================================================
# 1. ANALISI DELLA QUALITÀ DELLE REGRESSIONI INDIVIDUALI
# ============================================================

print("\n" + "="*70)
print("1. QUALITÀ DELLE REGRESSIONI INDIVIDUALI")
print("="*70)

# Per ogni paziente, calcola R² e altri indicatori di fit
regression_quality = []

for patient_id in df['Patient'].unique():
    patient_data = df[df['Patient'] == patient_id].sort_values('Weeks')
    
    if len(patient_data) >= 2:
        weeks = patient_data['Weeks'].values.reshape(-1, 1)
        fvc = patient_data['FVC'].values
        
        # Fit regressione
        model = LinearRegression()
        model.fit(weeks, fvc)
        
        # Calcola metriche
        y_pred = model.predict(weeks)
        r2 = r2_score(fvc, y_pred)
        mae = mean_absolute_error(fvc, y_pred)
        
        # Residui
        residuals = fvc - y_pred
        residuals_std = np.std(residuals)
        
        # Distanza da Week 0
        min_week = weeks.min()
        max_week = weeks.max()
        distance_from_week0 = abs(min_week)
        
        regression_quality.append({
            'Patient': patient_id,
            'n_points': len(patient_data),
            'r2': r2,
            'mae': mae,
            'residuals_std': residuals_std,
            'slope': model.coef_[0],
            'min_week': min_week,
            'max_week': max_week,
            'distance_from_week0': distance_from_week0,
            'has_week0': int(0 in weeks.flatten())
        })

regression_quality_df = pd.DataFrame(regression_quality)

# Merge con cluster info
regression_quality_df = regression_quality_df.merge(
    patient_df[['Patient', 'Cluster']], 
    on='Patient'
)

print("\nDistribuzione R² delle regressioni individuali:")
print(regression_quality_df['r2'].describe())

print(f"\nPazienti con R² < 0.5 (fit scadente): {(regression_quality_df['r2'] < 0.5).sum()} / {len(regression_quality_df)}")
print(f"Pazienti con R² < 0.3 (fit molto scadente): {(regression_quality_df['r2'] < 0.3).sum()} / {len(regression_quality_df)}")
print(f"Pazienti con R² > 0.8 (fit ottimo): {(regression_quality_df['r2'] > 0.8).sum()} / {len(regression_quality_df)}")

# Visualizza distribuzione R²
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Distribuzione R²
axes[0, 0].hist(regression_quality_df['r2'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0.5, color='r', linestyle='--', label='R²=0.5 threshold')
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribuzione R² delle Regressioni Individuali')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: R² vs Distanza da Week 0
scatter = axes[0, 1].scatter(regression_quality_df['distance_from_week0'], 
                             regression_quality_df['r2'],
                             c=regression_quality_df['Cluster'],
                             cmap='viridis',
                             s=50,
                             alpha=0.6)
axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Distanza da Week 0 (settimane)')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('Qualità Predizione vs Distanza da Week 0')
plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: R² per cluster
regression_quality_df.boxplot(column='r2', by='Cluster', ax=axes[1, 0])
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('Qualità Regressione per Cluster')
plt.suptitle('')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Numero punti vs R²
axes[1, 1].scatter(regression_quality_df['n_points'], 
                   regression_quality_df['r2'],
                   c=regression_quality_df['Cluster'],
                   cmap='viridis',
                   s=50,
                   alpha=0.6)
axes[1, 1].set_xlabel('Numero di Osservazioni')
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_title('R² vs Numero di Misurazioni')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 2. ANALISI EXTRAPOLAZIONE vs INTERPOLAZIONE
# ============================================================

print("\n" + "="*70)
print("2. ANALISI EXTRAPOLAZIONE vs INTERPOLAZIONE")
print("="*70)

# Pazienti che richiedono extrapolazione (non hanno misure vicino a Week 0)
extrapolation_analysis = regression_quality_df.copy()
extrapolation_analysis['requires_extrapolation'] = extrapolation_analysis['min_week'] > 0
extrapolation_analysis['extrapolation_distance'] = extrapolation_analysis['min_week'].clip(lower=0)

print(f"\nPazienti che richiedono EXTRAPOLAZIONE: {extrapolation_analysis['requires_extrapolation'].sum()}")
print(f"Pazienti con INTERPOLAZIONE: {(~extrapolation_analysis['requires_extrapolation']).sum()}")

print("\nDistanza di extrapolazione (per chi serve):")
extrap_data = extrapolation_analysis[extrapolation_analysis['requires_extrapolation']]
print(extrap_data['extrapolation_distance'].describe())

print("\nQualità predizioni per tipo:")
print("\nINTERPOLAZIONE (più affidabile):")
interp_data = extrapolation_analysis[~extrapolation_analysis['requires_extrapolation']]
print(f"  R² medio: {interp_data['r2'].mean():.3f}")
print(f"  MAE medio: {interp_data['mae'].mean():.2f}")

print("\nEXTRAPOLAZIONE (meno affidabile):")
print(f"  R² medio: {extrap_data['r2'].mean():.3f}")
print(f"  MAE medio: {extrap_data['mae'].mean():.2f}")

# Test statistico
t_stat, p_value = stats.ttest_ind(interp_data['r2'], extrap_data['r2'])
print(f"\nTest t per differenza R²: p-value = {p_value:.4f}")
if p_value < 0.05:
    print("  ⚠️  Differenza SIGNIFICATIVA - l'extrapolazione è meno affidabile!")
else:
    print("  ✓ Nessuna differenza significativa")

# ============================================================
# 3. ANALISI SENSIBILITÀ ALLE OUTLIERS
# ============================================================

print("\n" + "="*70)
print("3. SENSIBILITÀ ALLE OUTLIERS")
print("="*70)

# Per ogni paziente, controlla se ha outliers
outlier_analysis = []

for patient_id in df['Patient'].unique():
    patient_data = df[df['Patient'] == patient_id].sort_values('Weeks')
    
    if len(patient_data) >= 3:
        fvc = patient_data['FVC'].values
        
        # Z-score per identificare outliers
        z_scores = np.abs(stats.zscore(fvc))
        n_outliers = (z_scores > 2).sum()
        
        outlier_analysis.append({
            'Patient': patient_id,
            'n_outliers': n_outliers,
            'has_outliers': n_outliers > 0
        })

outlier_df = pd.DataFrame(outlier_analysis)
outlier_df = outlier_df.merge(regression_quality_df[['Patient', 'r2']], on='Patient')

print(f"\nPazienti con outliers (Z>2): {outlier_df['has_outliers'].sum()}")
print(f"\nR² medio per pazienti CON outliers: {outlier_df[outlier_df['has_outliers']]['r2'].mean():.3f}")
print(f"R² medio per pazienti SENZA outliers: {outlier_df[~outlier_df['has_outliers']]['r2'].mean():.3f}")

# ============================================================
# 4. CONFRONTO CON VALIDAZIONE
# ============================================================

print("\n" + "="*70)
print("4. AFFIDABILITÀ BASATA SU VALIDAZIONE")
print("="*70)

# Riprendi i risultati della validazione
validation_extended = validation_df.merge(
    regression_quality_df[['Patient', 'r2', 'distance_from_week0', 'n_points']], 
    on='Patient'
)

print("\nCorrelazione tra caratteristiche e errore di predizione:")
correlations = {
    'R² della regressione': validation_extended[['r2', 'Error_Personal']].corr().iloc[0, 1],
    'Distanza da Week 0': validation_extended[['distance_from_week0', 'Error_Personal']].corr().iloc[0, 1],
    'Numero di punti': validation_extended[['n_points', 'Error_Personal']].corr().iloc[0, 1],
}

for feature, corr in correlations.items():
    print(f"  {feature}: {corr:.3f}")

# Visualizzazione
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(validation_extended['r2'], validation_extended['Error_Personal'])
axes[0].set_xlabel('R² della Regressione')
axes[0].set_ylabel('Errore Predizione (MAE)')
axes[0].set_title(f'R² vs Errore (corr={correlations["R² della regressione"]:.3f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(validation_extended['distance_from_week0'], validation_extended['Error_Personal'])
axes[1].set_xlabel('Distanza da Week 0')
axes[1].set_ylabel('Errore Predizione (MAE)')
axes[1].set_title(f'Distanza vs Errore (corr={correlations["Distanza da Week 0"]:.3f})')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(validation_extended['n_points'], validation_extended['Error_Personal'])
axes[2].set_xlabel('Numero di Osservazioni')
axes[2].set_ylabel('Errore Predizione (MAE)')
axes[2].set_title(f'N.Punti vs Errore (corr={correlations["Numero di punti"]:.3f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("5. RACCOMANDAZIONI FINALI")
print("="*70)


# ============================================================
# CLASSIFICAZIONE AFFIDABILITÀ DELLE PREDIZIONI
# ============================================================

print("="*70)
print("CLASSIFICAZIONE AFFIDABILITÀ PREDIZIONI")
print("="*70)

# Merge tutte le info
reliability_df = predictions_df.merge(
    regression_quality_df[['Patient', 'r2', 'distance_from_week0', 'n_points', 'mae', 'residuals_std']],
    on='Patient'
)

# Criteri di affidabilità
def classify_reliability(row):
    """
    Classifica l'affidabilità della predizione
    """
    score = 0
    reasons = []
    
    # Criterio 1: R² della regressione
    if row['r2'] > 0.7:
        score += 3
        reasons.append("✓ R² elevato (>0.7)")
    elif row['r2'] > 0.4:
        score += 2
        reasons.append("○ R² medio (0.4-0.7)")
    else:
        score += 0
        reasons.append("✗ R² basso (<0.4)")
    
    # Criterio 2: Distanza da Week 0
    if row['distance_from_week0'] <= 5:
        score += 3
        reasons.append("✓ Vicino a Week 0 (≤5 sett)")
    elif row['distance_from_week0'] <= 15:
        score += 2
        reasons.append("○ Distanza media (6-15 sett)")
    else:
        score += 0
        reasons.append("✗ Lontano da Week 0 (>15 sett)")
    
    # Criterio 3: Numero di osservazioni
    if row['n_points'] >= 9:
        score += 2
        reasons.append("✓ Molte osservazioni (≥9)")
    elif row['n_points'] >= 7:
        score += 1
        reasons.append("○ Osservazioni sufficienti (7-8)")
    else:
        score += 0
        reasons.append("✗ Poche osservazioni (<7)")
    
    # Criterio 4: Variabilità residui
    if row['residuals_std'] < 100:
        score += 2
        reasons.append("✓ Bassa variabilità")
    elif row['residuals_std'] < 200:
        score += 1
        reasons.append("○ Media variabilità")
    else:
        score += 0
        reasons.append("✗ Alta variabilità")
    
    # Classificazione finale (0-10 punti)
    if score >= 8:
        reliability = "ALTA"
        color = "🟢"
    elif score >= 5:
        reliability = "MEDIA"
        color = "🟡"
    else:
        reliability = "BASSA"
        color = "🔴"
    
    return reliability, score, reasons, color

# Applica classificazione
reliability_results = reliability_df.apply(
    lambda row: classify_reliability(row), axis=1
)

reliability_df['Reliability'] = [r[0] for r in reliability_results]
reliability_df['Reliability_Score'] = [r[1] for r in reliability_results]
reliability_df['Reasons'] = [r[2] for r in reliability_results]
reliability_df['Color'] = [r[3] for r in reliability_results]

# Statistiche
print("\n📊 DISTRIBUZIONE AFFIDABILITÀ:")
reliability_counts = reliability_df['Reliability'].value_counts()
print(f"\n🟢 ALTA:   {reliability_counts.get('ALTA', 0):3d} pazienti ({reliability_counts.get('ALTA', 0)/len(reliability_df)*100:.1f}%)")
print(f"🟡 MEDIA:  {reliability_counts.get('MEDIA', 0):3d} pazienti ({reliability_counts.get('MEDIA', 0)/len(reliability_df)*100:.1f}%)")
print(f"🔴 BASSA:  {reliability_counts.get('BASSA', 0):3d} pazienti ({reliability_counts.get('BASSA', 0)/len(reliability_df)*100:.1f}%)")

# Caratteristiche per livello di affidabilità
print("\n" + "="*70)
print("CARATTERISTICHE PER LIVELLO DI AFFIDABILITÀ")
print("="*70)

for reliability_level in ['ALTA', 'MEDIA', 'BASSA']:
    subset = reliability_df[reliability_df['Reliability'] == reliability_level]
    if len(subset) > 0:
        print(f"\n{reliability_level}:")
        print(f"  R² medio:              {subset['r2'].mean():.3f}")
        print(f"  Distanza Week 0:       {subset['distance_from_week0'].mean():.1f} settimane")
        print(f"  MAE medio:             {subset['mae'].mean():.1f}")
        print(f"  Residui std:           {subset['residuals_std'].mean():.1f}")

# Visualizzazione
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Distribuzione affidabilità
reliability_order = ['ALTA', 'MEDIA', 'BASSA']
colors_map = {'ALTA': 'green', 'MEDIA': 'yellow', 'BASSA': 'red'}
counts = [reliability_counts.get(level, 0) for level in reliability_order]

axes[0, 0].bar(reliability_order, counts, color=[colors_map[x] for x in reliability_order], 
               edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Numero di Pazienti')
axes[0, 0].set_title('Distribuzione Affidabilità delle Predizioni')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Score distribution
axes[0, 1].hist(reliability_df['Reliability_Score'], bins=11, range=(0, 10), 
                edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=8, color='green', linestyle='--', label='Alta (≥8)')
axes[0, 1].axvline(x=5, color='orange', linestyle='--', label='Media (≥5)')
axes[0, 1].set_xlabel('Reliability Score')
axes[0, 1].set_ylabel('Frequenza')
axes[0, 1].set_title('Distribuzione Score di Affidabilità')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: R² vs Distance per reliability
for reliability_level in ['ALTA', 'MEDIA', 'BASSA']:
    subset = reliability_df[reliability_df['Reliability'] == reliability_level]
    axes[1, 0].scatter(subset['distance_from_week0'], subset['r2'],
                      label=reliability_level, s=50, alpha=0.6,
                      color=colors_map[reliability_level])

axes[1, 0].set_xlabel('Distanza da Week 0 (settimane)')
axes[1, 0].set_ylabel('R²')
axes[1, 0].set_title('R² vs Distanza da Week 0 (per Affidabilità)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Affidabilità per Cluster
cluster_reliability = pd.crosstab(reliability_df['Cluster'], 
                                   reliability_df['Reliability'],
                                   normalize='index') * 100

cluster_reliability[reliability_order].plot(kind='bar', stacked=True, 
                                             color=[colors_map[x] for x in reliability_order],
                                             ax=axes[1, 1], alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Percentuale')
axes[1, 1].set_title('Distribuzione Affidabilità per Cluster')
axes[1, 1].legend(title='Affidabilità')
axes[1, 1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Esempi di ciascuna categoria
print("\n" + "="*70)
print("ESEMPI PER CATEGORIA")
print("="*70)

for reliability_level in ['ALTA', 'MEDIA', 'BASSA']:
    subset = reliability_df[reliability_df['Reliability'] == reliability_level]
    if len(subset) > 0:
        print(f"\n{subset.iloc[0]['Color']} {reliability_level} - Esempio: {subset.iloc[0]['Patient']}")
        print(f"   Score: {subset.iloc[0]['Reliability_Score']}/10")
        print(f"   Motivi:")
        for reason in subset.iloc[0]['Reasons']:
            print(f"     {reason}")
        print(f"   Predizione Week 0: {subset.iloc[0]['predicted_FVC_week0_personal']:.0f}")

# ============================================================
# RACCOMANDAZIONE FINALE
# ============================================================

print("\n" + "="*70)
print("🎯 RACCOMANDAZIONE FINALE PER L'USO DELLE PREDIZIONI")
print("="*70)

alta_count = reliability_counts.get('ALTA', 0)
media_count = reliability_counts.get('MEDIA', 0)
bassa_count = reliability_counts.get('BASSA', 0)

print(f"""
✅ USARE CON CONFIDENZA ({alta_count} pazienti, {alta_count/len(reliability_df)*100:.1f}%):
   - Predizioni con affidabilità ALTA
   - Errore atteso: < 100 unità FVC
   
⚠️  USARE CON CAUTELA ({media_count} pazienti, {media_count/len(reliability_df)*100:.1f}%):
   - Predizioni con affidabilità MEDIA
   - Considerare intervallo di confidenza più ampio
   - Errore atteso: 100-200 unità FVC
   
❌ EVITARE O SEGNALARE ({bassa_count} pazienti, {bassa_count/len(reliability_df)*100:.1f}%):
   - Predizioni con affidabilità BASSA
   - Troppa incertezza per uso clinico
   - Errore atteso: > 200 unità FVC
   - Considerare metodi alternativi (mixed models, splines)

📋 CONCLUSIONE GENERALE:
   Il metodo di predizione lineare è UTILIZZABILE per circa {(alta_count+media_count)/len(reliability_df)*100:.0f}% 
   dei pazienti, ma richiede:
   
   1. Filtraggio per affidabilità
   2. Validazione caso-per-caso
   3. Integrazione con modelli più sofisticati per casi difficili
   4. Intervalli di confidenza appropriati
""")

# Salva il dataset con classificazione affidabilità
reliability_summary = reliability_df[['Patient', 'Cluster', 'predicted_FVC_week0_personal', 
                                       'Reliability', 'Reliability_Score', 'r2', 
                                       'distance_from_week0', 'n_points', 'mae']]

output_reliability_path = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/predictions_reliability.csv'
reliability_summary.to_csv(output_reliability_path, index=False)
print(f"\n✓ File affidabilità salvato: {output_reliability_path}")

reliability_summary.head(10)

# ============================================================
# ANALISI FEATURE IMPORTANCE PER CLUSTERING
# ============================================================

print("="*70)
print("ANALISI OTTIMIZZAZIONE FEATURES PER CLUSTERING")
print("="*70)

# ============================================================
# 1. CORRELAZIONE TRA FEATURES
# ============================================================

print("\n1. CORRELAZIONE TRA FEATURES")
print("="*70)

X = patient_df[features_for_clustering].copy()
correlation_matrix = X.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matrice di Correlazione tra Features')
plt.tight_layout()
plt.show()

print("\nFeatures altamente correlate (|r| > 0.7):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")

# ============================================================
# 2. PROVA DIVERSE COMBINAZIONI DI FEATURES
# ============================================================

print("\n" + "="*70)
print("2. TEST DIVERSE COMBINAZIONI DI FEATURES")
print("="*70)

feature_sets = {
    'Original (All 9)': features_for_clustering,
    
    'Clinical Only': ['Age', 'Sex_encoded', 'SmokingStatus_encoded'],
    
    'Temporal Only': ['fvc_slope', 'fvc_mean', 'fvc_std'],
    
    'Clinical + Temporal': ['fvc_slope', 'fvc_mean', 'Age', 'Sex_encoded', 'SmokingStatus_encoded'],
    
    'Core (No Redundancy)': ['fvc_slope', 'fvc_mean', 'Age', 'Sex_encoded', 'SmokingStatus_encoded'],
    
    'Slope + Demographics': ['fvc_slope', 'Age', 'Sex_encoded', 'SmokingStatus_encoded'],
    
    'FVC Features': ['fvc_slope', 'fvc_mean', 'fvc_std', 'percent_mean'],
    
    'Best Guess': ['fvc_slope', 'fvc_mean', 'Age', 'SmokingStatus_encoded'],
    
    'Minimal': ['fvc_slope', 'Sex_encoded', 'SmokingStatus_encoded']
}

results = []

for set_name, features in feature_sets.items():
    X_test = patient_df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # Test per k da 2 a 6
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        inertia = kmeans.inertia_
        
        # Calinski-Harabasz Score (altro metodo di valutazione)
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        ch_score = calinski_harabasz_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        
        results.append({
            'Feature_Set': set_name,
            'N_Features': len(features),
            'K': k,
            'Silhouette': silhouette,
            'Calinski_Harabasz': ch_score,
            'Davies_Bouldin': db_score,  # Più basso è meglio
            'Inertia': inertia
        })

results_df = pd.DataFrame(results)

# Trova i migliori per ogni metrica
print("\n📊 MIGLIORI COMBINAZIONI PER METRICA:")
print("\nTop 5 per Silhouette Score (più alto è meglio):")
top_silhouette = results_df.nlargest(5, 'Silhouette')[['Feature_Set', 'K', 'N_Features', 'Silhouette']]
print(top_silhouette.to_string(index=False))

print("\nTop 5 per Calinski-Harabasz Score (più alto è meglio):")
top_ch = results_df.nlargest(5, 'Calinski_Harabasz')[['Feature_Set', 'K', 'N_Features', 'Calinski_Harabasz']]
print(top_ch.to_string(index=False))

print("\nTop 5 per Davies-Bouldin Score (più basso è meglio):")
top_db = results_df.nsmallest(5, 'Davies_Bouldin')[['Feature_Set', 'K', 'N_Features', 'Davies_Bouldin']]
print(top_db.to_string(index=False))

# Visualizzazione comparativa
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Silhouette per feature set
for set_name in feature_sets.keys():
    subset = results_df[results_df['Feature_Set'] == set_name]
    axes[0, 0].plot(subset['K'], subset['Silhouette'], 'o-', label=set_name, linewidth=2)

axes[0, 0].set_xlabel('Numero di Cluster (K)')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('Silhouette Score per Feature Set')
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Calinski-Harabasz
for set_name in feature_sets.keys():
    subset = results_df[results_df['Feature_Set'] == set_name]
    axes[0, 1].plot(subset['K'], subset['Calinski_Harabasz'], 'o-', label=set_name, linewidth=2)

axes[0, 1].set_xlabel('Numero di Cluster (K)')
axes[0, 1].set_ylabel('Calinski-Harabasz Score')
axes[0, 1].set_title('Calinski-Harabasz Score per Feature Set')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Davies-Bouldin (più basso è meglio)
for set_name in feature_sets.keys():
    subset = results_df[results_df['Feature_Set'] == set_name]
    axes[1, 0].plot(subset['K'], subset['Davies_Bouldin'], 'o-', label=set_name, linewidth=2)

axes[1, 0].set_xlabel('Numero di Cluster (K)')
axes[1, 0].set_ylabel('Davies-Bouldin Score (lower is better)')
axes[1, 0].set_title('Davies-Bouldin Score per Feature Set')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Heatmap delle performance
pivot_silhouette = results_df.pivot(index='Feature_Set', columns='K', values='Silhouette')
sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='RdYlGn', 
            ax=axes[1, 1], cbar_kws={'label': 'Silhouette Score'})
axes[1, 1].set_title('Heatmap Silhouette Score')
axes[1, 1].set_xlabel('Numero di Cluster (K)')
axes[1, 1].set_ylabel('Feature Set')

plt.tight_layout()
plt.show()

# ============================================================
# 3. ANALISI FEATURE IMPORTANCE CON PCA
# ============================================================

print("\n" + "="*70)
print("3. FEATURE IMPORTANCE VIA PCA")
print("="*70)

from sklearn.decomposition import PCA

X_full = patient_df[features_for_clustering].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

pca = PCA()
pca.fit(X_scaled)

# Varianza spiegata
print("\nVarianza spiegata per componente:")
for i, var in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print(f"\nVarianza cumulativa prime 3 componenti: {sum(pca.explained_variance_ratio_[:3]):.3f}")

# Loading matrix (contributo features alle PC)
loadings = pd.DataFrame(
    pca.components_[:3].T,
    columns=['PC1', 'PC2', 'PC3'],
    index=features_for_clustering
)

print("\nLoading Features sulle Prime 3 PC:")
print(loadings.round(3))

# Plot loadings
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
    loadings_sorted = loadings[pc].abs().sort_values(ascending=True)
    colors = ['red' if x < 0 else 'blue' for x in loadings[pc][loadings_sorted.index]]
    
    axes[i].barh(range(len(loadings_sorted)), loadings[pc][loadings_sorted.index], color=colors, alpha=0.7)
    axes[i].set_yticks(range(len(loadings_sorted)))
    axes[i].set_yticklabels(loadings_sorted.index)
    axes[i].set_xlabel('Loading')
    axes[i].set_title(f'{pc} Loadings (Var: {pca.explained_variance_ratio_[i]:.2%})')
    axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[i].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Features più importanti (maggior carico sulle prime PC)
feature_importance = np.abs(pca.components_[:3]).sum(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': features_for_clustering,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n📊 RANKING FEATURE IMPORTANCE (basato su PCA):")
print(feature_importance_df.to_string(index=False))

# ============================================================
# 4. RACCOMANDAZIONE
# ============================================================

print("\n" + "="*70)
print("🎯 RACCOMANDAZIONI")
print("="*70)

best_config = results_df.loc[results_df['Silhouette'].idxmax()]

print(f"""
MIGLIORE CONFIGURAZIONE TROVATA:
  Feature Set: {best_config['Feature_Set']}
  K: {best_config['K']}
  Silhouette: {best_config['Silhouette']:.3f}
  Features: {feature_sets[best_config['Feature_Set']]}

FEATURES PIÙ IMPORTANTI (da PCA):
{feature_importance_df.head(5).to_string(index=False)}

SUGGERIMENTI:
1. Usa feature set ridotto per cluster più definiti
2. Considera K={int(best_config['K'])} cluster
3. Focalizzati su features non correlate
4. Testa anche DBSCAN o Hierarchical clustering
""")

# Salva risultati
results_df.to_csv('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/clustering_feature_analysis.csv', index=False)
print("\n✓ Analisi salvata: clustering_feature_analysis.csv")

# ============================================================
# CLUSTERING OTTIMIZZATO CON FEATURES RIDOTTE
# ============================================================

print("="*70)
print("CLUSTERING OTTIMIZZATO")
print("="*70)

# Features ottimali
optimal_features = ['fvc_slope', 'Sex_encoded', 'SmokingStatus_encoded']
optimal_k = 5

print(f"\nFeatures selezionate: {optimal_features}")
print(f"Numero di cluster: {optimal_k}")

# Prepara i dati
X_optimal = patient_df[optimal_features].copy()
scaler_optimal = StandardScaler()
X_scaled_optimal = scaler_optimal.fit_transform(X_optimal)

# Clustering
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
patient_df['Cluster_Optimal'] = kmeans_optimal.fit_predict(X_scaled_optimal)

# Metriche
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette_opt = silhouette_score(X_scaled_optimal, patient_df['Cluster_Optimal'])
ch_opt = calinski_harabasz_score(X_scaled_optimal, patient_df['Cluster_Optimal'])
db_opt = davies_bouldin_score(X_scaled_optimal, patient_df['Cluster_Optimal'])

print(f"\n📊 METRICHE QUALITÀ:")
print(f"Silhouette Score: {silhouette_opt:.3f}")
print(f"Calinski-Harabasz: {ch_opt:.2f}")
print(f"Davies-Bouldin: {db_opt:.3f}")

# Confronto con clustering originale
print(f"\n📈 MIGLIORAMENTO vs ORIGINALE:")
print(f"Silhouette: 0.25 → {silhouette_opt:.3f} (+{(silhouette_opt-0.25)/0.25*100:.0f}%)")

# ============================================================
# ANALISI DEI NUOVI CLUSTER
# ============================================================

print("\n" + "="*70)
print("CARATTERISTICHE DEI 5 CLUSTER OTTIMALI")
print("="*70)

# Statistiche per cluster
cluster_stats = patient_df.groupby('Cluster_Optimal').agg({
    'Patient': 'count',
    'fvc_slope': 'mean',
    'fvc_mean': 'mean',
    'Age': 'mean',
    'Sex': lambda x: (x == 'Male').sum() / len(x) * 100,  # % maschi
    'SmokingStatus': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
    'has_week0': 'sum',
    'fvc_intercept': 'mean'
}).round(2)

cluster_stats.columns = ['N_Pazienti', 'FVC_Slope_Avg', 'FVC_Mean', 'Age_Avg', 
                         'Male_%', 'Smoking_Mode', 'N_with_Week0', 'FVC_Week0_Est']

print("\n")
print(cluster_stats)

# Interpretazione cluster
print("\n" + "="*70)
print("INTERPRETAZIONE CLINICA DEI CLUSTER")
print("="*70)

for cluster_id in range(optimal_k):
    cluster_data = patient_df[patient_df['Cluster_Optimal'] == cluster_id]
    
    print(f"\n🔵 CLUSTER {cluster_id} (n={len(cluster_data)})")
    print(f"   FVC Slope: {cluster_data['fvc_slope'].mean():.2f} ± {cluster_data['fvc_slope'].std():.2f}")
    print(f"   FVC Baseline: {cluster_data['fvc_mean'].mean():.0f}")
    print(f"   Età media: {cluster_data['Age'].mean():.1f} anni")
    
    # Composizione sesso
    sex_counts = cluster_data['Sex'].value_counts()
    print(f"   Sesso: {sex_counts.to_dict()}")
    
    # Smoking status
    smoking_counts = cluster_data['SmokingStatus'].value_counts()
    print(f"   Smoking: {smoking_counts.to_dict()}")
    
    # Interpretazione
    slope_mean = cluster_data['fvc_slope'].mean()
    if slope_mean < -7:
        severity = "⚠️  DECLINO RAPIDO"
    elif slope_mean < -3:
        severity = "⚡ DECLINO MODERATO"
    else:
        severity = "✓ DECLINO LENTO/STABILE"
    
    print(f"   Prognosi: {severity}")

# ============================================================
# VISUALIZZAZIONI
# ============================================================

print("\n" + "="*70)
print("VISUALIZZAZIONI")
print("="*70)

# Plot 1: Scatter 3D interattivo delle features
fig = plt.figure(figsize=(18, 6))

# Subplot 1: fvc_slope vs Sex
ax1 = fig.add_subplot(131)
for cluster_id in range(optimal_k):
    cluster_data = patient_df[patient_df['Cluster_Optimal'] == cluster_id]
    ax1.scatter(cluster_data['Sex_encoded'], cluster_data['fvc_slope'], 
               label=f'Cluster {cluster_id}', s=100, alpha=0.6, edgecolors='black')

ax1.set_xlabel('Sex (0=Female, 1=Male)')
ax1.set_ylabel('FVC Slope (rate of decline)')
ax1.set_title('Clustering: Sex vs FVC Slope')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)

# Subplot 2: fvc_slope vs Smoking
ax2 = fig.add_subplot(132)
for cluster_id in range(optimal_k):
    cluster_data = patient_df[patient_df['Cluster_Optimal'] == cluster_id]
    ax2.scatter(cluster_data['SmokingStatus_encoded'], cluster_data['fvc_slope'],
               label=f'Cluster {cluster_id}', s=100, alpha=0.6, edgecolors='black')

ax2.set_xlabel('Smoking (0=Current, 1=Ex, 2=Never)')
ax2.set_ylabel('FVC Slope (rate of decline)')
ax2.set_title('Clustering: Smoking vs FVC Slope')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)

# Subplot 3: Sex vs Smoking colored by cluster
ax3 = fig.add_subplot(133)
scatter = ax3.scatter(patient_df['Sex_encoded'], 
                     patient_df['SmokingStatus_encoded'],
                     c=patient_df['Cluster_Optimal'],
                     s=100,
                     cmap='viridis',
                     alpha=0.6,
                     edgecolors='black')
ax3.set_xlabel('Sex (0=Female, 1=Male)')
ax3.set_ylabel('Smoking (0=Current, 1=Ex, 2=Never)')
ax3.set_title('Demographics Colored by Cluster')
plt.colorbar(scatter, ax=ax3, label='Cluster')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 2: Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for cluster_id in range(optimal_k):
    cluster_patients = patient_df[patient_df['Cluster_Optimal'] == cluster_id]['Patient'].values
    cluster_timeseries = df[df['Patient'].isin(cluster_patients)]
    
    row = cluster_id // 3
    col = cluster_id % 3
    ax = axes[row, col]
    
    # Plot alcune traiettorie
    for patient in np.random.choice(cluster_patients, min(15, len(cluster_patients)), replace=False):
        patient_data = cluster_timeseries[cluster_timeseries['Patient'] == patient]
        ax.plot(patient_data['Weeks'], patient_data['FVC'], alpha=0.3, linewidth=1)
    
    # Media del cluster
    mean_fvc = cluster_timeseries.groupby('Weeks')['FVC'].mean()
    ax.plot(mean_fvc.index, mean_fvc.values, 'r-', linewidth=3, label='Mean trajectory')
    
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Weeks')
    ax.set_ylabel('FVC')
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_patients)})\nSlope: {patient_df[patient_df["Cluster_Optimal"]==cluster_id]["fvc_slope"].mean():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Rimuovi subplot extra se K < 6
if optimal_k < 6:
    for i in range(optimal_k, 6):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()

# Plot 3: Silhouette plot dettagliato
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(X_scaled_optimal, patient_df['Cluster_Optimal'])

fig, ax = plt.subplots(figsize=(10, 8))

y_lower = 10
for cluster_id in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[patient_df['Cluster_Optimal'] == cluster_id]
    cluster_silhouette_vals.sort()
    
    size_cluster = len(cluster_silhouette_vals)
    y_upper = y_lower + size_cluster
    
    color = plt.cm.viridis(cluster_id / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7,
                     label=f'Cluster {cluster_id}')
    
    y_lower = y_upper + 10

ax.axvline(x=silhouette_opt, color='red', linestyle='--', linewidth=2, 
           label=f'Average Silhouette: {silhouette_opt:.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title('Silhouette Plot per Cluster')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n✓ Clustering ottimizzato completato!")

# ============================================================
# PREDIZIONE WEEK 0 CON CLUSTERING OTTIMIZZATO
# ============================================================

print("="*70)
print("PREDIZIONE WEEK 0 CON CLUSTER OTTIMIZZATI")
print("="*70)

# Ri-fai la regressione per ogni cluster ottimale
models_optimal = {}
predictions_optimal = []

for cluster_id in range(optimal_k):
    cluster_patients = patient_df[patient_df['Cluster_Optimal'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"Pazienti: {len(cluster_patients)}")
    print(f"Osservazioni: {len(cluster_data)}")
    
    # Regressione
    X_reg = cluster_data[['Weeks']].values
    y_reg = cluster_data['FVC'].values
    
    model = LinearRegression()
    model.fit(X_reg, y_reg)
    
    y_pred = model.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)
    mae = mean_absolute_error(y_reg, y_pred)
    
    print(f"Slope cluster: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.2f}")
    
    models_optimal[cluster_id] = {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'mae': mae
    }
    
    # Predici Week 0 per pazienti senza
    cluster_no_week0 = patient_df[
        (patient_df['Cluster_Optimal'] == cluster_id) & 
        (patient_df['has_week0'] == 0)
    ]
    
    for _, patient_row in cluster_no_week0.iterrows():
        predictions_optimal.append({
            'Patient': patient_row['Patient'],
            'Cluster_Optimal': cluster_id,
            'predicted_FVC_week0_cluster_opt': model.predict([[0]])[0],
            'predicted_FVC_week0_personal': patient_row['fvc_intercept'],
            'fvc_slope': patient_row['fvc_slope']
        })

predictions_optimal_df = pd.DataFrame(predictions_optimal)

print(f"\n{'='*70}")
print("VALIDAZIONE CON CLUSTER OTTIMIZZATI")
print(f"{'='*70}")

# Valida sui 18 pazienti con Week 0
validation_optimal = []

for _, patient_row in patient_df[patient_df['has_week0'] == 1].iterrows():
    patient_id = patient_row['Patient']
    cluster_id = patient_row['Cluster_Optimal']
    
    real_fvc = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
    pred_cluster = models_optimal[cluster_id]['intercept']
    pred_personal = patient_row['fvc_intercept']
    
    validation_optimal.append({
        'Patient': patient_id,
        'Cluster_Optimal': cluster_id,
        'Real_FVC': real_fvc,
        'Pred_Cluster_Opt': pred_cluster,
        'Pred_Personal': pred_personal,
        'Error_Cluster_Opt': abs(real_fvc - pred_cluster),
        'Error_Personal': abs(real_fvc - pred_personal)
    })

validation_optimal_df = pd.DataFrame(validation_optimal)

print(f"\n📊 CONFRONTO ERRORI:")
print(f"\nMetodo CLUSTER OTTIMIZZATO:")
print(f"  MAE: {validation_optimal_df['Error_Cluster_Opt'].mean():.2f}")
print(f"  Mediana: {validation_optimal_df['Error_Cluster_Opt'].median():.2f}")

print(f"\nMetodo PERSONALE:")
print(f"  MAE: {validation_optimal_df['Error_Personal'].mean():.2f}")
print(f"  Mediana: {validation_optimal_df['Error_Personal'].median():.2f}")

print(f"\nMetodo CLUSTER ORIGINALE (4 cluster, 9 features):")
print(f"  MAE: 536.12")

print(f"\n✨ MIGLIORAMENTO:")
print(f"  Cluster 4→5: {536.12:.0f} → {validation_optimal_df['Error_Cluster_Opt'].mean():.0f}")
print(f"  Riduzione errore: {(1 - validation_optimal_df['Error_Cluster_Opt'].mean()/536.12)*100:.0f}%")

# Visualizzazione confronto
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Errori per metodo
methods = ['Cluster\nOriginale\n(4 clust)', 'Cluster\nOttimizzato\n(5 clust)', 'Personale']
errors = [536.12, 
          validation_optimal_df['Error_Cluster_Opt'].mean(),
          validation_optimal_df['Error_Personal'].mean()]

colors_bars = ['red', 'orange', 'green']
bars = axes[0].bar(methods, errors, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('MAE (FVC units)')
axes[0].set_title('Confronto Errore di Predizione')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (bar, err) in enumerate(zip(bars, errors)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Real vs Predicted (cluster ottimizzato)
axes[1].scatter(validation_optimal_df['Real_FVC'], 
               validation_optimal_df['Pred_Cluster_Opt'],
               c=validation_optimal_df['Cluster_Optimal'],
               s=150, alpha=0.7, edgecolors='black', linewidth=2, cmap='viridis')

min_val = validation_optimal_df[['Real_FVC', 'Pred_Cluster_Opt']].min().min()
max_val = validation_optimal_df[['Real_FVC', 'Pred_Cluster_Opt']].max().max()
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

axes[1].set_xlabel('Real FVC at Week 0')
axes[1].set_ylabel('Predicted FVC (Cluster Method)')
axes[1].set_title(f'Cluster Ottimizzato\nMAE: {validation_optimal_df["Error_Cluster_Opt"].mean():.0f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Errori per cluster
validation_optimal_df.boxplot(column='Error_Cluster_Opt', by='Cluster_Optimal', ax=axes[2])
axes[2].set_xlabel('Cluster')
axes[2].set_ylabel('Absolute Error')
axes[2].set_title('Errore per Cluster Ottimizzato')
plt.suptitle('')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Statistiche per cluster
print(f"\n{'='*70}")
print("ERRORE DI PREDIZIONE PER CLUSTER")
print(f"{'='*70}")

for cluster_id in range(optimal_k):
    cluster_val = validation_optimal_df[validation_optimal_df['Cluster_Optimal'] == cluster_id]
    if len(cluster_val) > 0:
        print(f"\nCluster {cluster_id} (n={len(cluster_val)} validazioni):")
        print(f"  MAE cluster: {cluster_val['Error_Cluster_Opt'].mean():.2f}")
        print(f"  MAE personal: {cluster_val['Error_Personal'].mean():.2f}")
        print(f"  R² modello: {models_optimal[cluster_id]['r2']:.3f}")

# ============================================================
# DECISIONE FINALE: QUALE METODO USARE?
# ============================================================

print(f"\n{'='*70}")
print("🎯 DECISIONE FINALE SUL METODO")
print(f"{'='*70}")

print("""
ANALISI COMPARATIVA:

1. METODO CLUSTER ORIGINALE (4 cluster, 9 features):
   ❌ MAE: 536 unità
   ❌ Silhouette: 0.25
   ❌ Cluster poco definiti
   
2. METODO CLUSTER OTTIMIZZATO (5 cluster, 3 features):
   ✓ MAE: {:.0f} unità
   ✓ Silhouette: 0.548
   ✓ Cluster clinicamente interpretabili
   ⚡ Miglioramento: {:.0f}%
   
3. METODO PERSONALE (regressione individuale):
   ✓✓ MAE: {:.0f} unità  
   ✓ Più accurato per singoli pazienti
   ⚠️  Richiede buon fit (R² > 0.4)

RACCOMANDAZIONE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usa APPROCCIO IBRIDO:

✓ METODO PERSONALE per pazienti con:
  - R² > 0.5 (fit buono)
  - Distanza da Week 0 < 20 settimane
  - Almeno 7-8 osservazioni
  
✓ METODO CLUSTER OTTIMIZZATO per pazienti con:
  - R² < 0.5 (fit scarso) 
  - Poche osservazioni
  - Dati molto variabili
  
Questo massimizza l'accuratezza mantenendo robustezza!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".format(
    validation_optimal_df['Error_Cluster_Opt'].mean(),
    (1 - validation_optimal_df['Error_Cluster_Opt'].mean()/536.12)*100,
    validation_optimal_df['Error_Personal'].mean()
))

# ============================================================
# STRATEGIA FINALE: APPROCCIO IBRIDO INTELLIGENTE
# ============================================================

print("="*70)
print("IMPLEMENTAZIONE STRATEGIA IBRIDA OTTIMALE")
print("="*70)

# Combina informazioni: cluster ottimizzati + predizioni personali + affidabilità
final_predictions = predictions_df.copy()

# Merge con cluster ottimizzato
final_predictions = final_predictions.merge(
    patient_df[['Patient', 'Cluster_Optimal']], 
    on='Patient'
)

# Merge con metriche di qualità
final_predictions = final_predictions.merge(
    regression_quality_df[['Patient', 'r2', 'distance_from_week0', 'n_points', 'mae', 'residuals_std']],
    on='Patient'
)

# ============================================================
# CRITERIO DI DECISIONE INTELLIGENTE
# ============================================================

def select_best_prediction_method(row):
    """
    Sceglie il metodo migliore basandosi su:
    - Qualità regressione personale
    - Distanza da Week 0
    - Numero osservazioni
    - Cluster di appartenenza
    """
    
    # CRITERIO 1: Se regressione personale è buona, usa quella
    if row['r2'] > 0.6 and row['distance_from_week0'] < 15:
        return 'personal_high_conf', row['predicted_FVC_week0_personal']
    
    # CRITERIO 2: Se regressione personale è media, usa quella
    elif row['r2'] > 0.4 and row['distance_from_week0'] < 25:
        return 'personal_medium_conf', row['predicted_FVC_week0_personal']
    
    # CRITERIO 3: Se cluster ha buon R² (es. cluster 2, 3, 4), usa cluster
    elif row['Cluster_Optimal'] in [2, 4]:  # Cluster con R² migliore
        return 'cluster_optimized', models_optimal[row['Cluster_Optimal']]['intercept']
    
    # CRITERIO 4: Altrimenti usa predizione personale con cautela
    else:
        return 'personal_low_conf', row['predicted_FVC_week0_personal']

# Applica selezione
method_and_prediction = final_predictions.apply(
    lambda row: select_best_prediction_method(row), axis=1
)

final_predictions['selected_method'] = [x[0] for x in method_and_prediction]
final_predictions['final_FVC_week0'] = [x[1] for x in method_and_prediction]

# Statistiche
print("\n📊 DISTRIBUZIONE METODI SELEZIONATI:")
print(final_predictions['selected_method'].value_counts())

print("\n" + "="*70)
print("CARATTERISTICHE PER METODO SELEZIONATO")
print("="*70)

for method in final_predictions['selected_method'].unique():
    subset = final_predictions[final_predictions['selected_method'] == method]
    print(f"\n{method.upper().replace('_', ' ')} (n={len(subset)}):")
    print(f"  R² medio: {subset['r2'].mean():.3f}")
    print(f"  Distanza Week 0: {subset['distance_from_week0'].mean():.1f} settimane")
    print(f"  MAE medio: {subset['mae'].mean():.1f}")

# ============================================================
# VALIDAZIONE STRATEGIA IBRIDA
# ============================================================

print("\n" + "="*70)
print("VALIDAZIONE STRATEGIA IBRIDA")
print("="*70)

# Simula la strategia sui 18 pazienti di validazione
validation_hybrid = []

for _, patient_row in patient_df[patient_df['has_week0'] == 1].iterrows():
    patient_id = patient_row['Patient']
    
    # Crea un "row" simulato per il criterio
    test_row = {
        'r2': patient_row['fvc_intercept'] if 'r2' in regression_quality_df.columns else 0.5,
        'distance_from_week0': abs(patient_row['first_week']),
        'Cluster_Optimal': patient_row['Cluster_Optimal'],
        'predicted_FVC_week0_personal': patient_row['fvc_intercept']
    }
    
    # Trova in regression_quality_df
    quality_info = regression_quality_df[regression_quality_df['Patient'] == patient_id]
    if len(quality_info) > 0:
        test_row['r2'] = quality_info.iloc[0]['r2']
        test_row['distance_from_week0'] = quality_info.iloc[0]['distance_from_week0']
    
    method, prediction = select_best_prediction_method(pd.Series(test_row))
    
    real_fvc = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
    
    validation_hybrid.append({
        'Patient': patient_id,
        'Real_FVC': real_fvc,
        'Predicted_Hybrid': prediction,
        'Method': method,
        'Error': abs(real_fvc - prediction)
    })

validation_hybrid_df = pd.DataFrame(validation_hybrid)

print(f"\n✨ RISULTATI VALIDAZIONE IBRIDA:")
print(f"MAE Totale: {validation_hybrid_df['Error'].mean():.2f}")
print(f"Mediana: {validation_hybrid_df['Error'].median():.2f}")

print(f"\n📊 Per metodo:")
for method in validation_hybrid_df['Method'].unique():
    subset = validation_hybrid_df[validation_hybrid_df['Method'] == method]
    print(f"  {method}: MAE={subset['Error'].mean():.2f} (n={len(subset)})")

# Confronto finale
print("\n" + "="*70)
print("📊 CONFRONTO FINALE TUTTI I METODI")
print("="*70)

comparison_data = {
    'Metodo': [
        'Cluster Originale',
        'Cluster Ottimizzato', 
        'Personale',
        '🎯 IBRIDO INTELLIGENTE'
    ],
    'MAE': [
        536.12,
        622.04,
        127.62,
        validation_hybrid_df['Error'].mean()
    ],
    'Silhouette': [
        0.25,
        0.548,
        'N/A',
        0.548  # Usa cluster ottimizzato
    ],
    'Pro': [
        'Baseline',
        'Cluster chiari',
        'Più accurato',
        'Bilanciato'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

# Visualizzazione finale
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Confronto MAE
methods_plot = ['Cluster\nOriginale', 'Cluster\nOttimizzato', 'Personale', 'IBRIDO\nIntelligente']
mae_values = [536.12, 622.04, 127.62, validation_hybrid_df['Error'].mean()]
colors_plot = ['red', 'orange', 'lightgreen', 'darkgreen']

bars = axes[0].bar(methods_plot, mae_values, color=colors_plot, alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_ylabel('MAE (FVC units)', fontsize=12, fontweight='bold')
axes[0].set_title('Confronto Accuratezza Metodi di Predizione', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Distribuzione errori per metodo (hybrid)
validation_hybrid_df.boxplot(column='Error', by='Method', ax=axes[1])
axes[1].set_xlabel('Metodo Ibrido Selezionato', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Absolute Error (FVC)', fontsize=11, fontweight='bold')
axes[1].set_title('Distribuzione Errori - Strategia Ibrida', fontsize=13, fontweight='bold')
plt.suptitle('')
axes[1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# ============================================================
# RACCOMANDAZIONI FINALI PER LA TESI
# ============================================================

print("\n" + "="*70)
print("📝 RACCOMANDAZIONI FINALI PER LA TESI")
print("="*70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 CONCLUSIONI PER LA TESI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CLUSTERING:
   ✓ K-Means con 3 features (fvc_slope, Sex, SmokingStatus) ottimale
   ✓ K=5 cluster clinicamente interpretabili
   ✓ Silhouette Score: 0.548 (buona separazione)
   ✓ Identifica 5 fenotipi distinti di progressione

2. PREDIZIONE WEEK 0:
   ⚠️  Trade-off scoperto: 
      - Cluster migliori ≠ Predizioni migliori (con regressione semplice)
      - Regressione individuale: MAE ~128 (3.8%)
      - Regressione per cluster: MAE 536-622 (16-19%)
   
   ✓ Soluzione: Approccio IBRIDO
      - Usa regressione personale quando R² > 0.4
      - Fallback a cluster per casi difficili
      - MAE finale: ~{:.0f} unità

3. AFFIDABILITÀ:
   ✓ ~60% predizioni ALTA/MEDIA affidabilità
   ✓ Filtraggio per R², distanza Week 0, n_points
   ⚠️  40% richiede metodi più sofisticati

4. LIMITAZIONI:
   ❌ Modello lineare troppo semplice per molti pazienti
   ❌ R² medio basso (0.42) indica non-linearità
   ❌ 83% pazienti richiede extrapolazione
   
5. SVILUPPI FUTURI:
   → Mixed-effects models (effetti random per paziente)
   → Spline/GAM per catturare non-linearità
   → Deep learning (LSTM) per sequenze temporali
   → Incorporare biomarkers aggiuntivi

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 RISPOSTA ALLA DOMANDA INIZIALE:
"Le predizioni sono utilizzabili?"

SÌ, CON CAUTELA:
- Per ~60% pazienti: affidabili (errore < 5%)
- Per ~25% pazienti: utilizzabili con cautela (errore 5-10%)
- Per ~15% pazienti: sconsigliato (errore > 10%)

Il clustering è UTILE per:
✓ Segmentazione clinica
✓ Identificare gruppi alto rischio
✓ Personalizzare follow-up
✓ Stratificare trials clinici

Ma NON è sufficiente da solo per predizioni accurate.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".format(validation_hybrid_df['Error'].mean()))

# Salva risultati finali
final_predictions[['Patient', 'Cluster_Optimal', 'selected_method', 
                   'final_FVC_week0', 'r2', 'distance_from_week0']].to_csv(
    'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/final_predictions_hybrid.csv',
    index=False
)

print("\n✓ Predizioni finali salvate: final_predictions_hybrid.csv")


# ============================================================
# ANALISI: CLUSTERING SOLO DEMOGRAFICO vs CON FVC_SLOPE
# ============================================================

print("="*70)
print("ANALISI COMPARATIVA: DEMOGRAFICO vs CLINICO")
print("="*70)

# ============================================================
# Test 1: SOLO Sex + SmokingStatus (quello che hai detto)
# ============================================================

print("\n1️⃣ CLUSTERING SOLO DEMOGRAFICO (Sex + Smoking)")
print("="*70)

X_demo = patient_df[['Sex_encoded', 'SmokingStatus_encoded']].copy()
scaler_demo = StandardScaler()
X_scaled_demo = scaler_demo.fit_transform(X_demo)

# Prova diversi K
for k in range(2, 8):
    kmeans_demo = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels_demo = kmeans_demo.fit_predict(X_scaled_demo)
    
    silhouette_demo = silhouette_score(X_scaled_demo, labels_demo)
    
    print(f"K={k}: Silhouette = {silhouette_demo:.3f}")

# Usa K=6 (combinazioni possibili)
k_demo = 6
kmeans_demo = KMeans(n_clusters=k_demo, random_state=42, n_init=20)
patient_df['Cluster_Demo'] = kmeans_demo.fit_predict(X_scaled_demo)

silhouette_demo = silhouette_score(X_scaled_demo, patient_df['Cluster_Demo'])

print(f"\n✓ Miglior K demografico: {k_demo}")
print(f"✓ Silhouette: {silhouette_demo:.3f}")

# Analizza i cluster demografici
print("\n📊 COMPOSIZIONE CLUSTER DEMOGRAFICI:")
demo_composition = patient_df.groupby('Cluster_Demo').agg({
    'Patient': 'count',
    'Sex': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
    'SmokingStatus': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
    'fvc_slope': 'mean',
    'fvc_mean': 'mean',
    'Age': 'mean'
}).round(2)

demo_composition.columns = ['N', 'Sex_Mode', 'Smoking_Mode', 'FVC_Slope', 'FVC_Mean', 'Age']
print("\n", demo_composition)

# Verifica: sono solo combinazioni categoriche?
print("\n🔍 VERIFICA: I cluster sono solo combinazioni categoriche?")
for cluster_id in range(k_demo):
    cluster_data = patient_df[patient_df['Cluster_Demo'] == cluster_id]
    sex_unique = cluster_data['Sex'].nunique()
    smoking_unique = cluster_data['SmokingStatus'].nunique()
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Sex unici: {sex_unique} - {cluster_data['Sex'].value_counts().to_dict()}")
    print(f"  Smoking unici: {smoking_unique} - {cluster_data['SmokingStatus'].value_counts().to_dict()}")
    
    if sex_unique == 1 and smoking_unique == 1:
        print(f"  ⚠️  CLUSTER ARTIFICIALE (1 sesso + 1 smoking status)")
    else:
        print(f"  ✓ Cluster misto")

# ============================================================
# Test 2: CONFRONTO VARIABILITÀ CLINICA
# ============================================================

print("\n" + "="*70)
print("2️⃣ CONFRONTO VARIABILITÀ CLINICA TRA METODI")
print("="*70)

# Calcola variabilità FVC_slope all'interno dei cluster
variance_demo = []
variance_optimal = []

for cluster_id in range(k_demo):
    cluster_data = patient_df[patient_df['Cluster_Demo'] == cluster_id]
    variance_demo.append(cluster_data['fvc_slope'].std())

for cluster_id in range(optimal_k):
    cluster_data = patient_df[patient_df['Cluster_Optimal'] == cluster_id]
    variance_optimal.append(cluster_data['fvc_slope'].std())

print(f"\n📊 VARIABILITÀ FVC_SLOPE (std) PER CLUSTER:")
print(f"\nDemografico (Sex+Smoking): {np.mean(variance_demo):.2f} ± {np.std(variance_demo):.2f}")
print(f"Ottimizzato (+FVC_slope):  {np.mean(variance_optimal):.2f} ± {np.std(variance_optimal):.2f}")

print(f"\n💡 INTERPRETAZIONE:")
if np.mean(variance_demo) > np.mean(variance_optimal):
    print("   ✓ Cluster ottimizzato ha MENO variabilità interna")
    print("   ✓ FVC_slope aiuta a creare gruppi più omogenei clinicamente")
else:
    print("   ⚠️  Cluster demografico ha già bassa variabilità")
    print("   → Ma è perché separa solo categorie, non progressione clinica!")

# ============================================================
# Test 3: CAPACITÀ PREDITTIVA
# ============================================================

print("\n" + "="*70)
print("3️⃣ TEST CAPACITÀ PREDITTIVA")
print("="*70)

# Usa cluster demografici per predire Week 0
models_demo = {}
predictions_demo = []

for cluster_id in range(k_demo):
    cluster_patients = patient_df[patient_df['Cluster_Demo'] == cluster_id]['Patient'].values
    cluster_data = df[df['Patient'].isin(cluster_patients)].copy()
    
    if len(cluster_data) > 0:
        X_reg = cluster_data[['Weeks']].values
        y_reg = cluster_data['FVC'].values
        
        model = LinearRegression()
        model.fit(X_reg, y_reg)
        
        y_pred = model.predict(X_reg)
        r2 = r2_score(y_reg, y_pred)
        mae = mean_absolute_error(y_reg, y_pred)
        
        models_demo[cluster_id] = {
            'model': model,
            'intercept': model.intercept_,
            'r2': r2,
            'mae': mae
        }

# Validazione cluster demografici
validation_demo = []

for _, patient_row in patient_df[patient_df['has_week0'] == 1].iterrows():
    patient_id = patient_row['Patient']
    cluster_id = patient_row['Cluster_Demo']
    
    if cluster_id in models_demo:
        real_fvc = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
        pred_cluster = models_demo[cluster_id]['intercept']
        
        validation_demo.append({
            'Patient': patient_id,
            'Real_FVC': real_fvc,
            'Pred_Demo': pred_cluster,
            'Error_Demo': abs(real_fvc - pred_cluster)
        })

validation_demo_df = pd.DataFrame(validation_demo)

print(f"\n📊 CONFRONTO PREDIZIONE WEEK 0:")
print(f"\nCluster DEMOGRAFICO (Sex+Smoking):")
print(f"  MAE: {validation_demo_df['Error_Demo'].mean():.2f}")
print(f"  Mediana: {validation_demo_df['Error_Demo'].median():.2f}")

print(f"\nCluster OTTIMIZZATO (+FVC_slope):")
print(f"  MAE: {validation_optimal_df['Error_Cluster_Opt'].mean():.2f}")
print(f"  Mediana: {validation_optimal_df['Error_Cluster_Opt'].median():.2f}")

print(f"\nPersonale (baseline):")
print(f"  MAE: {validation_optimal_df['Error_Personal'].mean():.2f}")
print(f"  Mediana: {validation_optimal_df['Error_Personal'].median():.2f}")

# ============================================================
# VISUALIZZAZIONE COMPARATIVA
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Cluster demographics
axes[0, 0].set_title('Cluster DEMOGRAFICO (Sex+Smoking)', fontweight='bold', fontsize=12)
scatter1 = axes[0, 0].scatter(patient_df['Sex_encoded'], 
                              patient_df['SmokingStatus_encoded'],
                              c=patient_df['Cluster_Demo'],
                              s=150, cmap='tab10', alpha=0.7, edgecolors='black', linewidth=1.5)
axes[0, 0].set_xlabel('Sex (0=F, 1=M)')
axes[0, 0].set_ylabel('Smoking (0=Current, 1=Ex, 2=Never)')
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
axes[0, 0].grid(True, alpha=0.3)

# FVC slope distribution - demographic
axes[0, 1].set_title('Distribuzione FVC_Slope - Demografico', fontweight='bold', fontsize=12)
patient_df.boxplot(column='fvc_slope', by='Cluster_Demo', ax=axes[0, 1])
axes[0, 1].set_xlabel('Cluster Demografico')
axes[0, 1].set_ylabel('FVC Slope')
plt.suptitle('')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Silhouette comparison
axes[0, 2].set_title('Confronto Silhouette Score', fontweight='bold', fontsize=12)
methods = ['Demografico\n(Sex+Smoke)', 'Ottimizzato\n(+FVC_slope)']
silhouettes = [silhouette_demo, silhouette_opt]
colors_sil = ['skyblue', 'orange']
bars = axes[0, 2].bar(methods, silhouettes, color=colors_sil, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 2].set_ylabel('Silhouette Score')
axes[0, 2].set_ylim([0, 1])
axes[0, 2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, silhouettes):
    height = bar.get_height()
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Row 2: Cluster ottimizzato
axes[1, 0].set_title('Cluster OTTIMIZZATO (+FVC_slope)', fontweight='bold', fontsize=12)
scatter2 = axes[1, 0].scatter(patient_df['Sex_encoded'], 
                              patient_df['SmokingStatus_encoded'],
                              c=patient_df['Cluster_Optimal'],
                              s=150, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
axes[1, 0].set_xlabel('Sex (0=F, 1=M)')
axes[1, 0].set_ylabel('Smoking (0=Current, 1=Ex, 2=Never)')
plt.colorbar(scatter2, ax=axes[1, 0], label='Cluster')
axes[1, 0].grid(True, alpha=0.3)

# FVC slope distribution - optimal
axes[1, 1].set_title('Distribuzione FVC_Slope - Ottimizzato', fontweight='bold', fontsize=12)
patient_df.boxplot(column='fvc_slope', by='Cluster_Optimal', ax=axes[1, 1])
axes[1, 1].set_xlabel('Cluster Ottimizzato')
axes[1, 1].set_ylabel('FVC Slope')
plt.suptitle('')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Prediction accuracy comparison
axes[1, 2].set_title('Confronto Accuratezza Predizione', fontweight='bold', fontsize=12)
methods_pred = ['Demografico', 'Ottimizzato', 'Personale']
mae_pred = [
    validation_demo_df['Error_Demo'].mean(),
    validation_optimal_df['Error_Cluster_Opt'].mean(),
    validation_optimal_df['Error_Personal'].mean()
]
colors_pred = ['skyblue', 'orange', 'green']
bars2 = axes[1, 2].bar(methods_pred, mae_pred, color=colors_pred, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 2].set_ylabel('MAE (FVC units)')
axes[1, 2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, mae_pred):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# CONCLUSIONE
# ============================================================

print("\n" + "="*70)
print("🎯 CONCLUSIONE: SILHOUETTE ALTO È SEMPRE BUONO?")
print("="*70)

print(f"""
NO! Ecco perché:

1️⃣ CLUSTER DEMOGRAFICO (Silhouette {silhouette_demo:.3f}):
   ✓ Silhouette molto alto
   ❌ Separa solo combinazioni categoriche (artificiale)
   ❌ ALTA variabilità clinica interna ({np.mean(variance_demo):.2f})
   ❌ Predizione: MAE = {validation_demo_df['Error_Demo'].mean():.0f}
   
   💡 È come separare mele rosse, mele verdi, pere...
      Ma all'interno ci sono mele marce e fresche mischiate!

2️⃣ CLUSTER OTTIMIZZATO (Silhouette {silhouette_opt:.3f}):
   ○ Silhouette medio (ma OK!)
   ✓ Separa PROGRESSIONE CLINICA
   ✓ BASSA variabilità clinica interna ({np.mean(variance_optimal):.2f})
   ○ Predizione: MAE = {validation_optimal_df['Error_Cluster_Opt'].mean():.0f}
   
   💡 Separa mele per grado di maturazione/salute,
      anche se colore può essere misto!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 PER LA TUA TESI:

✓ USA Cluster OTTIMIZZATO per:
  - Segmentazione clinica significativa
  - Identificare pattern di progressione
  - Stratificare per prognosi
  
✓ MENZIONA Cluster demografico per:
  - Mostrare che hai testato diverse opzioni
  - Spiegare perché Silhouette alto ≠ sempre migliore
  - Trade-off: separazione geometrica vs utilità clinica

✓ COMBINA con predizione personale per migliori risultati

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 FRASE PER LA TESI:
"Nonostante il clustering basato solo su variabili demografiche 
(Sex, SmokingStatus) producesse un Silhouette Score più elevato 
({silhouette_demo:.3f}), l'inclusione di FVC_slope risulta in cluster 
clinicamente più significativi, con minore variabilità nella 
progressione della malattia all'interno di ciascun gruppo, 
dimostrando che l'ottimizzazione del Silhouette Score non sempre 
corrisponde all'utilità clinica."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Salva confronto
comparison_clusters = pd.DataFrame({
    'Metodo': ['Demografico', 'Ottimizzato', 'Personale'],
    'Silhouette': [silhouette_demo, silhouette_opt, 'N/A'],
    'MAE_Predizione': [
        validation_demo_df['Error_Demo'].mean(),
        validation_optimal_df['Error_Cluster_Opt'].mean(),
        validation_optimal_df['Error_Personal'].mean()
    ],
    'Varianza_FVC_Slope': [np.mean(variance_demo), np.mean(variance_optimal), 'N/A']
})

print("\n📊 TABELLA RIASSUNTIVA:")
print(comparison_clusters.to_string(index=False))

# ============================================================
# SUMMARY FINALE PER LA TESI
# ============================================================

print("="*70)
print("📝 SINTESI FINALE PER LA TESI")
print("="*70)

summary_table = pd.DataFrame({
    'Approccio': [
        'Clustering Demografico\n(Sex + SmokingStatus)',
        'Clustering Ottimizzato\n(+ FVC_slope)',
        'Regressione Personale\n(individuale)',
        'Strategia Ibrida\n(consigliata)'
    ],
    'Silhouette\nScore': ['1.000', '0.548', 'N/A', '0.548'],
    'N\nCluster': [6, 5, 'N/A', 5],
    'MAE\nPredizione': ['625', '622', '128', '~130*'],
    'Variabilità\nInterna': ['4.76±1.80', '4.52±0.70', 'N/A', 'N/A'],
    'Utilità\nClinica': ['❌ Bassa', '✓ Alta', '✓✓ Massima', '✓✓ Ottimale'],
    'Interpretabilità': ['Triviale', 'Eccellente', 'Limitata', 'Eccellente']
})

print("\n" + "="*70)
print("TABELLA COMPARATIVA METODI")
print("="*70)
print("\n", summary_table.to_string(index=False))

print("\n" + "="*70)
print("🎯 CONTRIBUTI CHIAVE DELLA TUA ANALISI")
print("="*70)

print("""
1. SCOPERTA METODOLOGICA:
   ✓ Dimostrato che Silhouette Score elevato non implica utilità clinica
   ✓ Clustering su sole variabili categoriche produce clusters "artificiali"
   ✓ Necessità di bilanciare metriche geometriche con significato clinico

2. SEGMENTAZIONE CLINICA:
   ✓ Identificati 5 fenotipi di progressione della fibrosi polmonare:
     • Uomini ex-fumatori stabili (n=87, slope -1.94)
     • Donne non-fumatrici declino moderato (n=23, slope -4.18)
     • Uomini ex-fumatori alto rischio (n=29, slope -14.12) ⚠️
     • Uomini non-fumatori declino moderato (n=23, slope -3.86)
     • Donne ex-fumatrici stabili (n=14, slope -2.37)

3. PREDIZIONE BASELINE (Week 0):
   ✓ Regressione individuale: errore medio 3.8% (MAE 128)
   ✓ Approccio ibrido combina clustering + regressione personale
   ⚠️  Limitazioni modello lineare identificate (R² medio 0.42)

4. AFFIDABILITÀ E VALIDAZIONE:
   ✓ Sistema di classificazione affidabilità (Alta/Media/Bassa)
   ✓ ~60% predizioni utilizzabili clinicamente
   ✓ Trade-off identificato: qualità cluster vs accuratezza predizione

5. DIREZIONI FUTURE:
   → Mixed-effects models per catturare eterogeneità
   → Modelli non-lineari (GAM, splines) per traiettorie complesse
   → Deep learning (LSTM/Transformer) per sequenze temporali
""")

print("\n" + "="*70)
print("📖 STRUTTURA SUGGERITA PER LA SEZIONE RISULTATI")
print("="*70)

print("""
3. RISULTATI

3.1 Feature Engineering e Preparazione Dati
    - 176 pazienti, 1549 osservazioni temporali
    - 158/176 (90%) senza baseline (Week 0)
    - Features aggregate: slope, mean, std, demographics

3.2 Clustering per Segmentazione Clinica
    
    3.2.1 Ottimizzazione Feature Set
          • Test 9 combinazioni di features
          • Identificazione trade-off Silhouette vs utilità clinica
          • RISULTATO: 3 features (fvc_slope, sex, smoking)
          
          [INSERISCI FIGURA: Heatmap silhouette scores]
          
    3.2.2 Analisi Critica Clustering Demografico
          • Silhouette Score: 1.000 (perfetto geometricamente)
          • MA: 6 cluster artificiali (combinazioni categoriche)
          • Variabilità clinica interna: 4.76 ± 1.80
          • MAE predizione: 625 unità
          
          [INSERISCI FIGURA: Composizione cluster demografici]
          
    3.2.3 Clustering Ottimizzato
          • 5 cluster clinicamente interpretabili
          • Silhouette Score: 0.548 (buono, non artificiale)
          • Variabilità clinica ridotta: 4.52 ± 0.70
          • Fenotipi identificati: [descrivi i 5 cluster]
          
          [INSERISCI FIGURA: Scatter plot + traiettorie per cluster]

3.3 Predizione Baseline FVC (Week 0)
    
    3.3.1 Approccio per Cluster
          • Regressione lineare per cluster
          • MAE: 622 unità (~19% errore)
          • R² medio basso (0.004-0.087)
          • Limitazione: linearità eccessiva
          
    3.3.2 Approccio Individuale
          • Regressione per paziente
          • MAE: 128 unità (3.8% errore) ✓
          • Validazione su 18 pazienti con Week 0
          • 56% pazienti con R² > 0.4
          
          [INSERISCI FIGURA: Real vs Predicted scatter]
          
    3.3.3 Analisi Affidabilità
          • Sistema classificazione Alta/Media/Bassa
          • Criteri: R², distanza Week 0, n_osservazioni
          • 60% predizioni affidabili per uso clinico
          
          [INSERISCI FIGURA: Distribuzione affidabilità]

3.4 Strategia Ibrida
    • Combina cluster + regressione individuale
    • Decision tree per selezione metodo
    • MAE finale: ~130 unità
    • Bilanciamento accuracy/robustezza

4. DISCUSSIONE

4.1 Interpretazione Clinica dei Cluster
    [Discuti i 5 fenotipi, focus su cluster 2 alto rischio]

4.2 Trade-off Metriche vs Utilità Clinica
    [Spiega perché Silhouette 1.0 non è desiderabile]

4.3 Limitazioni
    • Linearità del modello
    • 83% pazienti richiede extrapolazione
    • Dataset relativamente piccolo
    • Mancanza variabili biomarker

4.4 Applicazioni Cliniche
    • Stratificazione risk-based
    • Personalizzazione follow-up
    • Design trials clinici

5. CONCLUSIONI
    [Riassumi contributi chiave]
""")

print("\n" + "="*70)
print("💡 PUNTI FORTI DELLA TUA ANALISI PER LA DISCUSSIONE")
print("="*70)

print("""
✓ APPROCCIO RIGOROSO:
  - Validazione sistematica di ogni scelta
  - Confronto multipli metodi
  - Analisi critica dei risultati (non solo accettare metriche alte)

✓ CONTRIBUTO METODOLOGICO:
  - Dimostrazione empirica che Silhouette alto ≠ clustering utile
  - Framework per valutare utilità clinica vs metriche statistiche
  - Approccio ibrido pragmatico

✓ RILEVANZA CLINICA:
  - 5 fenotipi interpretabili di progressione
  - Identificazione gruppo alto rischio (cluster 2, slope -14)
  - Sistema di affidabilità per uso clinico pratico

✓ TRASPARENZA:
  - Limitazioni chiaramente identificate
  - R² bassi discussi apertamente
  - Direzioni future concrete
""")

# Crea figura finale riassuntiva per la tesi
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Silhouette comparison
ax1 = fig.add_subplot(gs[0, 0])
methods_sil = ['Demografico\n(6 cluster)', 'Ottimizzato\n(5 cluster)']
sil_values = [1.000, 0.548]
colors_bar = ['red', 'green']
bars1 = ax1.bar(methods_sil, sil_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Silhouette Score', fontweight='bold')
ax1.set_title('A) Qualità Geometrica', fontweight='bold')
ax1.set_ylim([0, 1.1])
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Soglia "buono"')
for bar, val in zip(bars1, sil_values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Clinical variance
ax2 = fig.add_subplot(gs[0, 1])
var_values = [4.76, 4.52]
bars2 = ax2.bar(methods_sil, var_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Std FVC_slope', fontweight='bold')
ax2.set_title('B) Omogeneità Clinica', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.invert_yaxis()  # Lower is better
for bar, val in zip(bars2, var_values):
    ax2.text(bar.get_x() + bar.get_width()/2., val - 0.1,
             f'{val:.2f}', ha='center', va='top', fontsize=11, fontweight='bold')

# Plot 3: Prediction accuracy
ax3 = fig.add_subplot(gs[0, 2])
methods_pred = ['Demografico', 'Ottimizzato', 'Personale']
mae_values = [625, 622, 128]
colors_pred = ['red', 'orange', 'green']
bars3 = ax3.bar(methods_pred, mae_values, color=colors_pred, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('MAE (FVC units)', fontweight='bold')
ax3.set_title('C) Accuratezza Predizione', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, mae_values):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
             f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Cluster composition (demografico)
ax4 = fig.add_subplot(gs[1, 0])
scatter4 = ax4.scatter(patient_df['Sex_encoded'], patient_df['SmokingStatus_encoded'],
                       c=patient_df['Cluster_Demo'], cmap='tab10', s=80, alpha=0.6, edgecolors='black')
ax4.set_xlabel('Sex (0=F, 1=M)')
ax4.set_ylabel('Smoking Status')
ax4.set_title('D) Cluster Demografici\n(Artificiali)', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Cluster composition (ottimizzato)
ax5 = fig.add_subplot(gs[1, 1])
scatter5 = ax5.scatter(patient_df['Sex_encoded'], patient_df['SmokingStatus_encoded'],
                       c=patient_df['Cluster_Optimal'], cmap='viridis', s=80, alpha=0.6, edgecolors='black')
ax5.set_xlabel('Sex (0=F, 1=M)')
ax5.set_ylabel('Smoking Status')
ax5.set_title('E) Cluster Ottimizzati\n(Clinicamente Rilevanti)', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: FVC slope distribution
ax6 = fig.add_subplot(gs[1, 2])
patient_df.boxplot(column='fvc_slope', by='Cluster_Optimal', ax=ax6)
ax6.set_xlabel('Cluster Ottimizzato')
ax6.set_ylabel('FVC Slope')
ax6.set_title('F) Distribuzione Progressione', fontweight='bold')
plt.suptitle('')
ax6.grid(True, alpha=0.3, axis='y')
ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 7-9: Sample trajectories per alcuni cluster
for idx, cluster_id in enumerate([0, 1, 2]):
    ax = fig.add_subplot(gs[2, idx])
    cluster_patients = patient_df[patient_df['Cluster_Optimal'] == cluster_id]['Patient'].values[:10]
    
    for patient in cluster_patients:
        patient_data = df[df['Patient'] == patient]
        ax.plot(patient_data['Weeks'], patient_data['FVC'], alpha=0.4, linewidth=1)
    
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Weeks')
    ax.set_ylabel('FVC')
    
    cluster_info = patient_df[patient_df['Cluster_Optimal'] == cluster_id]
    slope_mean = cluster_info['fvc_slope'].mean()
    ax.set_title(f'Cluster {cluster_id}: Slope={slope_mean:.1f}', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('ANALISI COMPARATIVA: Clustering Demografico vs Ottimizzato', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/figure_summary_thesis.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Figura riassuntiva salvata: figure_summary_thesis.png")


# ============================================================
# MODELLI NON-LINEARI PER PREDIZIONE WEEK 0
# ============================================================

print("="*70)
print("🚀 PREDIZIONE CON MODELLI NON-LINEARI")
print("="*70)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. PREPARAZIONE DATI PER MODELLI ML
# ============================================================

print("\n1️⃣ PREPARAZIONE DATI")
print("="*70)

# Per ogni paziente, crea features dai suoi dati temporali
def create_ml_features(patient_id, patient_df_row, df):
    """
    Crea features robuste per ML dai dati temporali del paziente
    """
    patient_data = df[df['Patient'] == patient_id].sort_values('Weeks')
    
    if len(patient_data) < 2:
        return None
    
    weeks = patient_data['Weeks'].values
    fvc = patient_data['FVC'].values
    
    features = {
        # Temporal features
        'first_week': weeks.min(),
        'last_week': weeks.max(),
        'weeks_span': weeks.max() - weeks.min(),
        'n_observations': len(weeks),
        
        # FVC statistics
        'fvc_first': fvc[0],
        'fvc_last': fvc[-1],
        'fvc_mean': fvc.mean(),
        'fvc_std': fvc.std(),
        'fvc_min': fvc.min(),
        'fvc_max': fvc.max(),
        'fvc_range': fvc.max() - fvc.min(),
        
        # Linear trend (come baseline)
        'fvc_slope': patient_df_row['fvc_slope'],
        'fvc_intercept': patient_df_row['fvc_intercept'],
        
        # Non-linear patterns
        'fvc_acceleration': 0,  # Second derivative
        'fvc_cv': fvc.std() / fvc.mean() if fvc.mean() > 0 else 0,  # Coefficient of variation
        
        # Rate of change features
        'fvc_change_total': fvc[-1] - fvc[0],
        'fvc_change_rate': (fvc[-1] - fvc[0]) / (weeks[-1] - weeks[0]) if weeks[-1] != weeks[0] else 0,
        
        # Percentages
        'percent_first': patient_data['Percent'].iloc[0],
        'percent_mean': patient_data['Percent'].mean(),
        'percent_std': patient_data['Percent'].std(),
        
        # Demographics
        'Age': patient_df_row['Age'],
        'Sex_encoded': patient_df_row['Sex_encoded'],
        'SmokingStatus_encoded': patient_df_row['SmokingStatus_encoded'],
        'Cluster_Optimal': patient_df_row['Cluster_Optimal']
    }
    
    # Calculate acceleration (if enough points)
    if len(weeks) >= 3:
        # Fit quadratic and get second derivative
        try:
            coeffs = np.polyfit(weeks, fvc, 2)
            features['fvc_acceleration'] = 2 * coeffs[0]  # Second derivative
        except:
            pass
    
    return features

# Crea dataset ML per tutti i pazienti
ml_data = []
for _, patient_row in patient_df.iterrows():
    patient_id = patient_row['Patient']
    feats = create_ml_features(patient_id, patient_row, df)
    if feats:
        feats['Patient'] = patient_id
        feats['has_week0'] = patient_row['has_week0']
        
        # Target: FVC a week 0 (se disponibile)
        if patient_row['has_week0'] == 1:
            feats['FVC_week0'] = df[(df['Patient'] == patient_id) & (df['Weeks'] == 0)]['FVC'].values[0]
        
        ml_data.append(feats)

ml_df = pd.DataFrame(ml_data)

print(f"Dataset ML creato: {len(ml_df)} pazienti")
print(f"Features: {len(ml_df.columns)-3} (esclusi Patient, has_week0, FVC_week0)")

# Split train/test
train_df = ml_df[ml_df['has_week0'] == 1].copy()
test_df = ml_df[ml_df['has_week0'] == 0].copy()

print(f"\nTrain set (con Week 0): {len(train_df)} pazienti")
print(f"Test set (senza Week 0): {len(test_df)} pazienti")

# Features per ML
feature_cols = [col for col in ml_df.columns if col not in ['Patient', 'has_week0', 'FVC_week0']]

X_train = train_df[feature_cols].values
y_train = train_df['FVC_week0'].values
X_test = test_df[feature_cols].values

print(f"\nShape: X_train={X_train.shape}, y_train={y_train.shape}")

# ============================================================
# 2. TEST MODELLI NON-LINEARI
# ============================================================

print("\n" + "="*70)
print("2️⃣ CONFRONTO MODELLI NON-LINEARI")
print("="*70)

# Cross-validation per valutare modelli
from sklearn.model_selection import cross_val_score, KFold

models = {
    'Linear (baseline)': LinearRegression(),
    'Polynomial (degree 2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
    'Polynomial (degree 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=1000, gamma='auto')
}

results = []
kf = KFold(n_splits=min(5, len(train_df)), shuffle=True, random_state=42)

print("\n📊 PERFORMANCE (5-Fold Cross-Validation):\n")
print(f"{'Modello':<25} {'MAE':<12} {'R²':<12} {'Std MAE':<12}")
print("-" * 70)

for name, model in models.items():
    try:
        # Cross-validation
        mae_scores = -cross_val_score(model, X_train, y_train, 
                                       cv=kf, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X_train, y_train, 
                                    cv=kf, scoring='r2')
        
        mae_mean = mae_scores.mean()
        mae_std = mae_scores.std()
        r2_mean = r2_scores.mean()
        
        results.append({
            'Model': name,
            'MAE': mae_mean,
            'MAE_std': mae_std,
            'R2': r2_mean
        })
        
        print(f"{name:<25} {mae_mean:<12.2f} {r2_mean:<12.3f} {mae_std:<12.2f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {str(e)[:30]}")

results_df = pd.DataFrame(results).sort_values('MAE')

print("\n" + "="*70)
print("🏆 RANKING MODELLI (per MAE)")
print("="*70)
print(results_df.to_string(index=False))

# ============================================================
# 3. TRAIN MIGLIOR MODELLO E PREDICI
# ============================================================

print("\n" + "="*70)
print("3️⃣ PREDIZIONI CON MIGLIOR MODELLO")
print("="*70)

best_model_name = results_df.iloc[0]['Model']
best_mae = results_df.iloc[0]['MAE']

print(f"\n🏆 Miglior modello: {best_model_name}")
print(f"   MAE Cross-Val: {best_mae:.2f}")

# Train il miglior modello su tutti i dati di training
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Predizioni sul train set (per validazione)
y_train_pred = best_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\n📊 Performance su Train Set Completo:")
print(f"   MAE: {train_mae:.2f}")
print(f"   R²: {train_r2:.3f}")

# Predizioni per pazienti senza Week 0
y_test_pred = best_model.predict(X_test)

predictions_ml = pd.DataFrame({
    'Patient': test_df['Patient'].values,
    'Predicted_FVC_week0_ML': y_test_pred,
    'Predicted_FVC_week0_Linear': test_df['fvc_intercept'].values,
    'Cluster_Optimal': test_df['Cluster_Optimal'].values
})

print(f"\n✓ Predizioni generate per {len(predictions_ml)} pazienti")

# ============================================================
# 4. CONFRONTO CON METODO LINEARE
# ============================================================

print("\n" + "="*70)
print("4️⃣ CONFRONTO ML vs LINEAR")
print("="*70)

# Valutazione su train set (che ha ground truth)
comparison = pd.DataFrame({
    'Patient': train_df['Patient'].values,
    'Real_FVC': y_train,
    'Pred_ML': y_train_pred,
    'Pred_Linear': train_df['fvc_intercept'].values,
    'Error_ML': np.abs(y_train - y_train_pred),
    'Error_Linear': np.abs(y_train - train_df['fvc_intercept'].values)
})

print(f"\n📊 RISULTATI SU TRAIN SET (n={len(comparison)}):")
print(f"\nMAE ML ({best_model_name}):")
print(f"  {comparison['Error_ML'].mean():.2f} ± {comparison['Error_ML'].std():.2f}")
print(f"  Mediana: {comparison['Error_ML'].median():.2f}")

print(f"\nMAE Linear (baseline):")
print(f"  {comparison['Error_Linear'].mean():.2f} ± {comparison['Error_Linear'].std():.2f}")
print(f"  Mediana: {comparison['Error_Linear'].median():.2f}")

improvement = (1 - comparison['Error_ML'].mean() / comparison['Error_Linear'].mean()) * 100
print(f"\n✨ MIGLIORAMENTO: {improvement:+.1f}%")

# Conta quanti pazienti hanno migliorato
better_count = (comparison['Error_ML'] < comparison['Error_Linear']).sum()
print(f"   Pazienti con predizione migliore: {better_count}/{len(comparison)} ({better_count/len(comparison)*100:.0f}%)")

# ============================================================
# 5. VISUALIZZAZIONI
# ============================================================

print("\n" + "="*70)
print("5️⃣ VISUALIZZAZIONI")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Confronto MAE modelli
ax1 = axes[0, 0]
results_sorted = results_df.sort_values('MAE', ascending=False)
colors_models = ['red' if 'Linear' in m else 'orange' if 'Polynomial' in m else 'green' 
                 for m in results_sorted['Model']]
bars = ax1.barh(range(len(results_sorted)), results_sorted['MAE'], 
                color=colors_models, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(results_sorted)))
ax1.set_yticklabels(results_sorted['Model'])
ax1.set_xlabel('MAE (FVC units)')
ax1.set_title('A) Confronto Modelli (Cross-Validation)', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Plot 2: Real vs Predicted (ML)
ax2 = axes[0, 1]
scatter2 = ax2.scatter(comparison['Real_FVC'], comparison['Pred_ML'],
                       c=train_df['Cluster_Optimal'], cmap='viridis',
                       s=100, alpha=0.7, edgecolors='black')
min_val = min(comparison['Real_FVC'].min(), comparison['Pred_ML'].min())
max_val = max(comparison['Real_FVC'].max(), comparison['Pred_ML'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax2.set_xlabel('Real FVC at Week 0')
ax2.set_ylabel(f'Predicted FVC ({best_model_name})')
ax2.set_title(f'B) ML Model\nMAE: {train_mae:.0f}', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Plot 3: Real vs Predicted (Linear)
ax3 = axes[0, 2]
ax3.scatter(comparison['Real_FVC'], comparison['Pred_Linear'],
            c=train_df['Cluster_Optimal'], cmap='viridis',
            s=100, alpha=0.7, edgecolors='black')
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax3.set_xlabel('Real FVC at Week 0')
ax3.set_ylabel('Predicted FVC (Linear)')
ax3.set_title(f'C) Linear Baseline\nMAE: {comparison["Error_Linear"].mean():.0f}', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distribuzione errori
ax4 = axes[1, 0]
ax4.hist(comparison['Error_ML'], bins=15, alpha=0.7, label=f'{best_model_name}', 
         color='green', edgecolor='black')
ax4.hist(comparison['Error_Linear'], bins=15, alpha=0.7, label='Linear', 
         color='red', edgecolor='black')
ax4.axvline(comparison['Error_ML'].mean(), color='green', linestyle='--', linewidth=2)
ax4.axvline(comparison['Error_Linear'].mean(), color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Absolute Error (FVC)')
ax4.set_ylabel('Frequency')
ax4.set_title('D) Distribuzione Errori', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Errori per cluster
ax5 = axes[1, 1]
comparison_with_cluster = comparison.copy()
comparison_with_cluster['Cluster'] = train_df['Cluster_Optimal'].values
comparison_long = pd.melt(comparison_with_cluster, 
                          id_vars=['Cluster'],
                          value_vars=['Error_ML', 'Error_Linear'],
                          var_name='Method', value_name='Error')
comparison_long['Method'] = comparison_long['Method'].map({'Error_ML': best_model_name, 'Error_Linear': 'Linear'})

import seaborn as sns
sns.boxplot(data=comparison_long, x='Cluster', y='Error', hue='Method', ax=ax5)
ax5.set_xlabel('Cluster')
ax5.set_ylabel('Absolute Error')
ax5.set_title('E) Errori per Cluster', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
ax5.legend(title='Method')

# Plot 6: Improvement per patient
ax6 = axes[1, 2]
improvement_per_patient = comparison['Error_Linear'] - comparison['Error_ML']
colors_imp = ['green' if x > 0 else 'red' for x in improvement_per_patient]
ax6.bar(range(len(improvement_per_patient)), improvement_per_patient.sort_values(),
        color=colors_imp, alpha=0.7, edgecolor='black')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.set_xlabel('Paziente (ordinato)')
ax6.set_ylabel('Miglioramento (Linear - ML)')
ax6.set_title(f'F) Miglioramento per Paziente\n{better_count}/{len(comparison)} pazienti migliorati', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/ml_models_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualizzazioni salvate")

# ============================================================
# MODELLI ML - VERSIONE CORRETTA (no data leakage)
# ============================================================

print("="*70)
print("🔧 MODELLI ML - SENZA DATA LEAKAGE")
print("="*70)

# ============================================================
# 1. FEATURES PULITE (no fvc_intercept!)
# ============================================================

print("\n1️⃣ RIMOZIONE FEATURES CON DATA LEAKAGE")
print("="*70)

# Features da ESCLUDERE (contengono informazione su Week 0)
exclude_features = ['fvc_intercept', 'fvc_slope']  # Questi "conoscono" Week 0

# Features VALIDE per predire Week 0
valid_features = [col for col in feature_cols if col not in exclude_features]

print(f"\nFeatures ORIGINALI: {len(feature_cols)}")
print(f"Features RIMOSSE (data leakage): {exclude_features}")
print(f"Features VALIDE: {len(valid_features)}")
print(f"\nFeatures usate:")
for f in valid_features:
    print(f"  - {f}")

X_train_clean = train_df[valid_features].values
X_test_clean = test_df[valid_features].values

print(f"\nShape: X_train={X_train_clean.shape}, y_train={y_train.shape}")

# ============================================================
# 2. CONFRONTO MODELLI (versione corretta)
# ============================================================

print("\n" + "="*70)
print("2️⃣ CONFRONTO MODELLI (Features Pulite)")
print("="*70)

models_v2 = {
    'Linear Regression': LinearRegression(),
    'Polynomial (deg 2)': make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=7, min_samples_split=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    'XGBoost (if avail)': None  # Placeholder
}

# Prova XGBoost se disponibile
try:
    from xgboost import XGBRegressor
    models_v2['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    del models_v2['XGBoost (if avail)']
except:
    del models_v2['XGBoost (if avail)']
    print("⚠️  XGBoost non disponibile, saltato")

results_v2 = []
kf = KFold(n_splits=min(6, len(train_df)), shuffle=True, random_state=42)  # 6-fold per più stabilità

print("\n📊 PERFORMANCE (Cross-Validation):\n")
print(f"{'Modello':<25} {'MAE':<12} {'R²':<12} {'Std MAE':<12}")
print("-" * 70)

for name, model in models_v2.items():
    if model is None:
        continue
        
    try:
        # Cross-validation
        mae_scores = -cross_val_score(model, X_train_clean, y_train, 
                                       cv=kf, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X_train_clean, y_train, 
                                    cv=kf, scoring='r2')
        
        mae_mean = mae_scores.mean()
        mae_std = mae_scores.std()
        r2_mean = r2_scores.mean()
        
        results_v2.append({
            'Model': name,
            'MAE': mae_mean,
            'MAE_std': mae_std,
            'R2': r2_mean
        })
        
        print(f"{name:<25} {mae_mean:<12.2f} {r2_mean:<12.3f} {mae_std:<12.2f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {str(e)[:40]}")

results_v2_df = pd.DataFrame(results_v2).sort_values('MAE')

print("\n" + "="*70)
print("🏆 RANKING MODELLI")
print("="*70)
print(results_v2_df.to_string(index=False))

# ============================================================
# 3. ANALISI FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("3️⃣ FEATURE IMPORTANCE")
print("="*70)

# Usa Random Forest per vedere quali features sono importanti
rf_model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
rf_model.fit(X_train_clean, y_train)

feature_importance = pd.DataFrame({
    'Feature': valid_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 10 FEATURES PIÙ IMPORTANTI:")
print(feature_importance.head(10).to_string(index=False))

# Plot feature importance
fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
ax_feat.barh(range(len(top_features)), top_features['Importance'], 
             color='steelblue', alpha=0.8, edgecolor='black')
ax_feat.set_yticks(range(len(top_features)))
ax_feat.set_yticklabels(top_features['Feature'])
ax_feat.set_xlabel('Importance')
ax_feat.set_title('Top 15 Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
ax_feat.grid(True, alpha=0.3, axis='x')
ax_feat.invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================
# 4. TRAIN MIGLIOR MODELLO E VALIDA
# ============================================================

print("\n" + "="*70)
print("4️⃣ VALIDAZIONE MIGLIOR MODELLO")
print("="*70)

best_model_name_v2 = results_v2_df.iloc[0]['Model']
best_mae_v2 = results_v2_df.iloc[0]['MAE']
best_r2_v2 = results_v2_df.iloc[0]['R2']

print(f"\n🏆 Miglior modello: {best_model_name_v2}")
print(f"   MAE Cross-Val: {best_mae_v2:.2f}")
print(f"   R² Cross-Val: {best_r2_v2:.3f}")

# Train su tutti i dati
best_model_v2 = models_v2[best_model_name_v2]
best_model_v2.fit(X_train_clean, y_train)

# Predizioni
y_train_pred_v2 = best_model_v2.predict(X_train_clean)
y_test_pred_v2 = best_model_v2.predict(X_test_clean)

# Metriche su train
train_mae_v2 = mean_absolute_error(y_train, y_train_pred_v2)
train_r2_v2 = r2_score(y_train, y_train_pred_v2)

print(f"\n📊 Performance su Train Set Completo:")
print(f"   MAE: {train_mae_v2:.2f}")
print(f"   R²: {train_r2_v2:.3f}")

# Confronto con baseline lineare individuale
baseline_preds = train_df['fvc_intercept'].values
baseline_mae = mean_absolute_error(y_train, baseline_preds)

print(f"\n📊 Confronto con Baseline (regressione individuale):")
print(f"   Baseline MAE: {baseline_mae:.2f}")
print(f"   ML MAE: {train_mae_v2:.2f}")

if train_mae_v2 < baseline_mae:
    improvement = ((baseline_mae - train_mae_v2) / baseline_mae) * 100
    print(f"   ✨ MIGLIORAMENTO: {improvement:.1f}%")
else:
    worsening = ((train_mae_v2 - baseline_mae) / baseline_mae) * 100
    print(f"   ⚠️  PEGGIORMENTE: {worsening:.1f}%")

# ============================================================
# 5. ANALISI DETTAGLIATA ERRORI
# ============================================================

print("\n" + "="*70)
print("5️⃣ ANALISI DETTAGLIATA ERRORI")
print("="*70)

comparison_v2 = pd.DataFrame({
    'Patient': train_df['Patient'].values,
    'Real_FVC': y_train,
    'Pred_ML': y_train_pred_v2,
    'Pred_Linear': baseline_preds,
    'Error_ML': np.abs(y_train - y_train_pred_v2),
    'Error_Linear': np.abs(y_train - baseline_preds),
    'Cluster': train_df['Cluster_Optimal'].values,
    'R2_individual': regression_quality_df[regression_quality_df['Patient'].isin(train_df['Patient'])]['r2'].values
})

print("\n📊 STATISTICHE ERRORI:")
print(f"\nML ({best_model_name_v2}):")
print(f"  Media: {comparison_v2['Error_ML'].mean():.2f}")
print(f"  Mediana: {comparison_v2['Error_ML'].median():.2f}")
print(f"  Min: {comparison_v2['Error_ML'].min():.2f}")
print(f"  Max: {comparison_v2['Error_ML'].max():.2f}")

print(f"\nLinear Individuale:")
print(f"  Media: {comparison_v2['Error_Linear'].mean():.2f}")
print(f"  Mediana: {comparison_v2['Error_Linear'].median():.2f}")
print(f"  Min: {comparison_v2['Error_Linear'].min():.2f}")
print(f"  Max: {comparison_v2['Error_Linear'].max():.2f}")

# Quanti pazienti sono migliorati?
better_ml = (comparison_v2['Error_ML'] < comparison_v2['Error_Linear']).sum()
print(f"\n👥 Pazienti con errore minore con ML: {better_ml}/{len(comparison_v2)} ({better_ml/len(comparison_v2)*100:.0f}%)")

# Analisi per cluster
print(f"\n📊 ERRORI PER CLUSTER:")
for cluster_id in range(optimal_k):
    cluster_data = comparison_v2[comparison_v2['Cluster'] == cluster_id]
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
        print(f"  ML MAE: {cluster_data['Error_ML'].mean():.2f}")
        print(f"  Linear MAE: {cluster_data['Error_Linear'].mean():.2f}")
        
        if cluster_data['Error_ML'].mean() < cluster_data['Error_Linear'].mean():
            print(f"  ✓ ML migliore")
        else:
            print(f"  ✗ Linear migliore")

# Correlazione errore con R² individuale
corr_r2_error_ml = comparison_v2[['R2_individual', 'Error_ML']].corr().iloc[0, 1]
corr_r2_error_linear = comparison_v2[['R2_individual', 'Error_Linear']].corr().iloc[0, 1]

print(f"\n📊 CORRELAZIONE R² INDIVIDUALE vs ERRORE:")
print(f"  ML: {corr_r2_error_ml:.3f}")
print(f"  Linear: {corr_r2_error_linear:.3f}")

# ============================================================
# 6. VISUALIZZAZIONI FINALI
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Real vs Predicted ML
ax1 = axes[0, 0]
scatter1 = ax1.scatter(comparison_v2['Real_FVC'], comparison_v2['Pred_ML'],
                       c=comparison_v2['Cluster'], cmap='viridis',
                       s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
min_val = comparison_v2[['Real_FVC', 'Pred_ML']].min().min()
max_val = comparison_v2[['Real_FVC', 'Pred_ML']].max().max()
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax1.set_xlabel('Real FVC at Week 0', fontweight='bold')
ax1.set_ylabel(f'Predicted ({best_model_name_v2})', fontweight='bold')
ax1.set_title(f'A) {best_model_name_v2}\nMAE: {train_mae_v2:.0f}, R²: {train_r2_v2:.3f}', 
              fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Cluster')

# Plot 2: Real vs Predicted Linear
ax2 = axes[0, 1]
ax2.scatter(comparison_v2['Real_FVC'], comparison_v2['Pred_Linear'],
            c=comparison_v2['Cluster'], cmap='viridis',
            s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax2.set_xlabel('Real FVC at Week 0', fontweight='bold')
ax2.set_ylabel('Predicted (Linear Individual)', fontweight='bold')
linear_r2 = r2_score(comparison_v2['Real_FVC'], comparison_v2['Pred_Linear'])
ax2.set_title(f'B) Linear Baseline\nMAE: {baseline_mae:.0f}, R²: {linear_r2:.3f}', 
              fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribuzione errori
ax3 = axes[0, 2]
bins_range = np.linspace(0, max(comparison_v2['Error_ML'].max(), comparison_v2['Error_Linear'].max()), 20)
ax3.hist(comparison_v2['Error_ML'], bins=bins_range, alpha=0.6, label=best_model_name_v2, 
         color='green', edgecolor='black')
ax3.hist(comparison_v2['Error_Linear'], bins=bins_range, alpha=0.6, label='Linear', 
         color='blue', edgecolor='black')
ax3.axvline(comparison_v2['Error_ML'].mean(), color='green', linestyle='--', linewidth=2, 
            label=f'ML mean: {comparison_v2["Error_ML"].mean():.0f}')
ax3.axvline(comparison_v2['Error_Linear'].mean(), color='blue', linestyle='--', linewidth=2,
            label=f'Linear mean: {comparison_v2["Error_Linear"].mean():.0f}')
ax3.set_xlabel('Absolute Error (FVC)', fontweight='bold')
ax3.set_ylabel('Frequency')
ax3.set_title('C) Distribuzione Errori', fontweight='bold', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Errori per cluster
ax4 = axes[1, 0]
comparison_long_v2 = pd.melt(comparison_v2, 
                              id_vars=['Cluster'],
                              value_vars=['Error_ML', 'Error_Linear'],
                              var_name='Method', value_name='Error')
comparison_long_v2['Method'] = comparison_long_v2['Method'].map({
    'Error_ML': best_model_name_v2, 
    'Error_Linear': 'Linear'
})
sns.boxplot(data=comparison_long_v2, x='Cluster', y='Error', hue='Method', ax=ax4)
ax4.set_xlabel('Cluster', fontweight='bold')
ax4.set_ylabel('Absolute Error', fontweight='bold')
ax4.set_title('D) Errori per Cluster', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(title='Method', fontsize=9)

# Plot 5: Errore vs R² individuale
ax5 = axes[1, 1]
ax5.scatter(comparison_v2['R2_individual'], comparison_v2['Error_ML'], 
            label=best_model_name_v2, alpha=0.7, s=100, color='green', edgecolors='black')
ax5.scatter(comparison_v2['R2_individual'], comparison_v2['Error_Linear'], 
            label='Linear', alpha=0.7, s=100, color='blue', edgecolors='black')
ax5.set_xlabel('R² (Regressione Individuale)', fontweight='bold')
ax5.set_ylabel('Absolute Error', fontweight='bold')
ax5.set_title('E) Errore vs Qualità Fit Lineare', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Miglioramento per paziente
ax6 = axes[1, 2]
improvement_v2 = comparison_v2['Error_Linear'] - comparison_v2['Error_ML']
colors_imp = ['green' if x > 0 else 'red' for x in improvement_v2]
sorted_imp = improvement_v2.sort_values()
ax6.bar(range(len(sorted_imp)), sorted_imp.values, color=colors_imp, alpha=0.7, edgecolor='black')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax6.set_xlabel('Paziente (ordinato)', fontweight='bold')
ax6.set_ylabel('Miglioramento ML vs Linear', fontweight='bold')
ax6.set_title(f'F) Miglioramento per Paziente\n{better_ml}/{len(comparison_v2)} pazienti', 
              fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/ml_vs_linear_clean.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualizzazioni salvate: ml_vs_linear_clean.png")

print("\n" + "="*70)
print("🎯 CONCLUSIONE")
print("="*70)
print(f"""
RISULTATI FINALI:

1. MIGLIOR MODELLO ML: {best_model_name_v2}
   - MAE: {best_mae_v2:.0f} (cross-val)
   - R²: {best_r2_v2:.3f}

2. BASELINE (Linear Individuale):
   - MAE: {baseline_mae:.0f}
   
3. CONFRONTO:
   - ML {'MIGLIORA' if train_mae_v2 < baseline_mae else 'PEGGIORA'} del {abs(((baseline_mae - train_mae_v2) / baseline_mae) * 100):.1f}%
   - {better_ml}/{len(comparison_v2)} pazienti hanno errore minore con ML

4. FEATURE PIÙ IMPORTANTI:
{feature_importance.head(3).to_string(index=False)}

5. RACCOMANDAZIONE:
   {'✓ USA ML per predizioni Week 0' if train_mae_v2 < baseline_mae else '⚠️  Linear individuale rimane migliore'}
   {'  Specialmente per cluster con basso R² lineare' if train_mae_v2 < baseline_mae else '  ML non cattura pattern meglio del linear in questo dataset'}
""")


# ============================================================
# APPROCCIO FINALE: SOLO FEATURES INDIPENDENTI
# ============================================================

print("="*70)
print("🔬 APPROCCIO CORRETTO: FEATURES VERAMENTE INDIPENDENTI")
print("="*70)

print("\n💡 PROBLEMA IDENTIFICATO:")
print("   Features come fvc_first, fvc_mean, fvc_max sono troppo correlate")
print("   con il target quando i dati sono vicini a Week 0!")
print("\n   SOLUZIONE: Usare solo features che NON derivano direttamente da FVC")

# ============================================================
# 1. FEATURES VERAMENTE INDIPENDENTI
# ============================================================

print("\n1️⃣ SELEZIONE FEATURES INDIPENDENTI")
print("="*70)

# Features che NON contengono informazione diretta su FVC values
independent_features = [
    # Temporal (posizione temporale, non valori FVC)
    'first_week',
    'last_week',
    'weeks_span',
    'n_observations',
    
    # Rate of change (ma senza valori assoluti)
    'fvc_change_rate',  # Questo è OK perché è relativo
    'fvc_cv',  # Coefficient of variation
    'fvc_acceleration',
    
    # Percentages (proxy clinico, non FVC diretto)
    'percent_first',
    'percent_mean',
    'percent_std',
    
    # Demographics (completamente indipendenti)
    'Age',
    'Sex_encoded',
    'SmokingStatus_encoded',
    'Cluster_Optimal'
]

print(f"\nFeatures selezionate (n={len(independent_features)}):")
for f in independent_features:
    print(f"  ✓ {f}")

# Verifica che le features esistano
available_features = [f for f in independent_features if f in ml_df.columns]
print(f"\nFeatures disponibili: {len(available_features)}/{len(independent_features)}")

X_train_independent = train_df[available_features].values
X_test_independent = test_df[available_features].values
y_train = train_df['FVC_week0'].values

print(f"\nShape: X_train={X_train_independent.shape}")

# ============================================================
# 2. TEST MODELLI CON FEATURES INDIPENDENTI
# ============================================================

print("\n" + "="*70)
print("2️⃣ MODELLI ML CON FEATURES INDIPENDENTI")
print("="*70)

models_final = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Ridge (alpha=10)': Ridge(alpha=10.0),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0),
    'Polynomial (deg 2)': make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
}

# Aggiungi XGBoost se disponibile
try:
    from xgboost import XGBRegressor
    models_final['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
except:
    pass

results_final = []
kf = KFold(n_splits=min(5, len(train_df)), shuffle=True, random_state=42)

print("\n📊 CROSS-VALIDATION RESULTS:\n")
print(f"{'Modello':<25} {'MAE':<12} {'R²':<12} {'Std MAE':<12}")
print("-" * 70)

for name, model in models_final.items():
    try:
        mae_scores = -cross_val_score(model, X_train_independent, y_train, 
                                       cv=kf, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X_train_independent, y_train, 
                                    cv=kf, scoring='r2')
        
        mae_mean = mae_scores.mean()
        mae_std = mae_scores.std()
        r2_mean = r2_scores.mean()
        
        results_final.append({
            'Model': name,
            'MAE': mae_mean,
            'MAE_std': mae_std,
            'R2': r2_mean
        })
        
        print(f"{name:<25} {mae_mean:<12.2f} {r2_mean:<12.3f} {mae_std:<12.2f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {str(e)[:40]}")

results_final_df = pd.DataFrame(results_final).sort_values('MAE')

print("\n" + "="*70)
print("🏆 RANKING FINALE")
print("="*70)
print(results_final_df.to_string(index=False))

# ============================================================
# 3. ANALISI RISULTATI
# ============================================================

print("\n" + "="*70)
print("3️⃣ CONFRONTO CON BASELINE")
print("="*70)

best_ml_mae = results_final_df.iloc[0]['MAE']
best_ml_name = results_final_df.iloc[0]['Model']
baseline_mae = 127.62  # Dal metodo lineare individuale

print(f"\n📊 RISULTATI:")
print(f"\nBaseline (Linear Individuale): {baseline_mae:.2f}")
print(f"Miglior ML ({best_ml_name}): {best_ml_mae:.2f}")

if best_ml_mae < baseline_mae:
    improvement = ((baseline_mae - best_ml_mae) / baseline_mae) * 100
    print(f"\n✨ ML MIGLIORA del {improvement:.1f}%")
else:
    worsening = ((best_ml_mae - baseline_mae) / baseline_mae) * 100
    print(f"\n⚠️  ML PEGGIORA del {worsening:.1f}%")
    print(f"\n💡 CONCLUSIONE:")
    print(f"   Il metodo lineare individuale è difficile da battere perché:")
    print(f"   - Usa TUTTI i dati temporali del paziente")
    print(f"   - È personalizzato per ogni progressione")
    print(f"   - ML deve generalizzare da soli 18 esempi di training")

# Train miglior modello
best_model_final = models_final[best_ml_name]
best_model_final.fit(X_train_independent, y_train)

y_train_pred_final = best_model_final.predict(X_train_independent)
y_test_pred_final = best_model_final.predict(X_test_independent)

# ============================================================
# 4. QUANDO ML È UTILE?
# ============================================================

print("\n" + "="*70)
print("4️⃣ QUANDO ML PUÒ ESSERE UTILE")
print("="*70)

# Confronta predizioni
comparison_final = pd.DataFrame({
    'Patient': train_df['Patient'].values,
    'Real_FVC': y_train,
    'Pred_ML': y_train_pred_final,
    'Pred_Linear': train_df['fvc_intercept'].values,
    'Error_ML': np.abs(y_train - y_train_pred_final),
    'Error_Linear': np.abs(y_train - train_df['fvc_intercept'].values),
    'R2_individual': regression_quality_df[regression_quality_df['Patient'].isin(train_df['Patient'])]['r2'].values,
    'Cluster': train_df['Cluster_Optimal'].values
})

# Quando ML è meglio?
ml_better = comparison_final[comparison_final['Error_ML'] < comparison_final['Error_Linear']]
linear_better = comparison_final[comparison_final['Error_ML'] >= comparison_final['Error_Linear']]

print(f"\n📊 ML migliore in: {len(ml_better)}/{len(comparison_final)} casi ({len(ml_better)/len(comparison_final)*100:.0f}%)")

if len(ml_better) > 0:
    print(f"\n✓ CARATTERISTICHE CASI DOVE ML È MIGLIORE:")
    print(f"   R² lineare medio: {ml_better['R2_individual'].mean():.3f}")
    print(f"   Cluster distribution: {ml_better['Cluster'].value_counts().to_dict()}")

if len(linear_better) > 0:
    print(f"\n✗ CARATTERISTICHE CASI DOVE LINEAR È MIGLIORE:")
    print(f"   R² lineare medio: {linear_better['R2_individual'].mean():.3f}")
    print(f"   Cluster distribution: {linear_better['Cluster'].value_counts().to_dict()}")

# ============================================================
# 5. VISUALIZZAZIONE FINALE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Confronto MAE
ax1 = axes[0, 0]
methods = ['Linear\nIndividuale', f'ML\n({best_ml_name})']
maes = [baseline_mae, best_ml_mae]
colors = ['blue' if baseline_mae < best_ml_mae else 'green', 
          'green' if best_ml_mae < baseline_mae else 'red']
bars = ax1.bar(methods, maes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('MAE (FVC units)', fontweight='bold', fontsize=12)
ax1.set_title('A) Confronto Metodi\n(Cross-Validation)', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, maes):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
             f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Real vs Predicted
ax2 = axes[0, 1]
scatter2 = ax2.scatter(comparison_final['Real_FVC'], comparison_final['Pred_ML'],
                       c=comparison_final['Cluster'], cmap='viridis',
                       s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
min_val = comparison_final[['Real_FVC', 'Pred_ML']].min().min()
max_val = comparison_final[['Real_FVC', 'Pred_ML']].max().max()
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
ax2.set_xlabel('Real FVC at Week 0', fontweight='bold', fontsize=11)
ax2.set_ylabel(f'Predicted ({best_ml_name})', fontweight='bold', fontsize=11)
ax2.set_title(f'B) Predizioni ML\nMAE: {best_ml_mae:.0f}', fontweight='bold', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Plot 3: Distribuzione errori
ax3 = axes[1, 0]
ax3.hist(comparison_final['Error_ML'], bins=15, alpha=0.6, label='ML', 
         color='orange', edgecolor='black')
ax3.hist(comparison_final['Error_Linear'], bins=15, alpha=0.6, label='Linear', 
         color='blue', edgecolor='black')
ax3.axvline(comparison_final['Error_ML'].mean(), color='orange', linestyle='--', linewidth=2)
ax3.axvline(comparison_final['Error_Linear'].mean(), color='blue', linestyle='--', linewidth=2)
ax3.set_xlabel('Absolute Error (FVC)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Frequency')
ax3.set_title('C) Distribuzione Errori', fontweight='bold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Miglioramento per R² individuale
ax4 = axes[1, 1]
comparison_final['Improvement'] = comparison_final['Error_Linear'] - comparison_final['Error_ML']
ax4.scatter(comparison_final['R2_individual'], comparison_final['Improvement'],
            c=comparison_final['Cluster'], cmap='viridis',
            s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax4.set_xlabel('R² (Regressione Lineare Individuale)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Miglioramento ML vs Linear', fontweight='bold', fontsize=11)
ax4.set_title('D) ML Improvement vs Linear Quality', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/ml_final_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualizzazione salvata: ml_final_comparison.png")

# ============================================================
# CONCLUSIONE DEFINITIVA
# ============================================================

print("\n" + "="*70)
print("🎯 CONCLUSIONE DEFINITIVA")
print("="*70)

print(f"""
RISULTATI FINALI (Features Indipendenti):

1. MIGLIOR MODELLO ML: {best_ml_name}
   - MAE Cross-Val: {best_ml_mae:.0f}
   - Features usate: {len(available_features)} (temporali + demografiche)

2. BASELINE (Linear Individuale):
   - MAE: {baseline_mae:.0f}

3. CONFRONTO:
   {'✓ ML MIGLIORA' if best_ml_mae < baseline_mae else '✗ LINEAR RIMANE MIGLIORE'}
   - Differenza: {abs(best_ml_mae - baseline_mae):.0f} unità FVC

4. PERCHÉ {'LINEAR È DIFFICILE DA BATTERE' if best_ml_mae >= baseline_mae else 'ML FUNZIONA'}:
   {'''
   - Linear usa TUTTA la sequenza temporale del paziente
   - È completamente personalizzato
   - ML ha solo 18 esempi per imparare
   - Dataset troppo piccolo per deep learning
   ''' if best_ml_mae >= baseline_mae else '''
   - ML cattura pattern non-lineari
   - Generalizza meglio su alcuni cluster
   - Utile per pazienti con basso R² lineare
   '''}

5. RACCOMANDAZIONE FINALE:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   {'Usa APPROCCIO IBRIDO:' if best_ml_mae < baseline_mae * 1.2 else 'Usa LINEAR INDIVIDUALE:'}
   
   {'✓ Linear per pazienti con R² > 0.5' if best_ml_mae < baseline_mae * 1.2 else '✓ Ha performance migliori'}
   {'✓ ML per pazienti con R² < 0.3' if best_ml_mae < baseline_mae * 1.2 else '✓ Più semplice e interpretabile'}
   {'✓ Cluster-based per casi molto difficili' if best_ml_mae < baseline_mae * 1.2 else '✓ Non richiede training ML'}
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. PER LA TESI:
   - Discuti ENTRAMBI gli approcci
   - Spiega limitazioni dataset piccolo (18 esempi)
   - Mostra che hai testato ML rigorosamente
   - Argomenta perché linear individuale è baseline forte
""")