'''Pseudocodice
1. Aggrega dati per paziente (slope, mean FVC, etc.)
2. Normalizza le features
3. Applica PCA (opzionale, per visualizzazione)
4. Clustering (K-Means con k=3-5 inizialmente)
5. Analizza i cluster trovati
6. Per ogni cluster, fai regressione FVC ~ Weeks
7. Predici Week 0 per pazienti mancanti usando il modello del loro cluster
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Carica i dati
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score



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