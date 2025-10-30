import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from collections import defaultdict
rng = np.random.RandomState(42)

# =========================
# STEP 0 — Load data
# =========================
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# =========================
# STEP 1 — Baseline, Slope e Baseline Week (slope su tempo centrato)
# =========================
def compute_baseline_slope_week(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values('Weeks').copy()
    baseline_week = g.iloc[0]['Weeks']
    baseline_fvc  = g.iloc[0]['FVC']

    # slope stimata su Δ settimane centrate; richiedi ≥3 punti per stabilità
    if len(g) >= 3:
        X = (g['Weeks'] - baseline_week).values.reshape(-1, 1)
        y = g['FVC'].values
        lr = LinearRegression()
        lr.fit(X, y)
        slope = float(lr.coef_[0])
    else:
        slope = 0.0

    return pd.Series({
        "Baseline_FVC": baseline_fvc,
        "Baseline_FVC_Slope": slope,
        "Baseline_Week": baseline_week
    })
    
#Returns Baseline FVC, SLOPE and BaselineWeek , for regression uses fvc  values based on the difference between values
#Fine as it doesn't involve the ct scan at week 0

baseline_df = df.groupby('Patient', as_index=False).apply(compute_baseline_slope_week)
df = df.merge(baseline_df, on='Patient', how='left')

# Coordinata temporale relativa
df['Delta_Weeks'] = df['Weeks'] - df['Baseline_Week']

# =========================
# STEP 2 — Split training e set per predire Week0 per chi la manca
# =========================
has_week0 = df[df['Weeks'] == 0]['Patient'].unique()
df_train = df[df['Patient'].isin(has_week0)]   # per valutazione equa usando pazienti con Week 0
patients_missing_week0 = df[~df['Patient'].isin(has_week0)]['Patient'].unique()

# Feature "statiche" per i pazienti senza Week0 (verranno usate a inference, con Delta_Weeks=0)
df_features = (df[df['Patient'].isin(patients_missing_week0)]
               .sort_values(['Patient','Weeks'])
               .groupby('Patient', as_index=False)
               .agg({
                   'Age':'first',
                   'Sex':'first',
                   'SmokingStatus':'first',
                   'Baseline_FVC':'first',
                   'Baseline_FVC_Slope':'first',
                   'Baseline_Week':'first'
               }))

# =========================
# STEP 3 — Definizione feature set e regressori
# (uso Delta_Weeks al posto di Weeks/Baseline_Week raw)
# =========================
model_features = {
    "Model_standard": ['Delta_Weeks', 'Age', 'SmokingStatus'],
    "Model_Baseline_FVC": ['Delta_Weeks', 'Age', 'SmokingStatus', 'Baseline_FVC'],
    "Model_Baseline_FVC_Slope": ['Delta_Weeks', 'Age', 'SmokingStatus', 'Baseline_FVC_Slope'],
    "Model_FVC+Slope": ['Delta_Weeks', 'Age', 'SmokingStatus', 'Baseline_FVC', 'Baseline_FVC_Slope'],
    # includo anche Baseline_Week se vuoi tenerne traccia esplicita (non è necessario)
    "Model_FVC+Slope+BL": ['Delta_Weeks','Age','SmokingStatus','Baseline_FVC','Baseline_FVC_Slope','Baseline_Week']
}

regressors = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

categorical = ['Sex', 'SmokingStatus']

def make_preprocess(features):
    cat_features = [f for f in features if f in categorical]
    num_features = [f for f in features if f not in categorical]
    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            # standardize: utile per LR; con OHE sparse uso with_mean=False
            ('scale', StandardScaler(with_mean=False))
        ]), num_features),
    ], remainder='drop')
    return preprocess

# Scorer multipli
scoring = {
    'r2': 'r2',
    'rmse': make_scorer(lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)), greater_is_better=False),
    'mae': make_scorer(mean_absolute_error, greater_is_better=False)
}

# =========================
# STEP 4 — Valutazione con GroupKFold (no leakage) ed esclusione righe baseline in CV
# =========================
results = []
gkf = GroupKFold(n_splits=5)

# Maschera per evitare target=feature: escludo righe con Weeks == Baseline_Week durante la valutazione
mask_eval = df_train['Weeks'] != df_train['Baseline_Week']

for reg_name, reg in regressors.items():
    for feat_name, feats in model_features.items():
        preprocess = make_preprocess(feats)
        model = Pipeline([
            ('preprocess', preprocess),
            ('regressor', reg)
        ])

        x_eval = df_train.loc[mask_eval, feats]
        y_eval = df_train.loc[mask_eval, 'FVC']
        groups_eval = df_train.loc[mask_eval, 'Patient']

        cvres = cross_validate(model, x_eval, y_eval,
                               cv=gkf, groups=groups_eval, scoring=scoring,
                               n_jobs=-1, return_train_score=False)

        r2_mean   = np.mean(cvres['test_r2'])
        rmse_mean = -np.mean(cvres['test_rmse'])
        mae_mean  = -np.mean(cvres['test_mae'])

        results.append({
            "Regressor": reg_name,
            "FeatureSet": feat_name,
            "NumFeatures": len(feats),
            "R2": r2_mean,
            "RMSE": rmse_mean,
            "MAE": mae_mean
        })

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\n📊 Confronto combinazioni modello + feature set (GroupKFold, no-baseline rows):")
print(results_df.head(10))

# Visualizzazione top 10 per R²
plt.figure(figsize=(12, 6))
labels = results_df.head(10).apply(lambda r: f"{r['Regressor']} ({r['FeatureSet']})", axis=1)
vals = results_df.head(10)['R2'].values
plt.barh(labels, vals)
plt.xlabel("R² (CV, GroupKFold)")
plt.title("Top 10 combinazioni modello + feature set")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =========================
# STEP 5 — Selezione best model e fit finale su TUTTE le righe di training
# (qui posso includere anche le baseline rows)
# =========================
best_row = results_df.iloc[0]
best_model_name = best_row['Regressor']
best_features_name = best_row['FeatureSet']
best_features = model_features[best_features_name]
print(f"\n🏆 Miglior modello: {best_model_name} — Feature set: {best_features_name} {best_features}")

# Istanzia il regressor
if best_model_name == "LinearRegression":
    regressor = LinearRegression()
elif best_model_name == "RandomForest":
    regressor = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
else:
    regressor = GradientBoostingRegressor(random_state=42)

preprocess_best = make_preprocess(best_features)
best_model = Pipeline([
    ('preprocess', preprocess_best),
    ('regressor', regressor)
])

x_train_all = df_train[best_features]
y_train_all = df_train['FVC']
best_model.fit(x_train_all, y_train_all)

# =========================
# STEP 6 — Modello dedicato per predire FVC alla Week 0 (fallback se pochi esempi)
# =========================
train_w0 = df[(df['Weeks'] == 0)].copy()
enough_week0 = train_w0['Patient'].nunique() >= 10  # soglia arbitraria per robustezza

w0_model = None
if enough_week0:
    # Imposta Delta_Weeks = 0 (già vero per Weeks==0, ma lo forziamo)
    train_w0['Delta_Weeks'] = 0
    x_w0 = train_w0[best_features]
    y_w0 = train_w0['FVC']
    groups_w0 = train_w0['Patient']

    # valuta (facoltativo; utile per log)
    cvres_w0 = cross_validate(best_model, x_w0, y_w0, cv=GroupKFold(n_splits=5),
                              groups=groups_w0, scoring=scoring, n_jobs=-1)
    print(f"\n📌 Week0-model CV — R2: {np.mean(cvres_w0['test_r2']):.3f} | "
          f"RMSE: {-np.mean(cvres_w0['test_rmse']):.1f} | MAE: {-np.mean(cvres_w0['test_mae']):.1f}")

    # fit finale su tutte le righe Week 0
    w0_model = Pipeline([
        ('preprocess', make_preprocess(best_features)),
        ('regressor', regressor)
    ])
    w0_model.fit(x_w0, y_w0)
else:
    print("\n⚠️ Pochi esempi con Weeks==0 → userò il best model globale come fallback per predire la Week 0.")

# =========================
# STEP 7 — Predizione Week 0 per pazienti che la mancano
# =========================
df_features_pred = df_features.copy()
df_features_pred['Delta_Weeks'] = 0  # a Week 0

model_for_w0 = w0_model if w0_model is not None else best_model
df_features_pred['Predicted_FVC_Week0'] = model_for_w0.predict(df_features_pred[best_features])

print("\n🩺 Prime 10 predizioni FVC @ Week 0 per pazienti senza Week 0:")
print(df_features_pred[['Patient', 'Predicted_FVC_Week0']].head(10))

# =========================
# STEP 8 — Visualizzazione del path della malattia per singolo paziente
#  - punti reali
#  - predizione Week 0 (se manca)
#  - retta “clinica” con pendenza = Baseline_FVC_Slope e intercetta = FVC(Week0 reale o predetto)
# =========================
# =========================
# STEP 8 — Visualizzazione del path della malattia per più pazienti
#  - punti reali
#  - predizione Week 0 (se manca)
#  - retta “clinica” con pendenza = Baseline_FVC_Slope e intercetta = FVC(Week0 reale o predetto)
# =========================
def plot_multiple_patients(patient_ids, show_clinical_line=True):
    """
    Visualize the FVC progression for multiple patients.
    
    Parameters:
        patient_ids (list): List of patient IDs to plot.
        show_clinical_line (bool): Whether to show the clinical line based on baseline FVC and slope.
    """
    for patient_id in patient_ids:
        # Check if patient exists in the dataset
        if patient_id not in df['Patient'].unique():
            print(f"Patient {patient_id} not found in the data.")
            continue
        
        patient_data = df[df['Patient'] == patient_id].sort_values('Weeks').copy()

        if patient_data.empty:
            print(f"Paziente {patient_id} non trovato.")
            continue

        # dati
        weeks = patient_data['Weeks'].values
        fvcs  = patient_data['FVC'].values

        # Crea il grafico
        plt.figure(figsize=(8, 4.5))
        plt.plot(weeks, fvcs, 'o-', label='FVC reale')

        # Predizione Week 0 se manca
        has_w0 = (patient_data['Weeks'] == 0).any()
        if not has_w0 and patient_id in df_features_pred['Patient'].values:
            pred_val = float(df_features_pred.loc[df_features_pred['Patient'] == patient_id,
                                                  'Predicted_FVC_Week0'].values[0])
            plt.scatter(0, pred_val, label='Predizione Week 0', marker='X', s=80)
            # Lineetta di collegamento al primo punto osservato
            plt.plot([0, weeks.min()], [pred_val, fvcs[0]], linestyle='--', alpha=0.6)

        # Regressione clinica: FVC_hat = FVC_w0 + slope * Δweeks
        if show_clinical_line:
            baseline_week = float(patient_data['Baseline_Week'].iloc[0])
            baseline_fvc = float(patient_data['Baseline_FVC'].iloc[0])
            baseline_slope = float(patient_data['Baseline_FVC_Slope'].iloc[0])

            # Disegna sull’intervallo [min(weeks), max(weeks)]
            grid_weeks = np.linspace(weeks.min(), weeks.max(), 100)
            fvc_line = baseline_fvc + baseline_slope * (grid_weeks - baseline_week)
            plt.plot(grid_weeks, fvc_line, linestyle=':', label='Linea clinica (Baseline FVC + Slope * weeks)')

        plt.title(f"Andamento FVC — Paziente {patient_id}")
        plt.xlabel("Weeks")
        plt.ylabel("FVC (ml)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Esegui la funzione per un gruppo di pazienti
patient_ids_to_plot = df['Patient'].unique()[:10]  # Prendi i primi 5 pazienti come esempio
plot_multiple_patients(patient_ids_to_plot)
