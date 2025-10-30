#Construct baseline for linear regression for patients with week 0
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import numpy as np
import argparse
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

# === Command-line arguments ===
parser = argparse.ArgumentParser(description="Run Linear or Polynomial Regression per patient")
parser.add_argument('--linear', action='store_true', help='Use linear regression')
parser.add_argument('--poly', type=int, help='Use polynomial regression of given degree')
parser.add_argument('--ridge', type=float, help='Ridge Regression (specify alpha)')
parser.add_argument('--lasso', type=float, help='Lasso Regression (specify alpha)')
parser.add_argument('--svr', action='store_true', help='Support Vector Regression')
args = parser.parse_args()

#---Determine model type
model_type = None
degree = None
alpha = None

if args.linear:
    model_type = 'linear'
elif args.poly:
    model_type = 'poly'
    degree = args.poly
else:
    raise ValueError("Specify one model: --linear, --poly DEGREE, --ridge ALPHA, --lasso ALPHA, --svr")


data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

# === Extract patients with week 0 ===
df_patients_id_week0 = df[df['Weeks'] == 0]['Patient']
df_patients_0 = df[df['Patient'].isin(df_patients_id_week0)]

predictions = []

for patient in df_patients_0['Patient'].unique():

    df_patient = df_patients_0[df_patients_0['Patient'] == patient]
    fvc_week0 = df_patient[df_patient['Weeks'] == 0]['FVC'].values
    X = df_patient['Weeks'].values.reshape(-1, 1)
    y = df_patient['FVC'].values

    # --- Model selection ---
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        fvc_pred = model.predict([[0]])
        Y_pred = model.predict(X)

    elif model_type == 'poly':
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        fvc_pred = model.predict(poly.transform([[0]]))
        Y_pred = model.predict(X_poly)


    mae = mean_absolute_error(fvc_week0, fvc_pred)
    
    
    predictions.append({
        'Patient': patient,
        'FVC_Week0_Predicted': fvc_pred[0],
        'FVC_Week0_Ground_Truth':fvc_week0[0],
        'MAE': mae
    })
    
    print(f"Patient: {patient} ----  {fvc_pred[0]}")
    
    # PLOT
    plt.figure(figsize=(8,6))
    plt.scatter(df_patient['Weeks'], df_patient['FVC'], color='blue', label='Dati reali',alpha=0.7)
    

    # Annotazioni per ogni punto reale
    for x, y in zip(df_patient['Weeks'], df_patient['FVC']):
        if x== 0:
            continue
        plt.text(x, y, f"({x}, {int(y)})", fontsize=8, ha='center', va='bottom', color='navy')
    
    if model_type == 'linear':
        plt.plot(df_patient['Weeks'],model.predict(df_patient[['Weeks']]), color='orange',linestyle='--',label='Linear regression')
    elif model_type =='poly':
        plt.plot(df_patient['Weeks'],model.predict(poly.transform(df_patient[['Weeks']])), color='orange',linestyle='--',label=f'Polynomial-{degree} regression')
    
    # Punto reale a Week 0
    plt.scatter(0, fvc_week0[0], color='green', s=100,
                label='FVC Week 0 (reale)', zorder=5)
    plt.text(0, fvc_week0[0], f"({0}, {int(fvc_week0[0])})", fontsize=9,
             ha='left', va='bottom', color='green')

    # Punto predetto a Week 0
    plt.scatter(0, fvc_pred[0], color='red', s=100,
                marker='x', label='FVC Week 0 (predetto)', zorder=5)
    plt.text(0, fvc_pred[0], f"({0}, {int(fvc_pred[0])})",
             fontsize=9, ha='left', va='top', color='red')
    
    # Dettagli grafico
    plt.title(f"{model_type} Regression per {patient}\nMAE: {mae:.2f}")
    plt.xlabel("Weeks")
    plt.ylabel("FVC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
predictions_df = pd.DataFrame(predictions)

print(f"✅ Predizioni Week 0 per {len(predictions_df)} pazienti")
print("\nStatistiche:")
print(predictions_df[['FVC_Week0_Predicted','MAE']].describe())
    

# Sort by MAE for clearer visualization
predictions_df_sorted = predictions_df.sort_values(by='MAE', ascending=False)

fig, axs = plt.subplots(1, 3, figsize=(18,5))

# Bar chart
axs[0].bar(predictions_df_sorted['Patient'], predictions_df_sorted['MAE'], color='skyblue')
axs[0].set_title("MAE per paziente")
axs[0].set_xlabel("Patient")
axs[0].set_ylabel("MAE")
axs[0].tick_params(axis='x', rotation=90)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Histogram
axs[1].hist(predictions_df['MAE'], bins=20, color='orange', edgecolor='black', alpha=0.7)
axs[1].set_title("Distribuzione dei valori di MAE")
axs[1].set_xlabel("MAE")
axs[1].set_ylabel("Numero di pazienti")
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Boxplot
axs[2].boxplot(predictions_df['MAE'], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red', linewidth=2))
axs[2].set_title("Boxplot MAE")
axs[2].set_ylabel("MAE")
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

