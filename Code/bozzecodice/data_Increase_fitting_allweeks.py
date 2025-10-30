#Increase amount of data for week 0
#Available data
#18 patients have data for week 0

#Linear regression to predict data for week 0
#Patient Weeks FVC Percent Age Sex SmokingStatus

#Predict FVC For Weeks = 0 for patients without the respective data
#Predict Data for data missing

#Weeks present in the excel file
all_weeks = [
    -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93,
    94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 107, 116, 117, 133
]
#We could plot the linear regression for all these weeks to have a general data for all patients but that would need data for training 

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

#All data
data_csv = 'C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/osic-pulmonary-fibrosis-progression/train.csv'
df = pd.read_csv(data_csv)

#Patients with week 0
has_week0 = df[df['Weeks']==0]['Patient'].unique()
df_train = df[df['Patient'].isin(has_week0)]


x_train = df_train[['Weeks','Age','Sex','SmokingStatus']]
y_train = df_train['FVC']

patient_missing_week0 = df[~df['Patient'].isin(has_week0)]['Patient'].unique()

# Aggregate features per patient (Age, Sex, SmokingStatus, Percent mean)
df_features = df[df['Patient'].isin(patient_missing_week0)].groupby('Patient').agg({
    'Age': 'first',
    'Sex': 'first',
    'SmokingStatus': 'first'
}).reset_index()

# Add Weeks column for prediction (Week 0)
df_features['Weeks'] = 0


categorical = ['Sex', 'SmokingStatus']
numeric = ['Weeks','Age']


preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])


model = Pipeline([
    ('preprocess', preprocess),
    ('linreg', LinearRegression())
])

# Cross-validation
y_pred_cv = cross_val_predict(model, x_train, y_train, cv=5)
r2 = r2_score(y_train, y_pred_cv)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))

print("📈 Performance (5-fold CV):")
print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")

# Plot real vs predicted
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_cv, alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel("FVC reale")
plt.ylabel("FVC predetto (CV)")
plt.title(f"Reale vs Predetto (R²={r2:.2f}, RMSE={rmse:.2f})")

plt.subplot(1, 2, 2)
residui = y_train - y_pred_cv
plt.hist(residui, bins=20, alpha=0.7)
plt.xlabel("Errore (FVC reale - predetto)")
plt.ylabel("Frequenza")
plt.title("Distribuzione errori")

plt.tight_layout()
plt.show()

# Fit model on full training data
model.fit(x_train, y_train)

# Predict Week 0 for missing patients
df_features['FVC_week0_pred'] = model.predict(df_features[categorical + numeric])

# Combine real and predicted Week 0 values
df_real = df_train[df_train['Weeks'] == 0][['Patient', 'FVC']].rename(columns={'FVC': 'FVC_week0'})
df_pred = df_features[['Patient', 'FVC_week0_pred']].rename(columns={'FVC_week0_pred': 'FVC_week0'})
df_final = pd.concat([df_real, df_pred])

print("\n✅ FVC settimana 0 (reali + stimati):")
print(df_final.head())

# Distribution plot
plt.figure(figsize=(6,4))
plt.hist(df_final['FVC_week0'], bins=20, alpha=0.7)
plt.xlabel("FVC settimana 0")
plt.ylabel("Numero di pazienti")
plt.title("Distribuzione FVC settimana 0 (reale + stimata)")
plt.show()