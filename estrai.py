import os
import csv
import ast
import shutil

# Percorsi di base
csv_path = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/Code/patient_event_slice.csv"
base_dir = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/CodeDevelopment/osic-pulmonary-fibrosis-progression/train"
output_dir = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/extracted"

# Creazione della cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

def safe_parse_slice_list(val):
    """Converte una stringa CSV in lista, gestendo errori e None."""
    if not val:
        return []
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                # Rimuove duplicati mantenendo ordine
                return list(dict.fromkeys(parsed))
            else:
                return []
        except Exception:
            return []
    elif isinstance(val, (list, tuple)):
        return list(dict.fromkeys(val))
    else:
        return []

# Leggi CSV e processa righe
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)  # usa DictReader per maggiore chiarezza
    for row in reader:
        patient_id = row['Patient'].strip()
        files_list = safe_parse_slice_list(row['slice_files'])

        if not files_list:
            print(f"⚠️ Nessun file da copiare per {patient_id}")
            continue

        # Cartella sorgente e destinazione
        source_folder = os.path.join(base_dir, patient_id)
        dest_folder = os.path.join(output_dir, patient_id)
        os.makedirs(dest_folder, exist_ok=True)

        # Copia i file, verifica se esistono
        for fname in files_list:
            src = os.path.join(source_folder, fname)
            dst = os.path.join(dest_folder, fname)

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"⚠️ File non trovato: {src}")

print("✅ Estrazione completata!")
