import requests
import pandas as pd
import sqlite3
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Laden des Penguins-Datensatzes
penguins = sns.load_dataset("penguins").dropna()

# 2. Speichern in eine SQLite-Datenbank
db_path = "penguins.db"
conn = sqlite3.connect(db_path)
penguins.to_sql("penguins", conn, if_exists="replace", index=False)
conn.close()

# 3. Feature Selection
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
label = "species"

# 4. Laden der Daten aus der Datenbank
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT * FROM penguins", conn)
conn.close()

# 5. Datenaufbereitung
df = df.dropna()
df[label] = df[label].astype("category").cat.codes  # Umwandlung in numerische Werte

# 6. Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.2, random_state=42)

# 7. Daten skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Modelltraining
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Modell speichern
joblib.dump((model, scaler), "penguin_classifier.pkl")

# 10. API-Daten abrufen und Vorhersage treffen
def fetch_new_penguin():
    url = "http://130.225.39.127:8000/new_penguin/"
    response = requests.get(url)
    if response.status_code == 200:
        new_data = response.json()
        new_df = pd.DataFrame([new_data])
        return new_df
    return None

def predict_species(new_df):
    model, scaler = joblib.load("penguin_classifier.pkl")
    new_df_scaled = scaler.transform(new_df[features])
    prediction = model.predict(new_df_scaled)
    return prediction

if __name__ == "__main__":
    new_penguin = fetch_new_penguin()
    if new_penguin is not None:
        species_prediction = predict_species(new_penguin)
        print("Predicted species:", species_prediction)