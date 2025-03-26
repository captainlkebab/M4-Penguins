import requests
import pandas as pd
import joblib

# Lade das gespeicherte Modell und den Skalierer
model, scaler = joblib.load("penguin_classifier.pkl")

# Mapping der Kategorien zurück zu Spezies-Namen
species_mapping = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}


# Features für das Modell
target_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Funktion zum Abrufen neuer Pinguindaten aus der API
def fetch_new_penguin():
    url = "http://130.225.39.127:8000/new_penguin/"
    response = requests.get(url)
    if response.status_code == 200:
        new_data = response.json()
        new_df = pd.DataFrame([new_data])
        return new_df
    return None

# Funktion zur Vorhersage der Spezies
def predict_species(new_df):
    new_df_scaled = scaler.transform(new_df[target_features])
    prediction = model.predict(new_df_scaled)
    species_name = species_mapping.get(prediction[0], "Unknown")
    return species_name

# Hauptfunktion zur Automatisierung
def main():
    new_penguin = fetch_new_penguin()
    if new_penguin is not None:
        species_prediction = predict_species(new_penguin)
        new_penguin["predicted_species"] = species_prediction
        print("Predicted species:", species_prediction)
        new_penguin.to_csv("prediction_results.csv", mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
