import warnings
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

warnings.filterwarnings('ignore')

def closest():
    # Caricamento del dataset dei Pokémon da un file CSV
    pokemon_data = pd.read_csv('src/Pokemon.csv')
    # Selezione delle colonne numeriche
    stats_columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Total"]
    numeric_data = pokemon_data[stats_columns]

    # Normalizzazione dei dati numerici
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Input delle statistiche da parte dell'utente
    nome = input("Inserisci il nome del Pokémon: ")
    print("Inserisci le statistiche del Pokémon:")
    user_stats = {
        "HP": float(input("HP: ")),
        "Attack": float(input("Attack: ")),
        "Defense": float(input("Defense: ")),
        "Sp. Atk": float(input("Sp. Atk: ")),
        "Sp. Def": float(input("Sp. Def: ")),
        "Speed": float(input("Speed: ")),
    }
    user_stats["Total"] = sum(user_stats.values())
    user_stats_values = np.array(list(user_stats.values())).reshape(1, -1)
    user_stats_scaled = scaler.transform(user_stats_values)

    # Ridurre il numero di cluster a 10 (o altro numero ragionevole)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(scaled_data)

    # Identificazione del cluster delle statistiche dell'utente
    user_cluster_label = kmeans.predict(user_stats_scaled)

    # Filtraggio dei Pokémon nel cluster
    cluster_indices = np.where(kmeans.labels_ == user_cluster_label[0])[0]
    cluster_pokemon = pokemon_data.iloc[cluster_indices]

    # Calcolo delle distanze tra le statistiche dell'utente e i Pokémon nel cluster
    cluster_scaled_data = scaled_data[cluster_indices]
    distances = np.linalg.norm(cluster_scaled_data - user_stats_scaled, axis=1)

    # Ordinamento dei Pokémon per distanza crescente
    sorted_indices = np.argsort(distances)

    # Selezione dei 5 Pokémon più vicini
    closest_pokemon = pokemon_data.iloc[cluster_indices[sorted_indices[:5]]]

    # Risultato
    print("\nI 5 Pokémon più simili alle statistiche fornite sono:")
    for i, pokemon in closest_pokemon.iterrows():
        print(f"- {pokemon['Name']}")

if __name__ == "__main__":
    closest()

