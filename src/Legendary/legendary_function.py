# legendary_function.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import pickle
with open('export_data.pkl', 'rb') as file:
    export_data = pickle.load(file)

# Import necessary objects
X_columns = export_data["X_columns"]
scaler = export_data["scaler"]
dtree = export_data["dtree"]
pokemon = export_data["pokemon"]

def leggendario():
    print("\nInserisci i dettagli del Pokémon per predire se è leggendario o no:")


    # Converte i tipi disponibili in minuscolo per un confronto case-insensitive
    tipi_disponibili = [tipo.lower() for tipo in pokemon['Type'].unique()]

    while True:
        tipo = input("Tipo (es. Grass, Fire, Water) o premi 0 per vedere tutti i tipi disponibili: ").lower()
        if tipo == '0':
            print("Tipi disponibili:", ', '.join(pokemon['Type'].unique()))  # Mostra i tipi originali
        elif tipo in tipi_disponibili:
            break
        else:
            print("Tipo non valido. Riprova.")

    def input_numerico(prompt):
        while True:
            try:
                valore = int(input(prompt))
                if 0 < valore <= 255:
                    return valore
                else:
                    print("Inserisci un valore tra 1 e 255.")
            except ValueError:
                print("Inserisci un numero valido.")

    hp = input_numerico("HP: ")
    attacco = input_numerico("Attack: ")
    difesa = input_numerico("Defense: ")
    attacco_sp = input_numerico("Special Attack: ")
    difesa_sp = input_numerico("Special Defense: ")
    velocità = input_numerico("Speed: ")
    generazione = input_numerico("Generation: ")
    total = hp + attacco + difesa + attacco_sp + difesa_sp + velocità

    # Creiamo una nuova riga con i dati forniti
    nuovo_pokemon = {"HP": hp, "Attack": attacco, "Defense": difesa, "Sp. Atk": attacco_sp, "Sp. Def": difesa_sp,
                     "Speed": velocità, "Generation": generazione, "Total": total, "#": 0}

    # Aggiungiamo colonne dummy per il tipo
    for col in X_columns:
        if "Type_" in col:
            nuovo_pokemon[col] = 1 if f"Type_{tipo}" == col else 0

    nuovo_pokemon_df = pd.DataFrame([nuovo_pokemon])

    for col in X_columns:
        if col not in nuovo_pokemon_df.columns:
            nuovo_pokemon_df[col] = 0

    nuovo_pokemon_df = nuovo_pokemon_df[X_columns]
    nuovo_pokemon_scaled = scaler.transform(nuovo_pokemon_df)

    predizione = dtree.predict(nuovo_pokemon_scaled)
    print("\nIl Pokémon è leggendario!" if predizione[0] == 1 else "\nIl Pokémon non è leggendario.")


if __name__ == "__main__":
    leggendario()