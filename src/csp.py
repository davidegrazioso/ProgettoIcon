import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Caricare il dataset
data = pd.read_csv("datasets/dataset.csv", sep='\t')

# Chiedere all'utente i parametri
num_pokemon = int(input("Quanti Pokémon vuoi nel team (2-6)? "))
while num_pokemon < 2 or num_pokemon > 6:
    num_pokemon = int(input("Inserisci un numero valido tra 2 e 6: "))

include_legendary = input("Vuoi includere Pokémon leggendari? (si/no): ").strip().lower()
while include_legendary not in ["si", "no"]:
    include_legendary = input("Risposta non valida. Vuoi includere Pokémon leggendari? (si/no): ").strip().lower()
include_legendary = include_legendary == "si"

print("Inserisci le statistiche da ottimizzare (es. 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed')")

def normalize_stat(stat):
    return stat.strip().lower().title() if stat.lower() != 'hp' else 'HP'

stat1 = normalize_stat(input("Inserisci la prima statistica da ottimizzare: "))
stat2 = normalize_stat(input("Inserisci la seconda statistica da ottimizzare: "))
stat3 = normalize_stat(input("Inserisci la terza statistica da ottimizzare: "))

# Filtrare il dataset in base alle preferenze dell'utente
filtered_data = data
if not include_legendary:
    filtered_data = filtered_data[filtered_data['Legendary'] == False]

# Ordinare il dataset per somma delle statistiche selezionate
filtered_data['Optimized_Stat'] = (
    filtered_data[stat1] + filtered_data[stat2] + filtered_data[stat3]
)
filtered_data = filtered_data.sort_values(by='Optimized_Stat', ascending=False)

# Selezionare i migliori Pokémon per formare il team
def create_team(data, num_pokemon):
    team = []
    type_counts = {}

    for _, row in data.iterrows():
        pokemon_type = row['Type']
        if type_counts.get(pokemon_type, 0) < 2:  # Non più di 2 Pokémon dello stesso tipo
            team.append(row['Name'])
            type_counts[pokemon_type] = type_counts.get(pokemon_type, 0) + 1

        if len(team) == num_pokemon:
            break

    return team

team = create_team(filtered_data, num_pokemon)

# Mostrare il team
if team:
    print("Il tuo team ottimizzato è:")
    for pokemon in team:
        print(pokemon)
else:
    print("Non è stato possibile creare un team con i vincoli impostati.")