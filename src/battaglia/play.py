import csv
import joblib


# Search Pokémon information in the Pokédex by name
def find_pokemon_data_by_name(name, pokedex):
    for pokemon in pokedex:
        if pokemon[1].lower() == name.lower():
            return [pokemon[0], pokemon[1], pokemon[4], pokemon[5], pokemon[6], pokemon[7], pokemon[8],
                    pokemon[9], pokemon[10]]
    return None  # Return None if the Pokémon is not found


# Predict victory percentage of Pokémon with trained model
def predict(name_first_pokemon, name_second_pokemon, pokedex):
    first_pokemon = find_pokemon_data_by_name(name_first_pokemon, pokedex)
    if not first_pokemon:
        print(f"Error: Pokémon with name '{name_first_pokemon}' not found in the Pokédex")
        return

    second_pokemon = find_pokemon_data_by_name(name_second_pokemon, pokedex)
    if not second_pokemon:
        print(f"Error: Pokémon with name '{name_second_pokemon}' not found in the Pokédex")
        return

    predict_model = joblib.load('datasets/battaglia/model_pokemon.mod')
    predict_first_pokemon = predict_model.predict(
        [[first_pokemon[2], first_pokemon[3], first_pokemon[4], first_pokemon[5],
          first_pokemon[6], first_pokemon[7], first_pokemon[8]]])
    predict_second_pokemon = predict_model.predict([[second_pokemon[2], second_pokemon[3], second_pokemon[4],
                                                     second_pokemon[5], second_pokemon[6], second_pokemon[7],
                                                     second_pokemon[8]]])

    print(f"({first_pokemon[0]}) {first_pokemon[1]} VS ({second_pokemon[0]}) {second_pokemon[1]}")

    if predict_first_pokemon > predict_second_pokemon:
        print(f"{first_pokemon[1]} è il vincitore!")
    elif predict_first_pokemon == predict_second_pokemon:
        print("È un pareggio!")
    else:
        print(f"{second_pokemon[1]} è il vincitore!")

def sfida():
    with open('datasets/Pokemon.csv', newline='') as csvfile:
        pokedex = list(csv.reader(csvfile))
        pokedex_iter = iter(pokedex)
        next(pokedex_iter)
        # Input Pokémon names
        name_first_pokemon = input("Inserisci il nome del primo Pokemon: ").strip().lower().title()
        name_second_pokemon = input("Inserisci il nome del secondo Pokemon: ").strip().lower().title()

        predict(name_first_pokemon, name_second_pokemon, pokedex)
# Main
if __name__ == "__main__":
    sfida()
