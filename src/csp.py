import pandas as pd
from constraint import Problem
import time
from concurrent.futures import ThreadPoolExecutor

def partenza():
    # Caricare il dataset
    data = pd.read_csv("datasets/dataset.csv", sep='\t')

    # Input
    num_pokemon = 3

    include_legendary = input("Vuoi includere Pokémon leggendari? (si/no): ").strip().lower() == "si"

    print("Inserisci le statistiche da ottimizzare (es. 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed')")

    def normalize_stat(stat):
        return stat.strip().lower().title() if stat.lower() != 'hp' else 'HP'

    valid_stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    stats = []

    while len(stats) < 3:
        stat = normalize_stat(input(f"Inserisci la statistica {len(stats) + 1}: "))
        if stat not in valid_stats:
            print("Statistica non valida. Riprova.")
        elif stat in stats:
            print("Hai già inserito questa statistica. Riprova.")
        else:
            stats.append(stat)

    stat1, stat2, stat3 = stats

    # Filtrare il dataset
    filtered_data = data.copy()
    if not include_legendary:
        filtered_data = filtered_data[filtered_data['Legendary'] == False]

    filtered_data['Optimized_Stat'] = (
        filtered_data[stat1] + filtered_data[stat2] + filtered_data[stat3]
    )

    # Ridurre il dataset ai primi 50 Pokémon più ottimizzati
    filtered_data = filtered_data.nlargest(50, 'Optimized_Stat')

    # Estrarre i dati necessari
    pokemon_names = filtered_data['Name'].tolist()
    pokemon_types = dict(zip(filtered_data['Name'], filtered_data['Type']))
    pokemon_stats = dict(zip(filtered_data['Name'], filtered_data['Optimized_Stat']))

    # Configurazione del problema
    problem = Problem()
    slots = [f"Slot_{i}" for i in range(num_pokemon)]
    problem.addVariables(slots, pokemon_names)

    # Vincolo: Tutti i Pokémon nel team devono essere diversi
    def all_different(*team):
        return len(set(team)) == len(team)

    problem.addConstraint(all_different, slots)

    # Vincolo: Non più di 2 Pokémon per tipo
    def max_two_per_type(*team):
        type_counts = {}
        for name in team:
            if name is None:
                continue
            pokemon_type = pokemon_types[name]
            for t in pokemon_type.split('/'):  # Gestione di tipi multipli
                type_counts[t] = type_counts.get(t, 0) + 1
                if type_counts[t] > 2:
                    return False
        return True

    problem.addConstraint(max_two_per_type, slots)

    # Funzione per calcolare il punteggio di un team
    def calculate_team_score(team):
        return sum(pokemon_stats[name] for name in team)

    # Funzione per trovare il team migliore in parallelo
    def find_best_team_parallel(problem, slots):
        best_team = None
        best_score = 0

        def evaluate_solution(solution):
            nonlocal best_team, best_score
            team = [solution[slot] for slot in slots]
            score = calculate_team_score(team)
            if score > best_score:
                best_team = team
                best_score = score

        # Multithreading per esplorare le soluzioni
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_solution, solution) for solution in problem.getSolutionIter()]
            for future in futures:
                future.result()  # Aspetta la terminazione di tutti i thread

        return best_team, best_score

    # Misurare il tempo di esecuzione
    start_time = time.time()

    best_team, best_score = find_best_team_parallel(problem, slots)

    end_time = time.time()
    #print(f"Tempo di calcolo: {end_time - start_time:.2f} secondi")

    # Mostrare il risultato
    if best_team:
        print("Il tuo team ottimizzato è:")
        for pokemon in best_team:
            print(f"{pokemon} ")
    else:
        print("Non è stato possibile trovare una soluzione con i vincoli forniti.")

if __name__ == "__main__":
    partenza()