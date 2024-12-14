import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Funzione per preprocessare i dati
def preprocess_data():
    # Caricamento del dataset Pokemon
    pokemon = pd.read_csv('datasets/Pokemon.csv')
    
    # Rimozione della colonna "Type 2"
    pokemon.drop(columns='Type 2', inplace=True)
    
    # Rinominare "Type 1" in "Type"
    pokemon.rename(columns={'Type 1': 'Type'}, inplace=True)
    
    # Salvataggio del dataset preprocessato
    pokemon.to_csv('datasets/pokemon_coded.csv', index=False)
    return pokemon

# Funzione per configurare i dati
def setup_data():
    # Configurazioni di visualizzazione di pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('mode.chained_assignment', None)

    # Caricamento del dataset preprocessato
    pokemons = pd.read_csv('datasets/pokemon_coded.csv', sep=',', encoding='latin-1')

    # Caricamento del dataset delle battaglie
    fights = pd.read_csv('datasets/combats.csv', sep=',', encoding='latin-1')

    # Calcolo del numero di combattimenti e vittorie
    nbFirstPosition = fights.groupby('First_pokemon')['Winner'].count()
    nbSecondPosition = fights.groupby('Second_pokemon')['Winner'].count()
    nbVictories = fights.groupby('Winner')['First_pokemon'].count()

    # Aggregazione dei dati
    aggregation = pd.DataFrame(index=nbVictories.index)
    aggregation['NBR_COMBATS'] = nbFirstPosition.add(nbSecondPosition, fill_value=0)
    aggregation['NB_VICTOIRES'] = nbVictories
    aggregation['POURCENTAGE_DE_VICTOIRE'] = aggregation['NB_VICTOIRES'] / aggregation['NBR_COMBATS']

    # Merge con il dataset dei Pokemon
    newPokedex = pokemons.merge(aggregation, left_on='#', right_index=True, how='left')
    newPokedex.fillna(0, inplace=True)  # Sostituisce i NaN con 0
    return newPokedex

# Funzione per normalizzare i dati numerici
def normalize_data(dataframe):
    # Selezione delle colonne numeriche
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    
    # Applicazione della normalizzazione Min-Max
    scaler = MinMaxScaler()
    dataframe[numeric_columns] = scaler.fit_transform(dataframe[numeric_columns])
    
    return dataframe

# Funzione per valutare il dataset
def evaluate_dataset(dataframe):
    print(f"\n--- Analisi del nuovo dataset ---\n")
    
    # Dimensioni del dataset
    print(f"Dimensioni: {dataframe.shape[0]} righe, {dataframe.shape[1]} colonne")
    
    # Valori mancanti
    print("\nValori mancanti:")
    missing_values = dataframe.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.any() else "Nessun valore mancante")


# Esecuzione del preprocessing
pokemon_coded = preprocess_data()

# Configurazione del dataset finale
dataset = setup_data()

# Normalizzazione dei dati
dataset = normalize_data(dataset)

# Salvataggio del dataset finale
dataset.to_csv('datasets/dataset.csv', sep='\t', index=False)

# Verifica del risultato
print("Preprocessing e normalizzazione completati. Dataset salvato in 'datasets/dataset.csv'.")
check = pd.read_csv('datasets/dataset.csv', sep='\t')
evaluate_dataset(check)

# Mostra alcune righe dei dati normalizzati
print("\nEsempio dei dati normalizzati:")
print(check.head())
