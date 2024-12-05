import joblib
import pandas as pnd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVC



# setup data
def setup_data():
    pnd.set_option('display.max_columns', None)
    pnd.set_option('mode.chained_assignment', None)

    # Dataframe
    pokemons = pnd.read_csv('datasets/Pokemon.csv', sep=',', encoding='latin-1')

    # transform legendary column to int
    pokemons['Legendary'] = (pokemons['Legendary'] == 'True').astype(int)

    # load fights
    fights = pnd.read_csv('datasets/combats.csv', sep=',', encoding='latin-1')

    nbFirstPosition = fights.groupby('First_pokemon').count()
    nbSecondPosition = fights.groupby('Second_pokemon').count()
    nbVictories = fights.groupby('Winner').count()

    aggregation = fights.groupby('Winner').count()
    aggregation.sort_index()

    aggregation['NBR_COMBATS'] = nbFirstPosition.Winner + nbSecondPosition.Winner
    aggregation['NB_VICTOIRES'] = nbVictories.First_pokemon

    # % of victory
    aggregation['POURCENTAGE_DE_VICTOIRE'] = nbVictories.First_pokemon / (
            nbFirstPosition.Winner + nbSecondPosition.Winner)

    newPokedex = pokemons.merge(aggregation, left_on='#', right_index=True, how='left')

    dataset = newPokedex
    dataset = dataset.dropna(axis=0, how='any')
    dataset.to_csv('datasets/dataset.csv', sep='\t')
    return dataset


# train and save model
def learn_and_save(dataset):
    # NIVEAU_ATTAQUE;NIVEAU_DEFFENSE;NIVEAU_ATTAQUE_SPECIALE;NIVEAU_DEFENSE_SPECIALE;VITESSE;NOMBRE_GENERATIONS
    X = dataset.iloc[:, 4:11].values
    Y = dataset.iloc[:, 16].values

    X_LEARN, X_VALIDATE, Y_LEARN, Y_VALIDATE = train_test_split(X, Y, test_size=0.2, random_state=0)

    algo1 = RandomForestRegressor()
    algo1.fit(X_LEARN, Y_LEARN)
    predictions = algo1.predict(X_VALIDATE)
    precision = r2_score(Y_VALIDATE, predictions)
    precision_learn = algo1.score(X_LEARN, Y_LEARN)
    print("=========== RANDOM FOREST REGRESSION ==========")
    print("Precision Learn : " + str(precision_learn))
    print("Precision Validation : " + str(precision))
    print("===============================================")

    file = 'datasets/model_pokemon.mod'
    joblib.dump(algo1, file)


# setup and train model
learn_and_save(setup_data())