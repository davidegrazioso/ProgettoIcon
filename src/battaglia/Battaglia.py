import joblib
import numpy as np
import pandas as pnd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')


# setup data
def setup_data():
    pnd.set_option('display.max_columns', None)
    pnd.set_option('mode.chained_assignment', None)

    # Dataframe
    pokemons = pnd.read_csv('datasets/pokemon_coded.csv', sep=',', encoding='latin-1')


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


    file = 'model_pokemon.mod'
    joblib.dump(algo1, file)

def evaluate_models(dataset):
    # Selezionare le feature e la target
    X = dataset.iloc[:, 4:11].values
    Y = dataset.iloc[:, 16].values

    # Suddividere il dataset in training e validation
    X_LEARN, X_VALIDATE, Y_LEARN, Y_VALIDATE = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Modelli con parametri ottimizzati
    models = {
        "Random Forest Regressor": (RandomForestRegressor(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        "Decision Tree Regressor": (tree.DecisionTreeRegressor(), {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        "SVM Regressor": (SVR(), {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1]
        }),
        "KNN Regressor": (KNeighborsRegressor(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }),
    }

    # Valutare ogni modello
    for model_name, (model, params) in models.items():
        # GridSearchCV per trovare i migliori parametri
        grid_search = GridSearchCV(model, params, cv=5, scoring='r2')
        grid_search.fit(X_LEARN, Y_LEARN)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_VALIDATE)

        # Calcolare precision learn e precision validation
        precision_learn = best_model.score(X_LEARN, Y_LEARN)
        precision_validation = r2_score(Y_VALIDATE, predictions)

        # Calcolare altre metriche di regressione
        mae = mean_absolute_error(Y_VALIDATE, predictions)
        mse = mean_squared_error(Y_VALIDATE, predictions)
        rmse = np.sqrt(mse)

        # Stampare i risultati
        print(f"=========== {model_name} ==========")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Precision Learn: {precision_learn:.4f}")
        print(f"Precision Validation: {precision_validation:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print("===================================")

# setup and train model
learn_and_save(setup_data())
evaluate_models(setup_data())