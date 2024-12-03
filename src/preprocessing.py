# Importazione delle librerie necessarie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

warnings.filterwarnings('ignore')

# Caricamento del dataset Pokémon
pokemon = pd.read_csv('src/Pokemon.csv')

# Esplorazione iniziale del dataset
print(f"The dataset contains {pokemon.shape[0]} samples and {pokemon.shape[1]} columns.")
print("Column names:", pokemon.columns)

# Visualizzazione iniziale dei dati
print(pokemon.head())

# Pulizia e gestione delle colonne
# Rimozione della colonna 'Type 2' per ridurre la complessità
pokemon.drop(columns=['Type 2'], inplace=True)

# Rinomina della colonna 'Type 1' in 'Type' per maggiore semplicità
pokemon.rename(columns={'Type 1': 'Type'}, inplace=True)

# Verifica di valori nulli
print("Null values per column:", pokemon.isnull().sum())

# Visualizzazione dei dati leggendari in base al tipo
plt.figure(figsize=(10, 5))
sns.countplot(data=pokemon, y='Type', hue='Legendary', palette='Set2')
plt.title("Type distribution by Legendary status")
plt.show()

# Distribuzione delle variabili numeriche
num_col = pokemon.drop(columns=['Name', 'Type'])
fig = plt.figure(figsize=(20, 20))
for i, var in enumerate(num_col):
    plt.subplot(4, 4, i + 1)
    sns.kdeplot(data=num_col, x=var, hue=pokemon['Legendary'], palette='dark')
plt.show()

# Preparazione dei dati per il machine learning
# One-hot encoding per la colonna 'Type'
pokemon_coded = pd.get_dummies(pokemon, columns=['Type'], drop_first=True)

# Label encoding per la colonna 'Legendary'
label_encoder = LabelEncoder()
pokemon_coded['Legendary'] = label_encoder.fit_transform(pokemon_coded['Legendary'])

# Rimozione della colonna 'Name'
pokemon_coded.drop(columns=['Name'], inplace=True)

# Esplorazione del dataset preprocessato
print("Sample of preprocessed data:")
print(pokemon_coded.sample(5))

# Dataset preprocessato pronto per l'addestramento
