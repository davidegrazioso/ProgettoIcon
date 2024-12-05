import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import warnings
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

warnings.filterwarnings('ignore')

def eda_analysis(pokemon_data):
    """
    Funzione per effettuare un'analisi esplorativa dei dati.
    """
    # Colonne numeriche
    stats_columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Total"]
    numeric_data = pokemon_data[stats_columns]

    # Controllo valori mancanti
    missing_values = pokemon_data.isnull().sum()
    print("Valori mancanti per colonna:")
    print(missing_values[missing_values > 0])

    # Distribuzione delle variabili categoriali
    categorical_columns = ["Type 1", "Type 2", "Legendary"]
    for column in categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=pokemon_data, x=column, order=pokemon_data[column].value_counts().index, palette="viridis")
        plt.title(f"Distribuzione della variabile '{column}'")
        plt.xticks(rotation=45)
        plt.show()

    # 1. Heatmap delle correlazioni
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap delle correlazioni")
    plt.show()

    # 2. Distribuzioni delle variabili numeriche
    numeric_data.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Distribuzioni delle variabili numeriche", fontsize=16)
    plt.show()

    # 3. Identificazione e visualizzazione degli outlier
    z_scores = zscore(numeric_data)
    outliers = (np.abs(z_scores) > 3).any(axis=1)
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=numeric_data, palette="Set2", orient="h")
    plt.title("Boxplot delle variabili numeriche per identificare gli outlier")
    plt.xlabel("Valori")
    plt.ylabel("Statistiche")
    plt.show()

    # 4. Relazioni fra variabili numeriche
    sns.pairplot(numeric_data, diag_kind="kde", plot_kws={"alpha": 0.6})
    plt.suptitle("Relazioni fra variabili numeriche", y=1.02)
    plt.show()

    # 5. Analisi della variabile target (se presente)
    # Suddividere la colonna 'Total' in intervalli
    bins = [180, 300, 420, 540, 660, 780]  # Intervalli di 'Total'
    labels = ['180-300', '301-420', '421-540', '541-660', '661-780']  # Etichette per i gruppi
    pokemon_data['Total_Group'] = pd.cut(pokemon_data['Total'], bins=bins, labels=labels, right=True)

    # 6. Analisi della variabile target rispetto alle variabili numeriche
    if 'Total' in pokemon_data.columns:
        for column in stats_columns:
            print(f"\n{column} vs Total Group")
            print(pokemon_data.groupby('Total_Group')[column].describe())
    
    # 7. metodo del gomito e Silhouette Score
    scale = StandardScaler()
    skills = stats_columns  # Assuming you want to use the stats_columns for clustering
    scaled_data = scale.fit_transform(pokemon_data[skills])
    inertias = []
    silhouette_scores = []
    max_clusters = 10  # Define the maximum number of clusters

    for k in range(2, max_clusters):  # Il Silhouette Score non è definito per k=1
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=10)        
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))    
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(2, max_clusters), inertias, 'b-o', label="Inertia (Metodo del gomito)")
    ax1.set_xlabel("Numero di cluster")
    ax1.set_ylabel("Inertia", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_title("Metodo del Gomito e Silhouette Score")

    ax2 = ax1.twinx()
    ax2.plot(range(2, max_clusters), silhouette_scores, 'r-o', label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.tight_layout()
    plt.legend(loc="upper left")
    plt.show()





# Esempio di utilizzo
if __name__ == "__main__":
    pokemon_data = pd.read_csv('datasets/Pokemon.csv')
    eda_analysis(pokemon_data)
