import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import warnings

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
    
    # 7. Metodo del gomito per trovare il numero ottimale di cluster
    Skills = [ 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed']
    scale = StandardScaler()
    StdScale = scale.fit_transform(pokemon_data[Skills])
    n_max_clusters = 15
    score = []
    for cluster in range(1,n_max_clusters):
        kmeans_f = KMeans(n_clusters = cluster, init="k-means++", random_state=10)
        kmeans_f.fit(StdScale)
        score.append(kmeans_f.inertia_)      
    plt.plot(range(1,n_max_clusters), score)
    plt.scatter(range(1,n_max_clusters), score)

    plt.title('The Elbow Chart')
    plt.xlabel('number of clusters')
    plt.ylabel('Inertia')
    n = 5
    plt.annotate(f"Number of clusters = {n}", xy=(n, score[n]), xytext=(n-3, score[n]*0.8), arrowprops=dict(arrowstyle="->"))
    plt.show()



# Esempio di utilizzo
if __name__ == "__main__":
    pokemon_data = pd.read_csv('src/Pokemon.csv')
    eda_analysis(pokemon_data)
