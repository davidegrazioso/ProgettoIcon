# Importare le librerie necessarie
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del dataset

df = pd.read_csv('datasets/dataset.csv', sep='\t')

# 1. Esplorazione dei dati
# Dimensioni del dataset
print(f"Dimensioni del dataset: {df.shape}")

# Tipi di dati
print("\nTipi di dati:")
print(df.dtypes)

# Controllo dei valori nulli
print("\nValori nulli:")
print(df.isnull().sum())

# 2. Analisi delle variabili
# Statistiche descrittive per variabili numeriche
print("\nStatistiche descrittive:")
print(df.describe())

# Distribuzione dei tipi di Pokémon
print("\nDistribuzione dei tipi di Pokémon:")
print(df['Type'].value_counts())

# Distribuzione dei Pokémon per generazione
print("\nDistribuzione per generazione:")
print(df['Generation'].value_counts())

# Pokémon leggendari
print("\nNumero di Pokémon leggendari:")
print(df['Legendary'].value_counts())

# 3. Analisi statistica
# Selezione delle colonne numeriche
numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Matrice di correlazione
correlation_matrix = numeric_columns.corr()

# Visualizzazione della matrice
print("\nMatrice di correlazione:")
print(correlation_matrix)


# 4. Visualizzazioni
# Configurazione grafici
sns.set(style="whitegrid")

# Distribuzione del punteggio totale
plt.figure(figsize=(10, 6))
sns.histplot(df['Total'], bins=20, kde=True, color='blue')
plt.title("Distribuzione del punteggio totale")
plt.xlabel("Total")
plt.ylabel("Frequenza")
plt.show()

# Percentuale di vittorie vs. Totale
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total', y='POURCENTAGE_DE_VICTOIRE', data=df, hue='Legendary', palette='viridis', alpha=0.7)
plt.title("Percentuale di vittorie vs. Totale")
plt.xlabel("Total")
plt.ylabel("Percentuale di vittorie")
plt.legend(title="Legendary")
plt.show()

# Numero di combattimenti vs. Percentuale di vittorie
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NBR_COMBATS', y='POURCENTAGE_DE_VICTOIRE', data=df, alpha=0.7)
plt.title("Numero di combattimenti vs. Percentuale di vittorie")
plt.xlabel("Numero di combattimenti")
plt.ylabel("Percentuale di vittorie")
plt.show()

# Distribuzione dei tipi di Pokémon
plt.figure(figsize=(14, 7))
sns.countplot(y='Type', data=df, order=df['Type'].value_counts().index, palette='muted')
plt.title("Distribuzione dei tipi di Pokémon")
plt.xlabel("Frequenza")
plt.ylabel("Tipo")
plt.show()

# Boxplot per Totale per Generazione
plt.figure(figsize=(12, 6))
sns.boxplot(x='Generation', y='Total', data=df, palette='coolwarm')
plt.title("Totale per Generazione")
plt.xlabel("Generazione")
plt.ylabel("Totale")
plt.show()

# Heatmap della matrice di correlazione
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Mappa di correlazione")
plt.show()

#metodo del gomito per clusterizzare i dati
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Pre-elaborazione dei dati
# Selezione delle colonne numeriche e rimozione dei NaN
numeric_columns = df.select_dtypes(include=['float64', 'int64']).dropna()

# Standardizzazione dei dati
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_columns)

# Metodo del Gomito
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Visualizzazione del Metodo del Gomito
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.title("Metodo del Gomito")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Inerzia")
plt.xticks(k_values)
plt.grid()
plt.show()

# Metodo della silhouette
silhouette_scores = []

for k in range(2, 11):  # L'indice silhouette richiede almeno 2 cluster
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, labels))

# Visualizzazione della silhouette
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='orange')
plt.title("Indice di Silhouette")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Valore di Silhouette")
plt.xticks(range(2, 11))
plt.grid()
plt.show()

# Scatter plot per le relazioni tra le variabili numeriche
# Scatter plot per le relazioni tra le variabili numeriche
plt.figure(figsize=(14, 10))
sns.pairplot(numeric_columns, diag_kind='kde', corner=True)
plt.suptitle("Relazioni tra Variabili Numeriche", y=1.02)
plt.show()

# Boxplot per individuare outliers
plt.figure(figsize=(14, 8))
numeric_columns.boxplot()
plt.title("Boxplot delle Variabili Numeriche (Outliers)")
plt.ylabel("Valori")
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

# Distribuzione della variabile target
plt.figure(figsize=(10, 6))
sns.histplot(df['POURCENTAGE_DE_VICTOIRE'], bins=20, kde=True, color='green')
plt.title("Distribuzione della Percentuale di Vittorie")
plt.xlabel("Percentuale di Vittorie")
plt.ylabel("Frequenza")
plt.show()

# Distribuzione delle variabili numeriche
numeric_columns.hist(figsize=(14, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Distribuzione delle Variabili Numeriche", y=1.02)
plt.show()

# Correlazione con la variabile target
target_corr = correlation_matrix['POURCENTAGE_DE_VICTOIRE'].sort_values(ascending=False)

# Visualizzazione della correlazione
plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
plt.title("Correlazioni con Percentuale di Vittorie")
plt.xlabel("Variabili")
plt.ylabel("Correlazione")
plt.xticks(rotation=45)
plt.show()

# Scatter plot per le due variabili più correlate con la target
most_correlated = target_corr.index[1:3]  # Prime due variabili più correlate
for col in most_correlated:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=col, y='POURCENTAGE_DE_VICTOIRE', alpha=0.7, color='teal')
    plt.title(f"Relazione tra {col} e Percentuale di Vittorie")
    plt.xlabel(col)
    plt.ylabel("Percentuale di Vittorie")
    plt.show()
