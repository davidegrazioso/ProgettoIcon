# %% [markdown]
# # **Details on DataSet**

# # Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# # Import Dataset
pokemon = pd.read_csv('datasets/Pokemon.csv')
#######print("The number of sample in dataset is {}.".format(pokemon.shape[0]))

#######pokemon.head()

# We have one column with the name # which we need to drop

#Checking the type of categories in Type1
#######pokemon['Type 1'].unique()
# Type 1 is a type that determines weakness/resistance to attacks. There are several categories in type 1 namely Grass, Fire, Water, Bug, Normal, Poison, Electric, Ground, Fairy, Fighting, Psychic, Rock, Ghost, Ice, Dragon, Dark, Steel, and Flying.

# checking the categories in type 2
#######pokemon['Type 2'].unique()
# Some Pokemon are dual type and have 2. There are several categories in type 1 namely Poison, Flying, Dragon, Ground, Fairy, Grass, Fightin, Psychic, Steel, Ice, Rock, Dark, Water, Electric, Fire, Ghost, Bug, and Normal.

# lets check how many nan value each column have
#######pokemon.isnull().sum()
pokemon.drop(columns='Type 2',inplace=True)
# Since we do not have use of null column so we will eliminate this column as well.

# # Data Visulization
#######plt.figure(figsize=(10,5))
#######sns.countplot(data=pokemon,y='Type 1',hue='Legendary',palette='Set2')
#######plt.title="type with legendary"

# Psychic and Dragon type are more probable to be legendary.
num_col=pokemon.drop(columns=['Name', 'Type 1'])
fig= plt.figure(figsize=(20,20))

for i, var in enumerate(num_col):
    plt.subplot(4, 4, i+1)
    sns.kdeplot(data=num_col,x=var,hue='Legendary',palette='dark')

#######plt.show()
#######plt.figure(figsize=(3,3))
#######sns.countplot(data=pokemon,y='Generation',hue='Legendary',palette='Accent')
#######sns.relplot(data=pokemon, kind="line",x="Legendary", y="Total",col="Type 1", col_wrap=6,height=2, aspect=.75, linewidth=3)
# No matter what the type the pokemon is, legendary will have higher total points.

# # Data Preprocessing

#######pokemon.sample(5)

# One hot encoding for Type column.
# Label Encoding for Legendary column.
pokemon.rename(columns={'Type 1':'Type'},inplace=True)

pokemon_coded = pd.get_dummies(pokemon,columns=['Type'],drop_first=True)
#######pokemon_coded.sample(5)
pokemon_coded.drop(columns=['Name'],inplace=True)
# Label Encoding our Legendary column.
encoder = LabelEncoder()
Y= encoder.fit_transform(pokemon_coded['Legendary'])
X = pokemon_coded.drop(columns=['Legendary'])

# # Split Train Data and Test Data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=25)
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)

# # Model Building

# **Logistic Regression**
log = LogisticRegression()
log.fit(x_train_scaled,y_train)

log_pred = log.predict(x_test_scaled)
#####print('Logistic Regression Classifier Accuracy Score: ',accuracy_score(y_test,log_pred)*100)
#####print("Classification Report:\n", classification_report(y_test,log_pred))

# Based on the classification report from logistic regression, it can be concluded that Legendary precision is 0.98, recall is 0.98, and F1 score is 0.98. For non-Legendary the precision is 0.77, recall is 0.77, and F1 score is 0.77. The accuracy of the Logistic Regression model is 96.25%. This means that the model successfully predicted the class correctly for 96.25% of all data used for evaluation.
confusion_matrix = metrics.confusion_matrix(y_test,log_pred)
metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1]).plot()
#######plt.show()

# **Decision Tree**
dtree= tree.DecisionTreeClassifier()
dtree.fit(x_train_scaled,y_train)
dtree_pred = dtree.predict(x_test_scaled)
#####print('Decision Tree Classifier Accuracy Score: ',accuracy_score(y_test,dtree_pred)*100)
#####print("Classification Report:\n", classification_report(y_test,dtree_pred))

# Based on the classification report from decicion tree, it can be concluded that Legendary precision is 0.98, recall is 0.95, and F1 score is 0.97. For non-Legendary the precision is 0.59, recall is 0.77, and F1 score is 0.67. The accuracy of the decision tree model is 93.75%.
confusion_matrix = metrics.confusion_matrix(y_test,dtree_pred)
#######metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1]).plot()
#######plt.show()

# **Random Forest**
rf = RandomForestClassifier()
rf.fit(x_train_scaled,y_train)

rf_pred = rf.predict(x_test_scaled)
#####print('Random Forest Classifier Accuracy Score: ',accuracy_score(y_test,rf_pred)*100)
#####print("Classification Report:\n", classification_report(y_test,rf_pred))

# Based on the classification report from random forest, it can be concluded that Legendary precision is 0.97, recall is 0.97, and F1 score is 0.97. For non-Legendary the precision is 0.69, recall is 0.69, and F1 score is 0.69. The accuracy of the random forest model is 95%.
confusion_matrix = metrics.confusion_matrix(y_test,rf_pred)
metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1]).plot()
#######plt.show()

# **Naive Bayes**
nb = GaussianNB()
nb.fit(x_train_scaled,y_train)

nb_pred = nb.predict(x_test_scaled)
#####print('Naive Bayes Classifier Accuracy Score: ',accuracy_score(y_test,nb_pred)*100)
#####print("Classification Report:\n", classification_report(y_test,nb_pred))

# Based on the classification report from naive bayes, it can be concluded that Legendary precision is 0.97, recall is 0.24, and F1 score is 0.39. For non-Legendary the precision is 0.10, recall is 0.92, and F1 score is 0.18. The accuracy of the random forest model is 30%, indicating that the model may not work well or there are problems that need to be fixed.
confusion_matrix = metrics.confusion_matrix(y_test,nb_pred)
metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1]).plot()
#######plt.show()

# **KNN**
k = 6
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(x_train_scaled,y_train)
knn_pred = knn.predict(x_test_scaled)
#####print('KNN Classifier Accuracy Score: ',accuracy_score(y_test,knn_pred)*100)
#####print("Classification Report:\n", classification_report(y_test,knn_pred))
# Based on the classification report from KNN, it can be concluded that Legendary precision is 0.93, recall is 1.00, and F1 score is 0.96. For non-Legendary the precision is 1.00, recall is 0.15, and F1 score is 0.27. The accuracy of the KNN model is 93.125%.
confusion_matrix = metrics.confusion_matrix(y_test,nb_pred)
metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1]).plot()
#######plt.show()

# **SVM**
svm = SVC()
svm.fit(x_train_scaled,y_train)

svm_pred = svm.predict(x_test_scaled)
#####print('SVM Classifier Accuracy Score: ',accuracy_score(y_test,svm_pred)*100)
#####print("Classification Report:\n", classification_report(y_test,svm_pred))

# Based on the classification report from SVM, it can be concluded that Legendary precision is 0.94, recall is 0.98, and F1 score is 0.96. For non-Legendary the precision is 0.57, recall is 0.31, and F1 score is 0.40. The accuracy of the KNN model is 92.5%.
confusion_matrix = metrics.confusion_matrix(y_test,svm_pred)
metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1]).plot()
#######plt.show()

# # Conclusion

# Using the logistic regression model has an accuracy rate of 96.25%, decision tree has an accuracy rate of 93.75%, random forest has an accuracy rate of 93.125%, naive bayes has an accuracy rate of 30%, KNN has an accuracy rate of 93.125%, and SVM has an accuracy rate of 92.5%. In this case, using the random forest and KNN methods will have the same accuracy rate of 93.125%. It can be concluded that using the logistic regression model has the highest accuracy rate of 96.25%.

def leggendario():
    print("\nInserisci i dettagli del Pokémon per predire se è leggendario o no:")

    nome = input("Nome: ")

    # Converte i tipi disponibili in minuscolo per un confronto case-insensitive
    tipi_disponibili = [tipo.lower() for tipo in pokemon['Type'].unique()]

    while True:
        tipo = input("Tipo (es. Grass, Fire, Water) o premi 0 per vedere tutti i tipi disponibili: ").lower()
        if tipo == '0':
            print("Tipi disponibili:", ', '.join(pokemon['Type'].unique()))  # Mostra i tipi originali
        elif tipo in tipi_disponibili:
            break
        else:
            print("Tipo non valido. Riprova.")


    def input_numerico(prompt):
        """
        Richiede all'utente di inserire un numero intero compreso tra 1 e 255.

        Argomenti:
        prompt (str): Il messaggio da visualizzare all'utente.

        Restituisce:
        int: Un numero intero compreso tra 1 e 255.

        Solleva:
        ValueError: Se l'input non è un numero valido.
        """
        while True:
            try:
                valore = int(input(prompt))
                if 0 < valore <= 255:
                    return valore
                else:
                    print("Inserisci un valore tra 1 e 255.")
            except ValueError:
                print("Inserisci un numero valido.")

    hp = input_numerico("HP: ")
    attacco = input_numerico("Attack: ")
    difesa = input_numerico("Defense: ")
    attacco_sp = input_numerico("Special Attack: ")
    difesa_sp = input_numerico("Special Defense: ")
    velocità = input_numerico("Speed: ")
    generazione = input_numerico("Generation: ")
    total = hp + attacco + difesa + attacco_sp + difesa_sp + velocità

    # Creiamo una nuova riga con i dati forniti
    nuovo_pokemon = {
        "HP": hp,
        "Attack": attacco,
        "Defense": difesa,
        "Sp. Atk": attacco_sp,
        "Sp. Def": difesa_sp,
        "Speed": velocità,
        "Generation": generazione,
        "Total": total,
        "#": 0  # Placeholder per l'ID
    }

    # Aggiungiamo colonne dummy per il tipo
    for col in X.columns:
        if "Type_" in col:
            nuovo_pokemon[col] = 1 if f"Type_{tipo}" == col else 0

    # Creiamo il DataFrame
    nuovo_pokemon_df = pd.DataFrame([nuovo_pokemon])

    # Allineiamo le colonne: aggiungiamo quelle mancanti e rimuoviamo quelle in eccesso
    for col in X.columns:
        if col not in nuovo_pokemon_df.columns:
            nuovo_pokemon_df[col] = 0  # Aggiungiamo colonne mancanti con valore 0

    # Ordiniamo le colonne per allinearle a quelle del modello
    nuovo_pokemon_df = nuovo_pokemon_df[X.columns]

    # Applichiamo la scalatura
    nuovo_pokemon_scaled = scaler.transform(nuovo_pokemon_df)

    # Previsione con il modello random forest
    predizione = dtree.predict(nuovo_pokemon_scaled)
    print("\nIl Pokémon è leggendario!" if predizione[0] == 1 else "\nIl Pokémon non è leggendario.")

if __name__ == "__main__":
    leggendario()