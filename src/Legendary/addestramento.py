# main_module.py
# %% [markdown]
# # **Details on DataSet**

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Walk through directories to list dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import Dataset
pokemon = pd.read_csv('datasets/pokemon_coded.csv')

print("The number of sample in dataset is {}.".format(pokemon.shape[0]))
pokemon.head()

# Visualization
plt.figure(figsize=(10, 5))
sns.countplot(data=pokemon, y='Type', hue='Legendary', palette='Set2')
plt.title = "Type with Legendary"
plt.show()

num_col = pokemon.drop(columns=['Name', 'Type'])
fig = plt.figure(figsize=(20, 20))
for i, var in enumerate(num_col):
    plt.subplot(4, 4, i + 1)
    sns.kdeplot(data=num_col, x=var, hue='Legendary', palette='dark')
plt.show()

plt.figure(figsize=(3, 3))
sns.countplot(data=pokemon, y='Generation', hue='Legendary', palette='Accent')
sns.relplot(data=pokemon, kind="line", x="Legendary", y="Total", col="Type", col_wrap=6, height=2, aspect=.75, linewidth=3)
plt.show()

# Encoding
pokemon_coded = pd.get_dummies(pokemon, columns=['Type'], drop_first=True)
Y = pokemon_coded['Legendary']
X = pokemon_coded.drop(columns=['Name', 'Legendary'])

# Splitting and Scaling
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=25)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Decision Tree
dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train_scaled, y_train)
dtree_pred = dtree.predict(x_test_scaled)
print('Decision Tree Accuracy:', accuracy_score(y_test, dtree_pred))
ConfusionMatrixDisplay.from_predictions(y_test, dtree_pred).plot()
plt.show()

# Random Forest
rf = RandomForestClassifier()
rf.fit(x_train_scaled, y_train)
rf_pred = rf.predict(x_test_scaled)
print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred).plot()
plt.show()

# Naive Bayes
nb = GaussianNB()
nb.fit(x_train_scaled, y_train)
nb_pred = nb.predict(x_test_scaled)
print('Naive Bayes Accuracy:', accuracy_score(y_test, nb_pred))
ConfusionMatrixDisplay.from_predictions(y_test, nb_pred).plot()
plt.show()

# KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train_scaled, y_train)
knn_pred = knn.predict(x_test_scaled)
print('KNN Accuracy:', accuracy_score(y_test, knn_pred))
ConfusionMatrixDisplay.from_predictions(y_test, knn_pred).plot()
plt.show()

# SVM
svm = SVC()
svm.fit(x_train_scaled, y_train)
svm_pred = svm.predict(x_test_scaled)
print('SVM Accuracy:', accuracy_score(y_test, svm_pred))
ConfusionMatrixDisplay.from_predictions(y_test, svm_pred).plot()
plt.show()

#  decision tree has an accuracy rate of 93.75%, random forest has an accuracy rate of 93.125%, 
# naive bayes has an accuracy rate of 30%, KNN has an accuracy rate of 93.125%, and SVM has an accuracy rate of 92.5%. 
# In this case, using the random forest and KNN methods will have the same accuracy rate of 93.125%. 
# Therefore, the decision tree method is the best method to use in this case.

# Export objects for `leggendario` function
export_data = {
    "X_columns": X.columns,
    "scaler": scaler,
    "dtree": dtree,
    "pokemon": pokemon
}

import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
with open('export_data.pkl', 'wb') as file:
    pickle.dump(export_data, file)
    # Function to evaluate model
    def evaluate_model(model, x_test, y_test):
        pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred)
        return accuracy, precision, recall, f1, roc_auc

    # Evaluate all models
    models = {
        "Decision Tree": dtree,
        "Random Forest": rf,
        "Naive Bayes": nb,
        "KNN": knn,
        "SVM": svm
    }

    for model_name, model in models.items():
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, x_test_scaled, y_test)
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")