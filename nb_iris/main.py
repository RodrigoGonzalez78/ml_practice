import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Cargar dataset Iris desde scikit-learn
iris = load_iris()

# Crear DataFrame con las características y etiquetas
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].apply(lambda x: iris.target_names[x])

# Mostrar las primeras filas
print(df.head())

# Dividir el dataset en conjunto de entrenamiento y prueba
X = iris.data  # largo/ancho de sépalo y pétalo
y = iris.target  # especie: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# Dividir el dataset en entrenamiento y prueba
# Usamos stratify para mantener la proporción de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar el modelo Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predicciones
y_pred = nb_model.predict(X_test)

# Precisión general
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy:.2f}")

# Reporte completo
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


conf_mat = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Naive Bayes")
plt.show()


