import pandas as pd
import seaborn as sns
import ml_functions

#Importar los datos
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")


# Seleccionamos las columnas relevantes
print(chicago_taxi_dataset.head(5))
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']] 


print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
print(training_df.describe(include='all'))


# Cual es el valor minimo y maximo de la tarifa?
max_fare = training_df['FARE'].max()
print("Cual es la tarifa maxima? \t\t\t\tRespuesta: ${fare:.2f}".format(fare = max_fare))

min_fare = training_df['FARE'].min()
print("Cual es la tarifa minima? \t\t\t\tAnswer: ${fare:.2f}".format(fare = min_fare))


#Cual es la distancia media de todos los viajes?
mean_distance = training_df['TRIP_MILES'].mean()
print("Cual es la distancia media de los viajes? \t\tRespuestas: {mean:.4f} miles".format(mean = mean_distance))

# Cuantas compañías de taxis hay en el dataset?
num_unique_companies =  training_df['COMPANY'].nunique()
print("Cuantas compañias hay en el dataset? \t\tRespuesta: {number}".format(number = num_unique_companies))

# Cual es el tipo de pago mas frecuente?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("Cual es el tipode pago mas Frecuente? \t\tRespuesta: {type}".format(type = most_freq_payment_type))


#Hay datos faltantes?
missing_values = training_df.isnull().sum().sum()
print("Hay datos faltantes? \t\t\t\tREspuesta:", "No" if missing_values == 0 else "Yes")


#Matrices de correlación
print(training_df.corr(numeric_only = True))

sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"]).savefig("pairplot.png")

# Entrenamiento del modelo
learning_rate = 0.001
epochs = 20
batch_size = 50


features = ['TRIP_MILES']
label = 'FARE'

model_1 = ml_functions.run_experiment(training_df, features, label, learning_rate, epochs, batch_size)