# === IMPORTACIÓN DE LIBRERÍAS ===
import io
import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

# === CARGA Y SELECCIÓN DE DATOS ===
rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

# Seleccionamos columnas relevantes del dataset
rice_dataset = rice_dataset_raw[[
    'Area', 'Perimeter', 'Major_Axis_Length',
    'Minor_Axis_Length', 'Eccentricity',
    'Convex_Area', 'Extent', 'Class',
]]

print("Resumen estadístico de las variables:")
print(rice_dataset.describe())

# === VISUALIZACIÓN EXPLORATORIA DE VARIABLES ===
for x_axis_data, y_axis_data in [
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', 'Minor_Axis_Length'),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'),
]:
    px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()

# === NORMALIZACIÓN DE VARIABLES NUMÉRICAS (Z-SCORE) ===
feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns

normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std

# Se conserva la clase original
normalized_dataset['Class'] = rice_dataset['Class']

print("\nPrimeras filas del dataset normalizado:")
print(normalized_dataset.head())

# === TRANSFORMAR LA CLASE A FORMATO BINARIO ===
keras.utils.set_random_seed(42)  # Para reproducibilidad

# Cammeo = 1, Osmancik = 0
normalized_dataset['Class_Bool'] = (
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)

# === DIVISIÓN DE DATOS EN ENTRENAMIENTO, VALIDACIÓN Y TEST ===
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)

train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

print("\nPrimeras filas del conjunto de test:")
print(test_data.head())

# === SEPARACIÓN DE FEATURES Y LABELS ===
label_columns = ['Class', 'Class_Bool']

train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

# === DEFINICIÓN DE FEATURES DE ENTRADA ===
input_features = ['Eccentricity', 'Major_Axis_Length', 'Area']

# === DEFINICIÓN DEL MODELO DE CLASIFICACIÓN ===
def create_model(settings: ml_edu.experiment.ExperimentSettings, metrics: list[keras.metrics.Metric]) -> keras.Model:
    """Crea y compila un modelo de clasificación binaria."""
    model_inputs = [keras.Input(name=feature, shape=(1,)) for feature in settings.input_features]
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)

    model_output = keras.layers.Dense(
        units=1, activation='sigmoid', name='dense_layer'
    )(concatenated_inputs)

    model = keras.Model(inputs=model_inputs, outputs=model_output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model

# === FUNCIÓN PARA ENTRENAR EL MODELO ===
def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Entrena el modelo y devuelve un objeto con métricas e historial."""
    features = {feature_name: np.array(dataset[feature_name]) for feature_name in settings.input_features}
    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

print('\nFunciones de modelo y entrenamiento definidas.')

# === CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO BASE ===
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(name='accuracy', threshold=settings.classification_threshold),
    keras.metrics.Precision(name='precision', thresholds=settings.classification_threshold),
    keras.metrics.Recall(name='recall', thresholds=settings.classification_threshold),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model = create_model(settings, metrics)
experiment = train_model('baseline', model, train_features, train_labels, settings)

# === VISUALIZACIÓN DE MÉTRICAS ===
ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
ml_edu.results.plot_experiment_metrics(experiment, ['auc'])

# === EVALUACIÓN FINAL EN TEST ===
def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
    print('\nComparando resultados en entrenamiento y test:')
    for metric, test_value in test_metrics.items():
        print('------')
        print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Test {metric}:  {test_value:.4f}')

test_metrics = experiment.evaluate(test_features, test_labels)
compare_train_test(experiment, test_metrics)

# === ENTRENAMIENTO CON TODAS LAS FEATURES ===
all_input_features = [
    'Eccentricity', 'Major_Axis_Length', 'Minor_Axis_Length',
    'Area', 'Perimeter', 'Convex_Area', 'Extent',
]

settings_all_features = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

metrics_all = [
    keras.metrics.BinaryAccuracy(name='accuracy', threshold=settings_all_features.classification_threshold),
    keras.metrics.Precision(name='precision', thresholds=settings_all_features.classification_threshold),
    keras.metrics.Recall(name='recall', thresholds=settings_all_features.classification_threshold),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model_all_features = create_model(settings_all_features, metrics_all)
experiment_all_features = train_model('all features', model_all_features, train_features, train_labels, settings_all_features)

# === GRÁFICOS DE RESULTADOS PARA TODAS LAS FEATURES ===
ml_edu.results.plot_experiment_metrics(experiment_all_features, ['accuracy', 'precision', 'recall'])
ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])

# === COMPARACIÓN FINAL ENTRE AMBOS MODELOS ===
test_metrics_all_features = experiment_all_features.evaluate(test_features, test_labels)
compare_train_test(experiment_all_features, test_metrics_all_features)

ml_edu.results.compare_experiment(
    [experiment, experiment_all_features],
    ['accuracy', 'auc'],
    test_features,
    test_labels
)
