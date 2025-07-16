
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

cancer_data = load_breast_cancer()
print(cancer_data['data'].shape)
print(cancer_data['feature_names'])

df= pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

print(df.head())

x=df[cancer_data['feature_names']].values
y=df['target'].values

model =LogisticRegression(solver='liblinear')
model.fit(x, y)

print("Prediccion para el primer ejemplo:", model.predict([x[0]]))
print(model.score(x, y))