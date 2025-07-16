import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

# Accuracy is 80% which means that 80% of the model’s predictions are correct
print("accuracy:", accuracy_score(y, y_pred))

# The precision is 78%, which we recall is the percent of the model’s positive predictions that are correct
print("precision:", precision_score(y, y_pred))

#The recall is 68%, which is the percent of the positive cases that the model predicted correctly.
print("recall:", recall_score(y, y_pred))

#The F1 score is 73%, which is an average of the precision and recall.
print("f1 score:", f1_score(y, y_pred))

#Scikit-learn reverses the confusion matrix to show the negative counts first!
print(confusion_matrix(y, y_pred))
