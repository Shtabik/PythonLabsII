import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error,r2_score, mean_absolute_error,classification_report, confusion_matrix


df = pd.read_csv('Car_Price.csv')
df = pd.get_dummies(df, columns=['Make', 'Fuel Type', 'Model', 'Transmission'], drop_first=True)
X = df.drop(['Price'], axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)

RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)
r2=r2_score(y_test, y_pred_test)
print(f"R^2:{r2}")
print(f"RMSE: {RMSE}")
print(f"MAE: {MAE}")
print(f"Средняя цена машин:{df['Price'].mean()}")

#задача классификации
df = pd.read_csv('mushrooms.csv')
X = df.drop(['class', 'odor', 'gill-color', 'stalk-root'], axis=1)
y = df['class']
y = (df['class'] == 'p').astype(int)

X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))
print("Количество обнуленных признаков:", np.sum(model.coef_ == 0))