import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,classification_report, confusion_matrix


df = pd.read_csv('Car_Price.csv')
X = df[['Mileage']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)

RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)
print(f"RMSE: {RMSE}")
print(f"MAE: {MAE}")
print(f"Средняя цена машин:{df['Price'].mean()}")

#задача классификации
y_cl = (df['Transmission'] == 'Automatic').astype(int)
X_cl = df[['Price', 'Mileage', 'Engine Size']]
X_train_cl, X_test_c, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.4, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train_cl, y_train_cl)
y_pred_cl = log_model.predict(X_test_c)
print(classification_report(y_test_cl, y_pred_cl, target_names=['Manual', 'Automatic']))
