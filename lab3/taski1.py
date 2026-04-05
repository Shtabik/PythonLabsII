import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error,classification_report, roc_curve, auc, accuracy_score)

df = pd.read_csv('Car_Price.csv')
df = df.drop(['Make', 'Model'], axis=1)
df = pd.get_dummies(df, columns=['Fuel Type', 'Transmission'], drop_first=True)
m_median = df['Mileage'].median()
y_median = df['Year'].median()
df['Condition'] = ((df['Mileage'] <= m_median) & (df['Year'] >= y_median)).astype(int)

X_reg = df.drop(['Price', 'Condition'], axis=1)
y_reg = df['Price']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = dt_regressor.predict(X_test_reg)
print(f"Средняя ошибка (MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"Квадратичная ошибка (RMSE): {root_mean_squared_error(y_test_reg, y_pred_reg):.2f}")

print("\n--- Задача классификации: Состояние (Condition) ---")

X_cl = df.drop(['Condition'], axis=1)
y_cl = df['Condition']

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.3, random_state=42, stratify=y_cl)
dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_classifier.fit(X_train_cl, y_train_cl)
y_pred_cl = dt_classifier.predict(X_test_cl)
y_proba = dt_classifier.predict_proba(X_test_cl)[:, 1]
print(f"Точность (Accuracy): {accuracy_score(y_test_cl, y_pred_cl):.4f}")
print(classification_report(y_test_cl, y_pred_cl))


fpr, tpr, _ = roc_curve(y_test_cl, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('Ложные срабатывания (FPR)')
plt.ylabel('Верные срабатывания (TPR)')
plt.title('ROC-кривая')
plt.legend()
plt.show()