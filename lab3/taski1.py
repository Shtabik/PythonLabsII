import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error,classification_report, roc_curve, auc, accuracy_score,r2_score)

df = pd.read_csv('Car_Price.csv')
df = pd.get_dummies(df, columns=['Fuel Type', 'Transmission'], drop_first=True)
X_reg = df.drop(['Price','Make','Model'], axis=1)
y_reg = df['Price']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
dt_regressor = DecisionTreeRegressor(max_depth=12,     min_samples_leaf=14, random_state=42)
dt_regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = dt_regressor.predict(X_test_reg)
print(f"R^2:{r2_score(y_test_reg, y_pred_reg)}")
print(f"Средняя ошибка (MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"Квадратичная ошибка (RMSE): {root_mean_squared_error(y_test_reg, y_pred_reg):.2f}")


df = pd.read_csv('mushrooms.csv')
X = df.drop(['class'], axis=1)
y = (df['class'] == 'p').astype(int)
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

dt_classifier = DecisionTreeClassifier(max_depth=4,ccp_alpha=0.01, random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
y_proba = dt_classifier.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='green', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Ложные срабатывания (FPR)')
plt.ylabel('Верные срабатывания (TPR)')
plt.title('ROC-кривая (Decision Tree)')
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.show()
