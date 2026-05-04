import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report

df = pd.read_csv('mushrooms.csv')
X = df.drop(['class'], axis=1)
X = pd.get_dummies(X)
y = (df['class'] == 'p').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf = RandomForestClassifier(
    n_estimators=70,
    max_depth=2,
    min_samples_leaf=40,
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)
ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(
    n_estimators=20,
    max_depth=2,
    learning_rate=0.01,
    random_state=42
)
gb.fit(X_train, y_train)

print("\n--- Отчет по Random Forest (с регуляризацией) ---")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("\n--- Отчет по AdaBoost (с регуляризацией) ---")
y_pred_ada = ada.predict(X_test)
print(classification_report(y_test, y_pred_ada))
print("\n--- Отчет по Gradient Boosting (с регуляризацией) ---")
y_pred_gb = gb.predict(X_test)
print(classification_report(y_test, y_pred_gb))

plt.figure(figsize=(10, 7))

y_proba_rf = rf.predict_proba(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf[:, 1])
plt.plot(fpr_rf, tpr_rf,  markersize=2, label=f'RF (AUC = {auc(fpr_rf, tpr_rf):.2f})')

# AdaBoost
y_proba_ada = ada.predict_proba(X_test)
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_proba_ada[:, 1])
plt.plot(fpr_ada, tpr_ada,  markersize=2, label=f'Ada (AUC = {auc(fpr_ada, tpr_ada):.2f})')

# Gradient Boosting
y_proba_gb = gb.predict_proba(X_test)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb[:, 1])
plt.plot(fpr_gb, tpr_gb,  markersize=2, label=f'GB (AUC = {auc(fpr_gb, tpr_gb):.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Ложные срабатывания (FPR)')
plt.ylabel('Верные срабатывания (TPR)')
plt.title('ROC-кривые ')
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.show()
print(f"Random Forest OOB Score: {rf.oob_score_:.4f}")