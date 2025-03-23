import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv('wdbc.data', header=None, names=column_names)

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop('id', axis=1)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_params = {
    'criterion': 'gini',
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'max_depth': 2,
    'random_state': 42
}

dt_original = DecisionTreeClassifier(**dt_params)
dt_original.fit(X_train_scaled, y_train)
y_pred_original = dt_original.predict(X_test_scaled)
f1_original = f1_score(y_test, y_pred_original)
precision_original = precision_score(y_test, y_pred_original)
recall_original = recall_score(y_test, y_pred_original)
cm_original = confusion_matrix(y_test, y_pred_original)
pca = PCA().fit(X_train_scaled)

pca_1 = PCA(n_components=1)
X_train_pca_1 = pca_1.fit_transform(X_train_scaled)
X_test_pca_1 = pca_1.transform(X_test_scaled)

dt_pca_1 = DecisionTreeClassifier(**dt_params)
dt_pca_1.fit(X_train_pca_1, y_train)
y_pred_pca_1 = dt_pca_1.predict(X_test_pca_1)

f1_pca_1 = f1_score(y_test, y_pred_pca_1)
precision_pca_1 = precision_score(y_test, y_pred_pca_1)
recall_pca_1 = recall_score(y_test, y_pred_pca_1)
cm_pca_1 = confusion_matrix(y_test, y_pred_pca_1)

pca_2 = PCA(n_components=2)
X_train_pca_2 = pca_2.fit_transform(X_train_scaled)
X_test_pca_2 = pca_2.transform(X_test_scaled)

dt_pca_2 = DecisionTreeClassifier(**dt_params)
dt_pca_2.fit(X_train_pca_2, y_train)
y_pred_pca_2 = dt_pca_2.predict(X_test_pca_2)

f1_pca_2 = f1_score(y_test, y_pred_pca_2)
precision_pca_2 = precision_score(y_test, y_pred_pca_2)
recall_pca_2 = recall_score(y_test, y_pred_pca_2)
cm_pca_2 = confusion_matrix(y_test, y_pred_pca_2)

def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tn, fp, fn, tp, tpr, fpr

tn_orig, fp_orig, fn_orig, tp_orig, tpr_orig, fpr_orig = calculate_metrics(cm_original)
tn_pca1, fp_pca1, fn_pca1, tp_pca1, tpr_pca1, fpr_pca1 = calculate_metrics(cm_pca_1)
tn_pca2, fp_pca2, fn_pca2, tp_pca2, tpr_pca2, fpr_pca2 = calculate_metrics(cm_pca_2)

print("Original Data (Continuous)：")
print(f"F1 Score: {f1_original:.4f}")
print(f"Precision: {precision_original:.4f}")
print(f"Recall: {recall_original:.4f}")
print("Confusion Matrix:")
print(cm_original)
print(f"True Positives (TP): {tp_orig}")
print(f"False Positives (FP): {fp_orig}")
print(f"True Positive Rate (TPR): {tpr_orig:.4f}")
print(f"False Positive Rate (FPR): {fpr_orig:.4f}")
print()

print("First Principal Component Only:")
print(f"F1 Score: {f1_pca_1:.4f}")
print(f"Precision: {precision_pca_1:.4f}")
print(f"Recall: {recall_pca_1:.4f}")
print("Confusion Matrix:")
print(cm_pca_1)
print(f"True Positives (TP): {tp_pca1}")
print(f"False Positives (FP): {fp_pca1}")
print(f"True Positive Rate (TPR): {tpr_pca1:.4f}")
print(f"False Positive Rate (FPR): {fpr_pca1:.4f}")
print()

print("First and Second Principal Components:")
print(f"F1 Score: {f1_pca_2:.4f}")
print(f"Precision: {precision_pca_2:.4f}")
print(f"Recall: {recall_pca_2:.4f}")
print("Confusion Matrix:")
print(cm_pca_2)
print(f"True Positives (TP): {tp_pca2}")
print(f"False Positives (FP): {fp_pca2}")
print(f"True Positive Rate (TPR): {tpr_pca2:.4f}")
print(f"False Positive Rate (FPR): {fpr_pca2:.4f}")
print()

print("Explained Variance Ratio:")
print(f"First Principal Component: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Second Principal Component: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Cumulative Variance (2 components): {pca.explained_variance_ratio_[:2].sum():.4f}")