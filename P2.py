import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import math

column_names = ['id', 'clump_thickness', 'uniformity_of_cell_size', 
               'uniformity_of_cell_shape', 'marginal_adhesion', 
               'single_epithelial_cell_size', 'bare_nuclei', 
               'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
data = pd.read_csv('breast-cancer-wisconsin.data', names=column_names, na_values='?')
data = data.dropna()
data = data.drop('id', axis=1)
data['class'] = data['class'].map({2: 0, 4: 1})
X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=2,
    min_samples_split=5,
    max_depth=2,
    random_state=42
)
dt.fit(X_train, y_train)
feature_index = dt.tree_.feature[0]
threshold = dt.tree_.threshold[0]
feature_name = X.columns[feature_index]
print(f"Feature selected for first split: {feature_name}")
print(f"Decision boundary value: {threshold}")

left_indices = X_train.iloc[:, feature_index] <= threshold
right_indices = X_train.iloc[:, feature_index] > threshold
left_y = y_train[left_indices]
right_y = y_train[right_indices]
left_count = len(left_y)
right_count = len(right_y)
total_count = len(y_train)
left_class_0 = sum(left_y == 0)
left_class_1 = sum(left_y == 1)
right_class_0 = sum(right_y == 0)
right_class_1 = sum(right_y == 1)

p_left = left_count / total_count
p_right = right_count / total_count
p_left_class_0 = left_class_0 / left_count if left_count > 0 else 0
p_left_class_1 = left_class_1 / left_count if left_count > 0 else 0
p_right_class_0 = right_class_0 / right_count if right_count > 0 else 0
p_right_class_1 = right_class_1 / right_count if right_count > 0 else 0

class_0_count = sum(y_train == 0)
class_1_count = sum(y_train == 1)
p_class_0 = class_0_count / total_count
p_class_1 = class_1_count / total_count
entropy_parent = -p_class_0 * (math.log2(p_class_0) if p_class_0 > 0 else 0) - p_class_1 * (math.log2(p_class_1) if p_class_1 > 0 else 0)

entropy_left = -p_left_class_0 * (math.log2(p_left_class_0) if p_left_class_0 > 0 else 0) - p_left_class_1 * (math.log2(p_left_class_1) if p_left_class_1 > 0 else 0)
entropy_right = -p_right_class_0 * (math.log2(p_right_class_0) if p_right_class_0 > 0 else 0) - p_right_class_1 * (math.log2(p_right_class_1) if p_right_class_1 > 0 else 0)
entropy_children = p_left * entropy_left + p_right * entropy_right
information_gain = entropy_parent - entropy_children

gini_parent = 1 - (p_class_0**2 + p_class_1**2)
gini_left = 1 - (p_left_class_0**2 + p_left_class_1**2)
gini_right = 1 - (p_right_class_0**2 + p_right_class_1**2)
gini_children = p_left * gini_left + p_right * gini_right
gini_gain = gini_parent - gini_children

misclass_parent = min(p_class_0, p_class_1)
misclass_left = min(p_left_class_0, p_left_class_1)
misclass_right = min(p_right_class_0, p_right_class_1)
misclass_children = p_left * misclass_left + p_right * misclass_right
misclass_gain = misclass_parent - misclass_children

print(f"\nEntropy of parent node: {entropy_parent:.4f}")
print(f"Entropy after split: {entropy_children:.4f}")
print(f"Information Gain: {information_gain:.4f}")

print(f"\nGini of parent node: {gini_parent:.4f}")
print(f"Gini after split: {gini_children:.4f}")
print(f"Gini Gain: {gini_gain:.4f}")

print(f"\nMisclassification Error of parent node: {misclass_parent:.4f}")
print(f"Misclassification Error after split: {misclass_children:.4f}")
print(f"Misclassification Error Gain: {misclass_gain:.4f}")