import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

iris = load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y
X = iris_df.drop('target', axis=1)
y = iris_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
max_depths = range(1, 6)
results = {}

for depth in max_depths:
    dt = DecisionTreeClassifier(
        min_samples_leaf=2,
        min_samples_split=5,
        max_depth=depth,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    precision_micro = precision_score(y_test, y_pred, average='micro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    results[depth] = {
        'precision': {
            'micro': precision_micro,
            'macro': precision_macro,
            'weighted': precision_weighted
        },
        'recall': {
            'micro': recall_micro,
            'macro': recall_macro,
            'weighted': recall_weighted
        },
        'f1': {
            'micro': f1_micro,
            'macro': f1_macro,
            'weighted': f1_weighted
        }
    }

for depth, metrics in results.items():
    print(f"Max Depth: {depth}")
    print(f"Precision (micro): {metrics['precision']['micro']:.4f}")
    print(f"Precision (macro): {metrics['precision']['macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision']['weighted']:.4f}")
    print(f"Recall (micro): {metrics['recall']['micro']:.4f}")
    print(f"Recall (macro): {metrics['recall']['macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall']['weighted']:.4f}")
    print(f"F1 (micro): {metrics['f1']['micro']:.4f}")
    print(f"F1 (macro): {metrics['f1']['macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1']['weighted']:.4f}")
    print()

recall_comparison = {depth: metrics['recall']['macro'] for depth, metrics in results.items()}
best_recall_depth = max(recall_comparison, key=recall_comparison.get)
precision_comparison = {depth: metrics['precision']['macro'] for depth, metrics in results.items()}
worst_precision_depth = min(precision_comparison, key=precision_comparison.get)
f1_comparison = {depth: metrics['f1']['macro'] for depth, metrics in results.items()}
best_f1_depth = max(f1_comparison, key=f1_comparison.get)

print(f"Best Recall at depth {best_recall_depth}: {recall_comparison[best_recall_depth]:.4f}")
print(f"Lowest Precision at depth {worst_precision_depth}: {precision_comparison[worst_precision_depth]:.4f}")
print(f"Best F1 Score at depth {best_f1_depth}: {f1_comparison[best_f1_depth]:.4f}")