from features import compute_vif_and_correlations, load_data_and_engineer
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


RANDOM_STATE = 42
TEST_SIZE = 0.25

X, y, categorical_cols, numeric_cols, df_full = load_data_and_engineer(
    csv_path="data/merged_data.csv",
    use_pruned_features=True,
)


# Preprocess (impute + OHE)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# RandomForest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("rf", rf),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)

print("\n=== Baseline Random Forest Results ===")
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nTest ROC-AUC (probabilities):", roc_auc_score(y_test, y_prob))

# Feature importances
# Gini importances in transformed space
try:
    feature_names_transformed = (
        model.named_steps["preprocessor"].get_feature_names_out()
    )
except AttributeError:
    ohe = (
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["ohe"]
    )
    feature_names_transformed = np.array(
        numeric_cols + list(ohe.get_feature_names_out(categorical_cols))
    )

importances = model.named_steps["rf"].feature_importances_
fi = (
    pd.DataFrame(
        {
            "feature": feature_names_transformed,
            "gini_importance": importances,
        }
    )
    .sort_values("gini_importance", ascending=False)
)

print("\nTop features by Gini importance (transformed space):")
print(fi.head(20).to_string(index=False))

# Permutation importance
perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="roc_auc",
)

feature_names_perm = np.array(X_test.columns)
perm_df = (
    pd.DataFrame(
        {
            "feature": feature_names_perm,
            "perm_importance_mean": perm.importances_mean,
        }
    )
    .sort_values("perm_importance_mean", ascending=False)
)

# Save model
import joblib
joblib.dump(model, "models/random_forest.pkl")
