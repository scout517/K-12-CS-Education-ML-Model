from features import load_data_and_engineer
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

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance


RANDOM_STATE = 42
TEST_SIZE = 0.25

X, y, categorical_cols, numeric_cols, df_full = load_data_and_engineer(
    csv_path="data/merged_data.csv",
    use_pruned_features=True,
)

# Preprocess and XGBoost Model
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        )
    ]
)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=1.0,
    n_jobs=-1,
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", xgb),
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)

print("\n=== Baseline xgboost Results ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nTest ROC-AUC:", roc_auc_score(y_test, y_prob))

# Feature importances
# Gini-style (gain) importances in the transformed space:
try:
    feature_names_transformed = (
        model.named_steps["preprocessor"].get_feature_names_out()
    )
except AttributeError:
    # Fallback if sklearn is older
    ohe = (
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["ohe"]
    )
    cat_features = list(ohe.get_feature_names_out(categorical_cols))
    feature_names_transformed = np.array(numeric_cols + cat_features)

importances = model.named_steps["xgb"].feature_importances_

fi = (
    pd.DataFrame(
        {
            "feature": feature_names_transformed,
            "gain_importance": importances,
        }
    )
    .sort_values("gain_importance", ascending=False)
)

print("\nTop features by XGBoost gain importance (transformed space):")
print(fi.head(20).to_string(index=False))

# Permutation importance on original feature space
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

print("\nTop features by permutation importance (ROC-AUC drop, original features):")
print(perm_df.head(20).to_string(index=False))

# Save model
import joblib
joblib.dump(model, "models/xgboost.pkl")

