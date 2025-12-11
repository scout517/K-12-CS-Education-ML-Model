import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Import the unified feature engineering function
from features import compute_vif_and_correlations, load_data_and_engineer  # make sure features.py is in the same folder or on PYTHONPATH

RANDOM_STATE = 42
TEST_SIZE = 0.25

X, y, categorical_cols, numeric_cols, df_full = load_data_and_engineer(
    csv_path="data/merged_data.csv",
    encoding="utf-8-sig",
    use_pruned_features=True,
)

print("Feature columns used:")
print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols)


# Multicollinearity + correlation diagnostics (VIF & corr)

# vif_df, corr_matrix, target_corr = compute_vif_and_correlations(
#     df=df_full,
#     numeric_cols=numeric_cols,
#     y=y,
#     corr_method="pearson",
# )

# print("\n=== VIF (using full dataset) ===")
# print(vif_df.to_string(index=False))

# print("\n=== Correlation matrix among numeric predictors ===")
# print(corr_matrix)

# print("\n=== Correlation of predictors with Offers CS? ===")
# print(target_corr.sort_values(ascending=False))

# Save to CSVs
# vif_df.to_csv("outputs/vif_logit_pruned.csv", index=False)
# corr_matrix.to_csv("outputs/corr_matrix_logit_pruned.csv")
# target_corr.to_csv("outputs/target_corr_logit_pruned.csv")

# Preprocessing: impute + scale + one-hot
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
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

# Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_STATE
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", log_reg),
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)

print("\n=== Baseline Logistic Regression Results ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nTest ROC-AUC (probabilities):", roc_auc_score(y_test, y_prob))

# Inspect coefficients
# Get feature names after preprocessing
ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["ohe"]
ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))
final_feature_names = numeric_cols + ohe_feature_names

coefs = model.named_steps["clf"].coef_[0]
coef_df = (
    pd.DataFrame({"feature": final_feature_names, "coef": coefs})
    .sort_values("coef", ascending=False)
)

print("\nTop positive coefficients (predicting Offers CS = 1):")
print(coef_df.head(15).to_string(index=False))

print("\nTop negative coefficients (predicting Offers CS = 0):")
print(coef_df.tail(15).to_string(index=False))

# Save model
import joblib
joblib.dump(model, "models/logistic_regression.pkl")
