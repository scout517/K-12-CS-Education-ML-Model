"""
visualizations.py

Generate visualizations for:
- Logistic Regression (baseline)
- Random Forest (baseline)
- XGBoost (baseline)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

from features import load_data_and_engineer  # your unified FE function

RANDOM_STATE = 42
TEST_SIZE = 0.25

sns.set(style="whitegrid")


def build_preprocessor(categorical_cols, numeric_cols):
    # Create the ColumnTransformer for numeric + categorical preprocessing.
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
    return preprocessor


def train_models():
    # Load data, do a simple 75/25 train–test split, build pipelines, train models.
    # Load features and target
    X, y, categorical_cols, numeric_cols, df_full = load_data_and_engineer(
        csv_path="data/merged_data.csv",
        encoding="utf-8-sig",
        use_pruned_features=True,
    )

    # Attach target to df_full for plotting distributions later
    df_full = df_full.copy()
    df_full["offers_cs_binary"] = y.values

    # Train/test split (75 / 25)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    # Define models
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
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

    # Pipelines
    lr_pipe = Pipeline([("preprocessor", preprocessor), ("clf", log_reg)])
    rf_pipe = Pipeline([("preprocessor", preprocessor), ("rf", rf)])
    xgb_pipe = Pipeline([("preprocessor", preprocessor), ("xgb", xgb)])

    # Fit on training set
    print("Fitting Logistic Regression...")
    lr_pipe.fit(X_train, y_train)

    print("Fitting Random Forest...")
    rf_pipe.fit(X_train, y_train)

    print("Fitting XGBoost...")
    xgb_pipe.fit(X_train, y_train)

    models = {
        "Logistic Regression": lr_pipe,
        "Random Forest": rf_pipe,
        "XGBoost": xgb_pipe,
    }

    splits = {
        "X_train": X_train,
        "X_val": X_test,   
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_test,  
        "y_test": y_test,
    }

    meta = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "df_full": df_full,
    }

    return models, splits, meta


# Plot functions
def plot_validation_accuracy_bar(models, splits, filename="images/fig_val_accuracy.png"):
    # Bar chart of accuracy for each model on the held-out 25% split.
    X_val = splits["X_val"]
    y_val = splits["y_val"]

    accuracies = {}
    for name, model in models.items():
        y_pred_val = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        accuracies[name] = acc
        print(f"Accuracy on 25% hold-out ({name}): {acc:.3f}")

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
    )
    plt.ylim(0, 1)
    plt.ylabel("Accuracy on hold-out set")
    plt.title("Accuracy by Model (25% hold-out)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_roc_curves(models, splits, filename="images/fig_roc_curves.png"):
    # ROC curves for all models on the test/hold-out set.
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    plt.figure(figsize=(6, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_confusion_matrices(models, splits, base_filename="images/fig_confusion_matrix"):
    # Normalized confusion matrices for each model on the test/hold-out set.
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize="true")

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["No CS", "Offers CS"],
            yticklabels=["No CS", "Offers CS"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Normalized Confusion Matrix\n{name}")
        plt.tight_layout()
        fname = f"{base_filename}_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, dpi=300)
        plt.close()


def get_feature_names_from_preprocessor(model, numeric_cols, categorical_cols):
    # Recover feature names after preprocessing (numeric and one-hot).
    preprocessor = model.named_steps["preprocessor"]
    num_features = numeric_cols.copy()
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_features = list(ohe.get_feature_names_out(categorical_cols))
    return num_features + cat_features


def get_transformed_data_and_feature_names(model, X, numeric_cols, categorical_cols):
    # Apply the model's preprocessor to X and return dense array and feature names.
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)

    # Densify if sparse
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = get_feature_names_from_preprocessor(
        model, numeric_cols, categorical_cols
    )
    return X_transformed, feature_names


def plot_permutation_importance(
    models,
    splits,
    model_key=None,
    meta=None,
    *,
    scoring="roc_auc",
    n_repeats=15,
    top_n=15,
    filename=None,
):
    
    # Plot permutation importance.
    
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    # Get feature names
    if hasattr(X_test, "columns"):
        feature_names = list(X_test.columns)
    elif meta is not None and "feature_names" in meta:
        feature_names = meta["feature_names"]
    else:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    # ALL MODELS IN ONE FIGURE
    if model_key is None or model_key == "all":
        model_items = list(models.items())
        n_models = len(model_items)

        if filename is None:
            filename = "images/fig_perm_importance_all_models.png"

        fig, axes = plt.subplots(
            1, n_models, figsize=(6 * n_models, 5), squeeze=False
        )
        axes = axes[0]  # 1 row

        for ax, (name, model) in zip(axes, model_items):
            print(f"Computing permutation importance for {name} (this may take a bit)...")
            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scoring=scoring,
            )

            perm_df = (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": result.importances_mean,
                })
                .sort_values("importance", ascending=False)
                .head(top_n)
            )

            sns.barplot(
                data=perm_df,
                y="feature",
                x="importance",
                orient="h",
                ax=ax,
            )

            # Clean up y labels (Pupil/Teacher Ratio text)
            labels = [
                label.get_text().replace(
                    "Pupil/Teacher Ratio [District] 2023-24",
                    "Pupil/Teacher Ratio",
                )
                for label in ax.get_yticklabels()
            ]
            ax.set_yticklabels(labels)

            ax.set_xlabel(f"Mean Decrease in {scoring.upper()}", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)
            ax.set_title(f"{name}", fontsize=14)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x:.2f}")
            )

        fig.suptitle("Permutation Importance Across Models", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        return

    # SINGLE MODEL
    model = models[model_key]

    if filename is None:
        safe_name = model_key.replace(" ", "_").lower()
        filename = f"images/fig_perm_importance_{safe_name}.png"

    print(f"Computing permutation importance for {model_key} (this may take a bit)...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring=scoring,
    )

    perm_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=perm_df,
        y="feature",
        x="importance",
        orient="h",
    )
    labels = [
        label.get_text().replace(
            "Pupil/Teacher Ratio [District] 2023-24",
            "Pupil/Teacher Ratio",
        )
        for label in ax.get_yticklabels()
    ]
    ax.set_yticklabels(labels)
    plt.xlabel(f"Mean Decrease in {scoring.upper()}", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.title(f"Permutation Importance ({model_key})", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_model_importances(models, meta, base_filename="images/fig_model_importance"):
    # Model-based feature importances for RF and XGBoost.
    categorical_cols = meta["categorical_cols"]
    numeric_cols = meta["numeric_cols"]

    for name in ["Random Forest", "XGBoost"]:
        model = models[name]
        if name == "Random Forest":
            est = model.named_steps["rf"]
        else:
            est = model.named_steps["xgb"]

        feature_names = get_feature_names_from_preprocessor(
            model, numeric_cols, categorical_cols
        )
        importances = est.feature_importances_

        imp_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(7, 5))
        sns.barplot(
            data=imp_df,
            y="feature",
            x="importance",
            orient="h",
        )
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top Features by {name} Importance")
        plt.tight_layout()
        fname = f"{base_filename}_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, dpi=300)
        plt.close()


def plot_logistic_coefficients(models, meta, filename_pos="images/fig_lr_coefs_pos.png", filename_neg="images/fig_lr_coefs_neg.png"):
    # Plot top positive and negative coefficients from logistic regression.
    lr_model = models["Logistic Regression"]
    categorical_cols = meta["categorical_cols"]
    numeric_cols = meta["numeric_cols"]

    feature_names = get_feature_names_from_preprocessor(
        lr_model, numeric_cols, categorical_cols
    )
    coefs = lr_model.named_steps["clf"].coef_[0]

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df_pos = coef_df.sort_values("coef", ascending=False).head(15)
    coef_df_neg = coef_df.sort_values("coef", ascending=True).head(15)

    # Positive
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=coef_df_pos,
        y="feature",
        x="coef",
        orient="h",
    )
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Top Positive Coefficients (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(filename_pos, dpi=300)
    plt.close()

    # Negative
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=coef_df_neg,
        y="feature",
        x="coef",
        orient="h",
    )
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Top Negative Coefficients (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(filename_neg, dpi=300)
    plt.close()

def plot_logistic_coefficients_combined(
    models,
    meta,
    filename="images/fig_lr_coefs_combined.png",
    top_n=5
):
    # Plot combined top positive and negative coefficients from logistic regression.

    lr_model = models["Logistic Regression"]
    categorical_cols = meta["categorical_cols"]
    numeric_cols = meta["numeric_cols"]

    # Get feature names after preprocessing
    feature_names = get_feature_names_from_preprocessor(
        lr_model, numeric_cols, categorical_cols
    )
    coefs = lr_model.named_steps["clf"].coef_[0]

    # Build dataframe
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})

    # Top positive and negative
    top_pos = coef_df.sort_values("coef", ascending=False).head(top_n)
    top_neg = coef_df.sort_values("coef", ascending=True).head(top_n)

    # Combine and sort for plotting
    combined = pd.concat([top_neg, top_pos], axis=0).sort_values("coef")

    # Plot combined figure
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=combined,
        y="feature",
        x="coef",
        orient="h"
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Coefficient", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.title(f"Top {top_n} Positive and Negative Coefficients (Logistic Regression)", fontsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_feature_distributions(meta, filename_prefix="images/fig_dist_"):
    # Distribution plots for key features by CS vs No-CS.
    df = meta["df_full"]

    df = df.copy()
    df["OffersCS"] = np.where(df["offers_cs_binary"] == 1, "Offers CS", "No CS")

    features_to_plot = [
        "poverty_rate_5_17",
        "diplomas_per_100",
        "rev_per_student_calc",
        "Pupil/Teacher Ratio [District] 2023-24",
        "pct_black",
        "pct_white",
    ]

    for feat in features_to_plot:
        if feat not in df.columns:
            continue

        plt.figure(figsize=(6, 4))
        sns.boxplot(
            data=df,
            x="OffersCS",
            y=feat,
        )
        plt.title(f"{feat} by CS Offering")
        plt.tight_layout()
        fname = f"{filename_prefix}{feat.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(fname, dpi=300)
        plt.close()


def compute_and_plot_shap_for_logistic_regression(
    models,
    splits,
    meta,
    max_samples=800,
    base_filename_prefix="images/fig_shap_",
):
    # Compute SHAP values for Logistic Regression only and generate summary + bar plots.

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    categorical_cols = meta["categorical_cols"]
    numeric_cols = meta["numeric_cols"]

    # Sample test set for visualization
    if len(X_test) > max_samples:
        X_test_sample, y_test_sample = stratified_sample(
            X_test, y_test, n=max_samples, random_state=RANDOM_STATE
        )
    else:
        X_test_sample, y_test_sample = X_test, y_test

    # Background from train
    if len(X_train) > max_samples:
        X_train_bg, _ = stratified_sample(
            X_train, y_train, n=max_samples, random_state=RANDOM_STATE
        )
    else:
        X_train_bg = X_train

    model_name = "Logistic Regression"
    pipe = models[model_name]

    X_bg_trans, feature_names = get_transformed_data_and_feature_names(
        pipe, X_train_bg, numeric_cols, categorical_cols
    )
    X_sample_trans, _ = get_transformed_data_and_feature_names(
        pipe, X_test_sample, numeric_cols, categorical_cols
    )

    print(f"\nComputing SHAP values for {model_name}")
    est = pipe.named_steps["clf"]
    explainer = shap.LinearExplainer(est, X_bg_trans)

    shap_values = explainer(X_sample_trans)

    # Summary
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values.values,
        X_sample_trans,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("SHAP Summary Plot - Logistic Regression", fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{base_filename_prefix}logistic_regression_summary.png", dpi=300)
    plt.close()

    # Bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values.values,
        X_sample_trans,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title("SHAP Global Importance - Logistic Regression", fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{base_filename_prefix}logistic_regression_bar.png", dpi=300)
    plt.close()


def stratified_sample(X, y, n, random_state=42):
    # Returns a stratified sample of size n from X, maintaining the class distribution of y.
    df = X.copy()
    df["_y"] = y.values

    samples = []
    for cls, cls_df in df.groupby("_y"):
        cls_n = max(1, int(n * len(cls_df) / len(df)))
        samples.append(cls_df.sample(cls_n, random_state=random_state))

    out = pd.concat(samples).sample(frac=1, random_state=random_state)
    y_out = out["_y"]
    X_out = out.drop(columns=["_y"])
    return X_out, y_out


# Main entry point
if __name__ == "__main__":
    models, splits, meta = train_models()

    # Accuracy bar chart
    plot_validation_accuracy_bar(models, splits, filename="images/fig_val_accuracy.png")

    # ROC curves 
    plot_roc_curves(models, splits, filename="images/fig_roc_curves.png")

    # Confusion matrices
    plot_confusion_matrices(models, splits, base_filename="images/fig_confusion_matrix")

    # Permutation importance
    plot_permutation_importance(models, splits, model_key="XGBoost")
    plot_permutation_importance(models, splits, model_key="Random Forest")
    plot_permutation_importance(models, splits, model_key="Logistic Regression")

    plot_permutation_importance(models, splits, model_key="all")

    # Model-based feature importances (RF & XGB)
    plot_model_importances(models, meta, base_filename="images/fig_model_importance")

    # Logistic regression coefficients
    plot_logistic_coefficients(models, meta)
    plot_logistic_coefficients_combined(models, meta)

    # Feature distributions by class
    plot_feature_distributions(meta)

    # SHAP for Logistic Regression only
    compute_and_plot_shap_for_logistic_regression(
        models,
        splits,
        meta,
        max_samples=800,
        base_filename_prefix="images/fig_shap_",
    )
