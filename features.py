import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Unified feature engineering for all models

def load_data_and_engineer(
    csv_path="data/merged_data.csv",
    encoding="utf-8-sig",
    use_pruned_features=True,
):
    """
    Load merged_data.csv, clean the target, engineer features,
    and return:
        X, y, categorical_cols, numeric_cols, df
    """


    df = pd.read_csv(csv_path, encoding=encoding)

    # Clean target column
    y = df["Offers CS?"]
    if y.dtype == "O":
        y = (
            y.astype(str)
             .str.strip()
             .str.lower()
             .map({
                 "yes": 1, "y": 1, "true": 1, "1": 1,
                 "no":  0, "n": 0, "false": 0, "0": 0
             })
        )
    y = y.astype(int)

 
    # Column aliases
    STATE = "State"

    STUDENTS_9_12 = "Grades 9-12 Students [District] 2023-24"
    MALE          = "Male Students [District] 2023-24"
    FEMALE        = "Female Students [District] 2023-24"
    AIAN          = "American Indian/Alaska Native Students [District] 2023-24"
    ASIAN         = "Asian or Asian/Pacific Islander Students [District] 2023-24"
    HISP          = "Hispanic Students [District] 2023-24"
    BLACK         = "Black or African American Students [District] 2023-24"
    WHITE         = "White Students [District] 2023-24"
    NH_PI         = "Nat. Hawaiian or Other Pacific Isl. Students [District] 2023-24"
    TWO_PLUS      = "Two or More Races Students [District] 2023-24"

    PTRATIO       = "Pupil/Teacher Ratio [District] 2023-24"
    FTE_TEACH     = "Full-Time Equivalent (FTE) Teachers [District] 2023-24"
    TOTAL_STAFF   = "Total Staff [District] 2023-24"
    LIBRARIANS    = "Librarians/media specialists [District] 2023-24"
    MEDIA_SUPPORT = "Media Support Staff [District] 2023-24"

    REV_LOCAL = "Total Revenue - Local Sources (TLOCREV) [District Finance] 2020-21"
    REV_STATE = "Total Revenue - State Sources (TSTREV) [District Finance] 2020-21"
    REV_FED   = "Total Revenue - Federal Sources (TFEDREV) [District Finance] 2020-21"
    REV_PER_PUPIL = (
        "Total Revenue (TOTALREV) per Pupil (V33) [District Finance] 2020-21"
    )

    DIPLOMAS  = "Diploma Recipients & Other Completers [District] 2022-23"
    POP_TOTAL = "Estimated Total Population"
    POP_5_17  = "Estimated Population 5-17"
    CHILDREN_POV = (
        "Estimated number of relevant children 5 to 17 years old in poverty who are related to the householder"
    )


    #Feature engineering helpers

    def safe_div(num, den):
        """Elementwise safe division for pandas Series, returns NaN where den is 0 or NaN."""
        return np.where(den.isna() | (den == 0), np.nan, num / den)

    # Total students as denominator for percentages
    race_cols = [AIAN, ASIAN, HISP, BLACK, WHITE, NH_PI, TWO_PLUS]
    race_sum = df[race_cols].sum(axis=1, skipna=True)
    mf_sum   = df[[MALE, FEMALE]].sum(axis=1, skipna=True)

    total_students = df[STUDENTS_9_12].copy()
    total_students = total_students.fillna(mf_sum)
    total_students = total_students.fillna(race_sum)

    # Revenue per student
    df["rev_total"] = (
        df[REV_LOCAL].fillna(0) +
        df[REV_STATE].fillna(0) +
        df[REV_FED].fillna(0)
    )
    df["rev_per_student_calc"] = safe_div(df["rev_total"], total_students)

    # Staff per 100 students
    df["staff_per_100"]       = 100 * safe_div(df[TOTAL_STAFF],   total_students)
    df["fte_per_100"]         = 100 * safe_div(df[FTE_TEACH],     total_students)
    df["librarians_per_100"]  = 100 * safe_div(df[LIBRARIANS],    total_students)
    df["media_support_per_100"] = 100 * safe_div(df[MEDIA_SUPPORT], total_students)

    # Race percentages
    df["pct_aian"]   = safe_div(df[AIAN],   total_students)
    df["pct_asian"]  = safe_div(df[ASIAN],  total_students)
    df["pct_hisp"]   = safe_div(df[HISP],   total_students)
    df["pct_black"]  = safe_div(df[BLACK],  total_students)
    df["pct_white"]  = safe_div(df[WHITE],  total_students)
    df["pct_nhpi"]   = safe_div(df[NH_PI],  total_students)
    df["pct_two"]    = safe_div(df[TWO_PLUS], total_students)

    # Poverty & diplomas
    df["poverty_rate_5_17"] = safe_div(df[CHILDREN_POV], df[POP_5_17])
    df["diplomas_per_100"]  = 100 * safe_div(df[DIPLOMAS], total_students)


    # Define engineered feature sets

    # "Full" engineered set (before pruning)
    full_engineered_cols = [
        "rev_per_student_calc",
        "staff_per_100",
        "fte_per_100",
        "librarians_per_100",
        "media_support_per_100",
        "pct_aian",
        "pct_asian",
        "pct_hisp",
        "pct_black",
        "pct_white",
        "pct_nhpi",
        "pct_two",
        "poverty_rate_5_17",
        "diplomas_per_100",
        PTRATIO,
        REV_PER_PUPIL,
        POP_TOTAL,
    ]

    # Final pruned numeric feature set we settled on
    pruned_numeric_cols = [
        "rev_per_student_calc",
        "pct_hisp",
        "pct_black",
        "pct_white",
        "poverty_rate_5_17",
        "diplomas_per_100",
        PTRATIO,
    ]


    # Choose feature set

    if use_pruned_features:
        numeric_cols = [c for c in pruned_numeric_cols if c in df.columns]
    else:
        # Keep all engineered cols that exist
        numeric_cols = [c for c in full_engineered_cols if c in df.columns]

    categorical_cols = [STATE] if STATE in df.columns else []

    feature_cols = categorical_cols + numeric_cols
    X = df[feature_cols].copy()

    return X, y, categorical_cols, numeric_cols, df



# VIF + correlation diagnostics
def compute_vif_and_correlations(df, numeric_cols, y, corr_method="pearson"):
    """
    Compute:
      - VIF for numeric predictors
      - Correlation matrix among numeric predictors
      - Correlation of each numeric predictor with the target y
    """
    #  Correlation matrix among predictors (uses all rows with any data)
    corr_matrix = df[numeric_cols].corr(method=corr_method)

    # Prepare data for VIF: drop rows with any NA or inf in numeric_cols
    X_vif = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    mask_complete = X_vif.notna().all(axis=1)
    X_vif = X_vif.loc[mask_complete]

    # Align y to the rows used for VIF / target correlations
    y_aligned = y.loc[X_vif.index]

    # Compute VIF
    vif_list = []
    if X_vif.shape[1] >= 2:
        for i, col in enumerate(X_vif.columns):
            vif_value = variance_inflation_factor(X_vif.values, i)
            vif_list.append({"feature": col, "VIF": vif_value})
        vif_df = pd.DataFrame(vif_list).sort_values("VIF", ascending=False).reset_index(drop=True)
    else:
        vif_df = pd.DataFrame(columns=["feature", "VIF"])

    # Correlation with target y (point-biserial == Pearson with 0/1 y)
    target_corr = pd.Series(
        {col: X_vif[col].corr(y_aligned, method=corr_method) for col in X_vif.columns},
        name=f"corr_with_{getattr(y, 'name', 'target')}"
    )

    return vif_df, corr_matrix, target_corr