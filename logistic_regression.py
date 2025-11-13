import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = pd.read_csv('data/merged_data.csv', encoding='utf-8-sig')

y = df['Offers CS?']

STATE = 'State'
STUDENTS_9_12 = 'Grades 9-12 Students [District] 2023-24'
MALE = 'Male Students [District] 2023-24'
FEMALE = 'Female Students [District] 2023-24'
AIAN = 'American Indian/Alaska Native Students [District] 2023-24'
ASIAN = 'Asian or Asian/Pacific Islander Students [District] 2023-24'
HISP = 'Hispanic Students [District] 2023-24'
BLACK = 'Black or African American Students [District] 2023-24'
WHITE = 'White Students [District] 2023-24'
NH_PI = 'Nat. Hawaiian or Other Pacific Isl. Students [District] 2023-24'
TWO_PLUS = 'Two or More Races Students [District] 2023-24'

PTRATIO = 'Pupil/Teacher Ratio [District] 2023-24'
FTE_TEACH = 'Full-Time Equivalent (FTE) Teachers [District] 2023-24'
TOTAL_STAFF = 'Total Staff [District] 2023-24'
LIBRARIANS = 'Librarians/media specialists [District] 2023-24'
MEDIA_SUPPORT = 'Media Support Staff [District] 2023-24'

REV_LOCAL = 'Total Revenue - Local Sources (TLOCREV) [District Finance] 2020-21'
REV_STATE = 'Total Revenue - State Sources (TSTREV) [District Finance] 2020-21'
REV_FED = 'Total Revenue - Federal Sources (TFEDREV) [District Finance] 2020-21'
REV_PER_PUPIL = 'Total Revenue (TOTALREV) per Pupil (V33) [District Finance] 2020-21'

DIPLOMAS = 'Diploma Recipients & Other Completers [District] 2022-23'
POP_TOTAL = 'Estimated Total Population'
POP_5_17 = 'Estimated Population 5-17'
CHILDREN_POV = 'Estimated number of relevant children 5 to 17 years old in poverty who are related to the householder'


# Feature engineering
def safe_div(num, den):
    return np.where((den is None) | (den == 0) | (np.isnan(den)), np.nan, num / den)

race_cols = [AIAN, ASIAN, HISP, BLACK, WHITE, NH_PI, TWO_PLUS]
race_sum = df[race_cols].sum(axis=1, skipna=True)
mf_sum = df[[MALE, FEMALE]].sum(axis=1, skipna=True)
total_students = df[STUDENTS_9_12].copy()
total_students = total_students.fillna(mf_sum)
total_students = total_students.fillna(race_sum)

df['rev_total'] = df[REV_LOCAL].fillna(0) + df[REV_STATE].fillna(0) + df[REV_FED].fillna(0)
df['rev_per_student_calc'] = safe_div(df['rev_total'], total_students)

df['staff_per_100'] = 100 * safe_div(df[TOTAL_STAFF], total_students)
df['fte_per_100'] = 100 * safe_div(df[FTE_TEACH], total_students)
df['librarians_per_100'] = 100 * safe_div(df[LIBRARIANS], total_students)
df['media_support_per_100'] = 100 * safe_div(df[MEDIA_SUPPORT], total_students)

df['pct_aian']  = safe_div(df[AIAN], total_students)
df['pct_asian'] = safe_div(df[ASIAN], total_students)
df['pct_hisp']  = safe_div(df[HISP], total_students)
df['pct_black'] = safe_div(df[BLACK], total_students)
df['pct_white'] = safe_div(df[WHITE], total_students)
df['pct_nhpi']  = safe_div(df[NH_PI], total_students)
df['pct_two']   = safe_div(df[TWO_PLUS], total_students)

df['poverty_rate_5_17'] = safe_div(df[CHILDREN_POV], df[POP_5_17])
df['diplomas_per_100'] = 100 * safe_div(df[DIPLOMAS], total_students)

# Base engineered list
engineered_cols = [
    'rev_per_student_calc', REV_PER_PUPIL,      # we'll drop REV_PER_PUPIL below
    'staff_per_100', 'fte_per_100',             # we'll drop fte_per_100 below
    'librarians_per_100', 'media_support_per_100',
    'pct_aian', 'pct_asian', 'pct_hisp', 'pct_black', 'pct_white', 'pct_nhpi', 'pct_two',
    'poverty_rate_5_17', 'diplomas_per_100',
    PTRATIO, POP_TOTAL
]
present_engineered = [c for c in engineered_cols if c in df.columns]
feature_cols = [STATE] + present_engineered

X = df[feature_cols].copy()

# PRUNE redundant/noisy features from VIF scores
# Keep: rev_per_student_calc + PTRATIO; drop: REV_PER_PUPIL + fte_per_100
# Keep race: White, Black, Hispanic only
drop_cols = [
    REV_PER_PUPIL, 'fte_per_100',               # redundant with funding/ratio
    'staff_per_100',                            # keep PTRATIO instead
    'media_support_per_100', 'librarians_per_100',
    POP_TOTAL, POP_5_17,
    'pct_aian', 'pct_asian', 'pct_nhpi', 'pct_two'  # trim race set
]
X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')


# VIF (optional diagnostic)
def print_vif_table(X_numeric: pd.DataFrame):
    num_df = X_numeric.copy()
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    try:
        vals = num_df.values
        vif = [variance_inflation_factor(vals, i) for i in range(vals.shape[1])]
        out = pd.DataFrame({'feature': num_df.columns, 'VIF': vif}).sort_values('VIF', ascending=False)
        print("\n== VIF (after pruning) ==")
        print(out.to_string(index=False))
    except Exception:
        corr = num_df.corr(numeric_only=True).abs()
        print("\n[statsmodels not available] Correlation matrix after pruning:")
        print(corr.to_string())

categorical_cols = [STATE] if STATE in X.columns else []
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print_vif_table(X[numeric_cols])

# Pipeline and tune C
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

param_grid = {'clf__C': [0.01, 0.1, 1, 3, 5, 10, 20, 50, 100]}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\nBest params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# Evaluate
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC (probabilities):", roc_auc_score(y_test, y_prob))

# Coefficients (drivers)
try:
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
except AttributeError:
    ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
    feature_names = np.array(numeric_cols + list(ohe.get_feature_names_out(categorical_cols)))

coefs = best_model.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs}).sort_values('coef', ascending=False)

print("\nTop positive drivers (predicting Offers CS = 1):")
print(coef_df.head(15).to_string(index=False))

print("\nTop negative drivers (predicting Offers CS = 0):")
print(coef_df.tail(15).to_string(index=False))
