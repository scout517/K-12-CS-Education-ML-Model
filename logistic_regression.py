import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data/merged_data.csv')

# print(df.head())

y = df['Offers CS?']

X = df[[
    'State',
    'Grades 9-12 Students [District] 2023-24',
    'Male Students [District] 2023-24',
    'Female Students [District] 2023-24',
    'American Indian/Alaska Native Students [District] 2023-24',
    'Asian or Asian/Pacific Islander Students [District] 2023-24',
    'Hispanic Students [District] 2023-24',
    'Black or African American Students [District] 2023-24',
    'White Students [District] 2023-24',
    'Nat. Hawaiian or Other Pacific Isl. Students [District] 2023-24',
    'Two or More Races Students [District] 2023-24',
    'Pupil/Teacher Ratio [District] 2023-24',
    'Full-Time Equivalent (FTE) Teachers [District] 2023-24',
    'Total Staff [District] 2023-24',
    'Librarians/media specialists [District] 2023-24',
    'Media Support Staff [District] 2023-24',
    'Total Revenue - Local Sources (TLOCREV) [District Finance] 2020-21',
    'Total Revenue - State Sources (TSTREV) [District Finance] 2020-21',
    'Total Revenue - Federal Sources (TFEDREV) [District Finance] 2020-21',
    'Total Revenue (TOTALREV) per Pupil (V33) [District Finance] 2020-21',
    'Diploma Recipients & Other Completers [District] 2022-23',
    'Estimated Total Population',
    'Estimated Population 5-17',
    'Estimated number of relevant children 5 to 17 years old in poverty who are related to the householder'
    ]]

categorical_cols = [
    'State'
]

numerical_cols = [item for item in X.columns if item != 'State']

def clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace(r'[^\d\.\-]', '', regex=True)  # keep only digits, dot, minus
    return pd.to_numeric(s, errors='coerce')

for col in numerical_cols:
    X[col] = clean_numeric(X[col])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))