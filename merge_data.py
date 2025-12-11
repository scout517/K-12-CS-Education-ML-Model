import pandas as pd

def read_csv_safe(path):
    """Try UTF-8 first, fall back to latin1 if needed."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def normalize_str(s: pd.Series) -> pd.Series:
    """Trim, uppercase state codes, collapse weird spaces/dashes."""
    s = s.astype(str).str.strip()
    s = s.str.replace("\u00a0", " ", regex=False).str.replace("\u202f", " ", regex=False)
    s = s.str.replace("–", "-", regex=False).str.replace("—", "-", regex=False).str.replace("−", "-", regex=False)
    return s

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Robust numeric cleaner:
      - remove NBSPs, commas, $, %
      - convert (123) -> -123
      - keep digits, dot, minus; drop everything else
    """
    s = series.astype(str).str.strip()
    s = (s.replace({"\u00a0": " ", "\u202f": " "})
           .str.replace(",", "", regex=False)
           .str.replace("$", "", regex=False)
           .str.replace("%", "", regex=False))
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# Columns you later feed into the model (numeric ones will be cleaned)
NUMERIC_COLS_TO_KEEP = [
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
]
CATEGORICAL_COLS_TO_KEEP = ['State']
LABEL_COL = 'Offers CS?'

df1 = read_csv_safe('data/Map_Full Data_data.csv')
df2 = read_csv_safe('data/School District Data.csv')
df3 = read_csv_safe('data/Poverty Data.csv')

for col in ['State']:
    if col in df1: df1[col] = normalize_str(df1[col])
for col in ['Agency ID', 'State Abbr', 'Agency Name']:
    if col in df2: df2[col] = normalize_str(df2[col])
for col in ['District ID', 'State Postal Code']:
    if col in df3: df3[col] = normalize_str(df3[col])


if 'Agency ID' in df2:
    df2['Agency ID'] = df2['Agency ID'].str.zfill(7) 
    df2['Agency ID Last5'] = df2['Agency ID'].str[-5:].str.zfill(5)
# df3: District IDs should be 5 digits
if 'District ID' in df3:
    df3['District ID'] = df3['District ID'].str[-5:].str.zfill(5)

# Dedup right-hand tables on merge keys
df2u = df2.drop_duplicates(['Agency Name', 'State Abbr'], keep='first')
df3u = df3.drop_duplicates(['District ID', 'State Postal Code'], keep='first')

# First merge: df1 + df2 on (district name, state)
merged_1_2 = pd.merge(
    df1,
    df2u,
    left_on=['School District Name', 'State'],
    right_on=['Agency Name', 'State Abbr'],
    how='inner',
    validate='m:1'
)

# Derive Agency ID Last5 for second merge
merged_1_2['Agency ID Last5'] = (
    merged_1_2['Agency ID'].astype(str).str.strip().str[-5:].str.zfill(5)
)

# Second merge: + df3 on (last5, state)
final_merged = pd.merge(
    merged_1_2,
    df3u,
    left_on=['Agency ID Last5', 'State'],
    right_on=['District ID', 'State Postal Code'],
    how='left',
    validate='m:1'
)

# De-dup rows by (district, school, state)
final_merged = final_merged.drop_duplicates(
    subset=['School District Name', 'School Name', 'State'],
    keep='first'
)

# Map label; keep only modeling columns + label
if LABEL_COL in final_merged.columns:
    final_merged[LABEL_COL] = final_merged[LABEL_COL].map({'Yes': 1, 'No': 0})

# Drop columns not needed
keep_cols = (
    ['School District Name', 'School Name', 'State', LABEL_COL] +
    NUMERIC_COLS_TO_KEEP
)
keep_cols = [c for c in keep_cols if c in final_merged.columns]
final_merged = final_merged[keep_cols].copy()

# Clean numerics
for col in NUMERIC_COLS_TO_KEEP:
    if col in final_merged.columns:
        final_merged[col] = clean_numeric_series(final_merged[col])

na_counts = final_merged[NUMERIC_COLS_TO_KEEP].isna().sum()
na_counts = na_counts[na_counts > 0]
if not na_counts.empty:
    print("\n[Info] Remaining NaNs after numeric cleaning (you can impute later in your ML pipeline):")
    print(na_counts.sort_values(ascending=False))

if LABEL_COL in final_merged.columns:
    final_merged = final_merged.dropna(subset=[LABEL_COL])

final_merged = final_merged.dropna()

# Save
final_merged.to_csv('merged_data.csv', index=False, encoding='utf-8-sig')
print("Wrote merged_data.csv with cleaned numerics and UTF-8 BOM.")
