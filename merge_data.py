import pandas as pd

# === 1. Load Data ===
df1 = pd.read_csv('data/Map_Full Data_data.csv', dtype={'State': str})
df2 = pd.read_csv('data/School District Data.csv', dtype={'Agency ID': str, 'State Abbr': str})
df3 = pd.read_csv('data/Poverty Data.csv', dtype={'District ID': str, 'State Postal Code': str})

# === 2. Clean Keys ===
df2['Agency ID'] = df2['Agency ID'].str.strip()
df2['Agency ID Last5'] = df2['Agency ID'].str[-5:].str.zfill(5)
df2['State Abbr'] = df2['State Abbr'].str.upper().str.strip()

df3['District ID'] = df3['District ID'].str.strip().str.zfill(5)
df3['State Postal Code'] = df3['State Postal Code'].str.upper().str.strip()

# === 3. Deduplicate Right-Hand Tables on Merge Keys ===
# (Keeps one row per Agency Name/State and per District ID/State Postal Code)
df2u = df2.drop_duplicates(['Agency Name', 'State Abbr'], keep='first')
df3u = df3.drop_duplicates(['District ID', 'State Postal Code'], keep='first')

# === 4. First Merge: df1 + df2 ===
merged_1_2 = pd.merge(
    df1,
    df2u,
    left_on=['School District Name', 'State'],
    right_on=['Agency Name', 'State Abbr'],
    how='inner',
    validate='m:1'  # each school should map to one district row
)

# merged_1_2.to_csv('merged_1_2.csv', index=False)

# Create consistent key for second merge
merged_1_2['Agency ID Last5'] = (
    merged_1_2['Agency ID']
    .astype(str)
    .str.strip()
    .str[-5:]
    .str.zfill(5)
)

# === 5. Second Merge: merged_1_2 + df3 ===
final_merged = pd.merge(
    merged_1_2,
    df3u,
    left_on=['Agency ID Last5', 'State'],
    right_on=['District ID', 'State Postal Code'],
    how='left',
    validate='m:1'  # many schools (left) → one district (right)
)

# === 6. Remove Any Duplicate (District, School) Rows ===
final_merged = final_merged.drop_duplicates(
    subset=['School District Name', 'School Name', 'State'],
    keep='first'
)

final_merged = final_merged.dropna()

final_merged.drop(columns=['Agency Name', 'County Number', 'State Postal Code', 'State FIPS Code', 
                           'District ID', 'Agency ID Last5', 'Name', 'is High School', 
                           'Grade Levels', 'teaches_cs', 'State Name'], inplace=True)

final_merged['Offers CS?'] = final_merged['Offers CS?'].map({'Yes': 1, 'No': 0})

# === 7. Save to File ===
final_merged.to_csv('merged_data.csv', index=False)