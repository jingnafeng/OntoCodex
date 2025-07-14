import pandas as pd
from difflib import get_close_matches, SequenceMatcher

# === Load input files ===
echo_df = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/Echo/ECHO_dic.csv")
loinc_cui = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/omop2obo/LOINC_CUI.csv")
loinc_omop = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/loinc_omop.csv")
snomed_omop = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/snomed_omop.csv")
measurement_omop = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/measurement_omop.csv")

# === Normalize column names ===
loinc_cui.columns = [col.upper().strip() for col in loinc_cui.columns]
loinc_omop.columns = [col.upper().strip() for col in loinc_omop.columns]
snomed_omop.columns = [col.upper().strip() for col in snomed_omop.columns]
measurement_omop.columns = [col.upper().strip() for col in measurement_omop.columns]

# === Extract LOINC code from URI ===
loinc_cui['LOINC'] = loinc_cui['CLASS ID'].str.extract(r'([0-9\-]+)$')

# === Prepare Echo Terms ===
echo_terms = echo_df['DISPLAY_NAME'].dropna().unique()

# === Build LOINC matching index ===
loinc_index = []
for _, row in loinc_cui.iterrows():
    label = str(row['PREFERRED LABEL']).strip()
    loinc_code = row['LOINC']
    loinc_index.append((label.lower(), loinc_code, label))

    if pd.notna(row['SYNONYMS']):
        for synonym in str(row['SYNONYMS']).split('|'):
            loinc_index.append((synonym.strip().lower(), loinc_code, label))

loinc_terms = [t[0] for t in loinc_index]

# === Stage 1: Match to LOINC (with measurement fallback) ===
stage1_results = []
unmatched_terms = []

for term in echo_terms:
    term_lower = term.lower()
    loinc_code = loinc_label = omop_id = match_type = None
    fuzzy_score = 0

    # Exact match
    exact = next((i for i in loinc_index if i[0] == term_lower), None)
    if exact:
        loinc_code, loinc_label = exact[1], exact[2]
        match_type = "exact"
        fuzzy_score = 100
    else:
        close = get_close_matches(term_lower, loinc_terms, n=1, cutoff=0.75)
        if close:
            matched = close[0]
            loinc_code, loinc_label = next((i[1], i[2]) for i in loinc_index if i[0] == matched)
            match_type = "fuzzy"
            fuzzy_score = round(SequenceMatcher(None, term_lower, matched).ratio() * 100)
        else:
            unmatched_terms.append(term)
            continue

    # Try LOINC → OMOP
    row = loinc_omop[loinc_omop['LOINC'] == loinc_code]
    if not row.empty:
        omop_id = row.iloc[0]['CONCEPT_ID']
        vocab = "LOINC"
    else:
        # Fallback: Try measurement_omop.csv
        meas_row = measurement_omop[(measurement_omop['VOCABULARY_ID'].str.upper().isin(["LOINC", "SNOMED", "ICD10CM"])) &
                                    (measurement_omop['CONCEPT_CODE'] == loinc_code)]

        omop_id = meas_row.iloc[0]['CONCEPT_ID'] if not meas_row.empty else None
        vocab = "MEASUREMENT" if omop_id else None

    stage1_results.append({
        'DISPLAY_NAME': term,
        'MATCHED_TERM': loinc_label,
        'LOINC_CODE': loinc_code,
        'OMOP_CONCEPT_ID': omop_id,
        'OMOP_VOCAB': vocab,
        'MATCH_SOURCE': "LOINC",
        'MATCH_TYPE': match_type,
        'FUZZY_SCORE': fuzzy_score
    })

# === Stage 2: Try SNOMED for unmatched ===
snomed_index = [(str(r['CONCEPT_NAME']).lower(), r['CONCEPT_ID'], r['CONCEPT_NAME']) for _, r in snomed_omop.iterrows()]
snomed_terms = [s[0] for s in snomed_index]

for term in unmatched_terms:
    term_lower = term.lower()
    fuzzy_score = 0
    match_type = None

    exact = next((s for s in snomed_index if s[0] == term_lower), None)
    if exact:
        omop_id, snomed_label = exact[1], exact[2]
        match_type = "exact"
        fuzzy_score = 100
    else:
        close = get_close_matches(term_lower, snomed_terms, n=1, cutoff=0.75)
        if close:
            match = close[0]
            omop_id, snomed_label = next((s[1], s[2]) for s in snomed_index if s[0] == match)
            match_type = "fuzzy"
            fuzzy_score = round(SequenceMatcher(None, term_lower, match).ratio() * 100)
        else:
            snomed_label = None
            omop_id = None
            match_type = "unmatched"
            fuzzy_score = 0

    stage1_results.append({
        'DISPLAY_NAME': term,
        'MATCHED_TERM': snomed_label,
        'LOINC_CODE': None,
        'OMOP_CONCEPT_ID': omop_id,
        'OMOP_VOCAB': "SNOMED" if omop_id else None,
        'MATCH_SOURCE': "SNOMED",
        'MATCH_TYPE': match_type,
        'FUZZY_SCORE': fuzzy_score
    })
# === Load domain-specific validation resources ===
valve_df = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/Echo/athena_cardiac_msm/valve.csv", delimiter='\t', dtype=str)
ejection_df = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/Echo/athena_cardiac_msm/ejection_fraction.csv", delimiter='\t', dtype=str)
ventricular_df = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/Echo/athena_cardiac_msm/ventricular.csv", delimiter='\t', dtype=str)

# Combine all rows into searchable strings
valve_text = valve_df.apply(lambda row: ' '.join(map(str, row.values)).lower(), axis=1).tolist()
ventricular_text = ventricular_df.apply(lambda row: ' '.join(map(str, row.values)).lower(), axis=1).tolist()
ejection_text = ejection_df.apply(lambda row: ' '.join(map(str, row.values)).lower(), axis=1).tolist()

# Validate each term against domain sources
def validate_cardio_sources(term):
    t = term.lower()
    if any(t in v for v in valve_text):
        return "valve.csv"
    elif any(t in v for v in ventricular_text):
        return "ventricular.csv"
    elif any(t in v for v in ejection_text):
        return "ejection_fraction.csv"
    else:
        return "Not Found"

# Add validation results
results = pd.DataFrame(stage1_results)
results['VALIDATED_IN'] = results['DISPLAY_NAME'].apply(validate_cardio_sources)

# === Save Outputs ===
results.to_csv("echo_loinc_snomed_omop_validated.csv", index=False)
results[results['MATCH_TYPE'] == 'unmatched'].to_csv("unmatched_terms.csv", index=False)

# === Report Summary ===
print("✅ Mapping Complete with Domain Validation")
print(f"  Total input terms:      {len(echo_terms)}")
print(f"  LOINC matched:          {len(echo_terms) - len(unmatched_terms)}")
print(f"  SNOMED fallback terms:  {len(unmatched_terms)}")
print("📁 Files saved:")
print("  ➜ echo_loinc_snomed_omop_validated.csv")
print("  ➜ unmatched_terms.csv")

# (base) feng.jingna@R5436928 Athena % python map_echo_fuzzy_loinc_omop.py
# ✅ Mapping Complete with Domain Validation
#   Total input terms:      1362
#   LOINC matched:          335
#   SNOMED fallback terms:  1027

# (base) feng.jingna@R5436928 Athena % wc unmatched_terms.csv
#      989    8953   86910 unmatched_terms.csv