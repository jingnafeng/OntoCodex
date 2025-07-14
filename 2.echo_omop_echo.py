import pandas as pd
import re
from difflib import SequenceMatcher

# Load input files
#echo_dic = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/unmatched_terms.csv")
echo_dic = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/Echo/ECHO_dic.csv")
omop_echo_dic = pd.read_csv("/Users/feng.jingna/macLocal/Vocabulary/Athena/output/Echo/OMOP_echo_dic_forMapping.csv")  # tab-separated sample


# -------------------------------
# Helper: normalize + sort tokens
# -------------------------------
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)      # remove punctuation
    return " ".join(sorted(text.split()))       # sort tokens alphabetically

# Normalize OMOP echo terms
omop_echo_dic['norm_name'] = omop_echo_dic['LOINC Long Common Name'].apply(normalize)

# ------------------------------------
# Ensure output columns exist and typed
# ------------------------------------
for col in ['MATCHED_TERM', 'LOINC_CODE', 'OMOP_CONCEPT_ID', 'OMOP_VOCAB',
            'MATCH_SOURCE', 'MATCH_TYPE', 'FUZZY_SCORE', 'VALIDATED_IN']:
    if col not in echo_dic.columns:
        echo_dic[col] = ""
    echo_dic[col] = echo_dic[col].astype(str)

# ----------------------
# Fuzzy Partial Matching
# ----------------------
for idx, row in echo_dic.iterrows():
    input_term = normalize(row['DISPLAY_NAME'])
    best_score = 0
    best_row = None

    for _, ref_row in omop_echo_dic.iterrows():
        ref_term = ref_row['norm_name']
        score = SequenceMatcher(None, input_term, ref_term).ratio()
        if score > best_score:
            best_score = score
            best_row = ref_row

    if best_score >= 0.6:
        echo_dic.at[idx, 'MATCHED_TERM'] = best_row['LOINC Long Common Name']
        echo_dic.at[idx, 'LOINC_CODE'] = best_row['LOINC Code']
        echo_dic.at[idx, 'OMOP_CONCEPT_ID'] = str(best_row['concept_id'])
        echo_dic.at[idx, 'OMOP_VOCAB'] = "LOINC"
        echo_dic.at[idx, 'MATCH_SOURCE'] = "LOINC"
        echo_dic.at[idx, 'MATCH_TYPE'] = "fuzzy_partial"
        echo_dic.at[idx, 'FUZZY_SCORE'] = f"{best_score:.3f}"
        echo_dic.at[idx, 'VALIDATED_IN'] = "Auto-mapped"
    else:
        echo_dic.at[idx, 'MATCH_TYPE'] = "unmatched"
        echo_dic.at[idx, 'FUZZY_SCORE'] = "0"
        echo_dic.at[idx, 'VALIDATED_IN'] = "Not Found"

# -----------------------
# Save the mapped results
# -----------------------
echo_dic.to_csv("echo_dic_mapped.csv", index=False)
print("✅ Mapping complete. Output saved to: echo_dic_mapped.csv")

# -----------------------
# Summary count
# -----------------------
mapped_count = (echo_dic['MATCH_TYPE'] != 'unmatched').sum()
unmapped_count = (echo_dic['MATCH_TYPE'] == 'unmatched').sum()
total = len(echo_dic)

print(f"📊 Mapping Summary:")
print(f"🔹 Total terms processed: {total}")
print(f"✅ Mapped terms:          {mapped_count}")
print(f"❌ Unmapped terms:        {unmapped_count}")

# (base) feng.jingna@R5436928 Athena % python echo_omop_echo.py
# ✅ Mapping complete. Output saved to: echo_dic_mapped.csv
# 📊 Mapping Summary:
# 🔹 Total terms processed: 988
# ✅ Mapped terms:          232
# ❌ Unmapped terms:        756

## For full echo_dic.csv result:
# ✅ Mapping complete. Output saved to: echo_dic_mapped.csv
# 📊 Mapping Summary:
# 🔹 Total terms processed: 1365
# ✅ Mapped terms:          403
# ❌ Unmapped terms:        962