import io
import pandas as pd
import difflib
import unicodedata

# Define the expected fields
EXPECTED_FIELDS = ["designation", "unit", "pu", "lot"]

FIELD_SYNONYMS = {
    "designation": ["designation", "désignation", "article", "libellé", "description", "item"],
    "unit": ["unit", "unité", "u", "unite"],
    "pu": ["pu", "prix unitaire", "prix", "unit price", "p.u."],
    "lot": ["lot", "section", "groupe", "group"],
}

def normalize(text):
    text = str(text).strip().lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def infer_field_mapping(headers, field_synonyms=FIELD_SYNONYMS):
    mapping = {}
    norm_headers = [normalize(h) for h in headers]
    for field, synonyms in field_synonyms.items():
        found = None
        for synonym in synonyms:
            matches = difflib.get_close_matches(normalize(synonym), norm_headers, n=1, cutoff=0.7)
            if matches:
                found = headers[norm_headers.index(matches[0])]
                break
        mapping[field] = found
    return mapping

def infer_column_mapping(columns):
    mapping = {}
    for field in EXPECTED_FIELDS:
        match = difflib.get_close_matches(field, columns, n=1, cutoff=0.6)
        if match:
            mapping[field] = match[0]
        else:
            mapping[field] = None
    return mapping

def extract_data_from_excel(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    mapping = infer_field_mapping(df.columns)
    records = []
    for _, row in df.iterrows():
        record = {}
        for field, col in mapping.items():
            record[field] = row[col] if col and col in row and pd.notna(row[col]) else ""
        records.append(record)
    return records
