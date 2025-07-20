import io
import pandas as pd
import difflib
import unicodedata
from sentence_transformers import SentenceTransformer
import numpy as np

# Define the expected fields
EXPECTED_FIELDS = ["designation", "unit", "pu", "lot"]

FIELD_SYNONYMS = {
    "designation": ["designation", "désignation", "article", "libellé", "description", "item"],
    "unit": ["unit", "unité", "u", "unite"],
    "pu": ["pu", "prix unitaire", "prix", "unit price", "p.u."],
    "lot": ["lot", "section", "groupe", "group"],
}

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text):
    text = str(text).strip().lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def vectorize(text):
    return model.encode(str(text))

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

def semantic_column_mapping(headers):
    mapping = {}
    header_vecs = {h: vectorize(h) for h in headers}
    for h, h_vec in header_vecs.items():
        best_field = None
        best_score = -1
        for field, synonyms in FIELD_SYNONYMS.items():
            for syn in synonyms:
                syn_vec = vectorize(syn)
                score = np.dot(h_vec, syn_vec) / (np.linalg.norm(h_vec) * np.linalg.norm(syn_vec))
                if score > best_score:
                    best_score = score
                    best_field = field
        mapping[h] = best_field if best_score > 0.7 else None  # seuil ajustable
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
    # Essaye d'abord le fuzzy mapping
    mapping = infer_field_mapping(df.columns)
    # Si certains champs ne sont pas trouvés, complète avec le semantic mapping
    if not all(mapping.values()):
        semantic_map = semantic_column_mapping(df.columns)
        for k, v in mapping.items():
            if not v:
                # Cherche la première colonne qui a ce champ en semantic mapping
                for col, field in semantic_map.items():
                    if field == k:
                        mapping[k] = col
                        break
    records = []
    for _, row in df.iterrows():
        record = {}
        for field, col in mapping.items():
            record[field] = row[col] if col and col in row and pd.notna(row[col]) else ""
        records.append(record)
    return records
