import io
import pandas as pd
import difflib
import unicodedata
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def gpt_map_columns(column_names):
    prompt = (
        "Voici une liste de colonnes extraites d'un tableau Excel :\n"
        f"{', '.join(column_names)}\n"
        "Pour chaque colonne, indique à quel champ logique elle correspond parmi : designation, unit, pu, lot. "
        "Réponds sous la forme d'un dictionnaire Python où la clé est le nom de colonne et la valeur est le champ logique ou 'autre'."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    import ast
    mapping = ast.literal_eval(response.choices[0].message.content)
    return mapping

def extract_data_from_excel(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    columns = list(df.columns)
    mapping = gpt_map_columns(columns)
    reverse_map = {v: k for k, v in mapping.items() if v in ["designation", "unit", "pu", "lot"]}
    records = []
    for _, row in df.iterrows():
        record = {}
        for field in ["designation", "unit", "pu", "lot"]:
            col = reverse_map.get(field)
            record[field] = row[col] if col and col in row and pd.notna(row[col]) else ""
        records.append(record)
    return records
