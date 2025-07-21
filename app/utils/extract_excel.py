import io
import pandas as pd
import difflib
import unicodedata
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import os
import logging

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the expected fields
EXPECTED_FIELDS = ["designation", "unit", "pu", "lot"]

FIELD_SYNONYMS = {
    "designation": ["designation", "désignation", "article", "libellé", "description", "item"],
    "unit": ["unit", "unité", "u", "unite"],
    "pu": ["pu", "prix unitaire", "prix", "unit price", "p.u."],
    "lot": ["lot", "section", "groupe", "group", "type", "catégorie"],
}

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger("extract_excel")
logger.setLevel(logging.INFO)

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

def find_header_row(df, field_synonyms=FIELD_SYNONYMS):
    for i in range(min(30, len(df))):  # Scan first 30 rows
        row = df.iloc[i]
        # Normalize all cell values
        norm_cells = [normalize(str(cell)) for cell in row.values]
        # Try fuzzy/semantic mapping
        for field, synonyms in field_synonyms.items():
            for syn in synonyms:
                if any(normalize(syn) in cell for cell in norm_cells):
                    return i
    return 0  # fallback: first row

def extract_data_from_excel(file_bytes):
    try:
        logger.info("Lecture brute du fichier Excel sans header...")
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
        logger.info(f"Premières lignes brutes :\n{df_raw.head(5)}")
    except Exception as e:
        logger.error(f"Erreur lecture brute Excel : {e}")
        return []

    try:
        header_row_idx = find_header_row(df_raw)
        logger.info(f"Ligne d'entête détectée : {header_row_idx}")
    except Exception as e:
        logger.error(f"Erreur détection header : {e}")
        return []

    try:
        df = pd.read_excel(io.BytesIO(file_bytes), header=header_row_idx)
        logger.info(f"Colonnes détectées : {list(df.columns)}")
    except Exception as e:
        logger.error(f"Erreur lecture Excel avec header : {e}")
        return []

    try:
        mapping = gpt_map_columns(list(df.columns))
        logger.info(f"Mapping GPT : {mapping}")
    except Exception as e:
        logger.error(f"Erreur GPT mapping : {e}")
        mapping = {}

    reverse_map = {v: k for k, v in mapping.items() if v in ["designation", "unit", "pu", "lot"]}
    logger.info(f"Reverse map utilisé : {reverse_map}")

    records = []
    for idx, row in df.iterrows():
        record = {}
        for field in ["designation", "unit", "pu", "lot"]:
            col = reverse_map.get(field)
            record[field] = row[col] if col and col in row and pd.notna(row[col]) else ""
        if any(record.values()):
            records.append(record)
        else:
            logger.warning(f"Ligne vide ou non reconnue à l'index {idx}: {row}")

    logger.info(f"Nombre de lignes extraites : {len(records)}")
    if records:
        logger.info(f"Exemple de lignes extraites : {records[:3]}")
    else:
        logger.warning("Aucune donnée extraite du fichier Excel.")

    return records

def gpt_extract_table(file_bytes):
    try:
        table_as_text = pd.read_excel(io.BytesIO(file_bytes), header=None).to_csv(index=False, header=False, sep="\t")
        logger.info(f"Table brute envoyée à GPT :\n{table_as_text[:500]}")
    except Exception as e:
        logger.error(f"Erreur lecture brute Excel pour GPT : {e}")
        return []

    prompt = (
        "Voici le contenu d'un tableau Excel :\n"
        f"{table_as_text}\n"
        "Pour chaque ligne, indique à quel champ logique chaque colonne correspond parmi : designation, unit, pu, lot. "
        "Retourne une liste de dictionnaires Python, chaque dictionnaire représentant une ligne structurée."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        import ast
        records = ast.literal_eval(response.choices[0].message.content)
        logger.info(f"Réponse GPT (extrait) : {str(records)[:500]}")
    except Exception as e:
        logger.error(f"Erreur GPT extraction table : {e}")
        records = []

    return records
