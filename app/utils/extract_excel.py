import io
import pandas as pd
import difflib
import unicodedata
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import os
import logging
from typing import List, Dict

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

def gpt_map_columns(column_names: List[str]) -> Dict[str, str]:
    prompt = (
        "Voici une liste de colonnes extraites d'un tableau Excel (basée sur les 15 premières lignes) :\n"
        f"{', '.join(column_names)}\n"
        "Pour chaque colonne, indique à quel champ logique elle correspond parmi : designation, unit, pu, lot. "
        "Si aucune correspondance n'est claire, utilise 'autre'. Réponds sous la forme d'un dictionnaire Python."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        import ast
        mapping = ast.literal_eval(response.choices[0].message.content)
        return {k: v for k, v in mapping.items() if v in EXPECTED_FIELDS or v == "autre"}
    except (ValueError, SyntaxError) as e:
        logger.error(f"Erreur parsing GPT response: {e}")
        return {}

def find_header_row(df, field_synonyms=FIELD_SYNONYMS, max_rows=15):
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        norm_cells = [normalize(str(cell)) for cell in row.values]
        for field, synonyms in field_synonyms.items():
            for syn in synonyms:
                if any(normalize(syn) in cell for cell in norm_cells):
                    logger.info(f"En-tête détecté à la ligne {i} avec {norm_cells}")
                    return i
    logger.warning("Aucun en-tête détecté dans les 15 premières lignes, utilisation de la ligne 0.")
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
        mapping = infer_field_mapping(list(df.columns))
        logger.info(f"infer_field_mapping : {mapping}")
        if not any(value in EXPECTED_FIELDS for value in mapping.values()):
            mapping = gpt_map_columns(list(df.columns))
        logger.info(f"Mapping final : {mapping}")
        if not mapping:
            logger.error("Aucun mappage valide trouvé, extraction impossible.")
            return []
    except Exception as e:
        logger.error(f"Erreur mapping : {e}")
        return []

    reverse_map = {v: k for k, v in mapping.items() if v in EXPECTED_FIELDS}
    logger.info(f"Reverse map utilisé : {reverse_map}")

    if not reverse_map:
        logger.error("Aucun champ attendu mappé, vérifiez les en-têtes ou les synonymes.")
        return []

    records = []
    for idx, row in df.iterrows():
        record = {}
        for field in EXPECTED_FIELDS:
            col = reverse_map.get(field)
            value = row[col] if col and col in row and pd.notna(row[col]) else ""
            record[field] = float(value) if field == "pu" and isinstance(value, (int, float, str)) and value.replace('.', '').replace(',', '').replace('-', '').isdigit() else str(value)
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