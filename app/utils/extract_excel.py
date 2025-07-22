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

# Convertir tous les synonymes en minuscules pour insensible à la casse
FIELD_SYNONYMS = {
    "designation": [s.lower() for s in ["designation", "désignation", "article", "libellé", "description", "item", "ouvrage"]],
    "unit": [s.lower() for s in ["unit", "unité", "u", "unite", "m²", "m3", "u"]],
    "pu": [s.lower() for s in ["pu", "prix unitaire", "prix", "unit price", "p.u.", "cout", "coût", "prix ht", "montant"]],
    "lot": [s.lower() for s in ["lot", "section", "groupe", "group", "type", "catégorie", "phase", "gros œuvre", "gros oeuvres"]],
}

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger("extract_excel")
logger.setLevel(logging.INFO)

def normalize(text):
    text = str(text).strip().lower()  # Déjà insensible à la casse
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def vectorize(text):
    return model.encode(str(text))

def infer_field_mapping(headers, field_synonyms=FIELD_SYNONYMS):
    mapping = {}
    norm_headers = [normalize(h) for h in headers]  # Déjà en minuscules
    for field, synonyms in field_synonyms.items():
        found = None
        for synonym in synonyms:
            matches = difflib.get_close_matches(synonym, norm_headers, n=1, cutoff=0.7)  # Comparaison avec synonymes en minuscules
            if matches:
                found = headers[norm_headers.index(matches[0])]
                break
        mapping[field] = found
    return mapping

def gpt_map_columns(column_names: List[str]) -> Dict[str, str]:
    # Inclure FIELD_SYNONYMS dans le prompt
    synonyms_str = "\n".join([f"{field}: {', '.join(synonyms)}" for field, synonyms in FIELD_SYNONYMS.items()])
    # Forcer les noms de colonnes en minuscules dans le prompt pour consistance
    prompt = (
        "Voici une liste de colonnes extraites d'un tableau Excel :\n"
        f"{', '.join(str(col).lower() for col in column_names)}\n"
        "Voici les synonymes définis pour mapper les colonnes aux champs logiques :\n"
        f"{synonyms_str}\n"
        "Utilise ces synonymes comme base principale pour mapper chaque colonne à un champ logique parmi : designation, unit, pu, lot. "
        "Si une colonne correspond à un synonyme d'un champ, mappe-la à ce champ. "
        "Si aucune correspondance claire n'est trouvée avec les synonymes, utilise ta propre connaissance pour mapper ou utilise 'autre'. "
        "Réponds sous la forme d'un dictionnaire Python, par exemple : {'colonne1': 'designation', 'colonne2': 'unit', ...}."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        import ast
        mapping = ast.literal_eval(response.choices[0].message.content)
        # Assurer que les clés du mappage sont en minuscules pour consistance
        return {k.lower(): v for k, v in mapping.items() if v in EXPECTED_FIELDS or v == "autre"}
    except (ValueError, SyntaxError) as e:
        logger.error(f"Erreur parsing GPT response: {e}. Réponse brute : {response.choices[0].message.content}")
        # Tentative de récupération : si la réponse est du texte, extraire manuellement
        content = response.choices[0].message.content.lower().strip()
        if content.startswith("{") and content.endswith("}"):
            try:
                return {k.strip(): v for k, v in ast.literal_eval(content).items() if v in EXPECTED_FIELDS or v == "autre"}
            except (ValueError, SyntaxError):
                return {}
        return {}

def find_header_row(df, field_synonyms=FIELD_SYNONYMS, max_rows=100):
    # Initialiser i à 0 et prioriser la ligne 0
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        norm_cells = [normalize(str(cell)) for cell in row.values if pd.notna(cell)]
        if len(norm_cells) > 0:  # Au moins 1 colonne non vide
            matches_count = 0
            for field, synonyms in field_synonyms.items():
                for syn in synonyms:
                    if any(normalize(syn) in cell for cell in norm_cells):
                        matches_count += 1
                        break
            if matches_count >= 2:  # Seulement pour les lignes suivantes, exiger 2 correspondances
                logger.info(f"En-tête détecté à la ligne {i} avec {norm_cells}")
                return i
    logger.warning(f"Aucun en-tête valide détecté dans les {max_rows} premières lignes, utilisation de la ligne 0.")
    return 0

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
        df.columns = [str(col).lower() for col in df.columns]  # Forcer les en-têtes en minuscules
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
            if field == "pu" and isinstance(value, (int, float)):
                record[field] = float(value)
            elif field == "pu" and isinstance(value, str):
                cleaned_value = value.replace('.', '').replace(',', '').replace('-', '')
                record[field] = float(value) if cleaned_value.isdigit() else str(value)
            else:
                record[field] = str(value)
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