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
    logger.info(f"Début du mappage avec en-têtes : {headers}")
    mapping = {}
    norm_headers = [normalize(h) for h in headers]  # Déjà en minuscules
    for field, synonyms in field_synonyms.items():
        found = None
        for synonym in synonyms:
            matches = difflib.get_close_matches(synonym, norm_headers, n=1, cutoff=0.7)
            if matches:
                found = headers[norm_headers.index(matches[0])]
                break
        mapping[field] = found
    logger.info(f"Résultat du mappage : {mapping}")
    return mapping

def gpt_map_columns(column_names: List[str]) -> Dict[str, str]:
    logger.info(f"Début du mappage GPT avec colonnes : {column_names}")
    synonyms_str = "\n".join([f"{field}: {', '.join(synonyms)}" for field, synonyms in FIELD_SYNONYMS.items()])
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
    logger.info(f"Réponse brute de ChatGPT : {response.choices[0].message.content}")
    try:
        import ast
        mapping = ast.literal_eval(response.choices[0].message.content)
        logger.info(f"Mappage GPT analysé avec succès : {mapping}")
        return {k.lower(): v for k, v in mapping.items() if v in EXPECTED_FIELDS or v == "autre"}
    except (ValueError, SyntaxError) as e:
        logger.error(f"Erreur parsing GPT response : {e}. Réponse brute : {response.choices[0].message.content}")
        content = response.choices[0].message.content.lower().strip()
        if content.startswith("{") and content.endswith("}"):
            try:
                mapping = {k.strip(): v for k, v in ast.literal_eval(content).items() if v in EXPECTED_FIELDS or v == "autre"}
                logger.info(f"Mappage GPT récupéré : {mapping}")
                return mapping
            except (ValueError, SyntaxError) as e2:
                logger.error(f"Erreur dans la récupération du mappage : {e2}")
                return {}
        logger.error("Échec total du parsing GPT, mappage vide retourné.")
        return {}

def find_header_row(df, max_rows=200):
    logger.info(f"Début de la détection de l'en-tête sur {min(max_rows, len(df))} lignes")
    logger.info(f"Premières lignes pour analyse : \n{df.head(min(max_rows, len(df))).to_string()}")
    # Détection heuristique basée sur les données
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        non_null_count = row.notna().sum()
        logger.debug(f"Ligne {i} : {non_null_count} colonnes non vides")
        if non_null_count >= 3:
            next_row = df.iloc[i + 1] if i + 1 < len(df) else None
            if next_row is not None and next_row.notna().sum() > 2:
                data_types = [type(cell).__name__ for cell in next_row if pd.notna(cell)]
                logger.debug(f"Types de données ligne suivante {i+1} : {data_types}")
                if any(t in ["int64", "float64", "datetime64"] for t in data_types):
                    excerpt = df.head(min(max_rows, len(df))).to_string(index=True)
                    prompt = (
                        f"Voici un extrait des premières lignes d'un fichier Excel :\n{excerpt}\n"
                        f"La ligne {i} a été détectée comme en-tête basée sur des règles (au moins 3 colonnes non vides suivies de données). "
                        "Confirme si c'est correct en vérifiant si elle contient des titres de colonnes cohérents suivis de données. "
                        "Réponds 'oui' ou 'non'."
                    )
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    logger.info(f"Validation ChatGPT pour ligne {i} : {response.choices[0].message.content}")
                    if "oui" in response.choices[0].message.content.lower():
                        logger.info(f"En-tête confirmé à la ligne {i} par validation ChatGPT")
                        return i
                    else:
                        logger.warning(f"Validation ChatGPT échouée pour ligne {i}")
    # Fallback à la détection par ChatGPT si les règles échouent
    logger.info("Passage au fallback ChatGPT pour détection d'en-tête")
    header_row_idx = gpt_detect_header_row(df, max_rows)
    if header_row_idx >= 0:
        logger.info(f"En-tête détecté à la ligne {header_row_idx} par fallback ChatGPT")
        return header_row_idx
    logger.warning(f"Aucun en-tête valide détecté dans les {max_rows} premières lignes, utilisation de la ligne 0.")
    return 0

def gpt_detect_header_row(df, max_rows=20):
    logger.info(f"Début de la détection d'en-tête par ChatGPT sur {min(max_rows, len(df))} lignes")
    excerpt = df.head(min(max_rows, len(df))).to_string(index=True)
    logger.debug(f"Extrait envoyé à ChatGPT : \n{excerpt}")
    prompt = (
        f"Voici un extrait des premières lignes d'un fichier Excel (index de ligne inclus) :\n"
        f"{excerpt}\n"
        "Identifie la ligne qui contient les en-têtes du tableau. Les en-têtes sont généralement la première ligne avec des titres de colonnes cohérents "
        "(par exemple, 'designation', 'unit', 'pu', 'lot' ou des termes similaires) suivie de données cohérentes sur les lignes suivantes. "
        "Ignore les lignes avec des titres généraux ou des descriptions (comme 'Bibliothèque de prix'). "
        "Réponds uniquement avec le numéro de la ligne (index basé sur 0) où se trouve l'en-tête, par exemple : 0, 1, 2, etc."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    logger.info(f"Réponse brute de ChatGPT : {response.choices[0].message.content}")
    try:
        header_row_idx = int(response.choices[0].message.content.strip())
        if 0 <= header_row_idx < min(max_rows, len(df)):
            logger.info(f"En-tête détecté à la ligne {header_row_idx} par ChatGPT")
            return header_row_idx
        else:
            logger.warning(f"Indice d'en-tête {header_row_idx} invalide, utilisation de 0.")
            return 0
    except (ValueError, IndexError) as e:
        logger.error(f"Erreur parsing réponse ChatGPT pour en-tête : {e}. Réponse brute : {response.choices[0].message.content}")
        return 0

def extract_data_from_excel(file_bytes):
    logger.info("Début de l'extraction des données du fichier Excel")
    try:
        logger.info("Lecture brute du fichier Excel sans header...")
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
        logger.info(f"Premières lignes brutes lues avec succès : \n{df_raw.head(5).to_string()}")
    except Exception as e:
        logger.error(f"Erreur lors de la lecture brute du fichier Excel : {e}")
        return []

    try:
        logger.info("Détection de la ligne d'en-tête...")
        header_row_idx = find_header_row(df_raw)
        logger.info(f"Ligne d'en-tête détectée : {header_row_idx}")
    except Exception as e:
        logger.error(f"Erreur lors de la détection de l'en-tête : {e}")
        return []

    try:
        logger.info(f"Lecture du fichier avec en-tête à la ligne {header_row_idx}...")
        df = pd.read_excel(io.BytesIO(file_bytes), header=header_row_idx)
        df.columns = [str(col).lower() for col in df.columns]  # Forcer les en-têtes en minuscules
        logger.info(f"Colonnes détectées : {list(df.columns)}")
    except Exception as e:
        logger.error(f"Erreur lors de la lecture avec en-tête : {e}")
        return []

    try:
        logger.info("Début du mappage des colonnes...")
        mapping = infer_field_mapping(list(df.columns))
        logger.info(f"Résultat du mappage initial : {mapping}")
        if not any(value in EXPECTED_FIELDS for value in mapping.values()):
            logger.info("Passage au mappage GPT car aucun mappage initial valide")
            mapping = gpt_map_columns(list(df.columns))
        logger.info(f"Mapping final : {mapping}")
        if not mapping:
            logger.error("Aucun mappage valide trouvé, extraction impossible.")
            return []
    except Exception as e:
        logger.error(f"Erreur lors du mappage : {e}")
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
