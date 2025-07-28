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
import time
from app.vector_store.weaviate_client import get_model, search_documents

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the expected fields
EXPECTED_FIELDS = ["designation", "unit", "pu", "lot"]

# Convertir tous les synonymes en minuscules pour insensible à la casse
FIELD_SYNONYMS = {
    "designation": [s.lower() for s in ["designation", "désignation", "article", "libellé", "description", "item", "type", "description ouvrage"]],
    "unit": [s.lower() for s in ["unit", "unité", "u", "unite", "m²", "m3", "u", "unité du détail"]],
    "pu": [s.lower() for s in ["pu", "prix unitaire", "prix", "unit price", "p.u.", "cout", "coût", "prix ht", "montant", "prix de revient", "prix de revient du détail"]],
    "lot": [s.lower() for s in ["lot", "section", "groupe", "group", "type", "catégorie", "phase", "gros œuvre", "gros oeuvres", "catégorie"]],
    # "quantity": [s.lower() for s in ["quantité", "quantite", "qte", "qté", "nombre", "volume", "qty", "quantity", "quantité totale"]]
}

model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger("extract_excel")
logger.setLevel(logging.INFO)

def normalize(text):
    text = str(text).strip().lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def vectorize(text):
    return model.encode(str(text))

def infer_field_mapping(headers, field_synonyms=FIELD_SYNONYMS):
    logger.info(f"Début du mappage avec en-têtes : {headers}")
    mapping = {}
    norm_headers = [normalize(h) for h in headers]
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
    FIELD_SYNONYMS = {
        "designation": [s.lower() for s in ["designation", "désignation", "article", "libellé", "description", "item", "type", "description ouvrage", "description composant"]],
        "unit": [s.lower() for s in ["unit", "unité", "u", "unite", "m²", "m3", "u", "unité du détail", "unité"]],
        "pu": [s.lower() for s in ["pu", "prix unitaire", "prix", "unit price", "p.u.", "cout", "coût", "prix ht", "montant", "prix de revient", "prix de revient du détail", "prix unitaires", "prix totaux"]],
        "lot": [s.lower() for s in ["lot", "section", "groupe", "group", "type", "catégorie", "phase", "gros œuvre", "gros oeuvres", "catégorie"]],
    }
    synonyms_str = "\n".join([f"{field}: {', '.join(synonyms)}" for field, synonyms in FIELD_SYNONYMS.items()])
    prompt = (
        "Voici une liste de colonnes extraites d'un tableau Excel :\n"
        f"{', '.join(str(col).lower() for col in column_names)}\n"
        "Voici les synonymes définis pour mapper les colonnes aux champs logiques :\n"
        f"{synonyms_str}\n"
        "Utilise ces synonymes comme base principale pour mapper chaque colonne à un champ logique parmi : designation, unit, pu, lot. "
        "Si une colonne correspond à un synonyme d'un champ, mappe-la à ce champ. Exemples utiles : 'description composant' pourrait être 'designation', "
        "'unité du détail' pourrait être 'unit', 'prix de revient' pourrait être 'pu', 'type' pourrait être 'lot'. "
        "Si aucune correspondance claire n'est trouvée avec les synonymes, utilise 'autre'. "
        "Réponds uniquement avec un dictionnaire Python, par exemple : {'colonne1': 'designation', 'colonne2': 'unit', ...}, sans texte narratif."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        logger.info(f"Réponse brute de ChatGPT : {response.choices[0].message.content}")
        import ast
        # Extraire le dictionnaire en ignorant le texte narratif
        content = response.choices[0].message.content.strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx != -1 and end_idx != 0 and start_idx < end_idx:
            dict_content = content[start_idx:end_idx]
            try:
                mapping = ast.literal_eval(dict_content)
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Parsing initial échoué avec ast.literal_eval : {e}. Tentative avec json.loads.")
                import json
                mapping = json.loads(dict_content.replace("'", '"'))
        else:
            logger.error(f"Réponse invalide, pas de dictionnaire détecté : {content}")
            return {}
        logger.info(f"Mappage GPT analysé avec succès : {mapping}")
        # Prioriser le mappage souhaité et gérer les doublons
        final_mapping = {}
        priority_fields = {"description composant": "designation", "unité du détail": "unit", "prix de revient": "pu", "type": "lot"}
        used_columns = set()
        # Appliquer les priorités d'abord
        for col, target_field in priority_fields.items():
            if col.lower() in [c.lower() for c in column_names]:
                final_mapping[col.lower()] = target_field
                used_columns.add(col.lower())
        # Mapper les autres colonnes restantes
        for col, mapped_field in mapping.items():
            col_lower = col.lower()
            if col_lower not in used_columns and mapped_field in EXPECTED_FIELDS:
                final_mapping[col_lower] = mapped_field
                used_columns.add(col_lower)
        # Ajouter les "autre" si pas encore mappés
        for col, mapped_field in mapping.items():
            col_lower = col.lower()
            if col_lower not in used_columns and mapped_field == "autre":
                final_mapping[col_lower] = "autre"
                used_columns.add(col_lower)
        logger.info(f"Mappage final GPT : {final_mapping}")
        return final_mapping
    except (ValueError, SyntaxError, json.JSONDecodeError) as e:
        logger.error(f"Erreur parsing GPT response : {e}. Réponse brute : {response.choices[0].message.content}")
        return {}
    except openai.RateLimitError as e:
        logger.error(f"Erreur de limite de taux ChatGPT : {e}. Mappage vide retourné.")
        return {}

def find_header_row(df, max_rows=30):
    logger.info(f"Début de la détection de l'en-tête sur {min(max_rows, len(df))} lignes")
    logger.info(f"Premières lignes pour analyse : \n{df.head(min(max_rows, len(df))).to_string()}")
    # Calculer le nombre moyen de colonnes non vides comme seuil dynamique
    total_non_null = sum(row.notna().sum() for _, row in df.head(max_rows).iterrows())
    avg_non_null = total_non_null / min(max_rows, len(df))
    threshold = max(3, int(avg_non_null * 0.8))  # Seuil dynamique, minimum 3
    logger.info(f"Seuil dynamique de colonnes non vides : {threshold}")

    # Détection heuristique basée sur la densité, la transition et l'absence de nombres dans l'en-tête
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        non_null_count = row.notna().sum()
        logger.debug(f"Ligne {i} : {non_null_count} colonnes non vides")
        if non_null_count >= threshold:
            # Vérifier si la ligne contient principalement du texte (pas de nombres)
            is_header = all(not pd.to_numeric(cell, errors='coerce') == cell for cell in row if pd.notna(cell))
            if is_header:
                next_row = df.iloc[i + 1] if i + 1 < len(df) else None
                if next_row is not None and next_row.notna().sum() > 1:
                    data_types = [type(cell).__name__ for cell in next_row if pd.notna(cell)]
                    logger.debug(f"Types de données ligne suivante {i+1} : {data_types}")
                    if any(t in ["int64", "float64", "object", "datetime64"] for t in data_types):
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": f"Voici un extrait des lignes autour de la ligne {i} d'un fichier Excel :\n{df.iloc[max(0, i-2):i+3].to_string(index=True)}\nLa ligne {i} a été détectée comme en-tête basée sur des règles (au moins {threshold} colonnes non vides sans nombres, suivies de données). Confirme si c'est correct en vérifiant si elle contient des titres de colonnes cohérents suivis de données. Réponds 'oui' ou 'non'."}],
                                temperature=0
                            )
                            logger.info(f"Validation ChatGPT pour ligne {i} : {response.choices[0].message.content}")
                            if "oui" in response.choices[0].message.content.lower():
                                logger.info(f"En-tête confirmé à la ligne {i} par validation ChatGPT")
                                return i
                            else:
                                logger.warning(f"Validation ChatGPT échouée pour ligne {i}")
                        except openai.RateLimitError as e:
                            logger.error(f"Erreur de limite de taux ChatGPT : {e}. Passage à la validation statistique.")
                            break
    # Fallback statistique si ChatGPT échoue ou n'est pas utilisé
    logger.info("Passage à la détection statistique en raison de l'échec ou absence de ChatGPT")
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        non_null_count = row.notna().sum()
        if non_null_count >= threshold and i + 1 < len(df):
            next_row = df.iloc[i + 1]
            if next_row.notna().sum() > 1 and any(pd.to_numeric(cell, errors='coerce') is not pd.NaT or isinstance(cell, str) for cell in next_row if pd.notna(cell)):
                logger.info(f"En-tête détecté à la ligne {i} par détection statistique")
                return i
    # Fallback final à ChatGPT avec extrait réduit
    logger.info("Passage au fallback ChatGPT avec extrait réduit")
    try:
        header_row_idx = gpt_detect_header_row(df, max_rows=15)
        if header_row_idx >= 0:
            logger.info(f"En-tête détecté à la ligne {header_row_idx} par fallback ChatGPT")
            return header_row_idx
    except openai.RateLimitError as e:
        logger.error(f"Erreur de limite de taux ChatGPT dans fallback : {e}. Utilisation de 0.")
    logger.warning(f"Aucun en-tête valide détecté dans les {max_rows} premières lignes, utilisation de la ligne 0.")
    return 0
    
def gpt_detect_header_row(df, max_rows=15):
    logger.info(f"Début de la détection d'en-tête par ChatGPT sur {min(max_rows, len(df))} lignes")
    excerpt = df.head(min(max_rows, len(df))).to_string(index=True)
    logger.debug(f"Extrait envoyé à ChatGPT : \n{excerpt}")
    prompt = (
        f"Voici un extrait des premières lignes d'un fichier Excel (index de ligne inclus) :\n"
        f"{excerpt}\n"
        "Identifie la ligne qui contient les en-têtes du tableau. Les en-têtes sont généralement la première ligne avec des titres de colonnes cohérents "
        "suivie de données cohérentes (textes, nombres, etc.). Ignore les lignes avec des titres généraux ou des descriptions. "
        "Réponds uniquement avec le numéro de la ligne (index basé sur 0)."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        logger.info(f"Réponse brute de ChatGPT : {response.choices[0].message.content}")
        header_row_idx = int(response.choices[0].message.content.strip())
        if 0 <= header_row_idx < min(max_rows, len(df)):
            logger.info(f"En-tête détecté à la ligne {header_row_idx} par ChatGPT")
            return header_row_idx
        else:
            logger.warning(f"Indice d'en-tête {header_row_idx} invalide, utilisation de 0.")
            return 0
    except openai.RateLimitError as e:
        logger.error(f"Erreur de limite de taux ChatGPT : {e}. Retour à 0.")
        return 0
    except (ValueError, IndexError) as e:
        logger.error(f"Erreur parsing réponse ChatGPT pour en-tête : {e}. Réponse brute : {response.choices[0].message.content}")
        return 0

def extract_data_from_excel(file_bytes, columns_map=None):
    logger.info("Début de l'extraction des données du fichier Excel")
    try:
        logger.info("Lecture brute du fichier Excel sans header...")
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
        logger.info(f"Premières lignes brutes lues avec succès : \n{df_raw.head(5).to_string()}")
    except Exception as e:
        logger.error(f"Erreur lors de la lecture brute du fichier Excel : {e}")
        return []

    header_row_idx = find_header_row(df_raw)
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), header=header_row_idx)
        original_columns = [str(col) for col in df.columns]
    except Exception as e:
        logger.error(f"Erreur lors de la lecture avec en-tête : {e}")
        return []

    # Utiliser uniquement columns_map si fourni
    if columns_map and isinstance(columns_map, dict):
        reverse_map = {v: k for k, v in columns_map.items() if v in EXPECTED_FIELDS and k in original_columns}
        logger.info(f"Reverse map fourni par l'utilisateur : {reverse_map}")
    else:
        mapping = infer_field_mapping([col.lower() for col in original_columns])
        if not (mapping and any(mapping.get(k) for k in EXPECTED_FIELDS)):
            mapping = gpt_map_columns([col.lower() for col in original_columns])
        reverse_map = {mapping.get(k, k): k for k in EXPECTED_FIELDS if mapping.get(k) is not None}
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
        # Ne pas inclure la ligne si le champ 'designation' est vide
        if not record.get("designation"):
            continue
        # Skip row if both 'unit' and 'pu' are empty
        if not record.get("unit") and not record.get("pu"):
            continue
        records.append(record)

    logger.info(f"Nombre de lignes extraites : {len(records)}")
    if records:
        logger.info(f"Exemple de lignes extraites : {records[:3]}")
    else:
        logger.warning("Aucune donnée extraite du fichier Excel.")

    return records

def extract_columns_and_reverse_map(file_bytes):
    logger.info("Début de l'extraction des colonnes pour mapping manuel")
    try:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    except Exception as e:
        logger.error(f"Erreur lors de la lecture brute du fichier Excel : {e}")
        return {}

    header_row_idx = find_header_row(df_raw)
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), header=header_row_idx)
        original_columns = [str(col) for col in df.columns]  # Keep original case
    except Exception as e:
        logger.error(f"Erreur lors de la lecture avec en-tête : {e}")
        return {}

    mapping = infer_field_mapping([col.lower() for col in original_columns])
    if not (mapping and any(mapping.get(k) for k in EXPECTED_FIELDS)):
        mapping = gpt_map_columns([col.lower() for col in original_columns])

    # Build result: for each column, if it matches a mapped value, show the field, else empty
    result = {}
    for col in original_columns:
        mapped_field = ""
        for field, mapped_col in mapping.items():
            if mapped_col and mapped_col.lower() == col.lower():
                mapped_field = field
                break
        result[col] = mapped_field

    logger.info(f"JSON mapping manuel proposé : {result}")
    return result

def dpgf_extract_data_from_excel(file_bytes, columns_map=None, user_id=None):
    logger.info("Début de l'extraction des données du fichier Excel")
    try:
        # Charger le fichier Excel une seule fois sans header
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
        logger.info(f"Premières lignes brutes lues avec succès : \n{df_raw.head(5).to_string()}")
    except Exception as e:
        logger.error(f"Erreur lors de la lecture brute du fichier Excel : {e}")
        return []

    # Détecter l'index de l'en-tête
    header_row_idx = find_header_row(df_raw)
    # Appliquer l'en-tête directement sur le DataFrame brut
    df_raw.columns = df_raw.iloc[header_row_idx]
    df = df_raw.drop(range(header_row_idx + 1)).reset_index(drop=True)
    original_columns = [str(col) for col in df.columns]

    # Utiliser uniquement columns_map si fourni
    if columns_map and isinstance(columns_map, dict):
        reverse_map = {v: k for k, v in columns_map.items() if v in EXPECTED_FIELDS and k in original_columns}
        logger.info(f"Reverse map fourni par l'utilisateur : {reverse_map}")
    else:
        mapping = infer_field_mapping([col.lower() for col in original_columns])
        if not (mapping and any(mapping.get(k) for k in EXPECTED_FIELDS)):
            mapping = gpt_map_columns([col.lower() for col in original_columns])
        reverse_map = {mapping.get(k, k): k for k in EXPECTED_FIELDS if mapping.get(k) is not None}
        logger.info(f"Reverse map utilisé : {reverse_map}")

    if not reverse_map:
        logger.error("Aucun champ attendu mappé, vérifiez les en-têtes ou les synonymes.")
        return []

    designation_cache = {}
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
        # Ajout du champ quantity si présent
        quantity_col = reverse_map.get("quantity") if "quantity" in reverse_map else None
        if quantity_col and quantity_col in row and pd.notna(row[quantity_col]):
            try:
                record["quantity"] = float(row[quantity_col])
            except Exception:
                record["quantity"] = row[quantity_col]
        else:
            record["quantity"] = 1.0  # Valeur par défaut si non présente

        # Ne pas inclure la ligne si le champ 'designation' est vide
        if not record.get("designation"):
            continue
        # Skip row if both 'unit' and 'pu' are empty
        if not record.get("unit") and not record.get("pu"):
            continue

        # Recherche vectorielle dans BipArticle avec cache
        designation = record.get("designation")
        bip_match = None
        if designation and designation.strip():
            if designation in designation_cache:
                bip_match = designation_cache[designation]
            else:
                bip_match = search_documents(user_id, designation)
                designation_cache[designation] = bip_match
        record["bip_match"] = bip_match if bip_match else None

        # Calcul du coût total si bip_match trouvé et quantity présente
        pu_bip = bip_match.get("pu") if bip_match and "pu" in bip_match else None
        try:
            pu_bip = float(pu_bip) if pu_bip is not None else 0.0
        except Exception:
            pu_bip = 0.0
        try:
            quantity = float(record.get("quantity", 1.0))
        except Exception:
            quantity = 1.0
        record["cout_total"] = pu_bip * quantity

        records.append(record)

    logger.info(f"Nombre de lignes extraites : {len(records)}")
    if records:
        logger.info(f"Exemple de lignes extraites : {records[:3]}")
    else:
        logger.warning("Aucune donnée extraite du fichier Excel.")

    return records
