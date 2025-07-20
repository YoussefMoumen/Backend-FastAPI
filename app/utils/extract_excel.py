import io
import pandas as pd
import difflib

# Define the expected fields
EXPECTED_FIELDS = ["designation", "unit", "pu", "lot"]

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
    df.columns = [col.strip().lower() for col in df.columns]
    mapping = infer_column_mapping(df.columns)
    records = []
    for _, row in df.iterrows():
        record = {}
        for field, col in mapping.items():
            record[field] = row[col] if col and col in row else ""
        records.append(record)
    return records
