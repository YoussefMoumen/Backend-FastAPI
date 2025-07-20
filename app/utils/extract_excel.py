import io
import pandas as pd

def extract_data_from_excel(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df.astype(str).to_dict(orient="records")
