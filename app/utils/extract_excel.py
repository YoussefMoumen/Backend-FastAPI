import pandas as pd

def extract_data_from_excel(file):
    df = pd.read_excel(file)
    return df.astype(str).to_dict(orient="records")
