import requests

# Exemple pour uploader un BIP
files = {"file": open("bip.pdf", "rb")}
data = {"user_id": "testuser"}
resp = requests.post("http://localhost:8000/upload_bip", files=files, data=data)
print(resp.json())
