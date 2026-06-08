from fastapi.testclient import TestClient

from app import data
from app.main import app


client = TestClient(app)

# Testar att /ai/ask inte körs om inget dataset har laddats upp
# AI kedjan använder statistik från datasetet ock ska därför inte köras innan en csv fil har laddats upp
def test_ask_ai_without_dataset_returns_400():
    
    # Simulerar ett tomt dataset i minnet
    data.dataset = None

    response = client.post(
        "/ai/ask",
        json={"question": "What is the mean value of temp_c?"},
    )
    
    # Verifiera att vi får tillbaka rätt HTTP statuskod
    assert response.status_code == 400

    # Verifierar ett tydligt felmeddelande
    assert response.json()["detail"] == "No dataset has been uploaded."


# Testar att /data/upload stoppar filer som inte är CSV
# Detta skyddar endpointen från fel filtyp redan innan Pandas försöker läsa filen
def test_upload_rejects_non_csv_file():
    
    # Simulerar att en textfil laddats upp istället för csv fil
    response = client.post(
        "/data/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    
    # Verifierar att API : et stoppar filen med rätt statuskod
    assert response.status_code == 400
    
    # Till sist att rätt felmeddelande retuneras till klienten
    assert response.json()["detail"] == "Only CSV files are allowed."