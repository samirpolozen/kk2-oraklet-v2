"""
Huvudfilen för FastAPi applikation, här skapas API: et och alla endpoints som används,
ladda upp data, hämta statestik och frågor till AI modellen.
"""

from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from app import data

from app.chain.pipeline import ai_pipeline
# from app.schemas import DatasetMetadata, kollar på den sen
from app.schemas import (
    DatasetMetadata,
    PromptBuilderInput,
    QuestionRequest,
    QuestionResponse,
)

# Skapar FastAPi där alla endpoints kopplas till
app = FastAPI()

# Hälsokontroll för att säkerställa att API: et körs
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# Tar emot csv fil och läser in med Pandas och sparar datasetet i minnet
# Retunerar metadata om datasetet som laddas upp
@app.post("/data/upload", response_model=DatasetMetadata)
async def upload_data(file: UploadFile = File(...)) -> DatasetMetadata:
    # Endast csv filer accepteras
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    # Läs innehållet från uppladad fil
    content = await file.read()
    # Edge case,  stoppa tomma filer innan Pandas läser den
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        # Omvandla filens bytes till text som kan läsas av Pandas
        decoded_content = content.decode("utf-8")
        #Läser csv filen och skapar Pandas DataFrame
        df = pd.read_csv(StringIO(decoded_content))
    # Edge case, fel teckenkodning    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV file must use UTF-8 encoding.")
    # Edge case, filen finns men ingen läsbar csv data
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file contains no readable data.")
    # Fångar övriga fel vid inläsning av csv filen
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV file.")
    # Sparar datasetet
    data.dataset = df
    # Grundläggande metadata om datasetet
    return DatasetMetadata(
        rows=len(df),
        columns=list(df.columns),
        dtypes={column: str(dtype) for column, dtype in df.dtypes.items()},
    )

# Hämtar statistik från datasetet som har laddats upp
@app.get("/data/stats")
def get_stats():
    if data.dataset is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset has been uploaded."
        )

    return data.dataset.describe(include="all").fillna("").to_dict()

# Tar emot en fråga från användaren och skickar den genom AI kedjan
# Datasetet måste vara uppladdat först för kedjan använder statistiken
@app.post("/ai/ask", response_model=QuestionResponse)
def ask_ai(request: QuestionRequest) -> QuestionResponse:
    if data.dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset has been uploaded."
        )

    try:
        stats = data.dataset.describe(include="all").fillna("").to_dict()

        chain_input = PromptBuilderInput(
            question=request.question,
            stats=stats,
        )

        result = ai_pipeline.invoke(chain_input)

        return QuestionResponse(
            question=result.question,
            answer=result.answer,
            model=result.model,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed: {str(error)}"
        )