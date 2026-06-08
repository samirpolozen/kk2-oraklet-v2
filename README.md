# KK2 - oraklet

En FastApi applikation som kombinerar dataanalys med Pandas, Pydantic och AI genom en lokal språkmodell 'SmolLM'.

## Projektbeskrivning

Syftet med projektet är att skapa ett Rest API bland annat kan: 

- Ladda upp och analysera CSV filer med Pandas
- Generera beskrivande statistik
- Använda SmolLM för att generera svar
- Bearbeta hela flödet genom en typad Runnable kedja

AI flödet består av en Runnable kedja med tre steg:

```text
PromtBuilder -> LLMRunner -> ResponseParser
```

Varje steg använder Pydantic modeller för typad indata och utdata.
---

## Installation

1. Klona projektet:

```bash
git clone <repository-url>
cd kk2-oraklet-v2
```

2. Installera beroenden:

```bash
uv sync
```

3. Starta aplikationen: 

```bash
uv run uvicorn app.main:app --reload
```

4. Öppna Swagger:

```text
http://127.0.0.1:8000/docs
```
---

## Testning

Alla tester kör med 

```bash
uv run pytest app/tests/ -v
```

Projektet innehåller tester för:
- Runnable steg
- Endpoint validering
- Felhantering
- Mockad AI kedja
---

## API endpoints

### Get /health
Kontrollerar att API:t är igång.

Exempel på svar:

```json
{
  "status": "ok"
}
```

### POST /data/upload
Laddar upp ett CSV-dataset och returnerar metadata.

###  GET /data/stats
Retunerar beskrivande statistik från det uppladdande datasetet.

### POST /ai/ask
Tar emot en fråga och skickar den genom Runnable-kedjan.

Exempel på en förfrågan:

```json
{
  "question": "What is the mean value of temp_c?"
}
```

Exempel på svar:

```json
{
  "question": "What is the mean value of temp_c?",
  "answer": "The mean value of temp_c is 6.4.",
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct"
}
```
---

## Projektstruktur 

```text
app/
├── chain/
│   ├── runnable.py
│   ├── steps.py
│   └── pipeline.py
├── tests/
│   ├── test_chain.py
│   └── test_endpoints.py
├── data.py
├── main.py
└── schemas.py
```
---

## Antaganden 

- Dataset lagras temporärt i minnet
- Endast CSV filer accepteras 
- SmolLm körs lokalt via transformers.pipeline
- Applikationen är en prototyp för utbildningssyfte och inte avseed för ex personuppgifter eller produktion.
---

## Tekniker 

- Python
- FastAPI
- Pandas
- Pydantic
- Transformers
- Pytest
- SmolLM32-135-Instruct

