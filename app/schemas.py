"""
Här är Pydantic modellerna för mitt API,
dem används för att ge struktur och validera data som skickas mellan klienten och API
"""

from pydantic import BaseModel, Field

from typing import Any

# Metadata om datasetet som returneras efter uppladdning
class DatasetMetadata(BaseModel):
    rows: int
    columns: list[str]
    dtypes: dict[str, str]

# Modellerna nedan används för /ai/ask endpointen.
# Dessa två beskriver frågan som kommer in och svaret som API:et skickar tillbaka.
class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)

# Används av API: et mot klienten
class QuestionResponse(BaseModel):
    question: str
    answer: str
    model: str


# Dessa två modeller används i början av Runnable kedjan.
# PromptBuilder tar fråga och statistiken och bygger en färdig prompt till AI.
class PromptBuilderInput(BaseModel):
    question: str = Field(min_length=1)
    stats: dict[str, Any]


class PromptBuilderOutput(BaseModel):
    question: str
    prompt: str


# Dessa två modeller används efter att AI modellen har körts.
# LLMRunnerOutput är modellens råa svar och ResponseParserOutput är färdiga svaret.
class LLMRunnerOutput(BaseModel):
    question: str
    raw_output: str
    model: str

# Används internt i Runnable kedjan, efter att råsvaret har parsats.
class ResponseParserOutput(BaseModel):
    question: str
    answer: str
    model: str