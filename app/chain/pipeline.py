from app.chain.steps import (
    PromptBuilder,
    LLMRunner,
    ResponseParser,
)
# Bygger ihop hela Runnable-kedjan
# Fråga -> Prompt -> Modell -> Färdigt svar

# I denna fil samlas hela AI flödet på ett ställe
# Istället för att kedjan ska byggas i main.py byggs den här, för att logiken blir enklare att återanvända och underhålla
# Varje steg ansvarar för en specifik del av bearbetningen :
# PromtBuilder - tar användarens fråga och datasets statistik och bygger en promt till modellen
# LLMRunner - skickar promten till SmolLM och hämtar svar
# ResponseParser - Städar modellens råa svar till ett läsbart svar

ai_pipeline = (
    PromptBuilder()
    | LLMRunner()
    | ResponseParser()
)