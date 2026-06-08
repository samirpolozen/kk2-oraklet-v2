from typing import ClassVar

from transformers import pipeline

from app.chain.runnable import Runnable
from app.schemas import (
    LLMRunnerOutput,
    PromptBuilderInput,
    PromptBuilderOutput,
    ResponseParserOutput,
)


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

 # Promtbuilder är vårt första steg i kedjan, den förbereder datan för modellen 
 # Tar emot frågan om dataset statistiken och bygger en strukturerad promt
class PromptBuilder(Runnable[PromptBuilderInput, PromptBuilderOutput]):
    # Invoke metoden bygger upp en läsbar text av statistiken som senare skickas till modellen   
    def invoke(self, data: PromptBuilderInput) -> PromptBuilderOutput:
        formatted_stats = ""
          
        for column, values in data.stats.items():
            formatted_stats += f"\nColumn: {column}\n"

            if isinstance(values, dict):
                for key, value in values.items():
                    formatted_stats += f"  {key}: {value}\n"
            else:
                formatted_stats += f"  {values}\n"
        # Invoke metoden bygger den slutliga promten med statistik och fråga, tanken är att styra modellen så att den bara använder datan vi skickar in
        prompt = f"""
You are a data analysis assistant.

Use ONLY the dataset statistics provided below.

The column names in the statistics are the available columns in the dataset.

When answering questions:
- Use "mean" when the user asks about average values.
- Use "max" when the user asks about highest or largest values.
- Use "min" when the user asks about lowest or smallest values.
- Use "count" when the user asks how many values exist.
- Use the available column names when matching the user's question.
- If a question refers to temperature, look for columns such as temp_c.
- If a question refers to rain or precipitation, look for columns such as precip.
- If a question refers to sunshine, look for columns such as sun_h.

If the question asks about temperature, only use temp_c.
If the question asks about sunshine, only use sun_h.
If the question asks about precipitation or rain, only use precip.

Important column mapping:
- temperature = temp_c
- rain / precipitation = precip
- sunshine / sun hours = sun_h
- city = city

Do not compare values between different columns.
Only use the column that matches the user's question.

Do not invent information.
Do not use knowledge outside the statistics.

If the answer cannot be determined from the data,
say exactly: "Not enough information."


Dataset statistics:
{formatted_stats}

User question:
{data.question}

Answer:
"""

        return PromptBuilderOutput(
            question=data.question,
            prompt=prompt.strip(),
        )

# Andra steget i Runnable kedjan
class LLMRunner(Runnable[PromptBuilderOutput, LLMRunnerOutput]):
    
    generator: ClassVar = None
    
    # Modellen laddas bara första gången den används, för att slippa ladda om den för varje fråga
    def _load_generator(self) -> None:
        if LLMRunner.generator is None:
            LLMRunner.generator = pipeline(
                "text-generation",
                model=MODEL_NAME,
            )
    
    # Invoke metoden skikckar promten till modellen och tar emot svaret
    def invoke(self, data: PromptBuilderOutput) -> LLMRunnerOutput:
        try:
            self._load_generator()

            result = LLMRunner.generator(
                data.prompt,
                max_new_tokens=150,
                return_full_text=False,
                do_sample=False,
            )

            raw_output = result[0]["generated_text"].strip()

            return LLMRunnerOutput(
                question=data.question,
                raw_output=raw_output,
                model=MODEL_NAME,
            )
        
        # Om något blir fel fångas felet upp så att API: et inte kraschar
        except Exception as error:
            return LLMRunnerOutput(
                question=data.question,
                raw_output=f"Model error: {str(error)}",
                model=MODEL_NAME,
            )

# Tredje steget i Runnable kedjan
class ResponseParser(Runnable[LLMRunnerOutput, ResponseParserOutput]):
    # Sista invoke metoden städar bort modellens råa svar innan det skickas tillbaka
    def invoke(self, data: LLMRunnerOutput) -> ResponseParserOutput:
        text = data.raw_output.strip()

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            answer = "No answer generated."
        else:
            answer = lines[0]

        prefixes = ["Answer:", "A:", "Response:", "-", "*"]

        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()

        answer = " ".join(answer.split())

        if not answer:
            answer = "No answer generated."
        
        # Retunerar det färdiga svaret till nästa steg eller API: et
        return ResponseParserOutput(
            question=data.question,
            answer=answer,
            model=data.model,
        )